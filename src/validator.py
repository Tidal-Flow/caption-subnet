import os
import time
import random
import argparse
import traceback
import base64
import bittensor as bt
from protocol import CaptionSegment, CaptionSynapse
from typing import List, Dict, Tuple
from datasets import load_dataset
import numpy as np
import torch
import jiwer
import io


class CaptionValidator:
    def __init__(self):
        self.config = self.get_config()
        self.setup_logging()
        self.setup_bittensor_objects()
        self.setup_dataset()
        self.scores = {}  # Dictionary to store scores by hotkey
        self.alpha = 0.05  # Learning rate for updating scores

    def get_config(self):
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--netuid", type=int, default=1, help="The chain subnet uid."
        )
        parser.add_argument(
            "--dataset_lang", 
            type=str, 
            default="multilang", 
            help="VoxPopuli dataset language to use (e.g., 'hr', 'multilang', 'en_accented')"
        )
        parser.add_argument(
            "--dataset_split", 
            type=str, 
            default="validation", 
            choices=["train", "validation", "test"],
            help="Dataset split to use"
        )
        parser.add_argument(
            "--sampling_interval", 
            type=int, 
            default=10, 
            help="Interval in seconds between sampling miners"
        )
        parser.add_argument(
        "--subtensor.network",
        type=str,
        default="local",
        help="Bittensor network to connect to (local/finney/test)"
    )
        parser.add_argument(
        "--subtensor.chain_endpoint",
        type=str,
        default="ws://127.0.0.1:9944",
        help="Chain endpoint for local deployment"
    )
        bt.subtensor.add_args(parser)
        bt.logging.add_args(parser)
        bt.wallet.add_args(parser)
        bt.dendrite.add_args(parser)
        config = bt.config(parser)
        return config

    def setup_logging(self):
        bt.logging(config=self.config)
        bt.logging.info(f"Running validator for subnet: {self.config.netuid} with {self.config.dataset_lang} dataset")

    def setup_bittensor_objects(self):
        self.wallet = bt.wallet(config=self.config)
        self.subtensor = bt.subtensor(config=self.config)
        self.metagraph = self.subtensor.metagraph(self.config.netuid)
        self.dendrite = bt.dendrite(wallet=self.wallet)
        
        # Initialize scores dictionary with default values
        for hotkey in self.metagraph.hotkeys:
            self.scores[hotkey] = 1.0

    def setup_dataset(self):
        """Load the VoxPopuli dataset"""
        bt.logging.info(f"Loading VoxPopuli dataset ({self.config.dataset_lang})...")
        try:
            self.dataset = load_dataset("facebook/voxpopuli", self.config.dataset_lang, split=self.config.dataset_split)
            bt.logging.info(f"Loaded {len(self.dataset)} samples from the dataset")
        except Exception as e:
            bt.logging.error(f"Error loading dataset: {e}")
            raise

    def preprocess_text(self, text: str) -> str:
        """Preprocess text for better comparison"""
        if not text:
            return ""
        # Convert to lowercase, remove extra spaces
        text = text.lower().strip()
        # Remove punctuation for better WER calculation
        for char in ",.!?;:\"'()[]{}":
            text = text.replace(char, "")
        # Normalize whitespace
        return " ".join(text.split())

    def evaluate_transcription(self, expected: str, actual: str) -> float:
        """
        Evaluate the transcription using Word Error Rate (WER)
        Returns a score between 0 and 1, where 1 is perfect
        """
        if not actual or not expected:
            return 0.0
            
        # Preprocess texts
        expected = self.preprocess_text(expected)
        actual = self.preprocess_text(actual)
        
        if not expected:
            return 0.0
            
        # Calculate WER
        try:
            wer = jiwer.wer(expected, actual)
            # Convert WER to score (1 - WER, clamped to [0, 1])
            score = max(0.0, min(1.0, 1.0 - wer))
            return score
        except Exception as e:
            bt.logging.error(f"Error calculating WER: {e}")
            return 0.0

    def encode_audio(self, audio_array: np.ndarray) -> str:
        """Encode audio array to base64 string"""
        try:
            # Convert to bytes
            audio_bytes = audio_array.tobytes()
            # Encode to base64
            audio_b64 = base64.b64encode(audio_bytes).decode('utf-8')
            return audio_b64
        except Exception as e:
            bt.logging.error(f"Error encoding audio: {e}")
            return ""

    def get_random_sample(self) -> Tuple[Dict, str]:
        """Get a random sample from the dataset"""
        sample_idx = random.randint(0, len(self.dataset) - 1)
        sample = self.dataset[sample_idx]
        
        # Get the audio data
        audio_array = sample['audio']['array']
        audio_b64 = self.encode_audio(audio_array)
        
        # Get the ground truth transcription
        ground_truth = sample['normalized_text']
        
        metadata = {
            'language': sample.get('language', 'unknown'),
            'sampling_rate': sample['audio']['sampling_rate']
        }
        
        return {
            'audio_data': audio_b64,
            'audio_id': sample['audio_id'],
            'metadata': metadata
        }, ground_truth

    def query_miners(self, sample: Dict, expected_transcription: str):
        """Query all miners with the sample and evaluate their responses"""
        bt.logging.info(f"Querying miners with audio_id: {sample['audio_id']}")
        
        # Create the synapse
        synapse = CaptionSynapse(
            audio_data=sample['audio_data'],
            audio_id=sample['audio_id'],
            metadata=sample['metadata']
        )
        
        # Get active axons from the metagraph
        axons = self.metagraph.axons
        
        # Query all miners
        responses = self.dendrite.query(
            axons=axons,
            synapse=synapse,
            deserialize=True,
            timeout=10.0
        )
        
        # Process responses and update scores
        for i, response in enumerate(responses):
            if response is not None and hasattr(response, 'transcription') and response.transcription:
                hotkey = self.metagraph.hotkeys[i]
                score = self.evaluate_transcription(expected_transcription, response.transcription)
                
                # Update the score with exponential moving average
                if hotkey in self.scores:
                    self.scores[hotkey] = (1 - self.alpha) * self.scores[hotkey] + self.alpha * score
                else:
                    self.scores[hotkey] = score
                    
                bt.logging.info(f"Miner {hotkey}: score={score:.4f}, new_avg={self.scores[hotkey]:.4f}")

    def update_weights(self):
        """Update the weights on the Bittensor network"""
        bt.logging.info("Updating weights on the network")
        
        # Convert scores dictionary to weights vector
        weights = torch.zeros(len(self.metagraph.hotkeys))
        
        for i, hotkey in enumerate(self.metagraph.hotkeys):
            weights[i] = self.scores.get(hotkey, 0)
        
        # Normalize weights
        if torch.sum(weights) > 0:
            weights = weights / torch.sum(weights)
        
        # Set weights
        try:
            self.subtensor.set_weights(
                netuid=self.config.netuid,
                wallet=self.wallet,
                uids=list(range(len(self.metagraph.hotkeys))),
                weights=weights,
                wait_for_inclusion=False
            )
            bt.logging.success("Successfully set weights")
        except Exception as e:
            bt.logging.error(f"Failed to set weights: {e}")

    def run(self):
        """Main validator loop"""
        bt.logging.info("Starting validator loop")
        
        try:
            while True:
                try:
                    # Get a random sample
                    sample, ground_truth = self.get_random_sample()
                    
                    # Query miners with the sample
                    self.query_miners(sample, ground_truth)
                    
                    # Update weights
                    self.update_weights()
                    
                    # Sleep before next round
                    time.sleep(self.config.sampling_interval)
                    
                except Exception as e:
                    bt.logging.error(f"Error in validation loop: {e}")
                    bt.logging.error(traceback.format_exc())
                    time.sleep(10)  # Wait before retrying
                    
        except KeyboardInterrupt:
            bt.logging.success("Validator stopped by keyboard interrupt")


# Run the validator
if __name__ == "__main__":
    validator = CaptionValidator()
    validator.run()