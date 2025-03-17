import os
import time
import argparse
import traceback
import bittensor as bt
import pandas as pd
import numpy as np
import uuid
import base64
import io
import soundfile as sf
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional
import threading
import torch
import asyncio
from transformers import WhisperProcessor, WhisperForConditionalGeneration

from protocol import STTSynapse

class STTMiner:
    def __init__(self):
        self.config = self.get_config()
        self.setup_logging()
        self.setup_bittensor_objects()
        
        # Job management - Initialize this BEFORE setup_job_database()
        self.job_lock = threading.Lock()
        
        self.setup_job_database()
        self.setup_whisper_model()
        
        # Miner state
        self.my_subnet_uid = self.metagraph.hotkeys.index(
            self.wallet.hotkey.ss58_address
        )

    def get_config(self):
        parser = argparse.ArgumentParser()
        
        # Add STT-specific arguments
        parser.add_argument(
            "--miner.csv_path", 
            type=str, 
            default="./miner_jobs.csv",
            help="Path to store the miner jobs CSV"
        )
        parser.add_argument(
            "--miner.whisper_model", 
            type=str, 
            default="openai/whisper-base",
            help="Whisper model to use for transcription"
        )
        parser.add_argument(
            "--miner.device", 
            type=str, 
            default="cuda" if torch.cuda.is_available() else "cpu",
            help="Device to run the model on (cuda/cpu)"
        )
        
        # Network and subnet arguments
        parser.add_argument(
            "--netuid", type=int, default=1, help="The chain subnet uid."
        )
        
        # Add standard Bittensor arguments
        bt.subtensor.add_args(parser)
        bt.logging.add_args(parser)
        bt.wallet.add_args(parser)
        bt.axon.add_args(parser)
        
        # Parse the config
        config = bt.config(parser)
        
        # Set up logging directory
        config.full_path = os.path.expanduser(
            "{}/{}/{}/netuid{}/{}".format(
                config.logging.logging_dir,
                config.wallet.name,
                config.wallet.hotkey_str,
                config.netuid,
                "miner",
            )
        )
        os.makedirs(config.full_path, exist_ok=True)
        return config

    def setup_logging(self):
        bt.logging(config=self.config, logging_dir=self.config.full_path)
        bt.logging.info(
            f"Running STT miner for subnet: {self.config.netuid} on network: {self.config.subtensor.network}"
        )
        bt.logging.info(self.config)

    def setup_bittensor_objects(self):
        bt.logging.info("Setting up Bittensor objects")
        
        # Initialize wallet
        self.wallet = bt.wallet(config=self.config)
        bt.logging.info(f"Wallet: {self.wallet}")
        
        # Initialize subtensor
        self.subtensor = bt.subtensor(config=self.config)
        bt.logging.info(f"Subtensor: {self.subtensor}")
        
        # Initialize metagraph
        self.metagraph = self.subtensor.metagraph(self.config.netuid)
        bt.logging.info(f"Metagraph: {self.metagraph}")
        
        # Verify miner is registered
        if self.wallet.hotkey.ss58_address not in self.metagraph.hotkeys:
            bt.logging.error(
                f"Your miner: {self.wallet} is not registered to chain connection: {self.subtensor} \n"
                f"Run 'btcli register' and try again."
            )
            exit()
        else:
            self.my_subnet_uid = self.metagraph.hotkeys.index(
                self.wallet.hotkey.ss58_address
            )
            bt.logging.info(f"Running miner on uid: {self.my_subnet_uid}")

    def setup_job_database(self):
        csv_path = self.config.miner.csv_path
        
        # Create or load the CSV file
        if os.path.exists(csv_path):
            self.jobs_df = pd.read_csv(csv_path)
            bt.logging.info(f"Loaded existing job database with {len(self.jobs_df)} entries")
        else:
            # Create a new DataFrame with the required columns
            self.jobs_df = pd.DataFrame(columns=[
                'job_id', 'job_status', 'job_accuracy', 'base64_audio', 
                'transcript_miner', 'gender', 'created_at', 'normalized_text',
                'language_miner', 'gender_miner', 'gender_confidence_miner'
            ])
            bt.logging.info("Created new job database")
            
        # Save the initial state
        self.save_jobs_df()

    def setup_whisper_model(self):
        """Initialize the Whisper model for speech recognition"""
        bt.logging.info(f"Loading Whisper model: {self.config.miner.whisper_model}")
        try:
            self.processor = WhisperProcessor.from_pretrained(self.config.miner.whisper_model)
            self.model = WhisperForConditionalGeneration.from_pretrained(self.config.miner.whisper_model)
            
            # Move model to the specified device
            self.model.to(self.config.miner.device)
            
            bt.logging.info(f"Whisper model loaded successfully on {self.config.miner.device}")
        except Exception as e:
            bt.logging.error(f"Failed to load Whisper model: {e}")
            traceback.print_exc()
            exit(1)

    def save_jobs_df(self):
        """Save the jobs DataFrame to CSV"""
        with self.job_lock:
            self.jobs_df.to_csv(self.config.miner.csv_path, index=False)
            bt.logging.debug(f"Saved job database with {len(self.jobs_df)} entries")

    def add_job(self, job_id: str, base64_audio: str, gender: Optional[str] = None):
        """Add a new job to the database"""
        with self.job_lock:
            # Check if job already exists
            if job_id in self.jobs_df['job_id'].values:
                bt.logging.warning(f"Job {job_id} already exists in database")
                return
            
            # Create a new job entry
            job = {
                'job_id': job_id,
                'job_status': 'not_done',
                'job_accuracy': None,
                'base64_audio': base64_audio,
                'transcript_miner': None,
                'gender': gender,
                'created_at': datetime.now().isoformat(),
                'normalized_text': None,  # Miners don't have ground truth
                'language_miner': None,
                'gender_miner': None,
                'gender_confidence_miner': None
            }
            
            # Add to DataFrame
            self.jobs_df = pd.concat([self.jobs_df, pd.DataFrame([job])], ignore_index=True)
            self.save_jobs_df()
            bt.logging.info(f"Added job {job_id} to database")

    def update_job(self, job_id: str, transcript: str, language: str, gender: str, gender_confidence: float):
        """Update a job with transcription results"""
        with self.job_lock:
            job_idx = self.jobs_df[self.jobs_df['job_id'] == job_id].index
            
            if len(job_idx) == 0:
                bt.logging.warning(f"Job {job_id} not found in database")
                return
            
            # Update job fields
            self.jobs_df.loc[job_idx, 'job_status'] = 'done'
            self.jobs_df.loc[job_idx, 'transcript_miner'] = transcript
            self.jobs_df.loc[job_idx, 'language_miner'] = language
            self.jobs_df.loc[job_idx, 'gender_miner'] = gender
            self.jobs_df.loc[job_idx, 'gender_confidence_miner'] = gender_confidence
            
            # Save changes
            self.save_jobs_df()
            bt.logging.info(f"Updated job {job_id} with transcription results")

    def transcribe_audio(self, base64_audio: str) -> Dict[str, Any]:
        """Transcribe audio using Whisper model"""
        try:
            # Decode base64 audio
            audio_bytes = base64.b64decode(base64_audio)
            audio_io = io.BytesIO(audio_bytes)
            
            # Load audio file
            audio_array, sample_rate = sf.read(audio_io)
            
            # Ensure audio is in the correct format for Whisper (mono, float32)
            if len(audio_array.shape) > 1:
                audio_array = audio_array.mean(axis=1)  # Convert to mono
            
            # Process with Whisper
            input_features = self.processor(
                audio_array, 
                sampling_rate=sample_rate, 
                return_tensors="pt"
            ).input_features.to(self.config.miner.device)
            
            # Generate transcription
            with torch.no_grad():
                predicted_ids = self.model.generate(input_features)
            
            # Decode the predicted IDs to text
            transcription = self.processor.batch_decode(
                predicted_ids, 
                skip_special_tokens=True
            )[0]
            
            # For this example, we'll use simple placeholders for language and gender detection
            # In a real implementation, you would use dedicated models for these tasks
            language = "en"  # Placeholder
            gender = "unknown"  # Placeholder
            gender_confidence = 0.5  # Placeholder
            
            return {
                'transcript': transcription,
                'language': language,
                'gender': gender,
                'gender_confidence': gender_confidence
            }
            
        except Exception as e:
            bt.logging.error(f"Error transcribing audio: {e}")
            traceback.print_exc()
            return {
                'transcript': None,
                'language': None,
                'gender': None,
                'gender_confidence': None
            }

    def run(self):
        """Main miner loop"""
        bt.logging.info("Starting miner loop")
        
        # Create and start the axon server
        axon = bt.axon(wallet=self.wallet, config=self.config)
        
        # Attach the STT processing function to the axon
        axon.attach(
            forward_fn=self.process_stt_request,
            blacklist_fn=None,  # No blacklist function for this example
            priority_fn=None,   # No priority function for this example
        )
        
        # Start the axon server
        axon.start()
        bt.logging.info(f"Axon server started on port {self.config.axon.port}")
        
        # Keep the miner running
        try:
            while True:
                # Periodically update the metagraph
                self.metagraph.sync()
                bt.logging.info(f"Block: {self.metagraph.block.item()} | Miners: {len(self.metagraph.axons)}")
                
                # Sleep to prevent overwhelming the network
                time.sleep(60)
                
        except KeyboardInterrupt:
            bt.logging.info("Keyboard interrupt detected, exiting")
        except Exception as e:
            bt.logging.error(f"Error in miner loop: {e}")
            traceback.print_exc()
        finally:
            # Stop the axon server
            axon.stop()
            bt.logging.info("Axon server stopped")

    def process_stt_request(self, synapse: STTSynapse) -> STTSynapse:
        """Process a speech-to-text request from a validator"""
        bt.logging.info(f"Received STT request for job {synapse.job_id}")
        
        try:
            # Record the start time for performance tracking
            start_time = time.time()
            
            # Add the job to our database
            self.add_job(synapse.job_id, synapse.base64_audio, synapse.gender)
            
            # Transcribe the audio
            result = self.transcribe_audio(synapse.base64_audio)
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            # Update the job in our database
            self.update_job(
                synapse.job_id,
                result['transcript'],
                result['language'],
                result['gender'],
                result['gender_confidence']
            )
            
            # Fill in the response fields
            synapse.transcript = result['transcript']
            synapse.language_detected = result['language']
            synapse.gender_detected = result['gender']
            synapse.gender_confidence = result['gender_confidence']
            synapse.processing_time = processing_time
            
            bt.logging.info(f"Completed STT request for job {synapse.job_id} in {processing_time:.2f}s")
            
        except Exception as e:
            bt.logging.error(f"Error processing STT request: {e}")
            traceback.print_exc()
            synapse.error = str(e)
        
        return synapse

if __name__ == "__main__":
    miner = STTMiner()
    miner.run()
