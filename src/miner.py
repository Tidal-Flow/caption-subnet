import os
import sys

# Add system Python path at the beginning to prioritize built-in modules
sys.path.insert(0, '/home/user/miniconda3/lib/python3.12')
import time
import argparse
import traceback
import base64
import bittensor as bt
import whisper
import torch
import io
from datasets import load_dataset
from protocol import CaptionSynapse, CaptionSegment
import numpy as np


class CaptionMiner:
    def __init__(self):
        self.config = self.get_config()
        self.setup_logging()
        self.setup_bittensor_objects()
        
        # Load the Whisper model
        bt.logging.info("Loading Whisper model...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.stt_model = whisper.load_model(self.config.whisper_model, device=device)
        bt.logging.info(f"Whisper model loaded on {device}")

    def get_config(self):
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--netuid", type=int, default=1, help="The chain subnet uid."
        )
        parser.add_argument(
            "--whisper_model", 
            type=str, 
            default="base", 
            choices=["tiny", "base", "small", "medium", "large"],
            help="Whisper model size to use for transcription"
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
        bt.axon.add_args(parser)
        config = bt.config(parser)
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
            f"Running CaptionMiner for subnet: {self.config.netuid} on network: {self.config.subtensor.network} with config:"
        )
        bt.logging.info(self.config)

    def setup_bittensor_objects(self):
        bt.logging.info("Setting up Bittensor objects.")
        self.wallet = bt.wallet(config=self.config)
        self.subtensor = bt.subtensor(config=self.config)
        self.metagraph = self.subtensor.metagraph(self.config.netuid)

        if self.wallet.hotkey.ss58_address not in self.metagraph.hotkeys:
            bt.logging.error(
                f"Your miner: {self.wallet} is not registered to chain connection: {self.subtensor} \nRun 'btcli register' and try again."
            )
            exit()
        else:
            self.my_subnet_uid = self.metagraph.hotkeys.index(
                self.wallet.hotkey.ss58_address
            )
            bt.logging.info(f"Running miner on uid: {self.my_subnet_uid}")

    def decode_audio(self, audio_data: str) -> np.ndarray:
        """Decodes base64 audio data to numpy array"""
        try:
            audio_bytes = base64.b64decode(audio_data)
            # Convert to numpy array - assuming 16kHz mono PCM format
            # In practice, you might need to use a library like librosa to handle different formats
            audio_array = np.frombuffer(audio_bytes, dtype=np.float32)
            return audio_array
        except Exception as e:
            bt.logging.error(f"Error decoding audio: {e}")
            return np.zeros(1000, dtype=np.float32)  # Return empty audio on error

    def handle_caption_request(self, synapse: CaptionSynapse) -> CaptionSynapse:
        """Process the audio data and return a transcription"""
        try:
            bt.logging.info(f"Received transcription request for audio_id: {synapse.audio_id}")
            
            # Decode the audio data
            audio_array = self.decode_audio(synapse.audio_data)
            
            # Transcribe using Whisper
            bt.logging.info("Transcribing with Whisper...")
            result = self.stt_model.transcribe(audio_array, language=synapse.metadata.get('language', None))
            
            # Extract the transcription text and timing information
            text = result["text"].strip()
            
            # Create segment with timing info if available
            if 'segments' in result and len(result['segments']) > 0:
                start_time = result['segments'][0]['start']
                end_time = result['segments'][-1]['end']
                segment = CaptionSegment(
                    text=text,
                    start_time=start_time,
                    end_time=end_time
                )
                bt.logging.info(f"Generated caption: {segment}")
            else:
                segment = CaptionSegment(text=text, start_time=0.0, end_time=0.0)
                bt.logging.info(f"Generated caption without timing info: {segment.text}")
            
            # Set the transcription in the synapse
            synapse.transcription = segment.text
            return synapse
            
        except Exception as e:
            bt.logging.error(f"Error processing caption request: {str(e)}")
            bt.logging.error(traceback.format_exc())
            # Return empty caption on error
            synapse.transcription = ""
            return synapse

    def blacklist_fn(self, synapse: CaptionSynapse) -> bool:
        """Function to determine if a request should be blacklisted."""
        # Implement basic validation to prevent abuse
        if not synapse.audio_data or len(synapse.audio_data) < 10:  # Very basic check
            return True
        return False

    def setup_axon(self):
        self.axon = bt.axon(wallet=self.wallet, config=self.config)
        self.axon.attach(
            forward_fn=self.handle_caption_request,
            blacklist_fn=self.blacklist_fn,
        )
        self.axon.serve(netuid=self.config.netuid, subtensor=self.subtensor)
        self.axon.start()

    def run(self):
        self.setup_axon()
        bt.logging.info("Starting CaptionMiner main loop")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            self.axon.stop()
            bt.logging.success("Miner killed by keyboard interrupt.")
        except Exception as e:
            bt.logging.error(f"Exception in miner loop: {str(e)}")
            bt.logging.error(traceback.format_exc())


# Run the miner.
if __name__ == "__main__":
    miner = CaptionMiner()
    miner.run()