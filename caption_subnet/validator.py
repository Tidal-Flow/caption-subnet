import os
import time
import random
import argparse
import traceback
import bittensor as bt
import pandas as pd
import numpy as np
import uuid
import base64
import datasets
import io
import soundfile as sf
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional
import jiwer
import asyncio
import threading

from protocol import STTSynapse

class STTValidator:
    def __init__(self):
        self.config = self.get_config()
        self.setup_logging()
        self.setup_bittensor_objects()
        self.setup_datasets()
        
       
        self.setup_job_database()
        
        # Validator state
        self.my_uid = self.metagraph.hotkeys.index(self.wallet.hotkey.ss58_address)
        self.scores = [1.0] * len(self.metagraph.S)
        self.last_update = self.subtensor.blocks_since_last_update(
            self.config.netuid, self.my_uid
        )
        self.tempo = self.subtensor.tempo(self.config.netuid)
        self.moving_avg_scores = [1.0] * len(self.metagraph.S)
        self.alpha = 0.1
        
        # Job rotation threshold
        self.job_rotation_threshold = 2  # Replace jobs after this many completions

    def get_config(self):
        parser = argparse.ArgumentParser()
        
        # Add STT-specific arguments
        parser.add_argument(
            "--validator.dataset_size", 
            type=int, 
            default=100,
            help="Number of samples to load from the dataset"
        )
        parser.add_argument(
            "--validator.csv_path", 
            type=str, 
            default="./validator_jobs.csv",
            help="Path to store the validator jobs CSV"
        )
        parser.add_argument(
            "--validator.job_batch_size", 
            type=int, 
            default=5,
            help="Number of jobs to process in each batch"
        )
        
        # Network and subnet arguments
        parser.add_argument(
            "--netuid", type=int, default=1, help="The chain subnet uid."
        )
        
        # Add standard Bittensor arguments
        bt.subtensor.add_args(parser)
        bt.logging.add_args(parser)
        bt.wallet.add_args(parser)
        
        # Parse the config
        config = bt.config(parser)
        
        # Set up logging directory
        config.full_path = os.path.expanduser(
            "{}/{}/{}/netuid{}/validator".format(
                config.logging.logging_dir,
                config.wallet.name,
                config.wallet.hotkey_str,
                config.netuid,
            )
        )
        os.makedirs(config.full_path, exist_ok=True)
        return config

    def setup_logging(self):
        bt.logging(config=self.config, logging_dir=self.config.full_path)
        bt.logging.info(
            f"Running STT validator for subnet: {self.config.netuid} on network: {self.config.subtensor.network}"
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
        
        # Initialize dendrite
        self.dendrite = bt.dendrite(wallet=self.wallet)
        bt.logging.info(f"Dendrite: {self.dendrite}")
        
        # Initialize metagraph
        self.metagraph = self.subtensor.metagraph(self.config.netuid)
        bt.logging.info(f"Metagraph: {self.metagraph}")
        
        # Verify validator is registered
        if self.wallet.hotkey.ss58_address not in self.metagraph.hotkeys:
            bt.logging.error(
                f"Your validator: {self.wallet} is not registered to chain connection: {self.subtensor} \n"
                f"Run 'btcli register' and try again."
            )
            exit()
        else:
            self.my_subnet_uid = self.metagraph.hotkeys.index(
                self.wallet.hotkey.ss58_address
            )
            bt.logging.info(f"Running validator on uid: {self.my_subnet_uid}")

    def setup_datasets(self):
        bt.logging.info("Loading VoxPopuli dataset")
        try:
            # Load a subset of the VoxPopuli dataset
            self.dataset = datasets.load_dataset(
                "facebook/voxpopuli", 
                "en", 
                split=f"train[:{self.config.validator.dataset_size}]"
            )
            bt.logging.info(f"Loaded {len(self.dataset)} samples from VoxPopuli")
        except Exception as e:
            bt.logging.error(f"Failed to load dataset: {e}")
            traceback.print_exc()
            exit(1)

    def setup_job_database(self):
        csv_path = self.config.validator.csv_path
        
        # Create or load the CSV file
        if os.path.exists(csv_path):
            self.jobs_df = pd.read_csv(csv_path)
            bt.logging.info(f"Loaded existing job database with {len(self.jobs_df)} entries")
        else:
            # Create a new DataFrame with the required columns
            self.jobs_df = pd.DataFrame(columns=[
                'job_id', 'job_status', 'job_accuracy', 'base64_audio', 
                'transcript_miner', 'gender', 'created_at', 'normalized_text',
                'language_miner', 'gender_miner', 'gender_confidence_miner',
                'miner_hotkey', 'dataset_index'  # Added miner_hotkey column
            ])
            bt.logging.info("Created new job database")
            
            # Initialize with jobs from the dataset
            self.add_new_jobs(self.config.validator.job_batch_size * 2)
            
        # Save the initial state
        self.save_jobs_df()

    def add_new_jobs(self, num_jobs: int):
        """Add new jobs from the dataset to the jobs DataFrame"""
        with self.job_lock:
            # Get indices of samples not yet in the jobs_df
            used_indices = set()
            if 'dataset_index' in self.jobs_df.columns:
                used_indices = set(self.jobs_df['dataset_index'].dropna().astype(int))
            
            available_indices = [i for i in range(len(self.dataset)) if i not in used_indices]
            
            if not available_indices:
                bt.logging.warning("No more available samples in the dataset")
                return
            
            # Select random samples from available indices
            num_to_add = min(num_jobs, len(available_indices))
            selected_indices = random.sample(available_indices, num_to_add)
            
            new_jobs = []
            for idx in selected_indices:
                sample = self.dataset[idx]
                
                # Convert audio array to WAV format and encode as base64
                audio_array = sample['audio']['array']
                sample_rate = sample['audio']['sampling_rate']
                
                # Convert to WAV and encode as base64
                buffer = io.BytesIO()
                sf.write(buffer, audio_array, sample_rate, format='wav')
                buffer.seek(0)
                base64_audio = base64.b64encode(buffer.read()).decode('utf-8')
                
                # Create a new job entry
                job = {
                    'job_id': str(uuid.uuid4()),
                    'job_status': 'not_done',
                    'job_accuracy': None,
                    'base64_audio': base64_audio,
                    'transcript_miner': None,
                    'gender': sample.get('gender', None),
                    'created_at': datetime.now().isoformat(),
                    'normalized_text': sample['normalized_text'],
                    'dataset_index': idx,
                    'language_miner': None,
                    'gender_miner': None,
                    'gender_confidence_miner': None,
                    'miner_hotkey': None,
                }
                new_jobs.append(job)
            
            # Add new jobs to the DataFrame
            self.jobs_df = pd.concat([self.jobs_df, pd.DataFrame(new_jobs)], ignore_index=True)
            bt.logging.info(f"Added {len(new_jobs)} new jobs to the database")

    def save_jobs_df(self):
        """Save the jobs DataFrame to CSV"""
        with self.job_lock:
            self.jobs_df.to_csv(self.config.validator.csv_path, index=False)
            bt.logging.debug(f"Saved job database with {len(self.jobs_df)} entries")

    def get_pending_jobs(self, batch_size: int) -> List[Dict]:
        """Get a batch of pending jobs"""
        with self.job_lock:
            pending_jobs = self.jobs_df[self.jobs_df['job_status'] == 'not_done']
            batch = pending_jobs.head(batch_size).to_dict('records')
            return batch

    def update_job_status(self, job_id: str, status: str, miner_response: Optional[Dict] = None, miner_hotkey: Optional[str] = None):
        """Update the status and results of a job"""
        with self.job_lock:
            job_idx = self.jobs_df[self.jobs_df['job_id'] == job_id].index
            
            if len(job_idx) == 0:
                bt.logging.warning(f"Job {job_id} not found in database")
                return
            
            # Update job status
            self.jobs_df.loc[job_idx, 'job_status'] = status
            
            # Store the miner hotkey that processed this job
            if miner_hotkey:
                self.jobs_df.loc[job_idx, 'miner_hotkey'] = miner_hotkey
            
            # If we have a miner response, update the relevant fields
            if miner_response and status == 'done':
                self.jobs_df.loc[job_idx, 'transcript_miner'] = miner_response.get('transcript')
                self.jobs_df.loc[job_idx, 'language_miner'] = miner_response.get('language_detected')
                self.jobs_df.loc[job_idx, 'gender_miner'] = miner_response.get('gender_detected')
                self.jobs_df.loc[job_idx, 'gender_confidence_miner'] = miner_response.get('gender_confidence')
                
                # Calculate WER if we have both transcripts
                if self.jobs_df.loc[job_idx, 'normalized_text'].values[0] and miner_response.get('transcript'):
                    ground_truth = self.jobs_df.loc[job_idx, 'normalized_text'].values[0]
                    hypothesis = miner_response.get('transcript')
                    
                    try:
                        wer = jiwer.wer(ground_truth, hypothesis)
                        accuracy = 1.0 - min(wer, 1.0)  # Convert WER to accuracy
                        self.jobs_df.loc[job_idx, 'job_accuracy'] = accuracy
                        bt.logging.info(f"Job {job_id} accuracy: {accuracy:.4f}")
                    except Exception as e:
                        bt.logging.error(f"Error calculating WER: {e}")
                        self.jobs_df.loc[job_idx, 'job_accuracy'] = 0.0
            
            # Save changes
            self.save_jobs_df()
            
            # Check if we need to rotate jobs
            completed_jobs = self.jobs_df[self.jobs_df['job_status'] == 'done']
            if len(completed_jobs) >= self.job_rotation_threshold:
                # Mark the oldest completed jobs as 'rotated'
                oldest_completed = completed_jobs.sort_values('created_at').head(self.job_rotation_threshold)
                self.jobs_df.loc[oldest_completed.index, 'job_status'] = 'rotated'
                
                # Add new jobs to replace the rotated ones
                self.add_new_jobs(len(oldest_completed))
                self.save_jobs_df()

    def calculate_scores(self, miner_hotkeys: List[str]) -> Dict[str, float]:
        """Calculate scores for miners based on their job performance"""
        with self.job_lock:
            # Get all completed jobs
            completed_jobs = self.jobs_df[self.jobs_df['job_status'] == 'done']
            
            # Initialize scores dictionary
            scores = {hotkey: 0.0 for hotkey in miner_hotkeys}
            job_counts = {hotkey: 0 for hotkey in miner_hotkeys}
            
            # Calculate scores based on job accuracy for each miner
            for _, job in completed_jobs.iterrows():
                miner_hotkey = job.get('miner_hotkey')
                
                # Skip if no miner hotkey or not in our list
                if not miner_hotkey or miner_hotkey not in miner_hotkeys:
                    continue
                    
                # Get job accuracy (or default to 0)
                accuracy = job.get('job_accuracy', 0.0)
                if accuracy is None:
                    accuracy = 0.0
                    
                # Add to miner's score
                scores[miner_hotkey] += accuracy
                job_counts[miner_hotkey] += 1
            
            # Calculate average score for each miner
            for hotkey in miner_hotkeys:
                if job_counts[hotkey] > 0:
                    scores[hotkey] = scores[hotkey] / job_counts[hotkey]
                else:
                    # If miner hasn't processed any jobs, give a default score
                    scores[hotkey] = 0.5
                
            # Log scores
            bt.logging.info("Miner scores:")
            for hotkey, score in scores.items():
                bt.logging.info(f"  {hotkey[:10]}...: {score:.4f} ({job_counts[hotkey]} jobs)")
                
            return scores

    def process_jobs(self, batch_size: int):
        """Process a batch of jobs by sending them to miners"""
        # Get pending jobs
        pending_jobs = self.get_pending_jobs(batch_size)
        
        if not pending_jobs:
            bt.logging.info("No pending jobs to process")
            return
        
        bt.logging.info(f"Processing {len(pending_jobs)} jobs")
        
        # Create synapse objects for each job
        synapses = []
        for job in pending_jobs:
            synapse = STTSynapse(
                job_id=job['job_id'],
                base64_audio=job['base64_audio'],
                audio_format='wav',
                gender=job['gender'],
                # Add a vpermit for authentication in a real implementation
            )
            synapses.append(synapse)
        
        # Query miners
        responses = self.dendrite.query(
            axons=self.metagraph.axons,
            synapse=synapses,
            timeout=30  # Longer timeout for audio processing
        )
        
        # Process responses
        for i, (job, response) in enumerate(zip(pending_jobs, responses)):
            # Get the miner's hotkey that processed this job
            axon_info = self.metagraph.axons[i % len(self.metagraph.axons)]
            miner_hotkey = axon_info.hotkey
            
            if response is None or response.error:
                error_msg = response.error if response else "No response"
                bt.logging.warning(f"Job {job['job_id']} failed: {error_msg}")
                self.update_job_status(job['job_id'], 'failed', miner_hotkey=miner_hotkey)
            else:
                bt.logging.info(f"Job {job['job_id']} completed successfully by miner {miner_hotkey[:10]}...")
                
                # Extract response data
                miner_response = {
                    'transcript': response.transcript,
                    'language_detected': response.language_detected,
                    'gender_detected': response.gender_detected,
                    'gender_confidence': response.gender_confidence,
                    'processing_time': response.processing_time
                }
                
                # Update job status with miner's response
                self.update_job_status(job['job_id'], 'done', miner_response, miner_hotkey)
                
                # Log some info about the result
                bt.logging.info(f"Transcript: {response.transcript[:50]}...")
                if job['normalized_text']:
                    wer = jiwer.wer(job['normalized_text'], response.transcript)
                    bt.logging.info(f"WER: {wer:.4f}")

    def update_weights(self):
        """Update weights on the Bittensor blockchain"""
        bt.logging.info("Updating weights on chain")
        
        # Get current block
        current_block = self.subtensor.block
        
        # Check if it's time to update weights
        blocks_since_update = self.subtensor.blocks_since_last_update(
            self.config.netuid, self.my_subnet_uid
        )
        
        if blocks_since_update < 100:  # Only update every 100 blocks
            bt.logging.info(f"Not updating weights. Blocks since last update: {blocks_since_update}")
            return
        
        # Calculate scores for all miners
        miner_hotkeys = self.metagraph.hotkeys
        scores = self.calculate_scores(miner_hotkeys)
        
        # Convert scores to weights (normalize)
        total_score = sum(scores.values())
        if total_score == 0:
            weights = [1.0 / len(miner_hotkeys) for _ in miner_hotkeys]
        else:
            weights = [scores.get(hotkey, 0.0) / total_score for hotkey in miner_hotkeys]
        
        # Update weights on chain
        bt.logging.info(f"Setting weights: min={min(weights):.4f}, max={max(weights):.4f}")
        
        try:
            result = self.subtensor.set_weights(
                netuid=self.config.netuid,
                wallet=self.wallet,
                uids=self.metagraph.uids,
                weights=weights,
                wait_for_inclusion=True,
            )
            if result:
                bt.logging.info("Successfully updated weights on chain")
            else:
                bt.logging.error("Failed to update weights on chain")
        except Exception as e:
            bt.logging.error(f"Error updating weights: {e}")
            traceback.print_exc()

    def run(self):
        """Main validator loop"""
        bt.logging.info("Starting validator loop")
        
        step = 0
        while True:
            try:
                # Process a batch of jobs
                self.process_jobs(self.config.validator.job_batch_size)
                
                # Periodically update the metagraph
                if step % 10 == 0:
                    self.metagraph.sync()
                    bt.logging.info(f"Block: {self.metagraph.block.item()} | Miners: {len(self.metagraph.axons)}")
                
                # Check if we should update weights
                if step % 20 == 0:
                    self.update_weights()
                
                # Sleep to prevent overwhelming the network
                time.sleep(5)
                step += 1
                
            except KeyboardInterrupt:
                bt.logging.info("Keyboard interrupt detected, exiting")
                break
            except Exception as e:
                bt.logging.error(f"Error in validator loop: {e}")
                traceback.print_exc()
                time.sleep(10)  # Wait a bit before retrying

if __name__ == "__main__":
    validator = STTValidator()
    validator.run()
