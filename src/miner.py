import os
import time
import argparse
import traceback
import base64
import bittensor as bt
from typing import Tuple
from protocol import CaptionSynapse, CaptionSegment
import whisper  # Assuming Whisper is the STT model being used


class CaptionMiner:
    def __init__(self):
        self.config = self.get_config()
        self.setup_logging()
        self.setup_bittensor_objects()
        self.stt_model = whisper.load_model("base")  # Load the STT model

    def get_config(self):
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--netuid", type=int, default=1, help="The chain subnet uid."
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

    def decode_audio(self, audio_data: str) -> bytes:
        return base64.b64decode(audio_data)

    def handle_caption_request(self, synapse: CaptionSynapse) -> CaptionSegment:
        audio_bytes = self.decode_audio(synapse.audio_data)
        transcription = self.stt_model.transcribe(audio_bytes)
        start_time = transcription['segments'][0]['start']
        end_time = transcription['segments'][-1]['end']
        caption_segment = CaptionSegment(
            text=transcription['text'],
            start_time=start_time,
            end_time=end_time
        )
        bt.logging.info(f"Processed audio, generated caption: {caption_segment.text}")
        return caption_segment

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
        while True:
            try:
                time.sleep(1)
            except KeyboardInterrupt:
                self.axon.stop()
                bt.logging.success("Miner killed by keyboard interrupt.")
                break
            except Exception as e:
                bt.logging.error(traceback.format_exc())
                continue


# Run the miner.
if __name__ == "__main__":
    miner = CaptionMiner()
    miner.run()