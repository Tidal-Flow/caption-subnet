import unittest
from src.miner import CaptionMiner
from src.protocol import CaptionSynapse

class TestCaptionMiner(unittest.TestCase):

    def setUp(self):
        self.miner = CaptionMiner()

    def test_audio_decoding(self):
        audio_data = "base64_encoded_audio_data"
        decoded_audio = self.miner.decode_audio(audio_data)
        self.assertIsNotNone(decoded_audio)
        self.assertIsInstance(decoded_audio, bytes)

    def test_handle_caption_synapse(self):
        synapse = CaptionSynapse(audio="base64_encoded_audio_data", metadata={})
        result = self.miner.handle_caption_synapse(synapse)
        self.assertIn("transcription", result)
        self.assertIsInstance(result["transcription"], str)

    def test_transcription_accuracy(self):
        audio_data = "base64_encoded_audio_data"
        expected_transcription = "This is a test transcription."
        self.miner.transcribe_audio = lambda x: expected_transcription  # Mocking the method
        result = self.miner.handle_caption_synapse(CaptionSynapse(audio=audio_data, metadata={}))
        self.assertEqual(result["transcription"], expected_transcription)

if __name__ == '__main__':
    unittest.main()