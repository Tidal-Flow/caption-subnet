# filepath: /bittensor-subnet/bittensor-subnet/tests/test_protocol.py
import unittest
from src.protocol import CaptionSegment, CaptionSynapse

class TestCaptionSegment(unittest.TestCase):
    def test_initialization(self):
        segment = CaptionSegment(start=0.0, end=5.0, text="Hello world")
        self.assertEqual(segment.start, 0.0)
        self.assertEqual(segment.end, 5.0)
        self.assertEqual(segment.text, "Hello world")

    def test_duration(self):
        segment = CaptionSegment(start=0.0, end=5.0, text="Hello world")
        self.assertEqual(segment.duration(), 5.0)

class TestCaptionSynapse(unittest.TestCase):
    def test_initialization(self):
        audio_data = "base64_encoded_audio"
        metadata = {"language": "en", "model": "whisper"}
        synapse = CaptionSynapse(audio=audio_data, metadata=metadata)
        self.assertEqual(synapse.audio, audio_data)
        self.assertEqual(synapse.metadata, metadata)

    def test_serialization(self):
        audio_data = "base64_encoded_audio"
        metadata = {"language": "en", "model": "whisper"}
        synapse = CaptionSynapse(audio=audio_data, metadata=metadata)
        serialized = synapse.serialize()
        self.assertIn("audio", serialized)
        self.assertIn("metadata", serialized)

if __name__ == '__main__':
    unittest.main()