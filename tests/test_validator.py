import unittest
from src.validator import CaptionValidator
from src.protocol import CaptionSynapse

class TestCaptionValidator(unittest.TestCase):

    def setUp(self):
        self.validator = CaptionValidator()
        self.audio_data = "base64_encoded_audio_data"
        self.synapse = CaptionSynapse(audio=self.audio_data)

    def test_score_miners(self):
        # Simulate responses from miners
        correct_caption = "This is a test caption."
        incorrect_caption = "This is a test capton."

        # Assume we have a method to simulate miner responses
        self.validator.receive_caption(self.synapse, correct_caption)
        self.validator.receive_caption(self.synapse, incorrect_caption)

        # Check the scoring logic
        scores = self.validator.scores
        self.assertEqual(scores[0], 1.0)  # Correct caption should score high
        self.assertLess(scores[1], 1.0)   # Incorrect caption should score lower

    def test_weight_setting(self):
        # Simulate setting weights based on scores
        self.validator.scores = [0.9, 0.1]
        self.validator.set_weights()

        # Check if weights are set correctly
        expected_weights = [0.9, 0.1]
        self.assertEqual(self.validator.weights, expected_weights)

if __name__ == '__main__':
    unittest.main()