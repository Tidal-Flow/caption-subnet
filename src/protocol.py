from bittensor.utils.weight_utils import serialize_forward_response
import base64

class CaptionSegment:
    """
    Represents a segment of transcribed text with start and end times.
    """

    def __init__(self, text: str, start_time: float = 0.0, end_time: float = 0.0):
        self.text = text
        self.start_time = start_time
        self.end_time = end_time
        
    def duration(self):
        """Returns the duration of the caption segment."""
        return self.end_time - self.start_time
        
    def __str__(self):
        return f"[{self.start_time:.2f}-{self.end_time:.2f}] {self.text}"


class CaptionSynapse:
    """
    Carries audio data and metadata for communication
    between the validator and miners.
    """

    def __init__(self, audio_data: str, audio_id: str = None, metadata: dict = None):
        self.audio_data = audio_data  # Base64 encoded audio data
        self.audio_id = audio_id      # ID from the VoxPopuli dataset
        self.metadata = metadata or {}
        self.transcription = None     # Will be filled by miners

    def serialize(self):
        """Serializes the synapse data for network transmission."""
        return {
            "audio_data": self.audio_data,
            "audio_id": self.audio_id,
            "metadata": self.metadata,
            "transcription": self.transcription if hasattr(self, "transcription") else None
        }
        
    def deserialize(self, data):
        """Deserializes the synapse data after network transmission."""
        if "audio_data" in data:
            self.audio_data = data["audio_data"]
        if "audio_id" in data:
            self.audio_id = data["audio_id"]
        if "metadata" in data:
            self.metadata = data["metadata"]
        if "transcription" in data:
            self.transcription = data["transcription"]
        return self