import base64
import json

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

    def to_dict(self):
        """Convert segment to dictionary for serialization."""
        return {
            "text": self.text,
            "start_time": self.start_time,
            "end_time": self.end_time
        }

    @classmethod
    def from_dict(cls, data):
        """Create segment from dictionary."""
        return cls(
            text=data["text"],
            start_time=data["start_time"],
            end_time=data["end_time"]
        )


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

    def serialize(self) -> bytes:
        """Serializes the synapse data for network transmission."""
        data = {
            "audio_data": self.audio_data,
            "audio_id": self.audio_id,
            "metadata": self.metadata,
            "transcription": self.transcription if hasattr(self, "transcription") else None
        }
        return json.dumps(data).encode('utf-8')
        
    @classmethod
    def deserialize(cls, data: bytes):
        """Deserializes the synapse data after network transmission."""
        if isinstance(data, bytes):
            data = json.loads(data.decode('utf-8'))
        elif isinstance(data, str):
            data = json.loads(data)
            
        instance = cls(
            audio_data=data.get("audio_data", ""),
            audio_id=data.get("audio_id", None),
            metadata=data.get("metadata", {})
        )
        if "transcription" in data:
            instance.transcription = data["transcription"]
        return instance