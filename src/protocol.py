class CaptionSegment:
    """
    Represents a segment of transcribed text with start and end times.
    """

    def __init__(self, start: float, end: float, text: str):
        self.start = start
        self.end = end
        self.text = text


class CaptionSynapse:
    """
    Carries base64-encoded audio and metadata for communication
    between the validator and miners.
    """

    def __init__(self, audio_base64: str, metadata: dict):
        self.audio_base64 = audio_base64
        self.metadata = metadata