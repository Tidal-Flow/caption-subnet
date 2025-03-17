import typing
import bittensor as bt
from typing import Optional, List, Dict, Any
import base64

class STTSynapse(bt.Synapse):
    """
    Protocol for Speech-to-Text (STT) tasks between validators and miners.
    
    Attributes:
        job_id: Unique identifier for the transcription job
        base64_audio: Base64 encoded audio data
        audio_format: Format of the audio (e.g., 'wav')
        language: Language code of the audio (if known)
        gender: Gender of the speaker (if known)
        vpermit: Verification permit for authentication
        
        # Response fields (filled by miners)
        transcript: Transcribed text from the audio
        language_detected: Language detected by the miner
        gender_detected: Gender detected by the miner
        gender_confidence: Confidence score for gender detection
        processing_time: Time taken to process the request (in seconds)
        error: Error message if processing failed
    """
    
    # Request fields (filled by validator)
    job_id: str
    base64_audio: str
    audio_format: str
    language: Optional[str] = None
    gender: Optional[str] = None
    vpermit: Optional[str] = None
    
    # Response fields (filled by miner)
    transcript: Optional[str] = None
    language_detected: Optional[str] = None
    gender_detected: Optional[str] = None
    gender_confidence: Optional[float] = None
    processing_time: Optional[float] = None
    error: Optional[str] = None
