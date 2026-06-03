"""
src/experiment/dummy_endpoints.py

Author: Jared Moore
Date: July, 2025

Contains dummy implementations of utility endpoint functions
"""

import base64


def transcribe_audio(audio_b64: str):
    """
    Dummy transcription: always returns a single dummy segment
    and the text "hello".
    """
    dummy_segments = [
        {
            "id": 0,
            "seek": 0,
            "start": 0.0,
            "end": 1.23,
            "text": "hello",
            "tokens": [],
            "temperature": 0.0,
            "avg_logprob": 0.0,
            "compression_ratio": 1.0,
            "no_speech_prob": 0.0,
        }
    ]
    words = [
        {
            "word": "hello",
            "start": 0.0,
            "end": 0.5,
        }
    ]
    full_json = {
        "text": "hello",
        "segments": dummy_segments,
        "language": "en",
        "words": words,
        "duration": 1.0,
    }
    return full_json, "hello"


def synthesize_audio(text: str) -> str:
    """
    Dummy TTS: returns a tiny WAV-header prefix, base64-encoded.
    """
    # A minimal WAV header fragment (“RIFF”)
    dummy_wav_bytes = b"RIFF$\x00\x00\x00WAVE"
    audio_b64 = base64.b64encode(dummy_wav_bytes).decode("utf-8")
    return f"data:audio/wav;base64,{audio_b64}"


def moderate_content(content: str) -> bool:
    """
    Dummy moderate: returns False; content is not flagged
    """
    return False


def is_refusal(content: str):
    """
    Dummy refusal: returns False
    """
    return False


def rephrase_participant_decision(decision: str) -> dict[str, str]:
    """
    Dummy participant decision rephrase: always returns a simple proposition.
    """
    if not decision or not isinstance(decision, str):
        return {"status": "error", "reason": "Please provide a valid decision."}
    return {"status": "ok", "proposition": "I should make a concrete decision soon."}
