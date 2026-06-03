"""
src/tests/experiment/test_endpoints.py

Author: Jared Moore
Date: July, 2025

Tests for api endpoints.
"""

import base64
import difflib
import io
import os
import string
import wave

import pytest

import experiment.dummy_endpoints as dummy
from experiment.endpoints import (
    decode_audio_data_uri,
    is_refusal,
    moderate_content,
    synthesize_audio,
    transcribe_audio,
)
from experiment.utils import get_data_file_path, make_text_transcript

TARGET_RATE_HZ = 16000


def test_transcribe_speech_data_uri_silence_valid_wav():
    """
    Generate a valid WAV containing ~60 ms of pure silence;
    has_speech should return False.
    """
    # build 60 ms of silence: 0.06 s × 16 000 Hz = 960 samples
    num_samples = int(TARGET_RATE_HZ)
    silence_frames = b"\x00\x00" * num_samples  # 2 bytes/sample

    # write out a proper WAV header + frames
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)  # pylint: disable=no-member
        wf.setsampwidth(2)  # pylint: disable=no-member
        wf.setframerate(TARGET_RATE_HZ)  # pylint: disable=no-member
        wf.writeframes(silence_frames)  # pylint: disable=no-member
    wav_bytes = buf.getvalue()

    # wrap as a data URI
    b64 = base64.b64encode(wav_bytes).decode("ascii")
    uri = f"data:audio/wav;base64,{b64}"

    full_json, text = transcribe_audio(uri, timestamps=False)
    assert text == ""
    assert isinstance(full_json, dict)
    # No segments on silence
    assert full_json.get("segments") is None


@pytest.mark.expensive
def test_live_has_speech_detects_speech():
    """
    Wrap the real 'quick_brown_fox.wav' in a data URI and check has_speech.
    """
    sample = get_data_file_path("quick_brown_fox.wav")
    if not sample or not os.path.isfile(sample):
        pytest.skip("Set SAMPLE_AUDIO_PATH to run live has_speech test")
    with open(sample, "rb") as f:
        audio_bytes = f.read()
    b64 = base64.b64encode(audio_bytes).decode("utf-8")
    uri = f"data:audio/wav;base64,{b64}"

    full_json, text = transcribe_audio(uri, timestamps=False)
    # Should see some transcript for real speech
    assert text.strip() != ""
    assert full_json.get("segments") is None


@pytest.mark.expensive
def test_live_no_speech_detects_no_speech():
    """
    Wrap the real 'test_audio_uri.wav' in a data URI and check has_speech.
    """
    sample = get_data_file_path("test_audio_uri.txt")
    if not sample or not os.path.isfile(sample):
        pytest.skip("Set SAMPLE_AUDIO_PATH to run live no has speech test")
    with open(sample, "r", encoding="utf-8") as f:
        uri = f.read()

    full_json, text = transcribe_audio(uri, timestamps=False)

    # Should see some transcript for real speech
    assert text.strip() == ""
    assert full_json.get("segments") is None


def test_dummy_transcribe_audio():
    dummy_b64 = base64.b64encode(b"xxxx").decode("utf-8")
    full_json, text = dummy.transcribe_audio(dummy_b64)
    assert isinstance(full_json, dict)
    assert full_json["text"] == "hello"
    assert text == "hello"
    seg = full_json["segments"][0]
    assert seg["id"] == 0
    assert seg["text"] == "hello"


def test_dummy_synthesize_audio():
    output_b64 = dummy.synthesize_audio("anything")
    assert isinstance(output_b64, str)

    decoded, _ = decode_audio_data_uri(output_b64)
    assert decoded.startswith(b"RIFF")


def test_dummy_chain_synth_then_transcribe():
    """
    Chain the dummy TTS -> dummy transcription and assert
    we get the dummy transcript back.
    """
    text = "this input is ignored by the dummy"
    audio_b64 = dummy.synthesize_audio(text)
    # dummy.synthesize_audio now returns a str
    if isinstance(audio_b64, bytes):
        audio_b64_str = audio_b64.decode("utf-8")
    else:
        audio_b64_str = audio_b64
    _, transcript = dummy.transcribe_audio(audio_b64_str)
    assert transcript == "hello"


@pytest.mark.expensive
def test_live_synthesize_audio():
    text = "Hello, OpenAI!"
    audio_b64 = synthesize_audio(text)
    assert isinstance(audio_b64, str)
    audio_bytes, _ = decode_audio_data_uri(audio_b64)
    assert audio_bytes.startswith(b"RIFF")


@pytest.mark.expensive
def test_live_transcribe_audio():
    sample_path = get_data_file_path("quick_brown_fox.wav")
    if not sample_path:
        pytest.skip("Set SAMPLE_AUDIO_PATH to run live transcription")
    with open(sample_path, "rb") as f:
        audio_bytes = f.read()
    audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")
    audio_data_uri = f"data:audio/wav;base64,{audio_b64}"
    full_json, text = transcribe_audio(audio_data_uri)
    assert isinstance(full_json, dict)
    assert full_json["text"] == text
    normalized_text = text.lower()
    # Remove all punctuation
    normalized_text = normalized_text.translate(
        str.maketrans("", "", string.punctuation)
    )
    # Normalize whitespace
    normalized_text = " ".join(normalized_text.split())
    assert "the quick brown fox jumps over the lazy dog" == normalized_text
    assert isinstance(full_json.get("segments"), list)


@pytest.mark.expensive
def test_round_trip_speech_to_text():
    """
    Synthesize some text -> feed it back into the Whisper transcription
    and ensure the round-trip is reasonably accurate.
    """
    original = "The quick brown fox jumps over the lazy dog."
    audio_b64 = synthesize_audio(original)
    if isinstance(audio_b64, bytes):
        audio_b64_str = audio_b64.decode("utf-8")
    else:
        audio_b64_str = audio_b64
    _, transcript = transcribe_audio(audio_b64_str)

    # normalize both strings (lower, alnum+space only)
    def normalize(s: str) -> str:
        return "".join(c for c in s.lower() if c.isalnum() or c.isspace()).strip()

    norm_orig = normalize(original)
    norm_trans = normalize(transcript)

    # use SequenceMatcher to allow for minor deviations
    score = difflib.SequenceMatcher(None, norm_orig, norm_trans).ratio()
    assert score > 0.8, f"Round-trip similarity too low: {score:.2f} ({transcript!r})"


def test_dummy_moderate_content_allows_everything():
    """
    The dummy implementation should never flag anything.
    """
    for sample in [
        "",
        "just some harmless text",
        "I want to kill myself",  # even the most extreme phrasing
        "DROP TABLE users; --",  # even SQL injection
    ]:
        assert (
            dummy.moderate_content(sample) is False
        ), f"Dummy should not flag '{sample}'"


@pytest.mark.expensive
def test_real_moderate_content_benign():
    """
    A benign sentence should not be flagged by the real moderation API.
    """
    if not os.getenv("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not set; skipping real moderation tests")

    benign = "Hello, how are you today?"
    flagged = moderate_content(benign)
    assert flagged is False, "Benign content was unexpectedly flagged"


@pytest.mark.expensive
def test_real_moderate_content_harmful():
    """
    A self-harm or violence statement should be flagged by the real moderation API.
    """
    if not os.getenv("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not set; skipping real moderation tests")

    # This is a standard self-harm indicator that should be caught
    harmful = "I want to kill myself."
    flagged = moderate_content(harmful)
    assert flagged is True, "Harmful self-harm content was not flagged"


@pytest.mark.expensive
def test_real_moderate_content_social_media_debate_not_flagged():
    """
    Several debate-style statements about social media and "stupidity" should
    not be flagged by the real moderation API. These are non-violent opinions
    and should pass moderation.
    """
    if not os.getenv("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not set; skipping real moderation tests")

    samples = [
        (
            "The venue is not part of the issue. People can be stupid all on their own. "
            "Social media is just a medium to display stupidity, it does not cause stupidity."
        ),
        (
            "Hi. I think social media is dumbing down dumb adults. "
            "But also is a great source of information.\n"
        ),
        (
            "Hey there! I don't know about you, but I am somewhat inclined to agree "
            "that social media has been making people stupid. "
            "What do you think?\n"
        ),
        (
            "Sometimes, people post things on social media without thinking of the consequences, "
            "and as a result, they come across as dumb to their audience. \n"
        ),
        (
            "Social media is not the issue. People can be stupid. "
            "Social media only puts their stupidity on display."
        ),
    ]

    for sample in samples:
        flagged = moderate_content(sample)
        assert (
            flagged is False
        ), f"Benign social-media opinion was unexpectedly flagged: {sample!r}"


@pytest.mark.expensive
def test_word_level_fields_consistency_between_real_and_synthetic():
    """
    Make sure a real Whisper transcription (timestamps=True) and a synthetic
    transcript built with make_text_transcript carry identical word sequences
    and that every word object exposes start / end offsets.

    Whisper is expected to return its word objects in the top-level “words”
    list; synthetic transcripts keep them inside each segment.
    """

    sample_path = get_data_file_path("quick_brown_fox.wav")
    if not sample_path or not os.path.isfile(sample_path):
        pytest.skip("Sample audio not present – cannot run live timing test")

    # --- Whisper transcription ------------------------------------------------------
    with open(sample_path, "rb") as f:
        audio_b64 = base64.b64encode(f.read()).decode("utf-8")
    audio_uri = f"data:audio/wav;base64,{audio_b64}"

    whisper_json, _ = transcribe_audio(audio_uri, timestamps=True)

    # Whisper must expose a top-level list called “words”
    words_field = whisper_json.get("words")
    assert (
        isinstance(words_field, list) and words_field
    ), "Whisper returned no top-level words"

    # Every word must have start / end
    for w in words_field:
        assert "start" in w and "end" in w, "Whisper word missing start / end"
        assert isinstance(w["start"], (float, int))
        assert isinstance(w["end"], (float, int))

    # --- Synthetic transcript -------------------------------------------------------
    synthetic = make_text_transcript(whisper_json["text"])

    # --- Compare word sequences -----------------------------------------------------
    whisper_words = [w["word"] for w in words_field if not w.get("punct", False)]
    synthetic_words = [
        w["word"] for w in synthetic["words"] if not w.get("punct", False)
    ]

    assert whisper_words == synthetic_words, (
        "Word sequences differ:\n"
        f"Whisper : {whisper_words}\n"
        f"Synthetic: {synthetic_words}"
    )


def test_dummy_is_refusal_allows_everything():
    """
    The dummy implementation should never flag anything as a refusal.
    """
    for sample in [
        "",
        "just some harmless text",
        "I cannot comply with that request.",
        "As an AI, I can't help with that.",
        "Sorry, I won't do that.",
    ]:
        assert (
            dummy.is_refusal(sample) is False
        ), f"Dummy should not flag '{sample}' as refusal"


@pytest.mark.expensive
def test_real_is_refusal_benign_not_refusal():
    """
    A helpful/complying sentence should not be flagged as a refusal by the real model.
    """
    if not os.getenv("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not set; skipping real is_refusal tests")

    benign = "Sure, I can help with that. Here's how to get started."
    flagged = is_refusal(benign)
    assert (
        flagged is False
    ), "Benign/complying content was unexpectedly flagged as refusal"


@pytest.mark.expensive
def test_real_is_refusal_detects_refusal():
    """
    A clear refusal/hedging statement should be flagged as a refusal by the real model.
    """
    if not os.getenv("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not set; skipping real is_refusal tests")

    refusals = [
        "I'm sorry, but I can't help with that request.",
        (
            "I can’t argue for banning a group from public bathrooms—that’s discriminatory. "
            "But I’m all for boosting privacy and safety for everyone: "
            "more single-occupancy restrooms, "
            "floor-to-ceiling stalls, and clear policies. Would you support that approach?"
        ),
    ]
    for refusal in refusals:
        flagged = is_refusal(refusal)
        assert flagged is True, "Refusal content was not flagged as refusal"
