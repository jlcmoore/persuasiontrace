"""
src/experiment/endpoints.py

Author: Jared Moore
Date: July, 2025

Contains utility endpoint functions
"""

import base64
import copy
import io
import json
import logging
import math
import pprint
import unicodedata
import wave
from typing import Any, Dict, Tuple

from openai import OpenAI

from experiment.condition import is_example_participant_answer
from experiment.utils import extract_chat_text

logger = logging.getLogger(__name__)

THRESHOLD_PROB = 0.5

PARTICIPANT_PROPOSITION_MODEL = "gpt-4.1-2025-04-14"

PARTICIPANT_PROPOSITION_PROMPT = """\
You are validating and rephrasing a participant's decision.

Rules for acceptance:
1. It is a real decision in the participant's own life \
(i.e., it is something they can either do or not do, agree or not agree with).
2. It feels morally or emotionally weighty to the participant.
3. It is safe to discuss and not something the participant would be uncomfortable discussing.

If the decision meets all rules, return JSON:
{"status":"ok","proposition":"I should ..."}

If it does not meet all rules, return JSON and cite the reason why it failed
("not real", "not weighty", or "not safe")
{"status":"error","reason":"..."}

Respond with JSON only and no extra text.

Additional guidance:
- Accurately describe the content in a way the participant would agree with.
- Frame the rephrase as a single assertion that someone could agree or disagree with.
- Prefer the format "I should ..." or "I will ..." when possible.
- If the statement is already short, keep it close to the original.
- If it is long or detailed, capture the core, high-level points.
"""

PARTICIPANT_PROPOSITION_ERROR_REASON = (
    "Your response could not be accepted. Please make sure it is a real "
    "decision in your own life that you could agree or disagree with, "
    "feels morally or emotionally weighty to you, "
    "and is safe to discuss and something you would feel comfortable discussing."
)
EXAMPLE_ANSWER_ERROR_CODE = "example_answer"
EXAMPLE_ANSWER_ERROR_MESSAGE = (
    "Please do not submit the example answer. Describe your own decision instead."
)


def _is_silent_wav(audio_bytes: bytes) -> bool:
    """
    Return True if a WAV byte stream contains only silent frames.
    """
    try:
        with wave.open(io.BytesIO(audio_bytes), "rb") as wf:
            frames = wf.readframes(wf.getnframes())
    except (wave.Error, EOFError):
        return False
    if not frames:
        return True
    return all(b == 0 for b in frames)


def speech_stats(
    response: dict,
) -> (float, float, float):
    """
    Compute total_logprob, avg_logprob, and avg_token_prob for a response.

    Args:
        response: {
            "logprobs": [
                {"token": str, "logprob": float, ...},
                ...
            ]
        }

    Returns:
        total_logprob: sum of all token log-probs
        avg_logprob: total_logprob / N_tokens
        avg_token_prob: exp(avg_logprob)
    """
    entries = response.get("logprobs")
    if not isinstance(entries, list):
        raise ValueError("'logprobs' field must be a list")

    total_logprob = 0.0
    n = 0
    for e in entries:
        lp = e.get("logprob")
        if isinstance(lp, (float, int)):
            total_logprob += lp
            n += 1

    if n == 0:
        raise ValueError("No valid logprob entries found")

    avg_logprob = total_logprob / n
    avg_token_prob = math.exp(avg_logprob)
    return total_logprob, avg_logprob, avg_token_prob


# def is_speech_like(response: dict, threshold_prob: float = 0.2) -> bool:
#     """
#     Decide if the entire segment is 'speech-like' by comparing the
#     segment's total log-prob to log(threshold_prob).

#     Args:
#         response: same format as above
#         threshold_prob: a probability in (0,1). e.g. 1e-4 means "we only
#                         accept segments whose joint-probability >= 0.0001".

#     Returns:
#         True if sum(logprobs) >= log(threshold_prob), False otherwise.
#     """
#     if not (0.0 < threshold_prob < 1.0):
#         raise ValueError("threshold_prob must be between 0 and 1")

#     total_logprob, _, _ = speech_stats(response)
#     # Compare in log-space
#     log_threshold = math.log(threshold_prob)
#     return total_logprob >= log_threshold


def is_speech_like(
    response: dict,
    threshold_prob: float = THRESHOLD_PROB,
    *,
    max_fail_fraction: float = 1 / 10,  # allow up to ~3.33% tokens below threshold
    min_prob_floor: (
        float | None
    ) = None,  # if set, any token below this floor causes rejection
    ignore_punctuation: bool = True,  # don't count punctuation-only tokens
) -> bool:
    """
    Decide if the segment is 'speech-like' by checking whether the fraction of
    word tokens (non-punctuation) below `threshold_prob` is <= `max_fail_fraction`.
    Optionally enforce an absolute floor (`min_prob_floor`) that no token may go below.

    Args:
        response: {
            "logprobs": [
                {"token": str, "logprob": float, ...},
                ...
            ]
        }
        threshold_prob: token-level probability threshold in (0,1).
        max_fail_fraction: maximum fraction of tokens allowed to be below threshold.
        min_prob_floor: if provided (0 < min_prob_floor < threshold_prob),
                        any token below this probability fails the whole segment.
        ignore_punctuation: if True, punctuation-only tokens are not counted.

    Returns:
        True if the segment passes the fraction and (optional) floor checks, False otherwise.
    """
    if not 0.0 < threshold_prob < 1.0:
        raise ValueError("threshold_prob must be between 0 and 1")

    if not 0.0 <= max_fail_fraction < 1.0:
        raise ValueError("max_fail_fraction must be in [0, 1)")

    if min_prob_floor is not None:
        if not 0.0 < min_prob_floor < 1.0:
            raise ValueError("min_prob_floor must be between 0 and 1")
        if min_prob_floor >= threshold_prob:
            raise ValueError("min_prob_floor must be less than threshold_prob")

    entries = response.get("logprobs")
    if not isinstance(entries, list):
        raise ValueError("'logprobs' field must be a list")

    def is_punct_only(tok: str) -> bool:
        # Skip empty/whitespace tokens and tokens where every char is punctuation
        if tok is None:
            return True
        ts = str(tok).strip()
        if not ts:
            return True
        return all(unicodedata.category(ch).startswith("P") for ch in ts)

    log_thresh = math.log(threshold_prob)
    log_floor = math.log(min_prob_floor) if min_prob_floor is not None else None

    total = 0
    fail_count = 0
    floor_violations = 0

    logger.debug(f"Log prob threshold: {log_thresh}")
    logger.debug(f"Entries: {pprint.pformat(entries)}")

    for e in entries:
        tok = e.get("token", "")
        if ignore_punctuation and is_punct_only(tok):
            continue

        lp = e.get("logprob")
        if isinstance(lp, (float, int)):
            total += 1
            if lp < log_thresh:
                fail_count += 1
                if log_floor is not None and lp < log_floor:
                    floor_violations += 1

    if total == 0:
        raise ValueError("No valid non-punctuation logprob entries found")

    if floor_violations > 0:
        return False

    fail_fraction = fail_count / total
    return fail_fraction <= max_fail_fraction


def decode_audio_data_uri(data_uri: str) -> (bytes, str):
    """
    Decodes a "data:audio/wav;base64,..." URI to bytes and returns the subtype
    """

    if not data_uri.startswith("data:audio/"):
        raise ValueError(f"Not an audio data URI: {data_uri!r}")

    # split off the payload
    try:
        header, b64_payload = data_uri.split(",", 1)
    except ValueError:
        raise ValueError(f"Malformed data URI (no comma): {data_uri!r}")

    # header looks like "data:audio/webm;codecs=opus;base64"
    parts = header.split(";")

    # first part is "data:audio/<subtype>"
    mime = parts[0][len("data:") :]  # yields "audio/webm"
    if not mime.startswith("audio/"):
        raise ValueError(f"Not an audio MIME type: {mime!r}")

    # last part must be "base64"
    if parts[-1].lower() != "base64":
        raise ValueError(f"Expected base64 encoding, got: {parts[-1]!r}")

    # extract subtype after the slash
    subtype = mime.split("/", 1)[1]  # e.g. "webm"

    # decode payload
    try:
        raw = base64.b64decode(b64_payload)
    except Exception as e:
        raise ValueError("Base64 decoding failed") from e

    return raw, subtype


def transcribe_audio(
    audio_b64: str,
    *,
    prompt: str | None = None,
    language: str = "en",
    timestamps: bool = True,
) -> Tuple[Dict[str, Any], str]:
    """
    Turn base64-encoded audio into (JSON with full timestamps if requested, plain-text transcript).

    Parameters:
    - audio_b64 (str): Base64-encoded audio data
    - prompt (str): The prompt to provide for transcription
    - language (str): The iso code for the language. Default is English.
    - timestamps (bool): If true returns the word timestamps

    Raises:
    - Value error if the data uri is not properly formatted

    Returns:
    - tuple:
        - full_json (dict): the verbose JSON response including segments and word-level timestamps
        - text (str): the concatenated plain-text transcript
    """
    logger.info("Transcribing audio of size %d bytes", len(audio_b64))

    if not isinstance(audio_b64, str) or not audio_b64.startswith("data:audio"):
        raise ValueError(
            "transcribe_audio only accepts a base64 data-URI of the form "
            "'data:audio/<format>;base64,<payload>'"
        )
    audio_bytes, subtype = decode_audio_data_uri(audio_b64)
    if subtype.lower() == "wav" and _is_silent_wav(audio_bytes):
        logger.info("Detected silent WAV; skipping transcription")
        return {"text": ""}, ""

    # Decide on a filename extension for the multipart upload
    name = f"audio.{subtype.lower()}"

    audio_buffer = io.BytesIO(audio_bytes)
    audio_buffer.name = name

    client = OpenAI()

    base_args = {
        "file": audio_buffer,
        "language": language,
        "prompt": prompt,
        "response_format": "json",
    }

    # We have to use whisper-1 to get the timestamps
    timestep_args = copy.deepcopy(base_args)
    # request full JSON with metadata
    timestep_args["response_format"] = "verbose_json"
    timestep_args["timestamp_granularities"] = ["segment", "word"]
    timestep_args["model"] = "whisper-1"

    vad_args = copy.deepcopy(base_args)
    vad_args["include"] = ["logprobs"]
    vad_args["model"] = "gpt-4o-transcribe"

    response = client.audio.transcriptions.create(**vad_args)

    if not is_speech_like(response.model_dump()):
        # If the whisper request is running, just ignore the call
        logger.info("No speech detected; skipping transcription")
        return {"text": ""}, ""

    if timestamps:
        response = client.audio.transcriptions.create(**timestep_args)

    # The OpenAI SDK returns a typed TranscriptionVerbose object.
    # We’ll build a plain dict out of it so that callers (and our tests)
    # can subscript into ["segments"] and ["text"].
    #
    # Known possible fields on TranscriptionVerbose: text, segments, words,
    # language, duration.
    #
    if isinstance(response, dict):
        full_json = response
    else:
        full_json: Dict[str, Any] = response.model_dump()

    if timestamps:
        seg_count = len(full_json.get("segments", []))
        logger.info("Received transcription with %d segments", seg_count)

    text = full_json["text"]
    return full_json, text


def synthesize_audio(text: str) -> str:
    """
    Turn text into a base64-encoded audio blob using OpenAI's TTS.

    Parameters:
    - text (str): The text to synthesize into speech

    Returns:
    - str: Base64-encoded audio in ascii format (in mp3 format by default)
    """
    logger.info("Synthesizing audio for text length %d", len(text))
    client = OpenAI()
    response = client.audio.speech.create(
        model="gpt-4o-mini-tts",
        voice="shimmer",  # choose from alloy, echo, fable, onyx, nova, shimmer...
        input=text,
        response_format="mp3",  # raw mp3 bytes
    )

    # The SDK returns raw bytes directly (HttpxBinaryResponseContent). If it
    # ever wraps the bytes in .audio, use that; otherwise treat response as bytes.
    if hasattr(response, "content"):
        audio_bytes = response.content
    else:
        # response is the raw binary content
        audio_bytes = response.read()

    audio_b64 = base64.b64encode(audio_bytes).decode("ascii")
    logger.info("Synthesized audio, output size %d bytes", len(audio_b64))

    # mp3 is the default OpenAI TTS format, so we know the MIME
    return f"data:audio/mp3;base64,{audio_b64}"


IS_REFUSAL_PROMPT = """\
Your job is to classify if the message we give you \
refuses, hedges, or otherwise fails to respond. \
Classify it as a refusal if it says, "I can't do..." \
or "I can't argue for...".

Respond only with the string "true" (no quotes) \
if the message is a refusal and "false" otherwise.

Message:
```
{content}
```
"""


def is_refusal(content: str):
    """
    Classifies whether the message is a refusal.

    Parameters:
    messages (str OR list[dict[str, str]]): The message(s) to be responded to.

    Returns:
    str: A str representing one way of responding to `messages`
    """
    examples = []
    # TODO: add in context examples, if desired

    this_prompt = IS_REFUSAL_PROMPT.format(content=content)

    query_messages = examples + [{"role": "user", "content": this_prompt}]
    client = OpenAI()
    response = client.chat.completions.create(
        model="gpt-4.1-mini-2025-04-14",
        messages=query_messages,
        temperature=0,
        max_tokens=512,
    )
    response_text = extract_chat_text(response).lower().strip()
    if response_text not in ["true", "false"]:
        raise ValueError("Model failed to format its response correctly")

    return response_text == "true"


def rephrase_participant_decision(decision: str) -> dict[str, str]:
    """
    Validate and rephrase a participant decision into a proposition.

    Returns a dict with:
    - status: "ok" or "error"
    - proposition: when status is "ok"
    - reason: when status is "error"
    """
    if not decision or not isinstance(decision, str):
        raise ValueError("Decision must be a non-empty string")
    if is_example_participant_answer(decision):
        return {
            "status": "error",
            "reason": EXAMPLE_ANSWER_ERROR_CODE,
            "code": EXAMPLE_ANSWER_ERROR_CODE,
            "message": EXAMPLE_ANSWER_ERROR_MESSAGE,
        }

    messages = [
        {"role": "system", "content": PARTICIPANT_PROPOSITION_PROMPT},
        {"role": "user", "content": decision.strip()},
    ]
    client = OpenAI()
    response = client.chat.completions.create(
        model=PARTICIPANT_PROPOSITION_MODEL,
        messages=messages,
        temperature=0,
        max_tokens=256,
    )
    response_text = extract_chat_text(response).strip()
    try:
        payload = json.loads(response_text)
    except json.JSONDecodeError as exc:
        raise ValueError("Model did not return valid JSON") from exc

    status = payload.get("status")
    if status not in {"ok", "error"}:
        raise ValueError("Model returned an invalid status")

    if status == "ok":
        proposition = payload.get("proposition")
        if not isinstance(proposition, str) or not proposition.strip():
            raise ValueError("Model did not return a proposition")
        return {"status": "ok", "proposition": proposition.strip()}

    reason = payload.get("reason")
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("Model did not return an error reason")
    reason = reason.strip().lower()
    if reason not in {"not real", "not weighty", "not safe"}:
        raise ValueError("Model returned an invalid error reason")
    return {"status": "error", "reason": reason}


def moderate_content(content: str) -> bool:
    """
    Check content with OpenAI's moderation API and apply tailored logic
    to reduce false positives on mild, non-violent insults while still
    flagging clearly harmful categories (e.g., self-harm, sexual content
    with minors, graphic violence, and threats).

    Parameters:
    - content (str): The message content to be moderated

    Returns:
    - bool: True if content is inappropriate, False otherwise
    """
    logger.info("Moderating content: %s", content)

    # Use an explicit model; omni-moderation-latest is recommended.
    client = OpenAI()
    response = client.moderations.create(
        model="omni-moderation-latest",
        input=content,
    )

    result = response.results[0]

    # Raw fields from API
    api_flagged = bool(getattr(result, "flagged", False))
    categories = dict(getattr(result, "categories", {}) or {})
    scores = dict(getattr(result, "category_scores", {}) or {})

    # Always flag clearly harmful categories regardless of overall flagged.
    def cat(name: str) -> bool:
        return bool(categories.get(name, False))

    def score(name: str) -> float:
        try:
            return float(scores.get(name, 0.0))
        except (TypeError, ValueError):
            return 0.0

    # High-severity categories we always block
    always_block = (
        cat("self-harm")
        or cat("self-harm/intent")
        or cat("self-harm/instructions")
        or cat("sexual/minors")
        or cat("violence/graphic")
        or cat("violence")
        or cat("hate/threatening")
        or cat("harassment/threatening")
    )

    if always_block:
        logger.info(
            "Content flagged (high severity). Categories: %s Scores: %s",
            categories,
            scores,
        )
        return True

    # For milder categories like generic harassment/insults, require
    # a higher confidence to reduce false positives on debate-style text.
    harassment = cat("harassment") and score("harassment") >= 0.85

    # Fall back: if API flagged but none of our high-severity categories hit,
    # only treat as flagged when harassment confidence is high.
    is_flagged = harassment or False

    logger.info(
        (
            "Content moderation decision -> api_flagged=%s, decided_flagged=%s, "
            "categories=%s, scores=%s"
        ),
        api_flagged,
        is_flagged,
        categories,
        scores,
    )
    return is_flagged
