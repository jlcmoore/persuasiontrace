"""
src/experiment/utils.py

Author: Jared Moore
Date: July, 2025

Contains utility functions used throughout the package.
"""

import importlib
import logging
import os
import random
import re
import sys
import textwrap
from enum import Enum
from typing import Any, Dict

import inflect
import textstat

import data

logger = logging.getLogger(__name__)

RESULTS_DIR = "results"


####


def extract_chat_text(response: Any) -> str:
    """
    Extract the assistant text content from an OpenAI chat completion response.
    """
    choices = getattr(response, "choices", None) or []
    if not choices:
        raise ValueError("OpenAI chat completion returned no choices")
    first = choices[0]
    message = getattr(first, "message", None)
    if isinstance(message, dict):
        content = message.get("content")
    else:
        content = getattr(message, "content", None)
    if not isinstance(content, str) or not content.strip():
        raise ValueError("OpenAI chat completion returned empty content")
    return content


#########

MODEL_NAMES_SHORT = {
    # NB: assuming no base models
    "meta-llama/Llama-2-70b-chat-hf": "llama2-70b",
    "meta-llama/Llama-3.1-8B-Instruct": "llama3.1-8b",
    "meta-llama/Llama-3.1-70B-Instruct": "llama3.1-70b",
    "meta-llama/Llama-3.1-405B-Instruct": "llama3.1-405b",
    "claude-3-5-sonnet-20240620": "claude3.5-sonnet",
    # TODO: add in more models
    "gpt-4o-2024-08-06": "gpt-4o",
    "o3-2025-04-16": "o3",
}

#########


def limit_to_n_characters(message: str, n: int) -> str:
    """Returns up to n characters of the message."""
    return message[0:n]


def limit_text_to_char_and_audio_budget(
    text: str,
    *,
    use_audio: bool,
    max_response_chars: int,
    max_audio_duration_s: int | None,
) -> str:
    """Limit text to a character budget and optional estimated audio budget.

    Args:
        text: Raw text to limit.
        use_audio: Whether audio duration constraints apply.
        max_response_chars: Maximum allowed characters.
        max_audio_duration_s: Maximum allowed estimated audio duration in seconds.

    Returns:
        The limited text, truncated by character and optional estimated duration.
    """
    limited = limit_to_n_characters(text, max_response_chars)
    if not use_audio or max_audio_duration_s is None:
        return limited

    est_transcript = make_text_transcript(limited)
    duration = float(est_transcript.get("duration", 0.0))
    if duration <= max_audio_duration_s or duration <= 0.0:
        return limited

    ratio = max_audio_duration_s / duration
    target_chars = max(1, int(len(limited) * ratio))
    return limit_to_n_characters(limited, target_chars)


def model_name_short(model_name: str) -> str:
    """Shortens the passed model name, if possible"""
    if model_name in MODEL_NAMES_SHORT:
        return MODEL_NAMES_SHORT[model_name]
    return model_name


#########


def get_data_file_path(filename) -> str:
    """
    For a given name of a package file (in src/data) returns the full file path for that file
    or errors if it does not exist.
    """
    data_path = importlib.resources.files(data) / filename
    if not os.path.exists(data_path):
        raise ValueError(f"Package file does not exist: {filename}")
    return str(data_path)


EXAMPLE_PROPOSITIONS_FILE = "example_propositions"


def set_logger(level, local_logger=None):
    """Convert the log level string to a logging level, raises an error if not a valid level"""
    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {level}")

    # Only configure if not already configured to avoid no-op
    if not logging.getLogger().hasHandlers():
        logging.basicConfig(
            level=logging.DEBUG,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[logging.StreamHandler(sys.stdout)],
        )

    if local_logger:
        local_logger.setLevel(numeric_level)


#########


def normalize_serial_sentence_values(
    raw_values: Any | None,
    *,
    context: str,
    round_id: int | None = None,
) -> list[float] | None:
    """Convert sentence-level belief values to floats, logging malformed inputs."""

    if raw_values is None:
        return None

    if not isinstance(raw_values, list):
        logger.error(
            "Non-list serial-question-sentence payload in %s for round %s",
            context,
            round_id,
        )
        return []

    cleaned: list[float] = []
    for belief in raw_values:
        try:
            cleaned.append(float(belief))
        except (TypeError, ValueError):
            logger.error(
                "Non-numeric serial-question-sentence value in %s for round %s",
                context,
                round_id,
            )
    return cleaned


#########


def normalize_message_highlight(
    raw_values: Any | None,
    *,
    context: str,
    round_id: int | None = None,
) -> list[dict[str, Any]] | None:
    """Convert per-message highlight payloads to a list of dicts."""

    if raw_values is None:
        return None
    if not isinstance(raw_values, list):
        logger.error(
            "Non-list message-highlight payload in %s for round %s",
            context,
            round_id,
        )
        return []

    cleaned: list[dict[str, Any]] = []
    for entry in raw_values:
        if isinstance(entry, dict):
            cleaned.append(dict(entry))
        else:
            logger.error(
                "Non-dict message-highlight entry in %s for round %s",
                context,
                round_id,
            )
    return cleaned


#########


def normalize_mouse_traces(
    raw_traces: Any | None,
    *,
    mapping_fn: "Callable[[float], float] | None" = None,
    backfill_timestamps: bool = True,
) -> list[list[dict[str, float]]] | None:
    """Normalize mouse traces to a JSON-serializable, analysis-ready format.

    Behavior:
    - Accepts nested lists of dicts with keys 'timestamp' and 'position'.
    - Converts 'position' from [0,100] to [0,1].
    - Optionally applies a mapping_fn to each [0,1] position (e.g., persuader-relative).
    - Converts timestamps to float and, when enabled (backfill_timestamps=True),
      performs in-segment forward-fill of missing values and backfills any leading
      None timestamps to 0.0. This preserves relative timing while ensuring a
      monotone, well-defined time series within each segment.

    Returns:
    - Nested list of dicts with float fields 'timestamp' and 'position', or None
      if raw_traces is None.
    """

    if raw_traces is None:
        return None
    result: list[list[dict[str, float]]] = []
    for seg in raw_traces if isinstance(raw_traces, list) else []:
        if not isinstance(seg, list):
            result.append([])
            continue
        timestamps: list[float | None] = []
        positions01: list[float] = []
        for pt in seg:
            if not (isinstance(pt, dict) and "position" in pt):
                continue
            try:
                pos01 = float(pt.get("position")) / 100.0
            except (TypeError, ValueError):
                continue
            if mapping_fn is not None:
                # Apply optional mapping; catch common numeric conversion errors only.
                try:
                    pos01 = float(mapping_fn(pos01))
                except (TypeError, ValueError):
                    # Leave pos01 unchanged on non-numeric outputs from mapping_fn
                    pass
            ts_val = pt.get("timestamp")
            try:
                ts = float(ts_val) if ts_val is not None else None
            except (TypeError, ValueError):
                ts = None
            positions01.append(pos01)
            timestamps.append(ts)

        if backfill_timestamps:
            last: float | None = None
            for i, ts in enumerate(timestamps):
                if ts is None and last is not None:
                    timestamps[i] = last
                else:
                    last = ts
            if timestamps and timestamps[0] is None:
                for i, ts in enumerate(timestamps):
                    if ts is None:
                        timestamps[i] = 0.0
                    else:
                        break

        seg_out: list[dict[str, float]] = []
        for ts, pos in zip(timestamps, positions01):
            if ts is None:
                continue
            seg_out.append({"timestamp": float(ts), "position": float(pos)})
        result.append(seg_out)

    return result


#########


def prefix_with_conjunction(prefix, conjunction, lst):
    """
    Convert a list of strings into a single string with `prefix` and `conjunction`
    before the last item.

    Parameters:
        lst (list of str): The list of strings to be joined.

    Returns:
        str: A single string with items separated by `prefix` and `conjunction`
            before the last item. Returns an empty string if the list is empty.
    """
    lst = list(lst)
    if len(lst) == 0:
        return ""
    if len(lst) == 1:
        return lst[0]
    return (prefix + " ").join(lst[:-1]) + " " + conjunction + " " + lst[-1]


def comma_with_and(lst):
    """
    Like `prefix_with_and` where prefix=',' and conjunction='and'
    """
    return prefix_with_conjunction(",", "and", lst)


def int_to_words(number):
    """
    Convert an integer to its word representation.

    Parameters:
        number (int): The integer to be converted.

    Returns:
        str: The word representation of the integer.
    """
    p = inflect.engine()
    return p.number_to_words(number)


def number_to_words_ordinal(n):
    """
    Convert an integer to its ordinal word representation.

    Parameters:
        n (int): The integer to be converted.

    Returns:
        str: The ordinal word representation of the integer.
    """
    p = inflect.engine()
    return p.number_to_words(p.ordinal(n))


def replace_json_chars(text):
    """
    Replaces senstive characters used in JSON parsing.

    Parameters:
        text (str): The string to replace.

    Returns:
        str: The replacement
    """
    return (
        text.replace("“", '"')
        .replace("”", '"')
        .replace("‘", '"')
        .replace("’", '"')
        .replace("'", '"')
        .replace("：", ":")
        .replace("，", ",")
        .replace("```json", "")
        .replace("```", "")
    )


#########


def updated_belief_after_persuader(
    current: float,
    supports: bool,
    persuader_text: str,
    *,
    jitter_range: tuple[float, float] = (0.85, 1.15),
) -> float:
    """Heuristic update of target belief after a persuader message.

    Rationale and shape of the heuristic:
    - We want a simple, content-agnostic update that (a) moves in the
      persuader's direction, (b) scales with message "effort/variety",
      and (c) stays bounded, stable and reproducible for toy runs.
    - We therefore compute two lightweight proxies from `persuader_text`:
        * length_factor in [0,1]: len(text)/900 capped at 1.0. Longer
          messages have more potential influence, but with diminishing
          returns beyond ~900 chars.
        * novelty in [0,1]: unique_token_count/total_token_count. More
          varied wording (less repetition) is credited slightly more.
    - We combine these into a small base delta in [~0.04, ~0.26]:
        base_delta = 0.04 + 0.14*length_factor + 0.08*novelty
      The constants keep movements modest and tunable, and ensure some
      minimum effect even for short, varied messages.
    - We multiply by a jitter factor U(jitter_range) (default 0.85–1.15)
      to avoid identical traces across similar conversations while keeping
      the effect bounded and visually stable.
    - Direction is + if the persuader supports the proposition and − if the
      persuader opposes. The result is then clipped to [0,1].

    Inputs:
    - current: current belief in [0,1]
    - supports: True if persuader supports the proposition; False if opposes
    - persuader_text: content of persuader's message
    - jitter_range: multiplicative noise bounds to avoid identical traces

    Returns a value clipped to [0,1]. This heuristic is intentionally simple
    (no semantic modeling) so it is fast, deterministic up to jitter, and easy
    to swap out later without touching storage code.
    """
    length_factor = min(len(persuader_text or "") / 900.0, 1.0)
    words = (persuader_text or "").split()
    novelty = len(set(w.lower() for w in words)) / (len(words) if words else 1)

    base_delta = 0.04 + 0.14 * length_factor + 0.08 * novelty

    # Jitter
    low, high = jitter_range
    jitter = random.uniform(low, high)

    delta = base_delta * jitter
    direction = 1.0 if supports else -1.0
    new_val = current + direction * delta
    return max(0.0, min(1.0, new_val))


# final belief selection now lives on Round.final_belief_from_measures


def interpolate_belief_sequence(start: float, end: float, steps: int) -> list[float]:
    """Interpolate beliefs from start to end over `steps` increments.

    Returns a list of length `steps` where i-th value is linear interpolation
    between start and end at i/steps. If steps <= 0 returns [].
    """
    if steps <= 0:
        return []
    vals: list[float] = []
    for i in range(1, steps + 1):
        t = i / steps
        vals.append((1 - t) * float(start) + t * float(end))
    return vals


def initial_belief_for_llm_target(low: float = 0.0, high: float = 1.0) -> float:
    """Choose an initial belief for an LLM target.

    Centralizes the policy so server and runners remain consistent.
    """
    return random.uniform(low, high)


def escape_string(value: str) -> str:
    """
    Escape characters in a string that could be problematic in filenames or directory names.

    Args:
        value (str): The string to escape.

    Returns:
        str: The escaped string.
    """
    if "&" in value or "\\" in value or "__" in value:
        raise TypeError("Values must not contain '&' nor '\\' nor '__'.")
    return value.replace("/", "__")


def unescape_string(value: str) -> str:
    """
    Unescape characters in a string that were previously escaped.

    Args:
        value (str): The string to unescape.

    Returns:
        str: The original string with escaped characters restored.
    """
    return value.replace("__", "/")


def dict_to_string(d: Dict[str, Any]) -> str:
    """
    Convert a dictionary into a string of key=value pairs separated by &,
    sorted by keys. Handles integers, booleans, strings, and None.

    Args:
        d (Dict[str, Any]): The dictionary to convert.

    Returns:
        str: A directory safe string representation of the dictionary.

    Raises:
        TypeError: If a value in the dictionary is not an int, bool, str, or None.
        Also if a string contains a `&` or a `=`
    """
    sanitized: Dict[str, Any] = {}

    # Validate and normalize each value type (including Enums -> their values)
    for key, value in d.items():
        normalized = value.value if isinstance(value, Enum) else value
        if not isinstance(normalized, (int, bool, str, type(None))):
            raise TypeError(
                f"Unsupported type for key '{key}': {type(normalized).__name__}"
            )
        value_str = str(normalized) if normalized is not None else "None"
        if "&" in value_str or "=" in value_str or "&" in key or "=" in key:
            raise ValueError("Cannot pass in values or keys with = or &")
        sanitized[key] = normalized

    # Ensure the dictionary is sorted by keys
    sorted_items = sorted(sanitized.items())

    # Create a keyword=value formatted string, converting `None` to a string
    kv_pairs = [
        f"{key}={escape_string(str(value)) if value is not None else 'None'}"
        for key, value in sorted_items
    ]

    result_string = "&".join(kv_pairs)

    return result_string


def string_to_dict(s: str) -> Dict[str, Any]:
    """
    Convert a string of key=value pairs separated by & back into a dictionary.
    Handles integers, booleans, strings, and None.

    Args:
        s (str): The string representation of the dictionary.

    Returns:
        Dict[str, Any]: The reconstructed dictionary.
    """

    # Define a helper function to convert string to correct type
    def convert_value(value: str) -> Any:
        value = unescape_string(value)  # Unescape strings first
        if value.isdigit():
            return int(value)
        if value.lower() == "true":
            return True
        if value.lower() == "false":
            return False
        if value == "None":
            return None
        return value

    # Split the string into key=value pairs
    kv_pairs = s.split("&")

    # Convert to dictionary with proper type handling
    return {
        key: convert_value(value)
        for key, value in (pair.split("=") for pair in kv_pairs)
    }


def seconds_to_min_sec(n: int) -> str:
    """
    Returns n as a string of x minutes and y seconds
    """
    sign = "-" if n < 0 else ""
    m, s = divmod(abs(int(n)), 60)
    return f"{sign}{m} minute{'s' if m != 1 else ''} {s} second{'s' if s != 1 else ''}"


######


def make_text_transcript(
    text: str,
    syllables_per_second: float = 4.0,
    min_word_duration: float = 0.08,
    inter_word_gap: float = 0.02,
    pause_map: dict[str, float] | None = None,
    language: str = "en",
) -> dict[str, Any]:
    """
    Create a synthetic transcript with time-aligned segments for plain text (no audio).

    - Longer words (more syllables) take longer.
    - Shorter words take less time.
    - Adds naturalistic pauses for punctuation and newlines.

    Returns a dict with:
      {
        "text": <original text>,
        "language": "en",
        "duration": <float seconds>,
        "words": [
          {
            "start": <float>,
            "end": <float>,
            "word": <segment_text>,
          },
          ...
        ]
      }
    """
    if pause_map is None:
        pause_map = {
            ",": 0.12,
            ";": 0.15,
            ":": 0.15,
            ".": 0.25,
            "!": 0.25,
            "?": 0.25,
            "—": 0.12,
            "-": 0.08,
            "…": 0.30,
            "...": 0.30,
            "\n": 0.30,
        }

    def estimate_syllables(word: str) -> int:
        syl = textstat.syllable_count(word)  # pylint: disable=no-member
        return max(1, int(syl))

    # Tokenize: keep ellipses, words with apostrophes, and single char punctuation
    token_pattern = re.compile(r"(\.\.\.|[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?|[^\s])")
    tokens = token_pattern.findall(text)

    all_words: list[dict[str, Any]] = []
    current_time = 0.0

    def is_sentence_break(tok: str) -> bool:
        return tok in (".", "!", "?", "\n", "...", "…")

    for i, tok in enumerate(tokens):
        is_word = bool(re.match(r"[A-Za-z0-9]", tok))

        if is_word:
            syl = estimate_syllables(tok)
            word_dur = max(min_word_duration, syl / syllables_per_second)
            word_start = current_time
            word_end = word_start + word_dur
            all_words.append(
                {
                    "start": round(word_start, 3),
                    "end": round(word_end, 3),
                    "word": tok,
                    "punct": False,
                }
            )
            current_time = word_end

            # Add a small inter-word gap unless next token is sentence-ending punctuation
            next_tok = tokens[i + 1] if i + 1 < len(tokens) else None
            if next_tok and not is_sentence_break(next_tok):
                current_time += inter_word_gap

        else:
            # Punctuation pause
            pause = pause_map.get(tok, 0.0)
            if pause > 0:
                # Record punctuation as a timed item (optional but useful for highlighting)
                punct_start = current_time
                punct_end = punct_start + pause
                all_words.append(
                    {
                        "start": round(punct_start, 3),
                        "end": round(punct_end, 3),
                        "word": tok,
                        "punct": True,
                    }
                )
                current_time = punct_end

    # If nothing was segmented (e.g., empty text), still return a minimal structure
    if not all_words and text.strip():
        all_words = [
            {
                "start": 0.0,
                "end": round(current_time, 3),
                "word": text.strip(),
            }
        ]

    return {
        "language": language,
        "duration": round(current_time, 3),
        "text": text,
        "words": all_words,
    }


def token_time_totals_verbose(
    transcript: dict[str, Any] | None,
) -> list[dict[str, float]] | None:
    """
    Convert Whisper verbose_json output into a list of
    {"text": <token>, "duration": <float>} dicts whose durations
    together cover the entire transcript.

    The duration of a token is defined as the interval from its own
    ``start`` time up to the *start time of the next token*
    (or, for the final token, up to ``transcript["duration"]`` if
    present, otherwise ``end - start``).

    If *transcript* is None return None.
    """

    if transcript is None:
        return None

    words = transcript.get("words")
    if not isinstance(words, list):
        raise ValueError("Expected Whisper verbose_json with top-level 'words' list")

    # Sort defensively by start time so that "next token" is well-defined
    words = sorted(words, key=lambda w: float(w["start"]))

    totals: list[dict[str, float]] = []

    overall_duration = float(transcript.get("duration", 0.0))
    n = len(words)

    for i, w in enumerate(words):
        start = float(w["start"])

        # Default duration – only used for the last token if we cannot
        # do better.
        fallback_dur = max(0.0, float(w["end"]) - start)

        if i + 1 < n:
            # Duration runs up to the *next* token's start
            next_start = float(words[i + 1]["start"])
            duration = max(0.0, next_start - start)
        else:  # last token
            if overall_duration > start:
                duration = overall_duration - start
            else:
                duration = fallback_dur

        text = w.get("text", w.get("word", ""))
        totals.append({"text": text, "duration": duration})

    return totals


def format_message_history(
    messages: list[dict[str, str]],
    base_indent: int = 2,
    width: int = 80,
) -> str:
    """
    Format messages as:
      <indent>Persuader: <text that wraps to same tab stop>
      <indent>Target:    <text that wraps to same tab stop>

    - Continuation lines align to the same tab stop.
    - A blank line is inserted after each message.
    - No line exceeds 'width' characters.
    """

    label_for = {
        "persuader": "Persuader:",
        "target": "Target:",
        "system": "System:",
        "assistant": "Assistant:",
        "user": "User:",
    }

    labels = [
        label_for.get(m.get("role", ""), f'{str(m.get("role","")).capitalize()}:')
        for m in messages
    ]
    if not labels:
        return ""

    label_width = max(len(lbl) for lbl in labels) + 1  # +1 space after label
    indent_str = " " * base_indent
    text_width = max(10, width - base_indent - label_width)

    out: list[str] = []

    for msg in messages:
        role = msg.get("role", "")
        raw = msg.get("content") or ""
        label = label_for.get(role, f"{role.capitalize()}:")
        label_col = label.ljust(label_width)

        if raw == "":
            # Emit labeled empty message
            out.append(f"{indent_str}{label_col}")
            out.append("")  # blank line after this message
            continue

        first_visual = True
        for para in raw.splitlines() or [""]:
            if para.strip() == "":
                # Preserve explicit blank lines inside a message,
                # aligned at the content tab stop
                if first_visual:
                    # If the first line is blank, still show the label once
                    out.append(f"{indent_str}{label_col}")
                    first_visual = False
                else:
                    out.append(f"{indent_str}{' ' * label_width}")
                continue

            wrapped = textwrap.fill(
                para,
                width=text_width,
                initial_indent="",
                subsequent_indent="",
                break_long_words=True,
                break_on_hyphens=True,
            )
            for subline in wrapped.splitlines():
                if first_visual:
                    out.append(f"{indent_str}{label_col}{subline}")
                    first_visual = False
                else:
                    out.append(f"{indent_str}{' ' * label_width}{subline}")

        # Blank line after each message
        out.append("")

    return "\n".join(out)
