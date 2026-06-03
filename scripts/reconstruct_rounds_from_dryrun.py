"""
scripts/reconstruct_rounds_from_dryrun.py

Reconstruct results files from the `read_database save-rounds --dry-run` text
output that was saved to a file (e.g., `saved_rounds.txt`).

Notes/assumptions:
- The dry-run prints a human-readable Condition header, followed by one or more
  Round summaries separated by a line of 20 spaces and 20 dashes.
- Player boundaries are NOT printed in the dry-run output. Therefore, we cannot
  recover which rounds belong to the same participant. This script writes each
  parsed Round as its own JSON list (i.e., one round per line) under the
  corresponding Condition directory.
- Some Round fields are not fully recoverable from the printed summaries
  (e.g., transcripts, chains_of_thought, reasoning_traces, full mouse_traces).
  These are left as defaults or omitted. The script will capture
  `proposition_during_round` if the dry-run output includes a
  "## Proposition During Round" section.

Usage:
    python scripts/reconstruct_rounds_from_dryrun.py \
        --input saved_rounds.txt \
        [--results-dir results]

This creates: results/<condition_dir>/<YYYY-MM-DD>.jsonl

Activate your venv first, e.g.:
    source env-continuouspersuasion/bin/activate
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import re
from dataclasses import dataclass
from typing import Any

# --------------------------- Parsing helpers ---------------------------


CONDITION_SEP_RE = re.compile(r"^-{80}\s*$")
ROUND_SEP_RE = re.compile(r"^\s{20}-{20}\s*$")


@dataclass
class ParsedCondition:
    """Minimal condition info parsed from the printed header line."""

    roles_str: str
    tags: list[str]

    def to_condition_dict(self) -> dict[str, Any]:
        """Convert parsed text into a dict approximating Condition fields.

        We recover enough for directory naming via Condition.to_dir format:
        - roles (human_persuader/target booleans, or llm_* model name short labels)
        - factual_domain / proposition_is_correct derived from tags
        - continuous_measure from tags like 'cont=serial' or 'cont=trace'
        - use_audio/show_transcript from 'audio' or 'audio+transcript'; else 'text'
        - synthetic_audio from 'synthetic'
        - control_dialogue from 'control'
        - turn_limit from 'turns=<n>'
        - minimum_turns from 'min_turns=<n>'

        NB: We map short LLM names to full identifiers using a best-effort map.
        """
        roles = _parse_roles(self.roles_str)

        # Defaults reflect Condition defaults in code
        cond: dict[str, Any] = {
            "roles": roles,
            "factual_domain": True,
            "proposition_is_correct": None,
            "continuous_measure": None,
            "synthetic_audio": False,
            "use_audio": False,
            "show_transcript": False,
            "control_dialogue": False,
            "turn_limit": None,
            "minimum_turns": None,
        }

        for t in self.tags:
            if t == "factual":
                cond["factual_domain"] = True
                cond["proposition_is_correct"] = None
            elif t == "non-factual":
                cond["factual_domain"] = False
                cond["proposition_is_correct"] = None
            elif t == "correct":
                cond["factual_domain"] = True
                cond["proposition_is_correct"] = True
            elif t == "incorrect":
                cond["factual_domain"] = True
                cond["proposition_is_correct"] = False
            elif t.startswith("cont="):
                cm = t.split("=", 1)[1]
                if cm == "serial":
                    cond["continuous_measure"] = "serial-questions"
                elif cm == "trace":
                    cond["continuous_measure"] = "mouse-trace"
                else:
                    cond["continuous_measure"] = cm
            elif t == "audio":
                cond["use_audio"] = True
                cond["show_transcript"] = False
            elif t == "audio+transcript":
                cond["use_audio"] = True
                cond["show_transcript"] = True
            elif t == "text":
                cond["use_audio"] = False
                cond["show_transcript"] = False
            elif t == "synthetic":
                cond["synthetic_audio"] = True
            elif t == "control":
                cond["control_dialogue"] = True
            elif t.startswith("turns="):
                try:
                    cond["turn_limit"] = int(t.split("=", 1)[1])
                except ValueError:
                    pass
            elif t.startswith("min_turns="):
                try:
                    cond["minimum_turns"] = int(t.split("=", 1)[1])
                except ValueError:
                    pass

        return cond


SHORT_TO_FULL_MODEL = {
    # best-effort inversion of src/experiment/utils.MODEL_NAMES_SHORT
    "llama2-70b": "meta-llama/Llama-2-70b-chat-hf",
    "llama3.1-8b": "meta-llama/Llama-3.1-8B-Instruct",
    "llama3.1-70b": "meta-llama/Llama-3.1-70B-Instruct",
    "llama3.1-405b": "meta-llama/Llama-3.1-405B-Instruct",
    "claude3.5-sonnet": "claude-3-5-sonnet-20240620",
    "gpt-4o": "gpt-4o-2024-08-06",
    "o3": "o3-2025-04-16",
}


def _parse_roles(roles_str: str) -> dict[str, Any]:
    """Parse roles like "o3 Persuader, Human Target" into a Roles-like dict.

    Returns dict suitable for Condition.roles model:
      { human_persuader: bool|int, human_target: bool|int,
        llm_persuader: str|None, llm_target: str|None }
    """
    # Example: "Human Persuader, Human Target" or "llama3.1-8b Persuader, Human Target"
    m = re.match(r"^\s*(.+?)\s+Persuader,\s+(.+?)\s+Target\s*$", roles_str)
    if not m:
        raise ValueError(f"Unrecognized roles line: {roles_str!r}")
    persuader_tok, target_tok = m.group(1), m.group(2)

    def human_like(tok: str) -> bool:
        return tok.startswith("Human")

    res: dict[str, Any] = {
        "human_persuader": False,
        "human_target": False,
        "llm_persuader": None,
        "llm_target": None,
    }

    if human_like(persuader_tok):
        res["human_persuader"] = True
    else:
        res["llm_persuader"] = SHORT_TO_FULL_MODEL.get(persuader_tok, persuader_tok)

    if human_like(target_tok):
        res["human_target"] = True
    else:
        # If printed (rare), map short LLM names for targets
        res["llm_target"] = SHORT_TO_FULL_MODEL.get(target_tok, target_tok)

    return res


def _parse_condition_header(line: str) -> ParsedCondition | None:
    """Parse the single-line Condition.__str__ output.

    Example input:
      "o3 Persuader, Human Target [non-factual, cont=serial, text, turns=10]"
    or without tags brackets if no tags were present.
    """
    line = line.strip()
    if not line:
        return None

    # Split optional tags in [ ... ]
    roles_part = line
    tags: list[str] = []
    m = re.match(r"^(.*)\s*\[(.*)\]\s*$", line)
    if m:
        roles_part = m.group(1).strip()
        raw_tags = m.group(2).strip()
        if raw_tags:
            tags = [t.strip() for t in raw_tags.split(",")]
    return ParsedCondition(roles_part, tags)


def _split_blocks(lines: list[str]) -> list[list[str]]:
    """Split the whole file into condition blocks on 80-dash separators."""
    blocks: list[list[str]] = []
    cur: list[str] = []
    for ln in lines:
        if CONDITION_SEP_RE.match(ln):
            if cur:
                blocks.append(cur)
                cur = []
        else:
            cur.append(ln)
    if cur:
        blocks.append(cur)
    return blocks


def _split_rounds(lines: list[str]) -> list[list[str]]:
    """Split a condition block into round summary chunks on 20sp+20dash lines."""
    rounds: list[list[str]] = []
    cur: list[str] = []
    for ln in lines:
        if ROUND_SEP_RE.match(ln):
            if cur:
                rounds.append(cur)
                cur = []
        else:
            cur.append(ln)
    if cur:
        rounds.append(cur)
    return rounds


def _parse_round_block(lines: list[str]) -> dict[str, Any] | None:
    """Parse a single Round.__str__ text block into a dict model_dump-like.

    Returns None if the block doesn't look like a Round.
    """
    text = "".join(lines)
    if "# Round Summary #" not in text:
        return None

    proposition = _extract_single_line_after_heading(lines, "## Proposition")
    proposition_during = _extract_single_line_after_heading(
        lines, "## Proposition During Round"
    )

    init_belief = _extract_scalar_from_prefix(lines, "- Initial:")
    final_belief = _extract_scalar_from_prefix(lines, "- Final:")

    stance = _extract_scalar_from_prefix(lines, "- Stance:")
    if stance is not None:
        persuader_supports = None
        s = str(stance).strip().lower()
        if s == "supports":
            persuader_supports = True
        elif s == "opposes":
            persuader_supports = False
    else:
        persuader_supports = None

    messages = _extract_messages(lines)

    rd: dict[str, Any] = {
        "proposition": proposition or "",
        "proposition_during_round": proposition_during or None,
        "target_initial_belief": (
            float(init_belief) if init_belief is not None else None
        ),
        "target_final_belief": (
            float(final_belief) if final_belief is not None else None
        ),
        "persuader_supports_proposition": persuader_supports,
        # fields we cannot reliably recover are left as defaults/omitted
        "messages": messages,
        # optional fields defaulting to None/[] as in the model
        "serial_questions": None,
        "mouse_traces": None,
        "transcripts": [],
        "chains_of_thought": [],
        "reasoning_traces": [],
        "target_ended_round": None,
        "timed_out": False,
    }
    return rd


def _extract_single_line_after_heading(lines: list[str], heading: str) -> str | None:
    """Find a heading line then return the very next non-empty line's text."""
    for i, ln in enumerate(lines):
        if ln.strip() == heading:
            # Next non-empty line is the value
            for j in range(i + 1, len(lines)):
                val = lines[j].rstrip("\n")
                if val.strip() == "":
                    continue
                return val.strip()
    return None


def _extract_scalar_from_prefix(lines: list[str], prefix: str) -> float | str | None:
    """Scan for a line starting with prefix and return the trailing token."""
    for ln in lines:
        s = ln.strip()
        if s.startswith(prefix):
            val = s[len(prefix) :].strip()
            return val
    return None


def _extract_messages(lines: list[str]) -> list[dict[str, str]]:
    """Reconstruct messages from the formatted "## Message History" block.

    Uses the printed layout rules:
    - two spaces indent
    - label padded to a uniform width, then content
    - blank line between messages
    """
    # Find start of message history
    start = None
    for i, ln in enumerate(lines):
        if ln.strip() == "## Message History":
            start = i + 1
            break
    if start is None:
        return []

    # Collect lines until the next heading or end of block
    msg_lines: list[str] = []
    for ln in lines[start:]:
        if ln.strip().startswith("## "):
            break
        msg_lines.append(ln.rstrip("\n"))

    # Split on blank lines
    chunks: list[list[str]] = []
    cur: list[str] = []
    for ln in msg_lines:
        if ln.strip() == "":
            if cur:
                chunks.append(cur)
                cur = []
        else:
            cur.append(ln)
    if cur:
        chunks.append(cur)

    messages: list[dict[str, str]] = []
    for ch in chunks:
        # First line has label and content
        m = re.match(r"^(\s*)([^:]+:)(\s*)(.*)$", ch[0])
        if not m:
            # if format unexpected, skip this chunk
            continue
        indent, label_with_colon, pad, content0 = m.groups()
        content_start_idx = len(indent) + len(label_with_colon) + len(pad)
        role = label_with_colon[:-1].lower()
        # Standardize roles back to persuader/target/system/user/assistant
        role_map = {
            "persuader": "persuader",
            "target": "target",
            "system": "system",
            "assistant": "assistant",
            "user": "user",
        }
        role_std = role_map.get(role, role)

        content_lines = [content0]
        for ln in ch[1:]:
            # Continuations are aligned to the same tab stop
            if len(ln) >= content_start_idx:
                content_lines.append(ln[content_start_idx:])
            else:
                content_lines.append(ln.strip())
        # Join lines with newlines to preserve paragraph breaks
        content = "\n".join(content_lines)
        messages.append({"role": role_std, "content": content})

    return messages


# ------------------------------ Main ----------------------------------


def main() -> None:
    """CLI entry to parse dry-run output and write results JSONL files."""
    parser = argparse.ArgumentParser(description="Rebuild results from dry-run text")
    parser.add_argument("--input", required=True, help="Path to saved_rounds.txt")
    parser.add_argument(
        "--results-dir", default="results", help="Directory to write results into"
    )
    parser.add_argument(
        "--date",
        default=_dt.date.today().isoformat(),
        help="Date string YYYY-MM-DD for output filename (default: today)",
    )
    args = parser.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        all_lines = f.readlines()

    # Split by conditions
    condition_blocks = _split_blocks(all_lines)
    if not condition_blocks:
        raise ValueError(
            "No condition blocks found in input. Is this the dry-run output?"
        )

    os.makedirs(args.results_dir, exist_ok=True)

    for block in condition_blocks:
        # The first non-empty line should be the condition header
        header_line = None
        for ln in block:
            if ln.strip():
                header_line = ln
                break
        if not header_line:
            continue

        parsed = _parse_condition_header(header_line)
        if not parsed:
            continue
        condition_dict = parsed.to_condition_dict()

        # Build a dir name similar to Condition.to_dir
        # roles part
        roles = condition_dict.pop("roles")
        roles_items = [(k, v) for k, v in roles.items() if v not in (False, None)]
        roles_items.sort(key=lambda kv: kv[0])
        roles_kv = [f"{k}={_escape_dir(str(v))}" for k, v in roles_items]
        roles_part = "&".join(roles_kv)

        # condition part (exclude defaults that are None/False)
        cond_items = [
            (k, v)
            for k, v in condition_dict.items()
            if v not in (None, False)  # keep True and concrete values
        ]
        cond_items.sort(key=lambda kv: kv[0])
        cond_kv = [f"{k}={_escape_dir(str(v))}" for k, v in cond_items]
        dir_name = roles_part + ("&" + "&".join(cond_kv) if cond_kv else "")

        # Gather round blocks within this condition block
        # Remove the header line and its immediate blank line if present
        start_idx = block.index(header_line) + 1
        rounds_area = block[start_idx:]
        round_blocks = _split_rounds(rounds_area)

        # Parse each round and emit as a list with a single round
        parsed_rounds = []
        for rb in round_blocks:
            rd = _parse_round_block(rb)
            if rd is not None:
                # Attach the reconstructed Condition so downstream loaders validate
                rd_with_cond = dict(rd)
                rd_with_cond["condition"] = {"roles": roles, **condition_dict}
                parsed_rounds.append(rd_with_cond)

        if not parsed_rounds:
            continue

        # Write JSONL file
        out_dir = os.path.join(args.results_dir, dir_name)
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"{args.date}.jsonl")
        with open(out_path, "w", encoding="utf-8") as outf:
            for rd in parsed_rounds:
                outf.write(json.dumps([rd], ensure_ascii=False) + "\n")

        print(f"Wrote {len(parsed_rounds)} rounds to {out_path}")


def _escape_dir(value: str) -> str:
    """Mimic src/experiment/utils.escape_string for directory-safe parts."""
    if any(x in value for x in ("&", "\\", "__")):
        # very conservative: replace instead of raising
        v = value.replace("\\", "-").replace("__", "_").replace("&", "+")
        return v.replace("/", "__")
    return value.replace("/", "__")


if __name__ == "__main__":
    main()
