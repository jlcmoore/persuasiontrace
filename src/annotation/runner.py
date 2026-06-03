"""
Annotate persuader messages from round results using LiteLLM.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator, Sequence

import litellm
from litellm.exceptions import (
    APIConnectionError,
    APIError,
    AuthenticationError,
    BadRequestError,
    InternalServerError,
    NotFoundError,
    RateLimitError,
)
from tqdm import tqdm

from annotation.io import AnnotationMessage, iter_persuader_messages
from annotation.prompt import (
    FEW_SHOT_EXAMPLES,
    RHETORIC_PROMPT,
    format_dialogue_for_prompt,
)
from experiment.cli_utils import add_min_date_arg
from experiment.condition_filters import (
    add_condition_filter_args,
    filters_from_args,
)

LITELLM_API_ERRORS = (
    APIConnectionError,
    APIError,
    AuthenticationError,
    BadRequestError,
    InternalServerError,
    NotFoundError,
    RateLimitError,
)

DEFAULT_MODEL = "gpt-5.1-2025-11-13"
DEFAULT_OUTPUT_DIR = Path("annotations")
DEFAULT_BATCH_SIZE = 25

AnnotationTargetKey = tuple[str, int, int, int]


@dataclass(frozen=True)
class RunnerConfig:
    """Configuration for a single annotation run."""

    model: str
    system_prompt: str
    timeout: int
    max_tokens: int
    num_retries: int


@dataclass(frozen=True)
class RunContext:
    """Prepared inputs for a single annotation run."""

    config: RunnerConfig
    messages: list[AnnotationMessage]
    output_path: Path
    meta: dict[str, object]


def build_system_prompt() -> str:
    """Return the system prompt used for annotation."""
    return f"{RHETORIC_PROMPT.rstrip()}\n\n{FEW_SHOT_EXAMPLES.rstrip()}"


def build_turns(round_obj: Any) -> list[dict[str, str]]:
    """Return a list of dialogue turns in prompt format."""
    turns: list[dict[str, str]] = []
    for message in round_obj.messages:
        role = str(message.get("role", ""))
        content = str(message.get("content", ""))
        turns.append({"speaker": role, "text": content})
    return turns


def build_prompt_text(*, round_obj: Any, target_turn_index: int) -> str:
    """Build the prompt text for one annotation request."""
    turns = build_turns(round_obj)
    return format_dialogue_for_prompt(turns, target_turn_index)


def to_litellm_messages(system_prompt: str, prompt_text: str) -> list[dict]:
    """Return LiteLLM chat messages for a single request."""
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt_text},
    ]


def extract_text_from_response(response: object) -> str:
    """Extract text content from a LiteLLM completion response."""
    if isinstance(response, dict):
        choices = response.get("choices") or []
        if choices:
            message = choices[0].get("message") or {}
            content = message.get("content")
            if isinstance(content, str):
                return content
        text = response.get("text") or response.get("content")
        if isinstance(text, str):
            return text
    if hasattr(response, "choices"):
        choices = getattr(response, "choices") or []
        if choices:
            message = getattr(choices[0], "message", None)
            content = getattr(message, "content", None)
            if isinstance(content, str):
                return content
    return ""


def build_record_from_response(
    *, message_item: AnnotationMessage, response: object
) -> dict[str, object]:
    """Build an annotation record from a LiteLLM response."""
    raw_response = extract_text_from_response(response)
    parsed: dict[str, object] | None = None
    error: str | None = None

    if raw_response:
        try:
            parsed = json.loads(raw_response)
        except json.JSONDecodeError as err:
            error = f"json_decode_error: {err}"
    else:
        error = "empty_response"

    return {
        "type": "annotation",
        "target": message_item.target.as_dict(),
        "condition": str(message_item.condition.as_non_id_role()),
        "target_text": message_item.message.get("content", ""),
        "response_text": raw_response,
        "parsed": parsed,
        "error": error,
    }


def build_error_record(
    *, message_item: AnnotationMessage, error: str
) -> dict[str, object]:
    """Build an error record when a batch call fails."""
    return {
        "type": "annotation",
        "target": message_item.target.as_dict(),
        "condition": str(message_item.condition.as_non_id_role()),
        "target_text": message_item.message.get("content", ""),
        "response_text": "",
        "parsed": None,
        "error": error,
    }


def collect_messages(
    *,
    min_date: str | None,
    condition_filters: dict[str, object] | None,
    limit: int | None,
    include_all_files: bool,
) -> list[AnnotationMessage]:
    """Collect annotation targets from round results."""
    messages = list(
        iter_persuader_messages(
            min_date=min_date,
            condition_filters=condition_filters,
            include_all_files=include_all_files,
        )
    )
    if limit is not None:
        messages = messages[: max(0, int(limit))]
    return messages


def sanitize_filename(text: str) -> str:
    """Return a filesystem-safe filename fragment."""
    safe = text.replace("/", "_").replace(":", "_").replace(" ", "_")
    return "".join(ch for ch in safe if ch.isalnum() or ch in {"_", "-", "."})


def write_jsonl_line(file_handle, payload: dict[str, object]) -> None:
    """Write one JSONL record to an open file handle."""
    file_handle.write(json.dumps(payload, ensure_ascii=True) + "\n")


def estimate_cost_summary(
    *,
    model: str,
    prompt_texts: list[str],
    system_prompt: str,
    max_completion_tokens: int,
) -> tuple[float, int, int]:
    """Return max cost estimate plus total prompt/completion token counts."""
    max_tokens = _read_model_max_tokens(model)
    if max_tokens is None:
        return 0.0, 0, 0

    total_cost = 0.0
    total_prompt_tokens = 0
    total_completion_tokens = 0
    for prompt_text in prompt_texts:
        prompt_tokens = _count_prompt_tokens(
            model=model,
            system_prompt=system_prompt,
            prompt_text=prompt_text,
        )
        if prompt_tokens is None:
            return 0.0, 0, 0
        total_prompt_tokens += prompt_tokens

        if max_tokens is None:
            available_completion = max_completion_tokens
        else:
            available_completion = max(max_tokens - prompt_tokens, 0)
            available_completion = min(available_completion, max_completion_tokens)
        total_completion_tokens += available_completion

        request_cost = _estimate_request_cost(
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=available_completion,
        )
        if request_cost is None:
            return 0.0, 0, 0
        total_cost += request_cost
    return round(total_cost, 6), total_prompt_tokens, total_completion_tokens


def _read_model_max_tokens(model: str) -> int | None:
    """Return the model max token count, or None on failure."""
    try:
        max_tokens_raw = litellm.get_max_tokens(model)
        if isinstance(max_tokens_raw, str):
            return int(max_tokens_raw)
        return int(max_tokens_raw) if max_tokens_raw is not None else None
    except (ValueError, TypeError, KeyError) as err:
        print(f"Cost estimation failed to read max tokens: {err}", file=sys.stderr)
        return None
    except LITELLM_API_ERRORS as err:
        print(f"Cost estimation failed to read max tokens: {err}", file=sys.stderr)
        return None


def _count_prompt_tokens(
    *, model: str, system_prompt: str, prompt_text: str
) -> int | None:
    """Return prompt token count, or None on failure."""
    messages = to_litellm_messages(system_prompt, prompt_text)
    try:
        return litellm.token_counter(model=model, messages=messages)
    except (ValueError, TypeError, KeyError) as err:
        print(f"Cost estimation failed to count tokens: {err}", file=sys.stderr)
        return None
    except LITELLM_API_ERRORS as err:
        print(f"Cost estimation failed to count tokens: {err}", file=sys.stderr)
        return None


def _estimate_request_cost(
    *, model: str, prompt_tokens: int, completion_tokens: int
) -> float | None:
    """Return estimated cost for one request, or None on failure."""
    try:
        prompt_cost, completion_cost = litellm.cost_per_token(
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )
    except (ValueError, TypeError, KeyError) as err:
        print(f"Cost estimation failed to compute costs: {err}", file=sys.stderr)
        return None
    except LITELLM_API_ERRORS as err:
        print(f"Cost estimation failed to compute costs: {err}", file=sys.stderr)
        return None
    return float(prompt_cost or 0.0) + float(completion_cost or 0.0)


def print_prompt_preview(*, system_prompt: str, prompt_text: str, model: str) -> None:
    """Print a human-readable prompt preview."""
    print("---")
    print(f"Model: {model}")
    print("---")
    print("System prompt:")
    print(system_prompt)
    print("---")
    print("User prompt:")
    print(prompt_text)
    print("---")


def write_dry_run_records(
    *,
    handle,
    messages: list[AnnotationMessage],
    config: RunnerConfig,
) -> None:
    """Write dry-run prompt records and print a preview/cost estimate."""
    prompt_texts = [
        build_prompt_text(
            round_obj=message_item.round_obj,
            target_turn_index=message_item.target.span.message_index,
        )
        for message_item in messages
    ]
    estimate, total_prompt_tokens, total_completion_tokens = estimate_cost_summary(
        model=config.model,
        prompt_texts=prompt_texts,
        system_prompt=config.system_prompt,
        max_completion_tokens=config.max_tokens,
    )
    print("---")
    print(f"Dry run: {len(messages)} messages.")
    print("---")
    sample_text = random.choice(prompt_texts)
    print_prompt_preview(
        system_prompt=config.system_prompt,
        prompt_text=sample_text,
        model=config.model,
    )
    print(f"Total prompt tokens: {total_prompt_tokens}")
    print(f"Total completion tokens (max): {total_completion_tokens}")
    print(f"Estimated max cost (USD): {estimate}")
    print("---")

    for message_item, prompt_text in zip(messages, prompt_texts, strict=True):
        record: dict[str, object] = {
            "type": "dry_run",
            "target": message_item.target.as_dict(),
            "condition": str(message_item.condition.as_non_id_role()),
            "target_text": message_item.message.get("content", ""),
            "prompt_text": prompt_text,
        }
        write_jsonl_line(handle, record)


def log_annotation_error(record: dict[str, object]) -> None:
    """Log annotation errors to stderr for quick diagnosis."""
    error = record.get("error")
    if not error:
        return
    target = record.get("target") or {}
    source_path = target.get("source_path", "")
    line_index = target.get("line_index", "")
    round_index = target.get("round_index", "")
    message_index = target.get("message_index", "")
    print(
        (
            "Annotation error: "
            f"{error} (source={source_path} "
            f"line={line_index} round={round_index} message={message_index})"
        ),
        file=sys.stderr,
    )


def update_error_counts(counts: dict[str, int], record: dict[str, object]) -> None:
    """Update error counts for annotation records."""
    error = record.get("error")
    if not error:
        return
    counts[str(error)] = counts.get(str(error), 0) + 1


def _parse_non_negative_int(value: object) -> int | None:
    """Parse a non-negative integer value from JSON payload fields."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value >= 0 else None
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        try:
            parsed = int(stripped)
        except ValueError:
            return None
        return parsed if parsed >= 0 else None
    return None


def target_key_from_record(record: dict[str, object]) -> AnnotationTargetKey | None:
    """Build a stable message target key from an annotation JSONL record."""
    target = record.get("target")
    if not isinstance(target, dict):
        return None
    source_path = target.get("source_path")
    if not isinstance(source_path, str) or not source_path:
        return None
    line_index = _parse_non_negative_int(target.get("line_index"))
    round_index = _parse_non_negative_int(target.get("round_index"))
    message_index = _parse_non_negative_int(target.get("message_index"))
    if line_index is None or round_index is None or message_index is None:
        return None
    return (source_path, line_index, round_index, message_index)


def target_key_from_message_item(
    message_item: AnnotationMessage,
) -> AnnotationTargetKey:
    """Build a stable message target key from an in-memory annotation message."""
    round_index = message_item.target.round_index
    span = message_item.target.span
    return (
        str(round_index.source_path),
        int(round_index.line_index),
        int(round_index.round_index),
        int(span.message_index),
    )


def load_existing_annotation_target_keys(path: Path) -> set[AnnotationTargetKey]:
    """
    Load successful annotation target keys from an existing JSONL output file.

    Failed rows are intentionally ignored so append-mode reruns can retry them.
    """
    if not path.exists():
        return set()
    existing_keys: set[AnnotationTargetKey] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                record = json.loads(stripped)
            except json.JSONDecodeError:
                continue
            if not isinstance(record, dict):
                continue
            if record.get("type") != "annotation":
                continue
            if record.get("error"):
                continue
            if not isinstance(record.get("parsed"), dict):
                continue
            target_key = target_key_from_record(record)
            if target_key is not None:
                existing_keys.add(target_key)
    return existing_keys


def ensure_trailing_newline(path: Path) -> None:
    """Ensure append writes start on a new JSONL line when file already exists."""
    if not path.exists() or path.stat().st_size == 0:
        return
    with path.open("rb") as handle:
        handle.seek(-1, 2)
        last_byte = handle.read(1)
    if last_byte != b"\n":
        with path.open("a", encoding="utf-8") as handle:
            handle.write("\n")


def resolve_output_path(args: argparse.Namespace) -> Path:
    """Return the output path for this run."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_tag = sanitize_filename(args.model)
    if args.output:
        return Path(args.output)
    return DEFAULT_OUTPUT_DIR / f"rhetoric_{model_tag}_{timestamp}.jsonl"


def build_meta_record(
    *,
    args: argparse.Namespace,
    system_prompt: str,
    condition_filters: dict[str, object],
) -> dict[str, object]:
    """Return a meta JSONL record for the annotation run."""
    return {
        "type": "meta",
        "generated_at": datetime.now().isoformat(),
        "model": args.model,
        "system_prompt": system_prompt,
        "min_date": args.min_date,
        "condition_filters": condition_filters,
        "dry_run": bool(args.dry_run),
        "append": bool(args.append),
        "all_files": bool(args.all_files),
        "max_workers": args.max_workers,
        "timeout": args.timeout,
        "max_tokens": args.max_tokens,
        "num_retries": args.num_retries,
    }


def chunk_items(
    items: Sequence[AnnotationMessage], *, batch_size: int
) -> Iterator[list[AnnotationMessage]]:
    """Yield items in fixed-size batches."""
    batch: list[AnnotationMessage] = []
    for item in items:
        batch.append(item)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def call_batch_completion(
    *, messages_list: list[list[dict[str, object]]], config: RunnerConfig
) -> list[object]:
    """Call LiteLLM batch completion with defaults."""
    request_params: dict[str, object] = {
        "model": config.model,
        "messages": messages_list,
        "timeout": int(config.timeout),
        "max_tokens": int(config.max_tokens),
    }
    apply_reasoning_defaults(config.model, request_params)
    return list(litellm.batch_completion(**request_params))


def build_batch_records(
    *,
    batch: list[AnnotationMessage],
    responses: list[object],
) -> list[dict[str, object]]:
    """Build annotation records from batch responses."""
    records: list[dict[str, object]] = []
    for message_item, response in zip(batch, responses, strict=False):
        records.append(
            build_record_from_response(
                message_item=message_item,
                response=response,
            )
        )
    if len(records) < len(batch):
        for message_item in batch[len(records) :]:
            records.append(
                build_error_record(
                    message_item=message_item,
                    error="missing_response",
                )
            )
    return records


def build_run_context(args: argparse.Namespace) -> RunContext:
    """Build the core data needed for an annotation run."""
    system_prompt = build_system_prompt()
    config = RunnerConfig(
        model=args.model,
        system_prompt=system_prompt,
        timeout=args.timeout,
        max_tokens=args.max_tokens,
        num_retries=args.num_retries,
    )
    condition_filters = filters_from_args(args)
    messages = collect_messages(
        min_date=args.min_date,
        condition_filters=condition_filters,
        limit=args.limit,
        include_all_files=bool(args.all_files),
    )
    output_path = resolve_output_path(args)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if args.append:
        existing_keys = load_existing_annotation_target_keys(output_path)
        messages = [
            message_item
            for message_item in messages
            if target_key_from_message_item(message_item) not in existing_keys
        ]
    meta = build_meta_record(
        args=args,
        system_prompt=system_prompt,
        condition_filters=condition_filters or {},
    )
    return RunContext(
        config=config,
        messages=messages,
        output_path=output_path,
        meta=meta,
    )


def run_annotations(args: argparse.Namespace) -> Path:
    """Run the annotation pipeline and return the output path."""
    run_context = build_run_context(args)
    if args.append:
        ensure_trailing_newline(run_context.output_path)
    write_mode = "a" if args.append else "w"
    with run_context.output_path.open(write_mode, encoding="utf-8") as handle:
        write_jsonl_line(handle, run_context.meta)
        if not run_context.messages:
            return run_context.output_path
        if args.dry_run:
            write_dry_run_records(
                handle=handle,
                messages=run_context.messages,
                config=run_context.config,
            )
            return run_context.output_path

        error_counts: dict[str, int] = {}
        progress = tqdm(total=len(run_context.messages), desc="Annotating")
        try:
            for batch in chunk_items(
                run_context.messages, batch_size=DEFAULT_BATCH_SIZE
            ):
                prompt_texts = [
                    build_prompt_text(
                        round_obj=message_item.round_obj,
                        target_turn_index=message_item.target.span.message_index,
                    )
                    for message_item in batch
                ]
                messages_list = [
                    to_litellm_messages(run_context.config.system_prompt, prompt_text)
                    for prompt_text in prompt_texts
                ]
                try:
                    responses = call_batch_completion(
                        messages_list=messages_list,
                        config=run_context.config,
                    )
                    records = build_batch_records(batch=batch, responses=responses)
                except LITELLM_API_ERRORS as err:
                    records = [
                        build_error_record(
                            message_item=message_item,
                            error=f"litellm_error: {err}",
                        )
                        for message_item in batch
                    ]

                for record in records:
                    log_annotation_error(record)
                    update_error_counts(error_counts, record)
                    write_jsonl_line(handle, record)
                    handle.flush()
                    progress.update(1)
        finally:
            progress.close()

        if error_counts:
            print("---")
            print("Annotation error summary:")
            for key, value in sorted(error_counts.items()):
                print(f"{key}: {value}")
            print("---")
    return run_context.output_path


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Annotate persuader messages with LiteLLM."
    )
    add_min_date_arg(parser)
    add_condition_filter_args(parser)
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="LiteLLM model name.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output JSONL path (defaults to annotations/).",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append new annotations to an existing output file and skip completed targets.",
    )
    parser.add_argument(
        "--all-files",
        action="store_true",
        help=(
            "Load all matching JSONL files per condition directory instead of "
            "only the newest matching file."
        ),
    )
    parser.add_argument("--limit", type=int, default=None, help="Limit messages.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Write prompts without calling the model.",
    )
    parser.add_argument("--timeout", type=int, default=60, help="Request timeout.")
    parser.add_argument("--max-tokens", type=int, default=512, help="Max tokens.")
    parser.add_argument("--max-workers", type=int, default=8, help="Thread count.")
    parser.add_argument("--num-retries", type=int, default=3, help="Retry count.")
    return parser.parse_args()


def apply_reasoning_defaults(model: str, params: dict[str, object]) -> None:
    """Set LiteLLM defaults with no reasoning for GPT-5.1 models."""
    if litellm.supports_reasoning(model):
        params.setdefault("temperature", 1)
        if "gpt-5.1" in model:
            params.setdefault("reasoning_effort", "none")
    else:
        params.setdefault("temperature", 0)


def main() -> None:
    """CLI entrypoint."""
    args = parse_args()
    output_path = run_annotations(args)
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
