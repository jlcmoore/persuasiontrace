"""Summarize Bayes-net survey movement metrics from round JSONL exports."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Compute target and non-target node delta summaries for Bayes-net "
            "survey rounds from one JSONL file or a condition directory."
        )
    )
    parser.add_argument(
        "input_path",
        type=Path,
        help="Path to one JSONL file or a condition directory containing JSONL files.",
    )
    parser.add_argument(
        "--dedupe-exact",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "When true, remove exact duplicate rounds by canonical JSON payload "
            "(default: true)."
        ),
    )
    parser.add_argument(
        "--output-format",
        choices=("text", "json"),
        default="text",
        help="Output format (default: text).",
    )
    return parser.parse_args()


def _mean_and_stderr(values: list[float]) -> tuple[float, float]:
    """Compute mean and standard error for numeric values.

    Args:
        values: Numeric values to summarize.

    Returns:
        Tuple ``(mean, stderr)``. Returns ``(nan, nan)`` for empty input.
    """
    count = len(values)
    if count == 0:
        return float("nan"), float("nan")
    mean_value = float(statistics.fmean(values))
    if count < 2:
        return mean_value, float("nan")
    std_value = float(statistics.stdev(values))
    stderr_value = float(std_value / math.sqrt(count))
    return mean_value, stderr_value


def _iter_jsonl_files(input_path: Path) -> list[Path]:
    """Resolve input path to a sorted list of JSONL files.

    Args:
        input_path: File or directory path provided by the user.

    Returns:
        Sorted JSONL file paths to read.

    Raises:
        FileNotFoundError: Input path does not exist.
        ValueError: Input path is not a JSONL file or directory with JSONL files.
    """
    if not input_path.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")
    if input_path.is_file():
        if input_path.suffix != ".jsonl":
            raise ValueError(f"Input file must end with .jsonl: {input_path}")
        return [input_path]
    if input_path.is_dir():
        jsonl_files = sorted(input_path.glob("*.jsonl"))
        if not jsonl_files:
            raise ValueError(f"No .jsonl files found under directory: {input_path}")
        return jsonl_files
    raise ValueError(f"Input path must be a file or directory: {input_path}")


def _payloads_from_line(parsed_line: object) -> list[dict[str, Any]]:
    """Extract round dictionaries from one decoded JSONL line.

    Args:
        parsed_line: Decoded JSON object from one line.

    Returns:
        List of round payload dictionaries.
    """
    payloads: list[dict[str, Any]] = []
    if isinstance(parsed_line, dict):
        payloads.append(parsed_line)
        return payloads
    if isinstance(parsed_line, list):
        for item in parsed_line:
            if isinstance(item, dict):
                payloads.append(item)
    return payloads


def _load_round_payloads(
    input_path: Path, *, dedupe_exact: bool
) -> tuple[list[dict[str, Any]], int]:
    """Load round payload dictionaries from JSONL inputs.

    Args:
        input_path: File or directory path containing JSONL data.
        dedupe_exact: Whether to remove exact duplicate payloads.

    Returns:
        Tuple ``(payloads, files_read)`` where ``payloads`` are parsed round
        dictionaries and ``files_read`` is the number of JSONL files processed.
    """
    files = _iter_jsonl_files(input_path)
    payloads: list[dict[str, Any]] = []
    seen: set[str] = set()
    for jsonl_path in files:
        with jsonl_path.open("r", encoding="utf-8") as handle:
            for raw_line in handle:
                stripped = raw_line.strip()
                if not stripped:
                    continue
                try:
                    parsed_line = json.loads(stripped)
                except json.JSONDecodeError:
                    continue
                for payload in _payloads_from_line(parsed_line):
                    if dedupe_exact:
                        key = json.dumps(
                            payload,
                            sort_keys=True,
                            separators=(",", ":"),
                            ensure_ascii=True,
                        )
                        if key in seen:
                            continue
                        seen.add(key)
                    payloads.append(payload)
    return payloads, len(files)


def _parse_probability(value: Any) -> float | None:
    """Normalize one value to a probability in ``[0, 1]``.

    Args:
        value: Raw candidate value.

    Returns:
        Normalized probability when valid, otherwise ``None``.
    """
    if not isinstance(value, (int, float)):
        return None
    parsed = float(value)
    if parsed < 0.0 or parsed > 1.0:
        return None
    return parsed


def _target_delta(payload: dict[str, Any]) -> float | None:
    """Compute persuader-relative target belief delta for one round.

    Args:
        payload: One round payload dictionary.

    Returns:
        Persuader-relative target delta, or ``None`` when required fields are
        missing or invalid.
    """
    initial = _parse_probability(payload.get("target_initial_belief"))
    final = _parse_probability(payload.get("target_final_belief"))
    supports = payload.get("persuader_supports_proposition")
    if initial is None or final is None or not isinstance(supports, bool):
        return None
    direction = 1.0 if supports else -1.0
    return (final - initial) * direction


def _non_target_node_deltas(payload: dict[str, Any]) -> list[float]:
    """Compute post-minus-pre deltas for non-target survey nodes.

    Args:
        payload: One round payload dictionary.

    Returns:
        List of deltas for node ids shared across initial and final node-belief
        mappings. Invalid entries are skipped.
    """
    initial = payload.get("target_initial_node_beliefs")
    final = payload.get("target_final_node_beliefs")
    if not isinstance(initial, dict) or not isinstance(final, dict):
        return []

    deltas: list[float] = []
    for node_id in sorted(set(initial).intersection(set(final))):
        start_value = _parse_probability(initial.get(node_id))
        end_value = _parse_probability(final.get(node_id))
        if start_value is None or end_value is None:
            continue
        deltas.append(end_value - start_value)
    return deltas


def _collect_target_deltas(payloads: list[dict[str, Any]]) -> list[float]:
    """Collect persuader-relative target deltas for all valid rounds.

    Args:
        payloads: Parsed round payload dictionaries.

    Returns:
        Persuader-relative target deltas.
    """
    deltas: list[float] = []
    for payload in payloads:
        delta = _target_delta(payload)
        if delta is not None:
            deltas.append(delta)
    return deltas


def _summarize_non_target_node_metrics(
    payloads: list[dict[str, Any]],
) -> dict[str, float | int]:
    """Summarize non-target node deltas across rounds.

    Args:
        payloads: Parsed round payload dictionaries.

    Returns:
        Dictionary of non-target node count and delta summary metrics.
    """
    node_pair_deltas: list[float] = []
    round_node_signed_means: list[float] = []
    round_node_abs_means: list[float] = []

    for payload in payloads:
        node_deltas = _non_target_node_deltas(payload)
        if not node_deltas:
            continue
        node_pair_deltas.extend(node_deltas)
        round_node_signed_means.append(float(statistics.fmean(node_deltas)))
        round_node_abs_means.append(
            float(statistics.fmean(abs(value) for value in node_deltas))
        )

    mean_node_delta, _ = _mean_and_stderr(node_pair_deltas)
    mean_abs_node_delta = (
        float(statistics.fmean(abs(value) for value in node_pair_deltas))
        if node_pair_deltas
        else float("nan")
    )
    mean_round_node_delta, stderr_round_node_delta = _mean_and_stderr(
        round_node_signed_means
    )
    mean_round_abs_node_delta, stderr_round_abs_node_delta = _mean_and_stderr(
        round_node_abs_means
    )

    return {
        "n_rounds_with_non_target_nodes": int(len(round_node_signed_means)),
        "n_non_target_node_pairs": int(len(node_pair_deltas)),
        "mean_non_target_node_delta": mean_node_delta,
        "mean_abs_non_target_node_delta": mean_abs_node_delta,
        "mean_round_non_target_node_delta": mean_round_node_delta,
        "stderr_mean_round_non_target_node_delta": stderr_round_node_delta,
        "mean_round_abs_non_target_node_delta": mean_round_abs_node_delta,
        "stderr_mean_round_abs_non_target_node_delta": stderr_round_abs_node_delta,
    }


def summarize_payloads(payloads: list[dict[str, Any]]) -> dict[str, float | int]:
    """Summarize target and non-target delta metrics.

    Args:
        payloads: Parsed round payload dictionaries.

    Returns:
        Summary dictionary with target and non-target node metrics.
    """
    target_deltas = _collect_target_deltas(payloads)
    node_metrics = _summarize_non_target_node_metrics(payloads)
    mean_delta, stderr_delta = _mean_and_stderr(target_deltas)

    return {
        "n_rounds": int(len(payloads)),
        "mean_delta": mean_delta,
        "stderr_mean_delta": stderr_delta,
        **node_metrics,
    }


def _format_text(
    *, summary: dict[str, float | int], files_read: int, input_path: Path
) -> str:
    """Format summary output for plain-text printing.

    Args:
        summary: Summary metrics dictionary.
        files_read: Number of JSONL files read.
        input_path: Original input path.

    Returns:
        Human-readable summary text.
    """
    lines = [
        f"input_path = {input_path}",
        f"files_read = {files_read}",
        f"n_rounds = {summary['n_rounds']}",
        f"n_rounds_with_non_target_nodes = {summary['n_rounds_with_non_target_nodes']}",
        f"n_non_target_node_pairs = {summary['n_non_target_node_pairs']}",
        f"mean_delta = {float(summary['mean_delta']):.6f}",
        f"SE(mean_delta) = {float(summary['stderr_mean_delta']):.6f}",
        (
            "mean_non_target_node_delta = "
            f"{float(summary['mean_non_target_node_delta']):.6f}"
        ),
        (
            "mean_abs_non_target_node_delta = "
            f"{float(summary['mean_abs_non_target_node_delta']):.6f}"
        ),
        (
            "SE(mean_round_non_target_node_delta) = "
            f"{float(summary['stderr_mean_round_non_target_node_delta']):.6f}"
        ),
        (
            "SE(mean_round_abs_non_target_node_delta) = "
            f"{float(summary['stderr_mean_round_abs_non_target_node_delta']):.6f}"
        ),
    ]
    return "\n".join(lines)


def main() -> None:
    """Run the Bayes-net survey summary CLI."""
    args = parse_args()
    payloads, files_read = _load_round_payloads(
        args.input_path, dedupe_exact=bool(args.dedupe_exact)
    )
    summary = summarize_payloads(payloads)
    output = {
        "input_path": str(args.input_path),
        "files_read": int(files_read),
        **summary,
    }

    if args.output_format == "json":
        print(json.dumps(output, sort_keys=True, ensure_ascii=True))
    else:
        print(
            _format_text(
                summary=summary, files_read=files_read, input_path=args.input_path
            )
        )


if __name__ == "__main__":
    main()
