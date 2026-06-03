"""Single-round LLM-judge target human-likeness scoring."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
from dataclasses import dataclass
from pathlib import Path

import litellm
import matplotlib.pyplot as plt
import numpy as np
from json_repair import loads as repair_json_loads
from pydantic import BaseModel, Field, ValidationError

from experiment import Condition, ContinuousMeasure, Round, load_round_results
from experiment.llm_batch import extract_text_from_response
from simulation.human_trajectory_clusters import extract_round_updates

from .simulator_common import (
    add_common_human_simulator_filter_args,
    add_include_vanilla_llm_target_arg,
    select_matched_human_structure_full_vanilla_from_args,
)
from .simulator_plot_style import (
    COMPARISON_CORPUS_COLOR_MAP,
    PAPER_RESULTS_FIGURE_SIZE_INCHES,
    comparison_corpus_is_visible_in_main_plots,
    comparison_corpus_sort_key,
    comparison_corpus_tick_label,
)
from .tables import print_table

DEFAULT_OUTPUT_PREFIX = Path("analysis/data/simulator_llm_judge")

# Use a non-interactive backend for CLI stability in headless environments.
plt.switch_backend("Agg")

JUDGE_SYSTEM_PROMPT = """\
You are evaluating one persuasion ROUND.

Your task is to score ONLY the TARGET participant's human-likeness.
Do NOT evaluate persuader quality. Do NOT reward or punish based on proposition
content, moral stance, truth, politics, or writing quality of the persuader.

Focus only on whether the TARGET's behavior appears human:
- target message style and interaction behavior
- turn-by-turn target belief dynamics and consistency with responses
- plausible variability/hesitation/commitment patterns for a human target

Return strict JSON with exactly:
{
  "reason": "<short text>",
  "confidence": <number in [0,1]>,
  "target_human_likeness": <number in [0,100]>
}
"""


@dataclass(frozen=True)
class JudgeRound:
    """Round payload for target human-likeness judging.

    Attributes:
        proposition: Proposition text.
        updates: Persuader-relative per-turn deltas.
        condition: Round condition metadata.
        target_initial_belief: Initial target belief.
        serial_questions: Raw target beliefs after each persuader turn.
        messages: Transcript as (role, content).
    """

    proposition: str
    updates: tuple[float, ...]
    condition: Condition
    target_initial_belief: float
    serial_questions: tuple[float, ...]
    messages: tuple[tuple[str, str], ...]


class JudgeScore(BaseModel):
    """Structured score output from the LLM judge."""

    reason: str = ""
    confidence: float = Field(ge=0.0, le=1.0)
    target_human_likeness: float = Field(ge=0.0, le=100.0)


def parse_args() -> argparse.Namespace:
    """Parse CLI args.

    Returns:
        Parsed argument namespace.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Score target human-likeness per round using an LLM judge and "
            "summarize by target type."
        )
    )
    add_common_human_simulator_filter_args(
        parser,
        include_results_dir=False,
        default_results_dir=None,
        include_proposition_match=True,
    )
    add_include_vanilla_llm_target_arg(
        parser,
        default=True,
        help_text="Include vanilla llm_target rounds as an additional corpus.",
    )
    parser.set_defaults(
        persuader_model="gpt-5-2025-08-07",
        turn_limit=4,
        participant_proposition="false",
        exclude_bn_survey=True,
    )
    parser.add_argument(
        "--max-rounds-per-corpus",
        type=int,
        default=0,
        help=(
            "Maximum rounds scored per corpus. Default 0 disables sampling "
            "(score all filtered rounds)."
        ),
    )
    parser.add_argument(
        "--max-messages",
        type=int,
        default=20,
        help="Maximum transcript messages included in the judge prompt.",
    )
    parser.add_argument(
        "--max-message-chars",
        type=int,
        default=220,
        help="Maximum characters per transcript message in prompt rendering.",
    )
    parser.add_argument(
        "--judge-model",
        type=str,
        default="gpt-5.4-2026-03-17",
        help="Model used for LLM judging.",
    )
    parser.add_argument(
        "--judge-temperature",
        type=float,
        default=0.0,
        help="Judge call temperature.",
    )
    parser.add_argument(
        "--judge-timeout-s",
        type=int,
        default=120,
        help="Judge call timeout in seconds.",
    )
    parser.add_argument(
        "--judge-max-retries",
        type=int,
        default=2,
        help="Judge call retry count.",
    )
    parser.add_argument(
        "--judge-max-tokens",
        type=int,
        default=180,
        help="Max completion tokens for judge output.",
    )
    parser.add_argument(
        "--judge-max-workers",
        type=int,
        default=64,
        help="Parallel workers for LiteLLM batch completion.",
    )
    parser.add_argument(
        "--debug-print-first-response",
        action="store_true",
        help="Print the first raw judge response text for each corpus.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Estimate token usage/cost and exit without judge model calls.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=17,
        help="Random seed for downsampling.",
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=DEFAULT_OUTPUT_PREFIX,
        help="Output prefix for CSV/JSONL and chart artifacts.",
    )
    parser.add_argument(
        "--plot-format",
        choices=["png", "pdf"],
        default="png",
        help="Output format for the summary bar chart.",
    )
    return parser.parse_args()


def _clean_text(text: str, max_chars: int) -> str:
    """Normalize one transcript message.

    Args:
        text: Raw message content.
        max_chars: Hard character cap.

    Returns:
        One-line cleaned text.
    """
    compact = " ".join(str(text).split())
    if len(compact) <= max_chars:
        return compact
    return compact[: max(1, max_chars)].rstrip() + "..."


def _judge_round_from_round_obj(round_obj: Round) -> JudgeRound | None:
    """Convert one round object to a judge row when valid.

    Args:
        round_obj: Parsed round model.

    Returns:
        Judge round row or None.
    """
    condition = round_obj.get_condition()
    if condition.continuous_measure != ContinuousMeasure.SERIAL_QUESTIONS:
        return None

    updates = extract_round_updates(round_obj)
    if updates is None:
        return None

    initial = round_obj.target_initial_belief
    serial_questions = round_obj.serial_questions
    if not isinstance(initial, (int, float)):
        return None
    if not isinstance(serial_questions, list) or not serial_questions:
        return None

    serial_clean: list[float] = []
    for value in list(serial_questions):
        if not isinstance(value, (int, float)):
            return None
        serial_clean.append(float(value))

    transcript: list[tuple[str, str]] = []
    for message in round_obj.messages:
        if not isinstance(message, dict):
            continue
        role = str(message.get("role") or "").strip().lower()
        if role not in {"persuader", "target"}:
            continue
        content = str(message.get("content") or "")
        transcript.append((role, content))
    if not transcript:
        return None

    return JudgeRound(
        proposition=round_obj.proposition,
        updates=updates,
        condition=condition,
        target_initial_belief=float(initial),
        serial_questions=tuple(serial_clean),
        messages=tuple(transcript),
    )


def load_rounds(*, min_date: str | None) -> list[JudgeRound]:
    """Load rounds using the canonical experiment loader.

    Args:
        min_date: Optional ISO date filter.

    Returns:
        Parsed judge rows.
    """
    condition_to_rounds = load_round_results(min_date=min_date, include_indices=False)
    out: list[JudgeRound] = []
    for round_groups in condition_to_rounds.values():
        for round_obj in itertools.chain.from_iterable(round_groups):
            row = _judge_round_from_round_obj(round_obj)
            if row is not None:
                out.append(row)
    return out


def render_round_prompt(
    round_row: JudgeRound,
    *,
    max_messages: int,
    max_message_chars: int,
) -> str:
    """Render a single-round judge prompt body.

    Args:
        round_row: Round payload.
        max_messages: Transcript message cap.
        max_message_chars: Per-message char cap.

    Returns:
        Prompt text.
    """
    transcript_lines: list[str] = []
    for role, content in list(round_row.messages[: max(1, max_messages)]):
        speaker = "Persuader" if role == "persuader" else "Target"
        transcript_lines.append(f"{speaker}: {_clean_text(content, max_message_chars)}")

    beliefs = [
        float(round_row.target_initial_belief),
        *list(round_row.serial_questions),
    ]
    deltas = [float(value) for value in round_row.updates]
    return (
        "Score the human-likeness of the TARGET only.\n\n"
        f"Proposition: {round_row.proposition}\n"
        f"Target belief trajectory (raw): {json.dumps(beliefs)}\n"
        f"Per-turn target deltas in persuader direction: {json.dumps(deltas)}\n"
        "Transcript:\n"
        f"{chr(10).join(transcript_lines)}\n\n"
        "Return strict JSON only."
    )


def _parse_score(text: str) -> JudgeScore | None:
    """Parse model output into JudgeScore.

    Args:
        text: Raw model output.

    Returns:
        Parsed score or None.
    """
    stripped = text.strip()
    if not stripped:
        return None
    try:
        repaired_payload = repair_json_loads(stripped)
        return JudgeScore.model_validate(repaired_payload)
    except (TypeError, ValueError, ValidationError):
        return None


def _sample_rows(
    rows: list[JudgeRound],
    *,
    max_rows: int,
    seed: int,
) -> list[JudgeRound]:
    """Optionally downsample rows uniformly without replacement.

    Args:
        rows: Candidate rows.
        max_rows: Maximum rows to keep. <=0 keeps all.
        seed: RNG seed.

    Returns:
        Selected rows.
    """
    if max_rows <= 0 or len(rows) <= max_rows:
        return list(rows)
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(rows))[: int(max_rows)]
    return [rows[int(index)] for index in indices]


def _token_count_for_messages(model: str, messages: list[dict[str, str]]) -> int:
    """Count prompt tokens with LiteLLM token_counter.

    Args:
        model: Judge model.
        messages: Prompt messages.

    Returns:
        Prompt token count.
    """
    return int(litellm.token_counter(model=model, messages=messages))


def _cost_for_model_tokens(
    model: str, prompt_tokens: int, completion_tokens: int
) -> float | None:
    """Estimate total USD cost from token counts.

    Args:
        model: Judge model.
        prompt_tokens: Prompt tokens.
        completion_tokens: Completion tokens.

    Returns:
        Estimated cost in USD, or None if unavailable.
    """
    try:
        prompt_cost, completion_cost = litellm.cost_per_token(
            model=model,
            prompt_tokens=int(prompt_tokens),
            completion_tokens=int(completion_tokens),
        )
    except (litellm.BadRequestError, ValueError, TypeError, KeyError):
        return None
    return float(prompt_cost) + float(completion_cost)


def _call_judge_batch(
    *,
    model: str,
    messages_list: list[list[dict[str, str]]],
    temperature: float,
    timeout_s: int,
    max_retries: int,
    max_tokens: int,
    max_workers: int,
) -> list[object]:
    """Call judge model in one LiteLLM batch request.

    Args:
        model: Judge model name.
        messages_list: Prompt batch.
        temperature: Sampling temperature.
        timeout_s: Request timeout in seconds.
        max_retries: Retry count.
        max_tokens: Max completion tokens.
        max_workers: Batch worker parallelism.

    Returns:
        Response objects in input order.
    """
    kwargs: dict[str, object] = {
        "model": model,
        "messages": messages_list,
        "timeout": int(timeout_s),
        "num_retries": int(max_retries),
        "max_tokens": int(max_tokens),
        "max_workers": int(max_workers),
    }

    if litellm.supports_reasoning(model):
        kwargs["temperature"] = 1
        kwargs["reasoning_effort"] = "none"
    else:
        kwargs["temperature"] = float(temperature)

    try:
        responses = litellm.batch_completion(**kwargs)
    except (RuntimeError, ValueError, OSError, litellm.OpenAIError):
        if "reasoning_effort" in kwargs:
            kwargs.pop("reasoning_effort")
            responses = litellm.batch_completion(**kwargs)
        else:
            raise

    if hasattr(responses, "__iter__") and not isinstance(responses, list):
        responses = list(responses)
    response_list = list(responses)
    if len(response_list) != len(messages_list):
        raise RuntimeError(
            "LiteLLM returned a different number of responses than requested: "
            f"requested={len(messages_list)} returned={len(response_list)} "
            f"model={model}"
        )
    return response_list


def _write_csv(path: Path, rows: list[dict[str, object]], columns: list[str]) -> None:
    """Write CSV rows.

    Args:
        path: Output path.
        rows: Data rows.
        columns: Column order.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    """Write JSONL rows.

    Args:
        path: Output path.
        rows: Serializable rows.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def _summary_for_scores(
    *,
    corpus: str,
    scored_rows: list[dict[str, object]],
) -> dict[str, object]:
    """Compute summary stats for one corpus.

    Args:
        corpus: Corpus label.
        scored_rows: Raw judged rows.

    Returns:
        Summary row.
    """
    scores = np.asarray(
        [
            float(row["target_human_likeness"])
            for row in scored_rows
            if np.isfinite(float(row["target_human_likeness"]))
        ],
        dtype=float,
    )
    confidences = np.asarray(
        [
            float(row["confidence"])
            for row in scored_rows
            if np.isfinite(float(row["confidence"]))
        ],
        dtype=float,
    )
    total = int(len(scored_rows))
    n_scored = int(scores.size)
    parse_failures = int(total - n_scored)

    if n_scored == 0:
        return {
            "corpus": corpus,
            "n_total": total,
            "n_scored": 0,
            "parse_failures": parse_failures,
            "mean_human_likeness": float("nan"),
            "median_human_likeness": float("nan"),
            "std_human_likeness": float("nan"),
            "se_human_likeness": float("nan"),
            "ci95_lo": float("nan"),
            "ci95_hi": float("nan"),
            "mean_confidence": float("nan"),
        }

    mean_score = float(np.mean(scores))
    std_score = float(np.std(scores, ddof=0))
    se_score = float(std_score / np.sqrt(scores.size))
    ci_lo = float(mean_score - 1.96 * se_score)
    ci_hi = float(mean_score + 1.96 * se_score)
    return {
        "corpus": corpus,
        "n_total": total,
        "n_scored": n_scored,
        "parse_failures": parse_failures,
        "mean_human_likeness": mean_score,
        "median_human_likeness": float(np.median(scores)),
        "std_human_likeness": std_score,
        "se_human_likeness": se_score,
        "ci95_lo": ci_lo,
        "ci95_hi": ci_hi,
        "mean_confidence": (
            float(np.mean(confidences)) if confidences.size > 0 else float("nan")
        ),
    }


def _plot_bar_chart(summary_rows: list[dict[str, object]], path: Path) -> None:
    """Plot bar chart of target human-likeness by corpus.

    Args:
        summary_rows: Summary rows.
        path: Destination image path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    plotted = [
        row
        for row in summary_rows
        if comparison_corpus_is_visible_in_main_plots(str(row.get("corpus", "")))
        if np.isfinite(float(row.get("mean_human_likeness", float("nan"))))
    ]
    plotted.sort(
        key=lambda row: (
            -float(row.get("mean_human_likeness", float("nan"))),
            comparison_corpus_sort_key(str(row.get("corpus", ""))),
        )
    )
    if not plotted:
        return

    labels = [comparison_corpus_tick_label(str(row["corpus"])) for row in plotted]
    values = np.asarray(
        [float(row["mean_human_likeness"]) for row in plotted],
        dtype=float,
    )
    errors = np.asarray(
        [1.96 * float(row["se_human_likeness"]) for row in plotted],
        dtype=float,
    )
    colors = [
        COMPARISON_CORPUS_COLOR_MAP.get(str(row["corpus"]), "#666666")
        for row in plotted
    ]

    x_pos = np.arange(len(plotted), dtype=float)
    fig, axis = plt.subplots(figsize=PAPER_RESULTS_FIGURE_SIZE_INCHES)
    axis.bar(x_pos, values, color=colors, alpha=0.92)
    axis.errorbar(
        x_pos,
        values,
        yerr=errors,
        fmt="none",
        ecolor="#222222",
        elinewidth=1.2,
        capsize=4,
    )
    axis.set_xticks(x_pos)
    axis.set_xticklabels(labels, rotation=0, ha="center")
    axis.tick_params(axis="x", labelsize=6.2, pad=1.2)
    axis.set_ylabel(r"Human-Likeness ($\rightarrow$)", fontsize=10)
    axis.set_ylim(0.0, 100.0)
    axis.grid(axis="y", linestyle=":", alpha=0.25)
    fig.subplots_adjust(left=0.26, bottom=0.18, right=0.98, top=0.96)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def main() -> None:
    """Run single-round target human-likeness scoring."""
    args = parse_args()
    all_rows = load_rounds(min_date=args.min_date)
    human_rows, structure_rows, full_rows, vanilla_rows = (
        select_matched_human_structure_full_vanilla_from_args(
            all_rows, args=args, include_vanilla=True
        )
    )

    if not human_rows:
        print("No human rounds available after filters.")
        return
    if not structure_rows or not full_rows:
        print("Need non-empty structure and full simulator corpora after filters.")
        return
    if args.include_vanilla_llm_target and not vanilla_rows:
        print("Vanilla corpus requested but empty after filters.")
        return

    corpora: list[tuple[str, list[JudgeRound]]] = [("human_reference", human_rows)]
    if args.include_vanilla_llm_target:
        corpora.append(("vanilla_llm_target", vanilla_rows))
    corpora.extend(
        [
            ("structure_target", structure_rows),
            ("full_simulated_target", full_rows),
        ]
    )

    raw_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    max_rows = int(args.max_rounds_per_corpus)

    if args.dry_run:
        dry_rows: list[dict[str, object]] = []
        total_rounds = 0
        total_prompt_tokens = 0
        completion_tokens_per_call = int(args.judge_max_tokens)
        total_completion_tokens = 0
        for index, (corpus, rows) in enumerate(corpora):
            sampled_rows = _sample_rows(
                rows,
                max_rows=max_rows,
                seed=int(args.seed + 1000 * (index + 1)),
            )
            prompt_tokens = 0
            for round_row in sampled_rows:
                prompt = render_round_prompt(
                    round_row,
                    max_messages=int(args.max_messages),
                    max_message_chars=int(args.max_message_chars),
                )
                messages = [
                    {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ]
                prompt_tokens += _token_count_for_messages(
                    str(args.judge_model), messages
                )
            rounds_n = int(len(sampled_rows))
            completion_tokens = rounds_n * completion_tokens_per_call
            estimated_cost = _cost_for_model_tokens(
                model=str(args.judge_model),
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
            )
            dry_rows.append(
                {
                    "corpus": corpus,
                    "rounds": rounds_n,
                    "prompt_tokens_estimate": int(prompt_tokens),
                    "completion_tokens_estimate": int(completion_tokens),
                    "estimated_cost_usd": estimated_cost,
                }
            )
            total_rounds += rounds_n
            total_prompt_tokens += prompt_tokens
            total_completion_tokens += completion_tokens

        total_cost = _cost_for_model_tokens(
            model=str(args.judge_model),
            prompt_tokens=total_prompt_tokens,
            completion_tokens=total_completion_tokens,
        )
        print_table(
            dry_rows,
            title="Dry Run Estimate (No Judge Calls)",
            columns=[
                "corpus",
                "rounds",
                "prompt_tokens_estimate",
                "completion_tokens_estimate",
                "estimated_cost_usd",
            ],
            aligns={
                "rounds": "right",
                "prompt_tokens_estimate": "right",
                "completion_tokens_estimate": "right",
                "estimated_cost_usd": "right",
            },
            formatters={
                "estimated_cost_usd": lambda value: (
                    f"{float(value):.6f}" if value is not None else ""
                ),
            },
        )
        print(
            "\nTotals: "
            f"rounds={total_rounds}, "
            f"prompt_tokens_estimate={total_prompt_tokens}, "
            f"completion_tokens_estimate={total_completion_tokens}, "
            f"estimated_cost_usd="
            f"{f'{float(total_cost):.6f}' if total_cost is not None else 'unknown'}"
        )

        prefix = Path(args.output_prefix)
        _write_csv(
            prefix.with_name(prefix.name + "_dry_run.csv"),
            dry_rows,
            [
                "corpus",
                "rounds",
                "prompt_tokens_estimate",
                "completion_tokens_estimate",
                "estimated_cost_usd",
            ],
        )
        _write_jsonl(
            prefix.with_name(prefix.name + "_dry_run.jsonl"),
            dry_rows,
        )
        return

    for index, (corpus, rows) in enumerate(corpora):
        sampled_rows = _sample_rows(
            rows,
            max_rows=max_rows,
            seed=int(args.seed + 1000 * (index + 1)),
        )
        messages_list: list[list[dict[str, str]]] = []
        for round_row in sampled_rows:
            prompt = render_round_prompt(
                round_row,
                max_messages=int(args.max_messages),
                max_message_chars=int(args.max_message_chars),
            )
            messages_list.append(
                [
                    {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ]
            )

        batch_responses: list[object] = []
        if messages_list:
            batch_responses = _call_judge_batch(
                model=str(args.judge_model),
                messages_list=messages_list,
                temperature=float(args.judge_temperature),
                timeout_s=int(args.judge_timeout_s),
                max_retries=int(args.judge_max_retries),
                max_tokens=int(args.judge_max_tokens),
                max_workers=int(args.judge_max_workers),
            )

        if args.debug_print_first_response and batch_responses:
            first = batch_responses[0]
            if first is None:
                print(f"[{corpus}] first_response=None")
            elif isinstance(first, Exception):
                print(f"[{corpus}] first_response_exception={first}")
            else:
                first_text = extract_text_from_response(first)
                snippet = " ".join(first_text.split())[:300]
                print(f"[{corpus}] first_response_snippet={snippet}")

        for round_index, round_row in enumerate(sampled_rows):
            response_obj = (
                batch_responses[round_index]
                if round_index < len(batch_responses)
                else None
            )
            if isinstance(response_obj, Exception):
                raise RuntimeError(
                    "Judge returned exception response object at "
                    f"corpus={corpus} round_index={round_index}: {response_obj}"
                )
            if response_obj is None:
                raise RuntimeError(
                    "Judge returned no response object at "
                    f"corpus={corpus} round_index={round_index}"
                )
            output_text = extract_text_from_response(response_obj)
            parsed = _parse_score(output_text)
            row: dict[str, object] = {
                "corpus": corpus,
                "round_index": int(round_index),
                "proposition": round_row.proposition,
                "n_turns": int(len(round_row.updates)),
                "reason": "",
                "confidence": float("nan"),
                "target_human_likeness": float("nan"),
                "raw_response": output_text,
            }
            if parsed is not None:
                row["reason"] = str(parsed.reason)
                row["confidence"] = float(parsed.confidence)
                row["target_human_likeness"] = float(parsed.target_human_likeness)
            raw_rows.append(row)

        corpus_rows = [row for row in raw_rows if str(row["corpus"]) == corpus]
        summary_rows.append(_summary_for_scores(corpus=corpus, scored_rows=corpus_rows))

    print_table(
        summary_rows,
        title="LLM-Judge Target Human-Likeness (Higher Is Better)",
        columns=[
            "corpus",
            "n_total",
            "n_scored",
            "parse_failures",
            "mean_human_likeness",
            "ci95_lo",
            "ci95_hi",
            "median_human_likeness",
            "mean_confidence",
        ],
        aligns={
            "n_total": "right",
            "n_scored": "right",
            "parse_failures": "right",
            "mean_human_likeness": "right",
            "ci95_lo": "right",
            "ci95_hi": "right",
            "median_human_likeness": "right",
            "mean_confidence": "right",
        },
        formatters={
            "mean_human_likeness": lambda value: (
                f"{float(value):.2f}" if np.isfinite(float(value)) else ""
            ),
            "ci95_lo": lambda value: (
                f"{float(value):.2f}" if np.isfinite(float(value)) else ""
            ),
            "ci95_hi": lambda value: (
                f"{float(value):.2f}" if np.isfinite(float(value)) else ""
            ),
            "median_human_likeness": lambda value: (
                f"{float(value):.2f}" if np.isfinite(float(value)) else ""
            ),
            "mean_confidence": lambda value: (
                f"{float(value):.3f}" if np.isfinite(float(value)) else ""
            ),
        },
    )

    prefix = Path(args.output_prefix)
    _write_csv(
        prefix.with_name(prefix.name + "_summary.csv"),
        summary_rows,
        [
            "corpus",
            "n_total",
            "n_scored",
            "parse_failures",
            "mean_human_likeness",
            "median_human_likeness",
            "std_human_likeness",
            "se_human_likeness",
            "ci95_lo",
            "ci95_hi",
            "mean_confidence",
        ],
    )
    _write_csv(
        prefix.with_name(prefix.name + "_round_scores.csv"),
        raw_rows,
        [
            "corpus",
            "round_index",
            "proposition",
            "n_turns",
            "reason",
            "confidence",
            "target_human_likeness",
            "raw_response",
        ],
    )
    _write_jsonl(
        prefix.with_name(prefix.name + "_round_scores.jsonl"),
        raw_rows,
    )
    _plot_bar_chart(
        summary_rows=summary_rows,
        path=prefix.with_name(prefix.name + f"_bar.{args.plot_format}"),
    )


if __name__ == "__main__":
    main()
