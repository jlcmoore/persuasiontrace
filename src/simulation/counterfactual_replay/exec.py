"""Execution and dry-run helpers for simulator counterfactual replay."""

from __future__ import annotations

import concurrent.futures
from pathlib import Path
from typing import Any, Callable


def chunked_jobs(jobs: list[Any], batch_size: int) -> list[list[Any]]:
    """Split jobs into fixed-size chunks."""
    return [jobs[i : i + batch_size] for i in range(0, len(jobs), batch_size)]


def dry_run_rows(jobs: list[Any]) -> list[dict[str, Any]]:
    """Build dry-run summary rows by corpus."""
    grouped: dict[str, list[Any]] = {}
    for job in jobs:
        grouped.setdefault(job.spec.corpus, []).append(job)

    rows: list[dict[str, Any]] = []
    for corpus, corpus_jobs in sorted(grouped.items(), key=lambda item: item[0]):
        target_backend = corpus_jobs[0].spec.target_backend
        turns_total = sum(len(job.persuader_messages) for job in corpus_jobs)
        calls_per_turn = 2 if target_backend == "simulated_target" else 1
        rows.append(
            {
                "corpus": corpus,
                "backend": target_backend,
                "jobs": int(len(corpus_jobs)),
                "turns": int(turns_total),
                "estimated_calls": int(turns_total * calls_per_turn),
            }
        )
    return rows


def write_dry_run_csv(
    output_prefix: Path,
    jobs: list[Any],
    write_csv_fn: Callable[[Path, list[dict[str, Any]], list[str]], None],
) -> Path:
    """Write one row per planned replay job for auditability."""
    rows: list[dict[str, Any]] = []
    for index, job in enumerate(jobs):
        rows.append(
            {
                "job_index": int(index),
                "corpus": job.spec.corpus,
                "target_backend": job.spec.target_backend,
                "target_model": job.spec.target_model,
                "persona": job.spec.persona.value,
                "replay_index": (
                    int(job.replay_index) if hasattr(job, "replay_index") else 0
                ),
                "turns": int(len(job.persuader_messages)),
                "source_path": str(job.row.source_path),
                "source_line_index": int(job.row.source_line_index),
                "source_round_index": (
                    int(job.row.source_round_index)
                    if job.row.source_round_index is not None
                    else None
                ),
                "proposition": job.row.proposition,
            }
        )
    path = Path(f"{output_prefix}_dry_run_jobs.csv")
    if rows:
        write_csv_fn(path, rows, list(rows[0].keys()))
    return path


def run_dry_run(
    output_prefix: Path,
    jobs: list[Any],
    *,
    write_csv_fn: Callable[[Path, list[dict[str, Any]], list[str]], None],
    print_table_fn: Callable[..., None],
) -> None:
    """Print dry-run plan and write planned-job artifact."""
    rows = dry_run_rows(jobs)
    print_table_fn(
        rows,
        columns=["corpus", "backend", "jobs", "turns", "estimated_calls"],
        title="Counterfactual Replay Dry Run",
        aligns={"jobs": "right", "turns": "right", "estimated_calls": "right"},
    )
    total_jobs = len(jobs)
    total_turns = sum(len(job.persuader_messages) for job in jobs)
    total_calls = sum(
        (2 if job.spec.target_backend == "simulated_target" else 1)
        * len(job.persuader_messages)
        for job in jobs
    )
    print(
        "Dry run totals:",
        f"jobs={total_jobs}",
        f"turns={total_turns}",
        f"estimated_calls={total_calls}",
    )
    dry_csv_path = write_dry_run_csv(output_prefix, jobs, write_csv_fn)
    print("Wrote dry-run output:", dry_csv_path)


def run_replay_batch(
    args: Any,
    jobs: list[Any],
    replay_fn: Callable[[Any], dict[str, Any] | None],
) -> list[dict[str, Any]]:
    """Execute one batch of jobs, optionally in parallel."""
    if args.max_workers <= 1 or len(jobs) <= 1:
        rows = [replay_fn(job) for job in jobs]
        return [row for row in rows if row is not None]

    indexed_jobs = list(enumerate(jobs))
    rows_by_index: dict[int, dict[str, Any]] = {}
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=int(args.max_workers)
    ) as executor:
        future_map = {
            executor.submit(replay_fn, job): index for index, job in indexed_jobs
        }
        for future in concurrent.futures.as_completed(future_map):
            index = future_map[future]
            row = future.result()
            if row is not None:
                rows_by_index[index] = row
    return [rows_by_index[index] for index in sorted(rows_by_index)]


def run_replay_jobs(
    args: Any,
    jobs: list[Any],
    replay_fn: Callable[[Any], dict[str, Any] | None],
) -> list[dict[str, Any]]:
    """Execute replay jobs in batches."""
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")
    if args.max_workers <= 0:
        raise ValueError("--max-workers must be positive.")

    rows: list[dict[str, Any]] = []
    for batch in chunked_jobs(jobs, int(args.batch_size)):
        rows.extend(run_replay_batch(args, batch, replay_fn))
    return rows
