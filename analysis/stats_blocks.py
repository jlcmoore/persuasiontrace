"""Statistical table blocks for round-level persuasion analyses."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from experiment import ContinuousMeasure

from .data_loading import persuader_relative_name
from .formatting import split_condition_label
from .stats import (
    bootstrap_mean_ci,
    holm_adjust,
    paired_t_test,
    significance_stars,
    welch_t_test,
)
from .tables import print_table


def write_table_csv(
    rows: list[dict[str, object]], columns: list[str], output_path: Path
) -> None:
    """Write rows to a CSV table.

    Args:
        rows: Table rows as dictionaries.
        columns: Output column order.
        output_path: CSV destination path.

    Returns:
        None.
    """
    output_path.parent.mkdir(exist_ok=True, parents=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved CSV: {output_path.resolve()}")


def print_condition_delta_summary(
    df: pd.DataFrame,
    *,
    delta_col: str,
    n_boot: int,
    output_path: Path | None = None,
) -> None:
    """Print and optionally export per-condition delta summaries.

    Args:
        df: Analysis dataframe.
        delta_col: Delta column name.
        n_boot: Number of bootstrap draws for confidence intervals.
        output_path: Optional CSV output path.

    Returns:
        None.
    """
    grouped = df.groupby("condition")[delta_col]
    rows = []
    for condition, values in grouped:
        arr = values.to_numpy(dtype=float)
        mean_val, ci_lo, ci_hi = bootstrap_mean_ci(arr, n_boot=n_boot)
        rows.append((condition, arr.size, mean_val, ci_lo, ci_hi))

    if not rows:
        print("No condition deltas available.")
        if output_path is not None:
            write_table_csv(
                [], ["Condition", "n", "mean", "ci_lo", "ci_hi"], output_path
            )
        return

    print("\nCondition delta summary")
    table_rows = [
        {
            "Condition": condition,
            "n": n,
            "mean": mean_val,
            "ci_lo": ci_lo,
            "ci_hi": ci_hi,
        }
        for condition, n, mean_val, ci_lo, ci_hi in rows
    ]
    print_table(
        table_rows,
        columns=["Condition", "n", "mean", "ci_lo", "ci_hi"],
        formatters={
            "mean": lambda val: f"{val:.3f}",
            "ci_lo": lambda val: f"{val:.3f}",
            "ci_hi": lambda val: f"{val:.3f}",
        },
        widths={"Condition": 50},
        aligns={"n": "right", "mean": "right", "ci_lo": "right", "ci_hi": "right"},
    )
    if output_path is not None:
        write_table_csv(
            table_rows, ["Condition", "n", "mean", "ci_lo", "ci_hi"], output_path
        )


def print_within_condition_tests(
    df: pd.DataFrame,
    *,
    initial_col: str,
    final_col: str,
    output_path: Path | None = None,
) -> None:
    """Print and optionally export paired pre/post tests by condition.

    Args:
        df: Analysis dataframe.
        initial_col: Initial belief column name.
        final_col: Final belief column name.
        output_path: Optional CSV output path.

    Returns:
        None.
    """
    print("\nWithin-condition pre/post tests (paired t-test, post > pre)")
    table_rows = []
    for condition, sub in df.groupby("condition"):
        init_vals = sub[initial_col].to_numpy(dtype=float)
        final_vals = sub[final_col].to_numpy(dtype=float)
        n, mean_diff, t_stat, p_val = paired_t_test(
            init_vals, final_vals, alternative="greater"
        )
        table_rows.append(
            {
                "Condition": condition,
                "n": n,
                "mean_diff": mean_diff,
                "t": t_stat,
                "p": p_val,
                "sig": significance_stars(p_val),
            }
        )
    print_table(
        table_rows,
        columns=["Condition", "n", "mean_diff", "t", "p", "sig"],
        formatters={
            "mean_diff": lambda val: f"{val:.3f}" if np.isfinite(val) else "nan",
            "t": lambda val: f"{val:.3f}" if np.isfinite(val) else "nan",
            "p": lambda val: f"{val:.4g}" if np.isfinite(val) else "nan",
        },
        widths={"Condition": 50},
        aligns={"n": "right", "mean_diff": "right", "t": "right", "p": "right"},
    )
    if output_path is not None:
        write_table_csv(
            table_rows,
            ["Condition", "n", "mean_diff", "t", "p", "sig"],
            output_path,
        )


def print_between_condition_tests(
    df: pd.DataFrame,
    *,
    delta_col: str,
    output_path: Path | None = None,
) -> None:
    """Print and optionally export pairwise Welch tests between conditions.

    Args:
        df: Analysis dataframe.
        delta_col: Delta column name.
        output_path: Optional CSV output path.

    Returns:
        None.
    """
    conditions = sorted(df["condition"].unique())
    pairs = []
    p_values = []
    for i, cond_a in enumerate(conditions):
        vals_a = df.loc[df["condition"] == cond_a, delta_col].to_numpy(dtype=float)
        vals_a = vals_a[np.isfinite(vals_a)]
        for cond_b in conditions[i + 1 :]:
            vals_b = df.loc[df["condition"] == cond_b, delta_col].to_numpy(dtype=float)
            vals_b = vals_b[np.isfinite(vals_b)]
            mean_diff, t_stat, p_val = welch_t_test(vals_a, vals_b)
            pairs.append((cond_a, cond_b, mean_diff, t_stat, p_val))
            p_values.append(p_val if np.isfinite(p_val) else 1.0)

    if not pairs:
        print("\nNo between-condition comparisons available.")
        if output_path is not None:
            write_table_csv(
                [],
                ["Condition A", "Condition B", "mean_diff", "t", "p", "p_adj", "sig"],
                output_path,
            )
        return

    p_adj = holm_adjust(p_values)

    print("\nBetween-condition delta tests (Welch t-test, Holm-corrected)")
    table_rows = []
    for (cond_a, cond_b, mean_diff, t_stat, p_val), p_corr in zip(pairs, p_adj):
        table_rows.append(
            {
                "Condition A": cond_a,
                "Condition B": cond_b,
                "mean_diff": mean_diff,
                "t": t_stat,
                "p": p_val,
                "p_adj": p_corr,
                "sig": significance_stars(p_corr),
            }
        )
    print_table(
        table_rows,
        columns=["Condition A", "Condition B", "mean_diff", "t", "p", "p_adj", "sig"],
        formatters={
            "mean_diff": lambda val: f"{val:.3f}" if np.isfinite(val) else "nan",
            "t": lambda val: f"{val:.3f}" if np.isfinite(val) else "nan",
            "p": lambda val: f"{val:.4g}" if np.isfinite(val) else "nan",
            "p_adj": lambda val: f"{val:.4g}" if np.isfinite(val) else "nan",
        },
        widths={"Condition A": 40, "Condition B": 40},
        aligns={"mean_diff": "right", "t": "right", "p": "right", "p_adj": "right"},
    )
    if output_path is not None:
        write_table_csv(
            table_rows,
            ["Condition A", "Condition B", "mean_diff", "t", "p", "p_adj", "sig"],
            output_path,
        )


def serial_first_vs_rest_df(
    df: pd.DataFrame, *, persuader_relative: bool
) -> pd.DataFrame:
    """Build one row per round with first-vs-rest serial deltas.

    Args:
        df: Analysis dataframe.
        persuader_relative: Whether to use persuader-relative belief values.

    Returns:
        Dataframe with first and rest deltas by condition.
    """
    serial_name = persuader_relative_name("serial", persuader_relative)
    initial_name = persuader_relative_name("initial", persuader_relative)
    final_name = persuader_relative_name("final", persuader_relative)

    serial_df = df[df["continuous"] == ContinuousMeasure.SERIAL_QUESTIONS].copy()
    if serial_df.empty or serial_name not in serial_df.columns:
        return pd.DataFrame()

    records: list[dict[str, object]] = []
    for _, row in serial_df.iterrows():
        serial_vals = row[serial_name] if isinstance(row[serial_name], list) else []
        if not serial_vals:
            continue
        initial_val = row.get(initial_name)
        final_val = row.get(final_name)
        if not isinstance(initial_val, (int, float)) or not isinstance(
            final_val, (int, float)
        ):
            continue
        series = [float(initial_val), *serial_vals, float(final_val)]
        deltas = [series[idx + 1] - series[idx] for idx in range(len(series) - 1)]
        if not deltas:
            continue
        first_delta = float(deltas[0])
        rest_delta = float(sum(deltas[1:])) if len(deltas) > 1 else 0.0
        condition = str(row["condition"])
        records.append(
            {
                "condition": condition,
                "condition_label": split_condition_label(condition),
                "first_delta": first_delta,
                "rest_delta": rest_delta,
            }
        )

    return pd.DataFrame(records)


def serial_first_vs_rest_table(
    df: pd.DataFrame, *, output_path: Path | None = None
) -> None:
    """Print and optionally export first-vs-rest significance tests.

    Args:
        df: First-vs-rest serial delta dataframe.
        output_path: Optional CSV output path.

    Returns:
        None.
    """
    if df.empty:
        print("No serial-question rounds found for the selected filters.")
        if output_path is not None:
            write_table_csv(
                [],
                [
                    "condition",
                    "n_rounds",
                    "n_nonzero",
                    "mean_first",
                    "mean_rest",
                    "median_first",
                    "median_rest",
                    "statistic",
                    "pvalue",
                    "median_diff",
                    "mean_diff",
                    "t_stat",
                    "t_pvalue",
                    "sig",
                ],
                output_path,
            )
        return

    rows: list[dict[str, object]] = []
    for condition, group in df.groupby("condition"):
        differences = group["first_delta"] - group["rest_delta"]
        nonzero = differences[differences != 0]
        if nonzero.empty:
            stat_value = np.nan
            p_value = np.nan
            effective_n = 0
        else:
            result = stats.wilcoxon(
                nonzero,
                alternative="greater",
                zero_method="wilcox",
                correction=False,
                method="auto",
            )
            stat_value = float(result.statistic)
            p_value = float(result.pvalue)
            effective_n = int(nonzero.shape[0])

        t_stat, t_pvalue = stats.ttest_rel(
            group["first_delta"], group["rest_delta"], nan_policy="omit"
        )
        mean_diff = float(np.mean(differences))
        if np.isfinite(t_pvalue) and mean_diff > 0:
            t_pvalue_one = float(t_pvalue) / 2
        else:
            t_pvalue_one = 1.0

        rows.append(
            {
                "condition": condition,
                "n_rounds": int(group.shape[0]),
                "n_nonzero": effective_n,
                "mean_first": float(np.mean(group["first_delta"])),
                "mean_rest": float(np.mean(group["rest_delta"])),
                "median_first": float(np.median(group["first_delta"])),
                "median_rest": float(np.median(group["rest_delta"])),
                "statistic": stat_value,
                "pvalue": p_value,
                "median_diff": float(np.median(differences)),
                "mean_diff": mean_diff,
                "t_stat": float(t_stat),
                "t_pvalue": t_pvalue_one,
            }
        )

    results = pd.DataFrame(rows).sort_values("condition")
    results["sig"] = results["pvalue"].apply(significance_stars)
    print("\nWilcoxon signed-rank test (first > rest):")
    table_rows = results.to_dict(orient="records")
    print_table(
        table_rows,
        columns=[
            "condition",
            "n_rounds",
            "n_nonzero",
            "mean_first",
            "mean_rest",
            "median_first",
            "median_rest",
            "statistic",
            "pvalue",
            "median_diff",
            "mean_diff",
            "t_stat",
            "t_pvalue",
            "sig",
        ],
        formatters={
            "mean_first": lambda val: f"{val:.3f}",
            "mean_rest": lambda val: f"{val:.3f}",
            "median_first": lambda val: f"{val:.3f}",
            "median_rest": lambda val: f"{val:.3f}",
            "statistic": lambda val: f"{val:.3f}" if pd.notna(val) else "nan",
            "pvalue": lambda val: f"{val:.4g}" if pd.notna(val) else "nan",
            "median_diff": lambda val: f"{val:.3f}",
            "mean_diff": lambda val: f"{val:.3f}",
            "t_stat": lambda val: f"{val:.3f}" if pd.notna(val) else "nan",
            "t_pvalue": lambda val: f"{val:.4g}" if pd.notna(val) else "nan",
        },
        widths={"condition": 60},
        aligns={
            "n_rounds": "right",
            "n_nonzero": "right",
            "mean_first": "right",
            "mean_rest": "right",
            "median_first": "right",
            "median_rest": "right",
            "statistic": "right",
            "pvalue": "right",
            "median_diff": "right",
            "mean_diff": "right",
            "t_stat": "right",
            "t_pvalue": "right",
        },
    )
    if output_path is not None:
        write_table_csv(
            table_rows,
            [
                "condition",
                "n_rounds",
                "n_nonzero",
                "mean_first",
                "mean_rest",
                "median_first",
                "median_rest",
                "statistic",
                "pvalue",
                "median_diff",
                "mean_diff",
                "t_stat",
                "t_pvalue",
                "sig",
            ],
            output_path,
        )
