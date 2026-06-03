# analysis/analysis.py
"""Run plotting and statistical blocks for persuasion round analyses."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import seaborn as sns

from experiment.cli_utils import add_min_date_arg
from experiment.condition_filters import add_condition_filter_args, filters_from_args

from .data_loading import load_dataframe, persuader_relative_name
from .plot_blocks import (
    plot_condition_bars,
    plot_mouse_mean_all_segment_aligned,
    plot_mouse_trace_segment_aligned,
    plot_pre_post_boxes,
    plot_proposition_hist,
    plot_serial_first_vs_rest,
    plot_serial_mean_all,
    plot_serial_questions,
)
from .stats_blocks import (
    print_between_condition_tests,
    print_condition_delta_summary,
    print_within_condition_tests,
    serial_first_vs_rest_df,
    serial_first_vs_rest_table,
)

FIG_DIR = Path(__file__).parent / "figures"
FIG_DIR.mkdir(exist_ok=True, parents=True)
DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True, parents=True)

PLOT_CHOICES = (
    "proposition_hist",
    "condition_bars",
    "pre_post_violins",
    "serial_by_condition",
    "serial_mean_all",
    "serial_first_vs_rest",
    "mouse_by_segment",
    "mouse_mean_by_segment",
)

STAT_CHOICES = (
    "condition_delta_summary",
    "within_condition_tests",
    "between_condition_tests",
    "serial_first_vs_rest_tests",
)


def _parse_selector(raw_value: str, *, choices: tuple[str, ...], flag: str) -> set[str]:
    """Parse comma-separated selector values for `--plots` or `--stats`.

    Args:
        raw_value: Raw CLI value.
        choices: Allowed values.
        flag: CLI flag name for error text.

    Returns:
        Set of selected values.

    Raises:
        ValueError: If unknown values are provided.
    """
    normalized = str(raw_value).strip().lower()
    if normalized == "all":
        return set(choices)

    selected = {
        token.strip().lower().replace("-", "_")
        for token in str(raw_value).split(",")
        if token.strip()
    }
    unknown = sorted(selected - set(choices))
    if unknown:
        raise ValueError(
            f"Unknown {flag} values: {', '.join(unknown)}. "
            f"Allowed: {', '.join(choices)}, all"
        )
    return selected


def _run_selected_plot_blocks(
    *,
    df: pd.DataFrame,
    selected_plots: set[str],
    show: bool,
    persuader_relative: bool,
    serial_first_df: pd.DataFrame | None,
) -> None:
    """Run selected plotting blocks.

    Args:
        df: Analysis dataframe.
        selected_plots: Selected plot keys.
        show: Whether to show figures interactively.
        persuader_relative: Whether to use persuader-relative values.
        serial_first_df: Optional first-vs-rest serial dataframe.

    Returns:
        None.
    """
    plot_jobs = {
        "proposition_hist": lambda: plot_proposition_hist(df, show, fig_dir=FIG_DIR),
        "condition_bars": lambda: plot_condition_bars(df, show, fig_dir=FIG_DIR),
        "pre_post_violins": lambda: plot_pre_post_boxes(
            df,
            show,
            persuader_relative,
            fig_dir=FIG_DIR,
        ),
        "serial_by_condition": lambda: plot_serial_questions(
            df,
            show,
            persuader_relative,
            fig_dir=FIG_DIR,
        ),
        "serial_mean_all": lambda: plot_serial_mean_all(
            df,
            show,
            persuader_relative,
            fig_dir=FIG_DIR,
        ),
        "mouse_by_segment": lambda: plot_mouse_trace_segment_aligned(
            df,
            show,
            persuader_relative,
            normalize_time=False,
            fig_dir=FIG_DIR,
        ),
        "mouse_mean_by_segment": lambda: plot_mouse_mean_all_segment_aligned(
            df,
            show,
            persuader_relative,
            normalize_time=True,
            fig_dir=FIG_DIR,
        ),
    }
    for key, job in plot_jobs.items():
        if key in selected_plots:
            job()
    if "serial_first_vs_rest" in selected_plots and serial_first_df is not None:
        plot_serial_first_vs_rest(serial_first_df, show, fig_dir=FIG_DIR)


def _run_selected_stat_blocks(
    *,
    df: pd.DataFrame,
    selected_stats: set[str],
    persuader_relative: bool,
    serial_first_df: pd.DataFrame | None,
) -> None:
    """Run selected statistical table blocks.

    Args:
        df: Analysis dataframe.
        selected_stats: Selected statistics keys.
        persuader_relative: Whether to use persuader-relative values.
        serial_first_df: Optional first-vs-rest serial dataframe.

    Returns:
        None.
    """
    initial_col = persuader_relative_name("initial", persuader_relative)
    final_col = persuader_relative_name("final", persuader_relative)
    delta_col = "delta_dir" if persuader_relative else "delta_raw"
    n_boot = 5000

    stat_jobs = {
        "condition_delta_summary": lambda: print_condition_delta_summary(
            df,
            delta_col=delta_col,
            n_boot=n_boot,
            output_path=DATA_DIR / "condition_delta_summary.csv",
        ),
        "within_condition_tests": lambda: print_within_condition_tests(
            df,
            initial_col=initial_col,
            final_col=final_col,
            output_path=DATA_DIR / "within_condition_tests.csv",
        ),
        "between_condition_tests": lambda: print_between_condition_tests(
            df,
            delta_col=delta_col,
            output_path=DATA_DIR / "between_condition_tests.csv",
        ),
    }
    for key, job in stat_jobs.items():
        if key in selected_stats:
            job()
    if "serial_first_vs_rest_tests" in selected_stats and serial_first_df is not None:
        serial_first_vs_rest_table(
            serial_first_df,
            output_path=DATA_DIR / "serial_first_vs_rest_tests.csv",
        )


def main() -> None:
    """Parse CLI args and run selected analysis blocks.

    Args:
        None.

    Returns:
        None.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--show", action="store_true", help="show figures as well")
    add_min_date_arg(parser)
    parser.add_argument(
        "--no-persuader-relative",
        action="store_true",
        help="Plot raw support instead of persuader-relative support",
    )
    parser.add_argument(
        "--plots",
        type=str,
        default="all",
        help=(
            "Comma-separated plot blocks to run. "
            "Use 'all' (default) or any of: " + ", ".join(PLOT_CHOICES)
        ),
    )
    parser.add_argument(
        "--stats",
        type=str,
        default="all",
        help=(
            "Comma-separated statistical table blocks to run. "
            "Use 'all' (default) or any of: " + ", ".join(STAT_CHOICES)
        ),
    )
    add_condition_filter_args(parser)
    args = parser.parse_args()

    persuader_relative = not args.no_persuader_relative
    filters = filters_from_args(args)
    try:
        selected_plots = _parse_selector(
            args.plots, choices=PLOT_CHOICES, flag="--plots"
        )
        selected_stats = _parse_selector(
            args.stats, choices=STAT_CHOICES, flag="--stats"
        )
    except ValueError as error:
        parser.error(str(error))

    df = load_dataframe(args.min_date, filters=filters)
    if df.empty:
        print("No rounds found – nothing to plot.")
        return

    print(
        f"Loaded {len(df)} finished rounds across {df['condition'].nunique()} conditions."
    )

    sns.set_style("whitegrid")

    needs_serial_first = bool(
        {"serial_first_vs_rest", "serial_first_vs_rest_tests"}
        & (selected_plots | selected_stats)
    )
    serial_first_df = (
        serial_first_vs_rest_df(df, persuader_relative=persuader_relative)
        if needs_serial_first
        else None
    )

    _run_selected_plot_blocks(
        df=df,
        selected_plots=selected_plots,
        show=args.show,
        persuader_relative=persuader_relative,
        serial_first_df=serial_first_df,
    )
    _run_selected_stat_blocks(
        df=df,
        selected_stats=selected_stats,
        persuader_relative=persuader_relative,
        serial_first_df=serial_first_df,
    )

    print(f"Figures saved to {FIG_DIR.resolve()}")


if __name__ == "__main__":
    main()
