"""Prepare and optionally fit an ordinal rhetoric regression on Salvi data.

This script merges:
1) annotation-regression features exported by `analysis.annotation_regression`,
and
2) Salvi round-sidecar metadata exported by `analysis.salvi_debategpt_to_rounds`.

The merged dataset preserves Likert outcomes for ordinal modeling while
reusing the existing annotation aggregation pipeline.
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import pandas as pd

DEFAULT_REGRESSION_CSV = Path("analysis/data/annotation_regression_salvi.csv")
DEFAULT_INDEX_CSV = Path("analysis/data/salvi_debategpt_round_index.csv")
DEFAULT_OUTPUT_CSV = Path("analysis/data/salvi_rhetoric_ordinal_dataset.csv")
DEFAULT_SUMMARY_CSV = Path("analysis/data/salvi_rhetoric_ordinal_summary.csv")
DEFAULT_R_SCRIPT = Path("analysis/salvi_ordinal_regression.R")
DEFAULT_RSCRIPT_BIN = "Rscript"


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments.

    Returns:
        Parsed argument namespace.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Merge Salvi sidecar metadata with annotation-regression features and "
            "optionally run ordinal regression."
        )
    )
    parser.add_argument(
        "--regression-csv",
        type=Path,
        default=DEFAULT_REGRESSION_CSV,
        help="Input CSV from analysis.annotation_regression.",
    )
    parser.add_argument(
        "--index-csv",
        type=Path,
        default=DEFAULT_INDEX_CSV,
        help="Sidecar index CSV from analysis.salvi_debategpt_to_rounds.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=DEFAULT_OUTPUT_CSV,
        help="Merged output dataset path for ordinal modeling.",
    )
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=DEFAULT_SUMMARY_CSV,
        help="Ordinal-model coefficient summary CSV output path.",
    )
    parser.add_argument(
        "--run-r",
        action="store_true",
        help="Run the ordinal R model after writing merged CSV.",
    )
    parser.add_argument(
        "--r-script",
        type=Path,
        default=DEFAULT_R_SCRIPT,
        help="R script used for ordinal model fitting.",
    )
    parser.add_argument(
        "--rscript-bin",
        default=DEFAULT_RSCRIPT_BIN,
        help="Rscript executable path.",
    )
    return parser.parse_args()


def _normalize_source_paths(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize source-path column to absolute-resolved paths.

    Args:
        frame: Dataframe containing `source_path`.

    Returns:
        Copy with normalized `source_path`.
    """
    normalized = frame.copy()
    normalized["source_path"] = normalized["source_path"].map(
        lambda value: str(Path(str(value)).resolve())
    )
    return normalized


def _load_inputs(
    *,
    regression_csv: Path,
    index_csv: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load regression and sidecar index datasets.

    Args:
        regression_csv: Annotation-regression CSV path.
        index_csv: Salvi sidecar index CSV path.

    Returns:
        Tuple of (regression dataframe, index dataframe).
    """
    regression_frame = pd.read_csv(regression_csv)
    index_frame = pd.read_csv(index_csv)
    regression_frame = _normalize_source_paths(regression_frame)
    index_frame = _normalize_source_paths(index_frame)
    return regression_frame, index_frame


def _validate_required_columns(frame: pd.DataFrame, required: list[str]) -> None:
    """Validate that all required columns are present.

    Args:
        frame: Input dataframe.
        required: Required column names.

    Raises:
        ValueError: If any required column is missing.
    """
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def build_ordinal_dataset(
    *,
    regression_frame: pd.DataFrame,
    index_frame: pd.DataFrame,
) -> pd.DataFrame:
    """Merge annotation features with Likert metadata and filter complete rows.

    Args:
        regression_frame: Output from `analysis.annotation_regression`.
        index_frame: Sidecar metadata from Salvi conversion.

    Returns:
        Merged modeling dataframe.
    """
    key_cols = ["source_path", "line_index", "round_index"]
    _validate_required_columns(regression_frame, key_cols)
    _validate_required_columns(index_frame, key_cols)
    merged = regression_frame.merge(
        index_frame,
        on=key_cols,
        how="inner",
        validate="one_to_one",
    )

    required_model_columns = [
        "agreement_post_likert",
        "agreement_pre_likert",
        "mean_logos_z",
        "mean_pathos_z",
        "mean_ethos_z",
    ]
    _validate_required_columns(merged, required_model_columns)
    cleaned = merged.dropna(subset=required_model_columns).copy()
    cleaned["agreement_post_likert"] = (
        cleaned["agreement_post_likert"].astype(int).clip(lower=1, upper=5)
    )
    cleaned["agreement_pre_likert"] = (
        cleaned["agreement_pre_likert"].astype(int).clip(lower=1, upper=5)
    )
    return cleaned


def write_dataset(path: Path, frame: pd.DataFrame) -> None:
    """Write merged modeling dataset to CSV.

    Args:
        path: Destination CSV path.
        frame: Dataframe to write.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def run_r_model(
    *,
    rscript_bin: str,
    r_script: Path,
    data_csv: Path,
    summary_csv: Path,
) -> None:
    """Run the ordinal R regression script.

    Args:
        rscript_bin: Rscript executable path.
        r_script: R script path.
        data_csv: Prepared modeling CSV path.
        summary_csv: Output summary CSV path.
    """
    cmd = [
        rscript_bin,
        str(r_script),
        "--data",
        str(data_csv),
        "--summary",
        str(summary_csv),
    ]
    subprocess.run(cmd, check=True)


def main() -> None:
    """Prepare the Salvi ordinal dataset and optionally run R modeling."""
    args = parse_args()
    regression_frame, index_frame = _load_inputs(
        regression_csv=args.regression_csv,
        index_csv=args.index_csv,
    )
    modeling_frame = build_ordinal_dataset(
        regression_frame=regression_frame,
        index_frame=index_frame,
    )
    write_dataset(args.output_csv, modeling_frame)
    print(f"wrote_dataset: {args.output_csv}")
    print(f"n_rows: {len(modeling_frame)}")
    if args.run_r:
        run_r_model(
            rscript_bin=str(args.rscript_bin),
            r_script=args.r_script,
            data_csv=args.output_csv,
            summary_csv=args.summary_csv,
        )
        print(f"wrote_summary: {args.summary_csv}")


if __name__ == "__main__":
    main()
