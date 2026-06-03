"""Human persuasion-trajectory cluster classifier utilities."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

DEFAULT_CLUSTER_MODEL_PATH = (
    Path(__file__).resolve().parent
    / "data"
    / "human_trajectory_cluster_model_k3_v1.json"
)


@dataclass(frozen=True)
class ClusterPrediction:
    """
    Prediction output for one belief-update trajectory.

    Attributes:
        cluster_id: Integer cluster id.
        cluster_name: Human-readable cluster label.
        distance: Euclidean distance to assigned scaled centroid.
    """

    cluster_id: int
    cluster_name: str
    distance: float


@dataclass(frozen=True)
class HumanTrajectoryClusterModel:
    """
    Parsed cluster model artifact used for assignment.

    Attributes:
        grid_points: Number of normalized cumulative trajectory points.
        feature_mode: Trajectory feature mode used by this model.
        fixed_turn_count: Fixed turn count for raw fixed-length mode.
        scaler_mean: Feature means used for z-scoring.
        scaler_scale: Feature scales used for z-scoring.
        centers_scaled: Cluster centers in scaled feature space.
        cluster_names: Mapping from cluster id to label.
        metadata: Extra model metadata for reporting/debugging.
    """

    grid_points: int
    feature_mode: Literal["normalized", "raw_fixed_length", "raw_padded"]
    fixed_turn_count: int | None
    scaler_mean: np.ndarray
    scaler_scale: np.ndarray
    centers_scaled: np.ndarray
    cluster_names: dict[int, str]
    metadata: dict[str, Any]

    @property
    def feature_dim(self) -> int:
        """
        Return expected feature dimension.

        Returns:
            Number of columns in feature vectors.
        """
        return int(self.scaler_mean.size)

    @property
    def cluster_ids(self) -> list[int]:
        """
        Return known cluster ids sorted ascending.

        Returns:
            Sorted list of cluster ids.
        """
        return sorted(self.cluster_names)

    @property
    def cluster_items(self) -> list[tuple[int, str]]:
        """
        Return sorted (id, name) cluster pairs.

        Returns:
            Sorted cluster id/name tuples.
        """
        return [
            (cluster_id, self.cluster_names[cluster_id])
            for cluster_id in self.cluster_ids
        ]


def load_human_trajectory_cluster_model(
    model_path: Path | None = None,
) -> HumanTrajectoryClusterModel:
    """
    Load human trajectory cluster model artifact.

    Args:
        model_path: Optional explicit model path. Uses package default when omitted.

    Returns:
        Parsed cluster model.
    """
    path = model_path or DEFAULT_CLUSTER_MODEL_PATH
    payload = json.loads(path.read_text(encoding="utf-8"))

    grid_points_raw = payload.get("grid_points")
    scaler_payload = payload.get("scaler")
    centers_payload = payload.get("centers_scaled")
    clusters_payload = payload.get("clusters")
    metadata_payload = payload.get("metadata", {})
    feature_mode_raw = payload.get("feature_mode", "normalized")
    fixed_turn_count_raw = payload.get("fixed_turn_count")

    if not isinstance(grid_points_raw, int) or grid_points_raw < 2:
        raise ValueError("Cluster model has invalid grid_points.")
    if feature_mode_raw not in {"normalized", "raw_fixed_length", "raw_padded"}:
        raise ValueError("Cluster model has unsupported feature_mode.")
    if fixed_turn_count_raw is not None and (
        not isinstance(fixed_turn_count_raw, int) or fixed_turn_count_raw < 1
    ):
        raise ValueError("Cluster model has invalid fixed_turn_count.")
    if feature_mode_raw in {"raw_fixed_length", "raw_padded"} and (
        fixed_turn_count_raw is None
    ):
        raise ValueError("raw_fixed_length/raw_padded mode requires fixed_turn_count.")
    if not isinstance(scaler_payload, dict):
        raise ValueError("Cluster model missing scaler payload.")
    if not isinstance(centers_payload, list) or not centers_payload:
        raise ValueError("Cluster model missing centers_scaled.")
    if not isinstance(clusters_payload, list) or not clusters_payload:
        raise ValueError("Cluster model missing clusters.")

    mean_raw = scaler_payload.get("mean")
    scale_raw = scaler_payload.get("scale")
    if not isinstance(mean_raw, list) or not isinstance(scale_raw, list):
        raise ValueError("Cluster model scaler is malformed.")
    scaler_mean = np.asarray(mean_raw, dtype=float)
    scaler_scale = np.asarray(scale_raw, dtype=float)
    if scaler_mean.size == 0 or scaler_mean.size != scaler_scale.size:
        raise ValueError("Cluster model scaler dimensions are invalid.")

    centers_scaled = np.asarray(centers_payload, dtype=float)
    if centers_scaled.ndim != 2 or centers_scaled.shape[1] != scaler_mean.size:
        raise ValueError("Cluster model center dimensions are invalid.")

    cluster_names: dict[int, str] = {}
    for item in clusters_payload:
        if not isinstance(item, dict):
            continue
        cluster_id_raw = item.get("id")
        cluster_name_raw = item.get("name")
        if not isinstance(cluster_id_raw, int) or not isinstance(cluster_name_raw, str):
            continue
        cluster_names[int(cluster_id_raw)] = cluster_name_raw
    if not cluster_names:
        raise ValueError("Cluster model has no valid cluster names.")
    if set(cluster_names) != set(range(int(centers_scaled.shape[0]))):
        raise ValueError("Cluster names do not match center ids.")

    return HumanTrajectoryClusterModel(
        grid_points=int(grid_points_raw),
        feature_mode=feature_mode_raw,
        fixed_turn_count=fixed_turn_count_raw,
        scaler_mean=scaler_mean,
        scaler_scale=scaler_scale,
        centers_scaled=centers_scaled,
        cluster_names=cluster_names,
        metadata=dict(metadata_payload) if isinstance(metadata_payload, dict) else {},
    )


def trajectory_feature_vector(
    updates: list[float] | tuple[float, ...],
    *,
    grid_points: int,
    feature_mode: Literal[
        "normalized", "raw_fixed_length", "raw_padded"
    ] = "normalized",
    fixed_turn_count: int | None = None,
) -> np.ndarray:
    """
    Build feature vector used by cluster classifier.

    Feature layout:
    - Normalized cumulative trajectory excluding x=0 (grid_points - 1 values)
    - Conversation length

    Args:
        updates: Persuader-relative per-message belief deltas.
        grid_points: Number of normalized cumulative grid points.
        feature_mode: Whether to use normalized interpolation, raw fixed-length,
            or raw padded cumulative trajectory features.
        fixed_turn_count: Required turn count when ``feature_mode`` is
            ``raw_fixed_length``.

    Returns:
        1D feature vector.
    """
    values = np.asarray(list(updates), dtype=float)
    if values.size == 0:
        raise ValueError("Cannot classify an empty update sequence.")
    cumulative = np.concatenate(
        [
            np.asarray([0.0], dtype=float),
            np.cumsum(values, dtype=float),
        ]
    )

    if feature_mode == "normalized":
        x_values = np.linspace(0.0, 1.0, cumulative.size, dtype=float)
        grid = np.linspace(0.0, 1.0, int(grid_points), dtype=float)
        curve_without_start = np.interp(grid, x_values, cumulative)[1:]
    elif feature_mode == "raw_fixed_length":
        if fixed_turn_count is None:
            raise ValueError("raw_fixed_length mode requires fixed_turn_count.")
        if values.size != int(fixed_turn_count):
            raise ValueError(
                "raw_fixed_length mode requires update count equal to fixed_turn_count."
            )
        curve_without_start = cumulative[1:]
    elif feature_mode == "raw_padded":
        if fixed_turn_count is None:
            raise ValueError("raw_padded mode requires fixed_turn_count.")
        if values.size > int(fixed_turn_count):
            raise ValueError(
                "raw_padded mode requires update count <= fixed_turn_count."
            )
        curve_without_start = cumulative[1:]
        if values.size < int(fixed_turn_count):
            tail_value = float(curve_without_start[-1])
            pad = np.full(
                (int(fixed_turn_count) - values.size,), tail_value, dtype=float
            )
            curve_without_start = np.concatenate([curve_without_start, pad], axis=0)
    else:
        raise ValueError(f"Unsupported feature_mode: {feature_mode}")

    extra = np.asarray([float(values.size)], dtype=float)
    return np.concatenate([curve_without_start, extra], axis=0)


def extract_round_updates(round_obj: Any) -> tuple[float, ...] | None:
    """
    Extract persuader-relative per-message belief deltas from a round-like object.

    Expected fields are compatible with `experiment.round.Round`:
    `target_initial_belief`, `serial_questions`, and `persuader_supports_proposition`.

    Args:
        round_obj: Round-like object with serial belief measurements.

    Returns:
        Tuple of deltas when valid, otherwise None.
    """
    initial = getattr(round_obj, "target_initial_belief", None)
    serial = getattr(round_obj, "serial_questions", None)
    supports = getattr(round_obj, "persuader_supports_proposition", None)

    if not isinstance(initial, (int, float)):
        return None
    if not isinstance(serial, list) or not serial:
        return None
    if not isinstance(supports, bool):
        return None

    values: list[float] = []
    for value in serial:
        if not isinstance(value, (int, float)):
            return None
        values.append(float(value))

    direction = 1.0 if supports else -1.0
    deltas: list[float] = []
    previous = float(initial)
    for value in values:
        deltas.append(direction * (float(value) - previous))
        previous = float(value)
    return tuple(deltas)


def classify_updates(
    updates: list[float] | tuple[float, ...],
    *,
    model: HumanTrajectoryClusterModel,
) -> ClusterPrediction:
    """
    Assign a trajectory to a human cluster.

    Args:
        updates: Persuader-relative per-message belief deltas.
        model: Loaded cluster model.

    Returns:
        Cluster prediction including id, name, and nearest-center distance.
    """
    feature_vector = trajectory_feature_vector(
        updates,
        grid_points=model.grid_points,
        feature_mode=model.feature_mode,
        fixed_turn_count=model.fixed_turn_count,
    )
    if feature_vector.size != model.feature_dim:
        raise ValueError(
            "Feature dimension mismatch: "
            f"got {feature_vector.size}, expected {model.feature_dim}."
        )
    z = (feature_vector - model.scaler_mean) / model.scaler_scale
    diffs = model.centers_scaled - z
    distances = np.sqrt(np.sum(np.square(diffs), axis=1))
    cluster_id = int(np.argmin(distances))
    return ClusterPrediction(
        cluster_id=cluster_id,
        cluster_name=model.cluster_names[cluster_id],
        distance=float(distances[cluster_id]),
    )


def classify_round(
    round_obj: Any,
    *,
    model: HumanTrajectoryClusterModel,
) -> ClusterPrediction | None:
    """
    Classify one round into a human trajectory cluster.

    Args:
        round_obj: Round-like object with serial-question belief measurements.
        model: Loaded cluster model.

    Returns:
        Cluster prediction when valid serial deltas can be extracted, otherwise None.
    """
    updates = extract_round_updates(round_obj)
    if updates is None:
        return None
    return classify_updates(updates, model=model)
