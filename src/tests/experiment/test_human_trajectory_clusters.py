"""Tests for human trajectory cluster classifier utilities."""

from __future__ import annotations

import numpy as np

from experiment.condition import Condition, Roles
from experiment.round import Round
from simulation.human_trajectory_clusters import (
    classify_round,
    classify_updates,
    extract_round_updates,
    load_human_trajectory_cluster_model,
    trajectory_feature_vector,
)


def test_load_human_trajectory_cluster_model_default() -> None:
    """
    Load default cluster model and validate key dimensions.
    """
    model = load_human_trajectory_cluster_model()
    assert model.grid_points >= 2
    assert model.feature_dim > 0
    assert model.centers_scaled.shape[0] == len(model.cluster_ids)
    assert model.centers_scaled.shape[1] == model.feature_dim


def test_trajectory_feature_vector_dimension_matches_model() -> None:
    """
    Feature extractor should produce vectors matching model feature size.
    """
    model = load_human_trajectory_cluster_model()
    vector = trajectory_feature_vector(
        [0.1, 0.05, -0.02], grid_points=model.grid_points
    )
    assert isinstance(vector, np.ndarray)
    assert vector.size == model.feature_dim


def test_classify_updates_returns_valid_cluster_prediction() -> None:
    """
    Classifier should return stable cluster ids and finite distances.
    """
    model = load_human_trajectory_cluster_model()
    updates = [0.22, 0.12, -0.03, -0.01]

    first = classify_updates(updates, model=model)
    second = classify_updates(updates, model=model)

    assert first.cluster_id in model.cluster_ids
    assert first.cluster_name == model.cluster_names[first.cluster_id]
    assert np.isfinite(first.distance)
    assert first.cluster_id == second.cluster_id
    assert first.cluster_name == second.cluster_name


def test_extract_round_updates_and_classify_round() -> None:
    """
    Round helper should recover deltas and classify a valid round object.
    """
    condition = Condition(
        roles=Roles(human_persuader=True, human_target=True),
        factual_domain=False,
        turn_limit=3,
    )
    round_obj = Round(
        condition=condition,
        proposition="Test proposition",
        persuader_supports_proposition=True,
        target_initial_belief=0.40,
        target_final_belief=0.55,
        serial_questions=[0.50, 0.45, 0.55],
    )
    updates = extract_round_updates(round_obj)
    assert updates is not None
    assert np.allclose(np.asarray(updates), np.asarray([0.10, -0.05, 0.10]))

    model = load_human_trajectory_cluster_model()
    prediction = classify_round(round_obj, model=model)
    assert prediction is not None
    assert prediction.cluster_id in model.cluster_ids
