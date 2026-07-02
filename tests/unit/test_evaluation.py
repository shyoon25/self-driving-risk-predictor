from __future__ import annotations

import math

import numpy as np

from scene_risk.forecasting.evaluation import ade, aggregate_forecast_metrics, fde
from tests.conftest import make_prediction


def test_perfect_prediction_zero_error() -> None:
    waypoints = np.array([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
    pred = make_prediction("a", waypoints, 1.5)
    gt = waypoints.copy()

    np.testing.assert_allclose(ade(pred, gt), 0.0)
    np.testing.assert_allclose(fde(pred, gt), 0.0)


def test_constant_offset_known_error() -> None:
    waypoints = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    pred = make_prediction("a", waypoints, 1.5)
    gt = waypoints + np.array([0.0, 2.0])  # 2 m off at every step

    np.testing.assert_allclose(ade(pred, gt), 2.0)
    np.testing.assert_allclose(fde(pred, gt), 2.0)


def test_partial_overlap_uses_shared_horizon() -> None:
    pred = make_prediction("a", np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]), 1.5)
    gt = np.array([[0.0, 1.0], [1.0, 1.0]])  # only 2 future steps observed

    np.testing.assert_allclose(ade(pred, gt), 1.0)
    np.testing.assert_allclose(fde(pred, gt), 1.0)  # final over the overlap (index 1)


def test_empty_ground_truth_is_nan() -> None:
    pred = make_prediction("a", np.array([[0.0, 0.0], [1.0, 0.0]]), 1.0)
    gt = np.empty((0, 2))

    assert math.isnan(ade(pred, gt))
    assert math.isnan(fde(pred, gt))


# --- aggregate_forecast_metrics --------------------------------------------


def test_aggregate_splits_moving_from_stationary() -> None:
    # (ade, fde, speed): one parked (0.0 m/s), one moving (2.0 m/s).
    metrics = aggregate_forecast_metrics([(0.1, 0.2, 0.0), (0.3, 0.6, 2.0)])

    # Moving subset excludes the stationary agent.
    assert metrics["n_moving_agent_predictions"] == 1
    np.testing.assert_allclose(metrics["mean_ade_moving"], 0.3)
    np.testing.assert_allclose(metrics["mean_fde_moving"], 0.6)

    # All-agent metrics include both, and aliases mirror the *_all fields.
    assert metrics["n_agent_predictions"] == 2
    np.testing.assert_allclose(metrics["mean_ade_all"], 0.2)
    np.testing.assert_allclose(metrics["mean_fde_all"], 0.4)
    assert metrics["mean_ade"] == metrics["mean_ade_all"]
    assert metrics["mean_fde"] == metrics["mean_fde_all"]
    assert metrics["moving_speed_threshold_mps"] == 0.5


def test_aggregate_empty_moving_set_is_none_not_nan() -> None:
    metrics = aggregate_forecast_metrics([(0.1, 0.2, 0.0)])  # only a parked agent

    assert metrics["n_moving_agent_predictions"] == 0
    assert metrics["mean_ade_moving"] is None
    assert metrics["mean_fde_moving"] is None
    # All-agent metrics still populated.
    assert metrics["n_agent_predictions"] == 1


def test_aggregate_ignores_nan_entries() -> None:
    # A moving agent with no observed future (NaN) must not be counted.
    metrics = aggregate_forecast_metrics([(0.1, 0.2, 2.0), (float("nan"), float("nan"), 3.0)])

    assert metrics["n_agent_predictions"] == 1
    assert metrics["n_moving_agent_predictions"] == 1
    np.testing.assert_allclose(metrics["mean_ade_all"], 0.1)


def test_aggregate_moving_threshold_is_strict() -> None:
    # Speed exactly at the threshold is NOT moving.
    metrics = aggregate_forecast_metrics([(0.1, 0.2, 0.5)])

    assert metrics["n_moving_agent_predictions"] == 0
    assert metrics["mean_ade_moving"] is None
