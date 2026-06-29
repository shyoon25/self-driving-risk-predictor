from __future__ import annotations

import math

import numpy as np

from scene_risk.forecasting.evaluation import ade, fde
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
