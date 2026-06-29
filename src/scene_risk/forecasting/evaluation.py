from __future__ import annotations

import numpy as np

from scene_risk.data.schemas import Prediction


def ade(pred: Prediction, gt_future: np.ndarray) -> float:
    """Average displacement error between a prediction and observed future positions.

    Mean L2 distance over the overlapping horizon
    ``T = min(len(pred.waypoints), len(gt_future))``. Returns ``float('nan')`` when
    there is no observed future to compare against (``T == 0``).

    Args:
        pred: forecast whose ``waypoints`` have shape ``(P, 2)``.
        gt_future: ground-truth future positions, shape ``(M, 2)``.
    """
    T = min(len(pred.waypoints), len(gt_future))
    if T == 0:
        return float("nan")
    errors = np.linalg.norm(pred.waypoints[:T] - gt_future[:T], axis=1)
    return float(errors.mean())


def fde(pred: Prediction, gt_future: np.ndarray) -> float:
    """Final displacement error at the last overlapping horizon step.

    L2 distance at step ``T - 1`` where ``T = min(len(pred.waypoints), len(gt_future))``.
    Returns ``float('nan')`` when there is no observed future to compare against.

    Args:
        pred: forecast whose ``waypoints`` have shape ``(P, 2)``.
        gt_future: ground-truth future positions, shape ``(M, 2)``.
    """
    T = min(len(pred.waypoints), len(gt_future))
    if T == 0:
        return float("nan")
    return float(np.linalg.norm(pred.waypoints[T - 1] - gt_future[T - 1]))
