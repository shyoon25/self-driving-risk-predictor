from __future__ import annotations

import math

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


def aggregate_forecast_metrics(
    per_agent: list[tuple[float, float, float]],
    moving_speed_mps: float = 0.5,
) -> dict[str, float | int | None]:
    """Summarize per-agent forecast errors into scene-level metrics.

    Each entry is ``(ade, fde, speed_mps)`` for one agent prediction. Entries whose
    ``ade``/``fde`` is NaN (no observed future to score against) are ignored. A "moving"
    agent is one whose observed speed exceeds ``moving_speed_mps``; reporting it separately
    keeps stationary agents from hiding forecasting difficulty.

    Means are ``None`` (not NaN) when their set is empty, so the JSON stays valid.
    ``mean_ade`` / ``mean_fde`` are kept as backwards-compatible aliases of the all-agent means.
    """

    def mean(xs: list[float]) -> float | None:
        return float(np.mean(xs)) if xs else None

    ade_all = [a for a, f, _ in per_agent if not math.isnan(a)]
    fde_all = [f for a, f, _ in per_agent if not math.isnan(f)]
    ade_moving = [a for a, f, s in per_agent if s > moving_speed_mps and not math.isnan(a)]
    fde_moving = [f for a, f, s in per_agent if s > moving_speed_mps and not math.isnan(f)]

    mean_ade_all = mean(ade_all)
    mean_fde_all = mean(fde_all)
    return {
        "n_agent_predictions": len(ade_all),
        "n_moving_agent_predictions": len(ade_moving),
        "moving_speed_threshold_mps": moving_speed_mps,
        "mean_ade_all": mean_ade_all,
        "mean_fde_all": mean_fde_all,
        "mean_ade_moving": mean(ade_moving),
        "mean_fde_moving": mean(fde_moving),
        # Backwards-compatible aliases.
        "mean_ade": mean_ade_all,
        "mean_fde": mean_fde_all,
    }
