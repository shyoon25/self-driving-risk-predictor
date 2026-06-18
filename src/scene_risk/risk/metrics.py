from __future__ import annotations

import numpy as np

from scene_risk.data.schemas import Prediction


def min_distance(ego: Prediction, agent: Prediction) -> float:
    """Minimum Euclidean separation between ego and agent predicted paths."""
    T = min(len(ego.waypoints), len(agent.waypoints))
    dists = np.linalg.norm(ego.waypoints[:T] - agent.waypoints[:T], axis=1)
    return float(dists.min())


def ttc(
    ego: Prediction,
    agent: Prediction,
    threshold_m: float = 2.0,
) -> float:
    """
    Time-to-collision estimate via predicted waypoints.

    Returns the earliest time (seconds) at which the predicted separation
    drops below threshold_m. Returns float('inf') when no collision is
    predicted over the horizon.
    """
    T = min(len(ego.waypoints), len(agent.waypoints))
    dt = ego.horizon_s / len(ego.waypoints)
    dists = np.linalg.norm(ego.waypoints[:T] - agent.waypoints[:T], axis=1)
    hits = np.nonzero(dists <= threshold_m)[0]
    if hits.size == 0:
        return float("inf")
    return float((int(hits[0]) + 1) * dt)
