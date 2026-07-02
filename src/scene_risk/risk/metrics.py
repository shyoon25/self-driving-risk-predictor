from __future__ import annotations

import numpy as np

from scene_risk.data.schemas import Prediction


def min_distance(ego: Prediction, agent: Prediction) -> float:
    """Minimum Euclidean separation between ego and agent predicted paths."""
    T = min(len(ego.waypoints), len(agent.waypoints))
    dists = np.linalg.norm(ego.waypoints[:T] - agent.waypoints[:T], axis=1)
    return float(dists.min())


def is_closing(ego: Prediction, agent: Prediction, margin_m: float = 0.5) -> bool:
    """Whether ego and agent are predicted to approach each other.

    Returns True when the minimum predicted separation is at least ``margin_m``
    smaller than the separation at the start of the horizon — i.e. the paths get
    meaningfully closer than they already are (an approach or crossing). Returns
    False for pairs that hold a roughly constant gap, such as an object parked
    alongside a stationary ego.
    """
    T = min(len(ego.waypoints), len(agent.waypoints))
    dists = np.linalg.norm(ego.waypoints[:T] - agent.waypoints[:T], axis=1)
    return bool(dists.min() < dists[0] - margin_m)


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
