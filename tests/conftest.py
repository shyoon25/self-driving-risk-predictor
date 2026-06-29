from __future__ import annotations

import numpy as np
import pytest

from scene_risk.data.schemas import (
    AgentCategory,
    AgentState,
    AgentTrajectory,
    Prediction,
)

_DT = 0.5


def make_prediction(agent_id: str, waypoints: np.ndarray, horizon_s: float) -> Prediction:
    """Build a Prediction from an array of future waypoints."""
    return Prediction(
        agent_id=agent_id,
        waypoints=np.asarray(waypoints, dtype=np.float64),
        horizon_s=horizon_s,
    )


def _trajectory(
    agent_id: str,
    start: np.ndarray,
    velocity: np.ndarray,
    n_frames: int = 5,
    category: AgentCategory = AgentCategory.VEHICLE,
) -> AgentTrajectory:
    states = []
    for i in range(n_frames):
        t = i * _DT
        states.append(
            AgentState(
                agent_id=agent_id,
                timestamp=t,
                position=start + velocity * t,
                velocity=velocity.copy(),
                heading=0.0,
                category=category,
                size=np.array([2.0, 4.5, 1.7], dtype=np.float64),
            )
        )
    return AgentTrajectory(agent_id=agent_id, category=category, states=states)


@pytest.fixture
def straight_trajectory() -> AgentTrajectory:
    """Agent at y=0 moving at [10, 0] m/s for 5 frames at 0.5 s intervals."""
    return _trajectory(
        "agent",
        np.array([0.0, 0.0]),
        np.array([10.0, 0.0]),
    )


@pytest.fixture
def stationary_ego() -> AgentTrajectory:
    """Ego stationary at the origin for 5 frames."""
    return _trajectory(
        "ego",
        np.array([0.0, 0.0]),
        np.array([0.0, 0.0]),
    )
