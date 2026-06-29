from __future__ import annotations

import numpy as np

from scene_risk.data.schemas import AgentTrajectory
from scene_risk.forecasting.constant_velocity import ConstantVelocityForecaster


def test_straight_line_rollout(straight_trajectory: AgentTrajectory) -> None:
    forecaster = ConstantVelocityForecaster(horizon_s=3.0, dt=0.5)
    pred = forecaster.predict(straight_trajectory)

    # 6 steps over a 3 s horizon at 0.5 s each.
    assert pred.waypoints.shape == (6, 2)
    last = straight_trajectory.states[-1]
    expected = np.array([last.position + last.velocity * 0.5 * (i + 1) for i in range(6)])
    np.testing.assert_allclose(pred.waypoints, expected)


def test_stationary_agent_stays_put(stationary_ego: AgentTrajectory) -> None:
    forecaster = ConstantVelocityForecaster(horizon_s=2.0, dt=0.5)
    pred = forecaster.predict(stationary_ego)

    np.testing.assert_allclose(pred.waypoints, np.zeros((4, 2)))


def test_horizon_stored(straight_trajectory: AgentTrajectory) -> None:
    forecaster = ConstantVelocityForecaster(horizon_s=5.0, dt=0.5)
    pred = forecaster.predict(straight_trajectory)

    assert pred.horizon_s == 5.0
    assert pred.agent_id == straight_trajectory.agent_id
