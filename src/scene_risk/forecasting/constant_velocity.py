from __future__ import annotations

import numpy as np

from scene_risk.data.schemas import AgentTrajectory, Prediction
from scene_risk.forecasting.base import Forecaster


class ConstantVelocityForecaster(Forecaster):
    """Rolls out the last observed velocity for a fixed horizon."""

    def __init__(self, horizon_s: float = 3.0, dt: float = 0.5) -> None:
        self.horizon_s = horizon_s
        self.dt = dt
        self._steps = max(1, round(horizon_s / dt))

    def predict(self, trajectory: AgentTrajectory) -> Prediction:
        last = trajectory.states[-1]
        waypoints = np.empty((self._steps, 2), dtype=np.float64)
        pos = last.position.copy()
        for i in range(self._steps):
            pos = pos + last.velocity * self.dt
            waypoints[i] = pos
        return Prediction(
            agent_id=trajectory.agent_id,
            waypoints=waypoints,
            horizon_s=self.horizon_s,
        )
