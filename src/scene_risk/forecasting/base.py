from __future__ import annotations

from abc import ABC, abstractmethod

from scene_risk.data.schemas import AgentTrajectory, Prediction


class Forecaster(ABC):
    @abstractmethod
    def predict(self, trajectory: AgentTrajectory) -> Prediction:
        """Return predicted future waypoints for a single agent."""
        ...
