from __future__ import annotations

from dataclasses import dataclass

from scene_risk.data.schemas import RiskLevel


@dataclass(frozen=True)
class RiskThresholds:
    ttc_critical: float = 1.5
    ttc_high: float = 3.0
    ttc_medium: float = 5.0
    min_dist_critical: float = 1.0
    min_dist_high: float = 3.0
    min_dist_medium: float = 8.0
    collision_threshold_m: float = 2.0
    closing_margin_m: float = 0.5

    def classify(self, ttc: float, min_dist: float) -> RiskLevel:
        if ttc <= self.ttc_critical or min_dist <= self.min_dist_critical:
            return RiskLevel.CRITICAL
        if ttc <= self.ttc_high or min_dist <= self.min_dist_high:
            return RiskLevel.HIGH
        if ttc <= self.ttc_medium or min_dist <= self.min_dist_medium:
            return RiskLevel.MEDIUM
        return RiskLevel.LOW
