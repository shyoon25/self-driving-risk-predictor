from __future__ import annotations

from scene_risk.data.schemas import AgentRisk, Prediction, RiskLevel, SceneRisk
from scene_risk.risk.levels import RiskThresholds
from scene_risk.risk.metrics import min_distance, ttc

_LEVEL_ORDER = [RiskLevel.LOW, RiskLevel.MEDIUM, RiskLevel.HIGH, RiskLevel.CRITICAL]
_LEVEL_SCORE = {lvl: i / (len(_LEVEL_ORDER) - 1) for i, lvl in enumerate(_LEVEL_ORDER)}


class RiskAssessor:
    def __init__(self, thresholds: RiskThresholds | None = None) -> None:
        self._thresh = thresholds or RiskThresholds()

    def assess_scene(
        self,
        ego_prediction: Prediction,
        agent_predictions: list[Prediction],
        scene_token: str,
        sample_token: str,
    ) -> SceneRisk:
        agent_risks: list[AgentRisk] = []
        for pred in agent_predictions:
            t = ttc(ego_prediction, pred, self._thresh.collision_threshold_m)
            d = min_distance(ego_prediction, pred)
            level = self._thresh.classify(t, d)
            agent_risks.append(
                AgentRisk(agent_id=pred.agent_id, ttc=t, min_distance=d, risk_level=level)
            )

        if agent_risks:
            worst = max(agent_risks, key=lambda r: _LEVEL_ORDER.index(r.risk_level))
            risk_label = worst.risk_level
        else:
            risk_label = RiskLevel.LOW

        return SceneRisk(
            scene_token=scene_token,
            sample_token=sample_token,
            agent_risks=agent_risks,
            scene_risk_score=_LEVEL_SCORE[risk_label],
            risk_label=risk_label,
        )
