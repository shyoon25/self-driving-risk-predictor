from __future__ import annotations

import numpy as np

from scene_risk.data.schemas import RiskLevel
from scene_risk.risk.assessor import RiskAssessor
from scene_risk.risk.metrics import min_distance, ttc
from tests.conftest import make_prediction

_HORIZON = 2.0


def _pred(agent_id: str, waypoints: list[list[float]]):
    return make_prediction(agent_id, np.array(waypoints, dtype=np.float64), _HORIZON)


# --- metrics ---------------------------------------------------------------


def test_ttc_head_on() -> None:
    ego = _pred("ego", [[1, 0], [2, 0], [3, 0], [4, 0]])
    agent = _pred("a", [[4, 0], [3, 0], [2, 0], [1, 0]])

    # dt = 2.0 / 4 = 0.5; separation drops to 1.0 m at index 1.
    np.testing.assert_allclose(ttc(ego, agent, threshold_m=2.0), 1.0)
    np.testing.assert_allclose(min_distance(ego, agent), 1.0)


def test_ttc_no_collision() -> None:
    ego = _pred("ego", [[0, 0], [1, 0], [2, 0], [3, 0]])
    agent = _pred("a", [[0, 10], [1, 10], [2, 10], [3, 10]])

    assert ttc(ego, agent, threshold_m=2.0) == float("inf")
    np.testing.assert_allclose(min_distance(ego, agent), 10.0)


def test_already_overlapping() -> None:
    ego = _pred("ego", [[0, 0], [1, 0]])
    agent = _pred("a", [[0, 0], [1, 0]])

    np.testing.assert_allclose(min_distance(ego, agent), 0.0)
    # collision at the very first step: dt = 2.0 / 2 = 1.0 -> ttc = 1.0.
    np.testing.assert_allclose(ttc(ego, agent, threshold_m=2.0), 1.0)


def test_parallel_paths_constant_offset() -> None:
    ego = _pred("ego", [[0, 0], [1, 0], [2, 0]])
    agent = _pred("a", [[0, 3], [1, 3], [2, 3]])

    np.testing.assert_allclose(min_distance(ego, agent), 3.0)
    assert ttc(ego, agent, threshold_m=2.0) == float("inf")


# --- assessor --------------------------------------------------------------


def test_assess_scene_critical() -> None:
    assessor = RiskAssessor()
    ego = _pred("ego", [[0, 0], [1, 0]])
    agent = _pred("a", [[0, 0], [1, 0]])  # overlapping -> min_dist 0 -> CRITICAL

    result = assessor.assess_scene(ego, [agent], "scene", "sample")

    assert result.risk_label == RiskLevel.CRITICAL
    assert result.agent_risks[0].risk_level == RiskLevel.CRITICAL
    assert 0.0 <= result.scene_risk_score <= 1.0


def test_assess_scene_low() -> None:
    assessor = RiskAssessor()
    ego = _pred("ego", [[0, 0], [1, 0], [2, 0]])
    agent = _pred("a", [[0, 50], [1, 50], [2, 50]])  # far -> LOW

    result = assessor.assess_scene(ego, [agent], "scene", "sample")

    assert result.risk_label == RiskLevel.LOW
    assert 0.0 <= result.scene_risk_score <= 1.0


def test_assess_scene_empty_agents() -> None:
    assessor = RiskAssessor()
    ego = _pred("ego", [[0, 0], [1, 0]])

    result = assessor.assess_scene(ego, [], "scene", "sample")

    assert result.agent_risks == []
    assert result.risk_label == RiskLevel.LOW
    assert result.scene_risk_score == 0.0
