from __future__ import annotations

import numpy as np

from scene_risk.data.schemas import RiskLevel
from scene_risk.risk.assessor import RiskAssessor
from scene_risk.risk.metrics import is_closing, min_distance, ttc
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


def test_ttc_pron_collision() -> None:
    pass


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


def test_is_closing_approaching() -> None:
    ego = _pred("ego", [[0, 0], [0, 0], [0, 0]])
    agent = _pred("a", [[5, 0], [3, 0], [1, 0]])  # walking toward ego

    assert is_closing(ego, agent) is True


def test_is_closing_constant_gap() -> None:
    ego = _pred("ego", [[0, 0], [0, 0], [0, 0]])
    agent = _pred("a", [[1.5, 0], [1.5, 0], [1.5, 0]])  # parked alongside

    assert is_closing(ego, agent) is False


# --- assessor --------------------------------------------------------------


def test_assess_scene_critical_head_on() -> None:
    assessor = RiskAssessor()
    ego = _pred("ego", [[1, 0], [2, 0], [3, 0], [4, 0]])
    agent = _pred("a", [[4, 0], [3, 0], [2, 0], [1, 0]])  # closing head-on -> CRITICAL

    result = assessor.assess_scene(ego, [agent], "scene", "sample")

    assert result.risk_label == RiskLevel.CRITICAL
    assert result.agent_risks[0].risk_level == RiskLevel.CRITICAL
    assert 0.0 <= result.scene_risk_score <= 1.0


def test_parked_object_beside_stationary_ego_stays_low() -> None:
    """Regression: a close but non-approaching agent must not raise the alarm."""
    assessor = RiskAssessor()
    ego = _pred("ego", [[0, 0], [0, 0], [0, 0]])  # stationary
    agent = _pred("a", [[1.0, 0], [1.0, 0], [1.0, 0]])  # parked 1 m away, constant gap

    result = assessor.assess_scene(ego, [agent], "scene", "sample")

    # min_distance (1 m) would classify CRITICAL, but the gate demotes it.
    np.testing.assert_allclose(result.agent_risks[0].min_distance, 1.0)
    assert result.agent_risks[0].risk_level == RiskLevel.LOW
    assert result.risk_label == RiskLevel.LOW


def test_approaching_agent_escalates() -> None:
    assessor = RiskAssessor()
    ego = _pred("ego", [[0, 0], [0, 0], [0, 0]])  # stationary
    agent = _pred("a", [[5, 0], [3, 0], [1, 0]])  # approaching to within 1 m

    result = assessor.assess_scene(ego, [agent], "scene", "sample")

    assert result.agent_risks[0].risk_level in (RiskLevel.HIGH, RiskLevel.CRITICAL)


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
