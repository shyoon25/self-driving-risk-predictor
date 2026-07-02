from __future__ import annotations

import numpy as np

from scene_risk.data.schemas import (
    AgentCategory,
    AgentRisk,
    AgentState,
    RiskLevel,
    SceneRisk,
)
from scene_risk.visualization.bev_renderer import BEVRenderer
from tests.conftest import make_prediction


def _state(agent_id: str, position: list[float]) -> AgentState:
    return AgentState(
        agent_id=agent_id,
        timestamp=0.0,
        position=np.array(position, dtype=np.float64),
        velocity=np.array([1.0, 0.0]),
        heading=0.0,
        category=AgentCategory.VEHICLE,
        size=np.array([2.0, 4.5, 1.7], dtype=np.float64),
    )


def test_render_returns_uint8_frame() -> None:
    renderer = BEVRenderer(canvas_size=800, range_m=50.0)
    ego = _state("ego", [0.0, 0.0])
    agent = _state("a", [10.0, 0.0])
    pred = make_prediction("a", np.array([[11.0, 0.0], [12.0, 0.0]]), 1.0)
    scene_risk = SceneRisk(
        scene_token="scene",
        sample_token="sample",
        agent_risks=[
            AgentRisk(agent_id="a", ttc=2.0, min_distance=10.0, risk_level=RiskLevel.MEDIUM)
        ],
        scene_risk_score=0.33,
        risk_label=RiskLevel.MEDIUM,
    )

    frame = renderer.render("sample", [agent], [pred], scene_risk, ego)

    assert frame.shape == (800, 800, 3)
    assert frame.dtype == np.uint8
    assert frame.any()  # non-empty: something was drawn


def _count_red(frame: np.ndarray) -> int:
    b, g, r = frame[:, :, 0], frame[:, :, 1], frame[:, :, 2]
    return int(np.count_nonzero((r > 200) & (b < 80) & (g < 80)))


def test_small_critical_agent_is_visible() -> None:
    """A pedestrian-sized CRITICAL agent must render as a clearly visible red mark.

    Without the min-size floor and highlight ring, a ~0.5 m object is only a
    handful of pixels; the fix guarantees a substantial red footprint.
    """
    renderer = BEVRenderer(canvas_size=800, range_m=50.0)
    ego = _state("ego", [0.0, 0.0])
    ped = _state("p", [10.0, 0.0])
    ped.size = np.array([0.5, 0.5, 1.8], dtype=np.float64)  # tiny
    pred = make_prediction("p", np.array([[9.0, 0.0], [1.0, 0.0]]), 1.0)
    scene_risk = SceneRisk(
        scene_token="scene",
        sample_token="sample",
        agent_risks=[
            AgentRisk(agent_id="p", ttc=0.5, min_distance=0.4, risk_level=RiskLevel.CRITICAL)
        ],
        scene_risk_score=1.0,
        risk_label=RiskLevel.CRITICAL,
    )

    frame = renderer.render("sample", [ped], [pred], scene_risk, ego)

    # 12x12 min box (144 px) plus the highlight ring — far more than a raw 4 px box.
    assert _count_red(frame) > 120
