from __future__ import annotations

import cv2  # type: ignore[import]
import numpy as np

from scene_risk.data.schemas import AgentRisk, AgentState, Prediction, RiskLevel, SceneRisk

_RISK_COLOR: dict[RiskLevel, tuple[int, int, int]] = {
    RiskLevel.LOW: (0, 200, 0),
    RiskLevel.MEDIUM: (0, 165, 255),
    RiskLevel.HIGH: (0, 0, 255),
    RiskLevel.CRITICAL: (0, 0, 180),
}
_EGO_COLOR: tuple[int, int, int] = (255, 255, 255)
_UNKNOWN_COLOR: tuple[int, int, int] = (140, 140, 140)


class BEVRenderer:
    """Renders a bird's-eye view frame for one sample."""

    def __init__(self, canvas_size: int = 800, range_m: float = 50.0) -> None:
        self._size = canvas_size
        self._range_m = range_m
        self._scale = canvas_size / (2.0 * range_m)

    def render(
        self,
        sample_token: str,
        current_states: list[AgentState],
        predictions: list[Prediction],
        scene_risk: SceneRisk,
        ego_state: AgentState,
    ) -> np.ndarray:
        canvas = np.full((self._size, self._size, 3), 20, dtype=np.uint8)
        self._draw_grid(canvas)

        risk_by_id: dict[str, AgentRisk] = {r.agent_id: r for r in scene_risk.agent_risks}
        pred_by_id: dict[str, Prediction] = {p.agent_id: p for p in predictions}

        for state in current_states:
            agent_risk = risk_by_id.get(state.agent_id)
            color = _RISK_COLOR[agent_risk.risk_level] if agent_risk else _UNKNOWN_COLOR
            self._draw_agent(
                canvas, state, ego_state.position, color, pred_by_id.get(state.agent_id)
            )

        self._draw_ego(canvas, ego_state)
        self._draw_hud(canvas, scene_risk, sample_token)
        return canvas

    def _to_px(self, pos: np.ndarray, ego_pos: np.ndarray) -> tuple[int, int]:
        rel = pos - ego_pos
        cx = self._size // 2 + int(rel[0] * self._scale)
        cy = self._size // 2 - int(rel[1] * self._scale)  # y-axis flipped for image coords
        return cx, cy

    def _draw_grid(self, canvas: np.ndarray) -> None:
        center = self._size // 2
        for r_m in (10, 20, 30, 40, 50):
            cv2.circle(canvas, (center, center), int(r_m * self._scale), (45, 45, 45), 1)

    def _draw_agent(
        self,
        canvas: np.ndarray,
        state: AgentState,
        ego_pos: np.ndarray,
        color: tuple[int, int, int],
        pred: Prediction | None,
    ) -> None:
        cx, cy = self._to_px(state.position, ego_pos)
        if not (0 <= cx < self._size and 0 <= cy < self._size):
            return

        w_px = max(4, int(state.size[0] * self._scale))
        l_px = max(4, int(state.size[1] * self._scale))
        box = cv2.boxPoints(((cx, cy), (l_px, w_px), -np.degrees(state.heading)))  # type: ignore[call-overload]
        cv2.drawContours(canvas, [box.astype(np.int32)], 0, color, -1)
        cv2.drawContours(canvas, [box.astype(np.int32)], 0, (255, 255, 255), 1)

        if pred is not None:
            prev_pt = (cx, cy)
            for wp in pred.waypoints:
                wp_px = self._to_px(wp, ego_pos)
                if 0 <= wp_px[0] < self._size and 0 <= wp_px[1] < self._size:
                    cv2.line(canvas, prev_pt, wp_px, color, 1)
                prev_pt = wp_px

    def _draw_ego(self, canvas: np.ndarray, ego_state: AgentState) -> None:
        center = (self._size // 2, self._size // 2)
        w_px = max(4, int(ego_state.size[0] * self._scale))
        l_px = max(4, int(ego_state.size[1] * self._scale))
        box = cv2.boxPoints((center, (l_px, w_px), 0.0))  # type: ignore[call-overload]
        cv2.drawContours(canvas, [box.astype(np.int32)], 0, _EGO_COLOR, -1)
        cv2.drawContours(canvas, [box.astype(np.int32)], 0, (0, 0, 0), 1)
        cv2.putText(
            canvas,
            "EGO",
            (center[0] - 12, center[1] - l_px // 2 - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            _EGO_COLOR,
            1,
        )

    def _draw_hud(self, canvas: np.ndarray, scene_risk: SceneRisk, sample_token: str) -> None:
        color = _RISK_COLOR[scene_risk.risk_label]
        cv2.putText(
            canvas,
            f"Risk: {scene_risk.risk_label.value}  score={scene_risk.scene_risk_score:.2f}",
            (10, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2,
        )
        cv2.putText(
            canvas,
            f"agents: {len(scene_risk.agent_risks)}",
            (10, 52),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (180, 180, 180),
            1,
        )
        cv2.putText(
            canvas,
            sample_token[:12],
            (10, self._size - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (100, 100, 100),
            1,
        )
