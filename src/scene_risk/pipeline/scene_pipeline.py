from __future__ import annotations

import json
import math
from pathlib import Path

import cv2  # type: ignore[import]
import numpy as np

from scene_risk.data.extractor import SceneExtractor
from scene_risk.data.loader import NuScenesLoader
from scene_risk.data.schemas import AgentState, AgentTrajectory, Prediction
from scene_risk.forecasting.base import Forecaster
from scene_risk.forecasting.constant_velocity import ConstantVelocityForecaster
from scene_risk.forecasting.evaluation import ade, fde
from scene_risk.risk.assessor import RiskAssessor
from scene_risk.visualization.bev_renderer import BEVRenderer

# Tolerance (seconds) for matching trajectory state timestamps to a sample timestamp.
_TS_EPS = 1e-6


class ScenePipeline:
    """Orchestrate data → forecasting → risk → visualization for a single scene.

    All four dependencies have production defaults; override them in tests via
    constructor injection.

    Assumption: the forecaster ``dt`` aligns with the nuScenes 2 Hz annotation rate,
    so predicted waypoints line up index-for-index with the agent's actual future
    states for the ADE/FDE evaluation.
    """

    def __init__(
        self,
        loader: NuScenesLoader,
        forecaster: Forecaster | None = None,
        assessor: RiskAssessor | None = None,
        renderer: BEVRenderer | None = None,
    ) -> None:
        self._loader = loader
        self._forecaster = forecaster or ConstantVelocityForecaster()
        self._assessor = assessor or RiskAssessor()
        self._renderer = renderer or BEVRenderer()
        self._extractor = SceneExtractor(loader.nusc)

    def run(self, scene_token: str, output_dir: Path) -> None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        trajectories = self._extractor.extract(scene_token)
        ego_traj = self._extractor.get_ego_trajectory(scene_token)
        traj_by_id = {t.agent_id: t for t in trajectories}
        sample_tokens = self._loader.get_sample_tokens(scene_token)

        ade_values: list[float] = []
        fde_values: list[float] = []

        for idx, sample_token in enumerate(sample_tokens):
            sample = self._loader.nusc.get("sample", sample_token)
            sample_ts = float(sample["timestamp"]) * 1e-6

            ego_state = _state_at(ego_traj, sample_ts)
            if ego_state is None:
                continue
            ego_pred = self._forecaster.predict(_slice_traj(ego_traj, sample_ts))

            instance_tokens = {
                str(self._loader.nusc.get("sample_annotation", ann)["instance_token"])
                for ann in sample["anns"]
            }

            current_states: list[AgentState] = []
            agent_preds: list[Prediction] = []
            for instance_token in instance_tokens:
                traj = traj_by_id.get(instance_token)
                if traj is None:
                    continue
                state = _state_at(traj, sample_ts)
                if state is None:
                    continue
                pred = self._forecaster.predict(_slice_traj(traj, sample_ts))
                current_states.append(state)
                agent_preds.append(pred)

                gt_future = _future_positions(traj, sample_ts, len(pred.waypoints))
                a = ade(pred, gt_future)
                f = fde(pred, gt_future)
                if not math.isnan(a):
                    ade_values.append(a)
                if not math.isnan(f):
                    fde_values.append(f)

            scene_risk = self._assessor.assess_scene(
                ego_pred, agent_preds, scene_token, sample_token
            )
            frame = self._renderer.render(
                sample_token, current_states, agent_preds, scene_risk, ego_state
            )
            cv2.imwrite(str(output_dir / f"frame_{idx:04d}.png"), frame)

        metrics = {
            "scene_token": scene_token,
            "n_samples": len(sample_tokens),
            "n_agent_predictions": len(ade_values),
            "mean_ade": float(np.mean(ade_values)) if ade_values else None,
            "mean_fde": float(np.mean(fde_values)) if fde_values else None,
        }
        (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))


def _state_at(trajectory: AgentTrajectory, timestamp: float) -> AgentState | None:
    """Return the last state with ``timestamp <= timestamp``, else ``None``."""
    latest: AgentState | None = None
    for state in trajectory.states:
        if state.timestamp <= timestamp + _TS_EPS:
            latest = state
        else:
            break
    return latest


def _slice_traj(trajectory: AgentTrajectory, timestamp: float) -> AgentTrajectory:
    """Sub-trajectory of states observed at or before ``timestamp`` (streaming view)."""
    states = [s for s in trajectory.states if s.timestamp <= timestamp + _TS_EPS]
    return AgentTrajectory(
        agent_id=trajectory.agent_id, category=trajectory.category, states=states
    )


def _future_positions(trajectory: AgentTrajectory, timestamp: float, n: int) -> np.ndarray:
    """First ``n`` actual positions strictly after ``timestamp`` as a ``(<=n, 2)`` array."""
    future = [s.position for s in trajectory.states if s.timestamp > timestamp + _TS_EPS]
    if not future:
        return np.empty((0, 2), dtype=np.float64)
    return np.array(future[:n], dtype=np.float64)
