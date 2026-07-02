from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from scene_risk.data.loader import NuScenesLoader
from scene_risk.pipeline.scene_pipeline import ScenePipeline

pytestmark = pytest.mark.skipif(
    os.environ.get("NUSCENES_DATAROOT") is None,
    reason="NUSCENES_DATAROOT not set",
)


def test_first_scene_produces_frames_and_metrics(tmp_path: Path) -> None:
    dataroot = os.environ["NUSCENES_DATAROOT"]
    loader = NuScenesLoader(dataroot, version="v1.0-mini")
    scene_token = loader.get_scene_tokens()[0]

    pipeline = ScenePipeline(loader)
    pipeline.run(scene_token, tmp_path)

    frames = sorted(tmp_path.glob("frame_*.png"))
    assert len(frames) >= 1

    metrics = json.loads((tmp_path / "metrics.json").read_text())
    assert metrics["scene_token"] == scene_token
    assert metrics["n_samples"] >= 1

    # Moving-agent breakdown is present and consistent with the all-agent aliases.
    assert "mean_ade_moving" in metrics
    assert "n_moving_agent_predictions" in metrics
    assert metrics["mean_ade"] == metrics["mean_ade_all"]
    assert metrics["mean_fde"] == metrics["mean_fde_all"]
