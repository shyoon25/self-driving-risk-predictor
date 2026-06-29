# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Goal

Self-Driving Scene Risk Predictor — a clean, resume-quality autonomous driving safety stack that predicts future agent behavior and estimates scene-level driving risk on the nuScenes dataset.

This is NOT an object detection project. nuScenes bounding-box annotations serve as detections; the focus is the reasoning stack above perception: trajectory extraction → forecasting → risk assessment → visualization.

## System Architecture

Data flows strictly one-way through five layers. No module may import from a downstream module.

```
nuScenes Annotations
        │
        ▼
  [ data ]          extract AgentTrajectory objects by instance_token
        │
        ▼
  [ forecasting ]   predict future waypoints → Prediction
        │
        ▼
  [ risk ]          TTC + min-distance → AgentRisk + SceneRisk
        │
        ▼
  [ visualization ] BEV PNG frames with risk overlays
        │
        ▼
  [ pipeline ]      orchestrate all layers per scene
```

**Import rules**:

- **Allowed**: each layer may import only from layers upstream of it. `forecasting` may import `data`; `risk` may import `data` and `forecasting`; `visualization` may import any of `data`, `forecasting`, `risk`; `pipeline` may import all of them.
- **Forbidden**: any import from a downstream layer (e.g. `data` importing `forecasting`, or `risk` importing `visualization`). No cyclic imports.

All inter-layer contracts are typed dataclasses in `src/scene_risk/data/schemas.py`. Never pass raw dicts between modules.

## Domain Context

- **Dataset**: nuScenes v1.0-mini (10 scenes, 2 Hz annotation rate)
- **Tracking**: `instance_token` is a persistent agent identity — no tracker needed
- **Coordinate frame**: global (world) frame for all positions and velocities. Risk is always computed in global coordinates; only `visualization` may transform to ego-centered BEV coordinates for rendering.
- **Ego vehicle**: extracted from `ego_pose` records via `LIDAR_TOP` sample data; not present in `sample["anns"]`
- **Velocity**: obtained via `nusc.box_velocity(ann_token)`; falls back to `[0, 0]` at scene boundaries

## Engineering Principles

- Full type annotations — `disallow_untyped_defs = true`
- `from __future__ import annotations` at the top of every source file
- `data.dataroot` is never committed — always overridden at runtime via Hydra
- Unit tests must not require nuScenes data; integration tests skip when `NUSCENES_DATAROOT` is unset
- ruff: line length 100, selects `E F I UP B SIM`

## Success Criteria

1. Load nuScenes mini scenes and extract per-agent trajectories by `instance_token`.
2. Forecast constant-velocity future paths for all visible agents and ego.
3. Compute per-agent TTC and minimum-distance scores relative to ego.
4. Produce a `SceneRisk` with per-agent breakdown and a scene-level label.
5. Render BEV PNG frames: oriented agent boxes, trajectory fans, risk color-coding.
6. Evaluate forecasting accuracy with ADE/FDE-style displacement metrics against observed future positions.

## CLAUDE.md Hierarchy

Read only the module file relevant to the current task. This file covers cross-cutting concerns only.

```
CLAUDE.md                               ← project-level (this file)
src/scene_risk/
  data/CLAUDE.md                        ← schemas, loader, extractor
  forecasting/CLAUDE.md                 ← forecaster ABC + CV implementation
  risk/CLAUDE.md                        ← metrics, thresholds, assessor
  visualization/CLAUDE.md               ← BEV renderer
  pipeline/CLAUDE.md                    ← scene orchestration + CLI
tests/CLAUDE.md                         ← test conventions and fixture rules
```

## Common Commands

| Task | Command |
|------|---------|
| Install | `pip install -e ".[dev]"` |
| Lint | `ruff check src tests scripts` |
| Format | `ruff format src tests scripts` |
| Type check | `mypy src` |
| Unit tests | `pytest tests/unit` |
| Integration tests | `NUSCENES_DATAROOT=/path pytest tests/integration` |
| Run pipeline | `python scripts/run_pipeline.py data.dataroot=/path scene_token=<tok>` |
