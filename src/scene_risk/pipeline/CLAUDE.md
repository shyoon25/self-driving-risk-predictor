# Pipeline Module

## Responsibility

Orchestrate all four upstream modules (data → forecasting → risk → visualization) for a single nuScenes scene. Iterate over every sample, run the full stack, write PNG frames to disk.

## Key Files

| File | Role |
|------|------|
| `scene_pipeline.py` | `ScenePipeline` — dependency-injected orchestrator |

Entry point: `scripts/run_pipeline.py` (Hydra-driven CLI, lives outside this module).

## Interface

```python
class ScenePipeline:
    def __init__(
        self,
        loader: NuScenesLoader,
        forecaster: Forecaster | None = None,   # defaults to ConstantVelocityForecaster
        assessor: RiskAssessor | None = None,   # defaults to RiskAssessor()
        renderer: BEVRenderer | None = None,    # defaults to BEVRenderer()
    ) -> None: ...

    def run(self, scene_token: str, output_dir: Path) -> None: ...
```

All four dependencies have production defaults; override them in tests via constructor injection.

## Processing Loop (per sample)

1. Look up `sample_ts` for the current `sample_token`.
2. Find ego `AgentState` at `sample_ts` from the pre-extracted ego trajectory.
3. Collect `instance_token`s from `sample["anns"]`.
4. For each visible agent: slice its full trajectory to `states where timestamp ≤ sample_ts`; call `forecaster.predict(sub_traj)`.
5. Forecast ego from its own trajectory slice.
6. `assessor.assess_scene(ego_pred, agent_preds, ...)` → `SceneRisk`.
7. `renderer.render(...)` → frame array; write to `output_dir/frame_{idx:04d}.png`.

**Trajectory slicing**: all trajectories are extracted once at scene start; the loop slices by timestamp to simulate a streaming perspective. This is O(n·m) but fine for 10-scene mini.

## Inputs

- `scene_token: str` — from `NuScenesLoader.get_scene_tokens()`
- `output_dir: Path` — created automatically if absent

## Outputs

- `output_dir/frame_0000.png` … `frame_NNNN.png`
- One frame per sample (nuScenes mini: 40 samples/scene at 2 Hz = 20 s)

## Key Assumptions

- Ego is never in `sample["anns"]` — handled separately via `get_ego_trajectory()`
- `_state_at(trajectory, timestamp)` returns the last known state ≤ timestamp; returns `None` only if trajectory has no states before that timestamp (rare edge case → sample is skipped)
- Agents not yet visible at a sample timestamp are correctly excluded because their first annotation timestamp will be greater than `sample_ts`

## CLI Usage

```bash
python scripts/run_pipeline.py \
  data.dataroot=/path/to/nuscenes \
  scene_token=<token>

# Override any config value:
python scripts/run_pipeline.py \
  data.dataroot=/path \
  scene_token=<token> \
  forecaster.horizon_s=5.0 \
  risk.ttc_critical=1.0
```

Output frames land in `outputs/<scene_token[:12]>/`.
