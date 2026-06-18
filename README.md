# Self-Driving Scene Risk Predictor

A production-quality autonomous driving safety stack that predicts future agent behavior and estimates scene-level driving risk on the [nuScenes](https://www.nuscenes.org/) dataset.

This is **not** an object detection project. nuScenes bounding-box annotations serve as detections; the focus is the reasoning stack above perception: trajectory extraction → forecasting → risk assessment → BEV visualization.

---

## Demo

<!-- Replace with actual output frame once pipeline runs end-to-end -->
> _BEV output frames will appear here after running the pipeline._

Each frame shows:
- **Ego vehicle** (white box) at canvas center
- **Agent boxes** color-coded by risk level — green (LOW) → orange (MEDIUM) → red (HIGH) → dark red (CRITICAL)
- **Predicted trajectory fans** extending 3 s into the future
- **Range rings** at 10, 20, 30, 40, 50 m
- **Risk HUD** with scene-level score and agent count

---

## Architecture

Data flows strictly one-way through five layers:

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

**Import rule:** `data ← forecasting ← risk ← visualization ← pipeline`. No module imports from a downstream module. All inter-layer contracts are typed dataclasses in `src/scene_risk/data/schemas.py`.

---

## Key Design Decisions

| Decision | Choice | Why |
|---|---|---|
| Agent tracking | `instance_token` as persistent ID | nuScenes guarantees it — no tracker needed |
| Forecasting | Constant velocity | Deterministic, interpretable, correct baseline |
| Risk metrics | TTC + min-distance | Industry-standard; no ground truth needed to validate visually |
| TTC method | Closest approach on predicted waypoints | Simple and accurate for straight-line forecasts |
| Config system | [Hydra](https://hydra.cc/) | Composable YAML overrides; swap forecaster/risk params without code changes |
| Package layout | `src/` | Prevents accidental relative imports; installable with `pip install -e .` |
| Type checking | mypy (`disallow_untyped_defs`) | Catches schema mismatches across module boundaries at dev time |
| Linting | ruff | Single tool for format + lint + import sort |

---

## Repository Structure

```
scene-risk/
├── configs/
│   ├── config.yaml                   # root Hydra config
│   ├── data/nuscenes.yaml            # dataroot, version
│   ├── forecaster/constant_velocity.yaml
│   └── risk/default.yaml             # TTC + distance thresholds
│
├── src/scene_risk/
│   ├── data/
│   │   ├── schemas.py                # ALL inter-module dataclasses
│   │   ├── loader.py                 # NuScenesLoader — wraps devkit
│   │   └── extractor.py             # SceneExtractor — ann → AgentTrajectory
│   ├── forecasting/
│   │   ├── base.py                   # Forecaster ABC
│   │   └── constant_velocity.py
│   ├── risk/
│   │   ├── metrics.py                # ttc(), min_distance() — pure functions
│   │   ├── levels.py                 # RiskThresholds + classify()
│   │   └── assessor.py              # RiskAssessor → SceneRisk
│   ├── visualization/
│   │   └── bev_renderer.py          # BEVRenderer → (H, W, 3) BGR array
│   └── pipeline/
│       └── scene_pipeline.py        # ScenePipeline — wires all modules
│
├── scripts/
│   └── run_pipeline.py              # Hydra CLI entry point
│
├── tests/
│   ├── conftest.py                  # synthetic fixtures (no nuScenes required)
│   ├── unit/                        # run without dataset
│   └── integration/                 # skipped without NUSCENES_DATAROOT
│
└── .github/workflows/ci.yml         # ruff + mypy + unit tests on push
```

---

## Setup

**Requirements:** Python 3.11+, nuScenes v1.0-mini dataset ([download](https://www.nuscenes.org/nuscenes#download))

```bash
# 1. Clone and install
git clone https://github.com/shyoon25/scene-risk.git
cd scene-risk
pip install -e ".[dev]"

# 2. Verify unit tests pass (no dataset needed)
pytest tests/unit
```

---

## Running the Pipeline

```bash
python scripts/run_pipeline.py \
  data.dataroot=/path/to/nuscenes \
  scene_token=<token>
```

Output frames are written to `outputs/<scene_token[:12]>/frame_0000.png` … `frame_NNNN.png`.

**Override any config value at the command line:**

```bash
# Longer forecast horizon, tighter TTC threshold
python scripts/run_pipeline.py \
  data.dataroot=/path/to/nuscenes \
  scene_token=<token> \
  forecaster.horizon_s=5.0 \
  risk.ttc_critical=1.0
```

**List available scene tokens:**

```python
from scene_risk.data.loader import NuScenesLoader
loader = NuScenesLoader(dataroot="/path/to/nuscenes")
print(loader.get_scene_tokens())
```

---

## Risk Classification

`RiskAssessor` classifies each agent using **OR logic** — breach either threshold to escalate:

| Level | TTC (s) | Min-distance (m) | Score |
|---|---|---|---|
| CRITICAL | ≤ 1.5 | ≤ 1.0 | 1.00 |
| HIGH | ≤ 3.0 | ≤ 3.0 | 0.67 |
| MEDIUM | ≤ 5.0 | ≤ 8.0 | 0.33 |
| LOW | — | — | 0.00 |

Scene-level label = highest agent label. Scene score = score of that label.

---

## Development

```bash
ruff check src tests scripts   # lint
ruff format src tests scripts  # format
mypy src                        # type check
pytest tests/unit               # unit tests (no dataset)
NUSCENES_DATAROOT=/path pytest tests/integration   # integration tests
```

---

## Extending

**Add a new forecaster:**
1. Subclass `Forecaster` in `src/scene_risk/forecasting/`
2. Add `configs/forecaster/<name>.yaml`
3. Wire into `scripts/run_pipeline.py` via Hydra switch
4. Add unit tests with synthetic trajectories — no nuScenes data required

---

## Tech Stack

- **Dataset:** nuScenes v1.0-mini (10 scenes, 2 Hz annotation rate)
- **Forecasting:** constant-velocity rollout, configurable horizon and timestep
- **Risk metrics:** time-to-collision (TTC) + minimum pairwise distance
- **Visualization:** OpenCV BEV renderer, ego-centered, y-axis flipped for image coords
- **Config:** Hydra + OmegaConf composable YAML hierarchy
- **Types:** Python 3.11 dataclasses, mypy strict annotations
- **Tooling:** ruff, pytest, GitHub Actions CI
