# Self-Driving Scene Risk Predictor

A clean, resume-quality autonomous driving safety stack that predicts future agent behavior and estimates scene-level driving risk on the [nuScenes](https://www.nuscenes.org/) dataset.

This is **not** an object detection project. nuScenes bounding-box annotations serve as detections; the focus is the reasoning stack above perception: trajectory extraction → forecasting → risk assessment → BEV visualization.

---

## Demo

Bird's-eye view of nuScenes mini scene `cc8c0bf5…`, one frame per sample (2 Hz), rendered by the pipeline:

<p align="center">
  <img src="docs/demo.gif" alt="BEV risk demo" width="480">
</p>

A single frame — the ringed bright-red box is a CRITICAL agent flagged on its predicted path:

<p align="center">
  <img src="docs/demo_frame.png" alt="BEV frame with a CRITICAL agent" width="480">
</p>

Each frame shows:
- **Ego vehicle** (white box) at canvas center, with **range rings** at 10–50 m
- **Agent boxes** color-coded by risk — green (LOW) → orange (MEDIUM) → orange-red (HIGH) → **bright red (CRITICAL)**
- **HIGH/CRITICAL agents** are enlarged and wrapped in a **highlight ring** so small objects (pedestrians, cones) stay visible, and are drawn on top
- **Predicted trajectory fans** extending 3 s into the future
- **Risk HUD** with scene-level label, score, total agent count, and `high/crit` count

Running the full scene also writes a `metrics.json` with forecasting accuracy (ADE/FDE) — see [Results](#results).

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
| Risk gating | Escalate only when closing | A constant-gap object (e.g. parked beside a stopped ego) stays LOW — avoids proximity-only false alarms |
| Forecast eval | ADE / FDE vs observed future | Quantifies forecast accuracy against ground truth per scene |
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
│   │   ├── constant_velocity.py
│   │   └── evaluation.py             # ade(), fde() — forecast accuracy metrics
│   ├── risk/
│   │   ├── metrics.py                # ttc(), min_distance(), is_closing() — pure functions
│   │   ├── levels.py                 # RiskThresholds + classify()
│   │   └── assessor.py              # RiskAssessor → SceneRisk (with closing-motion gate)
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
```

---

## Setup

**Requirements:** Python **3.11 or 3.12** (see note below), the GEOS native library, and the
nuScenes v1.0-mini dataset.

```bash
# 1. GEOS — required to build Shapely (a nuscenes-devkit dependency)
brew install geos                      # macOS; Linux: apt-get install libgeos-dev

# 2. Clone and install into an isolated venv
git clone https://github.com/shyoon25/scene-risk.git
cd scene-risk
python3.11 -m venv .venv && source .venv/bin/activate
GEOS_CONFIG="$(brew --prefix geos)/bin/geos-config" pip install -e ".[dev]"

# 3. Verify unit tests pass (no dataset needed)
pytest tests/unit
```

> **Python version:** use 3.11 or 3.12. On Python 3.14 the Hydra 1.3 CLI fails at startup
> (`ValueError: badly formed help string`, from 3.14's stricter `argparse`), and
> `nuscenes-devkit` pins `numpy<2`, which conflicts with packages requiring numpy 2 — a venv
> keeps that contained.

**Get the dataset:** download **Full dataset (v1.0) → Mini** (`v1.0-mini.tgz`, ~4 GB) from the
[nuScenes downloads page](https://www.nuscenes.org/nuscenes#download) — *not* the Panoptic/Lidarseg
add-ons — then extract it so `dataroot` contains `maps/ samples/ sweeps/ v1.0-mini/`:

```bash
mkdir -p ~/data/nuscenes
tar -xf ~/Downloads/v1.0-mini.tgz -C ~/data/nuscenes
```

---

## Running the Pipeline

```bash
python scripts/run_pipeline.py \
  data.dataroot=/path/to/nuscenes \
  scene_token=<token>
```

Output is written to `outputs/<scene_token[:12]>/`: one `frame_0000.png … frame_NNNN.png` per sample, plus a `metrics.json` summarizing forecasting accuracy for the scene.

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

**Closing-motion gate:** after classification, an agent is demoted back to LOW unless it is actually *approaching* — its predicted minimum separation drops at least `risk.closing_margin_m` (default 0.5 m) below where it started. This keeps a stationary object that merely sits close to a stopped ego from triggering a proximity-only alarm.

---

## Results

Forecasting accuracy on mini scene `cc8c0bf5…` (constant-velocity, 3 s horizon), written to `metrics.json`:

| Agents | Count | Mean ADE | Mean FDE |
|---|---|---|---|
| Parked (≤ 0.5 m/s) | 3,513 | 0.04 m | — |
| Moving (> 0.5 m/s) | 959 | 0.33 m | 0.70 m |
| **All** | 4,472 | **0.10 m** | **0.20 m** |

The scene is dominated by parked vehicles, where constant velocity is near-exact; the moving-agent numbers (sub-metre ADE, 0.7 m FDE over 3 s) are the meaningful measure of the baseline forecaster.

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
- **Risk metrics:** time-to-collision (TTC) + minimum pairwise distance, gated by closing motion
- **Forecast evaluation:** ADE / FDE against observed future positions, per scene
- **Visualization:** OpenCV BEV renderer, ego-centered, y-axis flipped for image coords
- **Config:** Hydra + OmegaConf composable YAML hierarchy
- **Types:** Python 3.11 dataclasses, mypy strict annotations
- **Tooling:** ruff, mypy, pytest
