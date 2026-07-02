# Risk Module

## Responsibility

Compute per-agent safety metrics relative to ego and aggregate them into a scene-level risk label and score. Operates entirely on `Prediction` objects — no nuScenes internals, no rendering.

## Key Files

| File | Role |
|------|------|
| `levels.py` | `RiskThresholds` — frozen dataclass of TTC/distance cut-offs; `classify()` method |
| `metrics.py` | `ttc()` and `min_distance()` — pure functions on waypoint arrays |
| `assessor.py` | `RiskAssessor` — iterates agents, calls metrics, aggregates to `SceneRisk` |

## Interface

```python
class RiskAssessor:
    def assess_scene(
        self,
        ego_prediction: Prediction,
        agent_predictions: list[Prediction],
        scene_token: str,
        sample_token: str,
    ) -> SceneRisk: ...
```

## Inputs

- `ego_prediction` — ego forecast at current sample
- `agent_predictions` — one `Prediction` per visible agent at current sample

## Outputs

- `SceneRisk` with:
  - `agent_risks: list[AgentRisk]` — per-agent TTC, min-distance, and `RiskLevel`
  - `scene_risk_score: float` in `[0.0, 1.0]`
  - `risk_label: RiskLevel` — highest agent label

## Metrics

**`ttc(ego, agent, threshold_m)`** — earliest time (seconds) where predicted separation ≤ `threshold_m`. Returns `float('inf')` if no collision predicted. Time = `(first_hit_index + 1) * dt` where `dt = horizon_s / T`.

**`min_distance(ego, agent)`** — minimum pairwise Euclidean distance over the shared forecast horizon `T = min(len(ego.waypoints), len(agent.waypoints))`.

**`is_closing(ego, agent, margin_m)`** — True when the minimum predicted separation is at least `margin_m` below the separation at the start of the horizon (an approach or crossing). False for a roughly constant gap.

Both functions are **pure** — no class state, no side effects. Safe to call in isolation for unit tests.

## Risk Classification

`RiskThresholds.classify(ttc, min_dist)` uses **OR logic**: breach either threshold to escalate.

Default thresholds (`configs/risk/default.yaml`):

| Level | TTC (s) | Min-dist (m) |
|-------|---------|-------------|
| CRITICAL | ≤ 1.5 | ≤ 1.0 |
| HIGH | ≤ 3.0 | ≤ 3.0 |
| MEDIUM | ≤ 5.0 | ≤ 8.0 |
| LOW | — | — |

Scene score mapping: `LOW=0.0`, `MEDIUM=0.33`, `HIGH=0.67`, `CRITICAL=1.0`.
Scene label = highest `RiskLevel` among all agents; empty scene → `LOW`, score `0.0`.

**Closing-motion gate**: after `classify()`, the assessor demotes any non-LOW agent back to `LOW` unless `is_closing()` holds. This prevents proximity alone (via either channel) from flagging an object that merely sits at a constant distance — the common false alarm when ego is stopped. `closing_margin_m` (default 0.5 m) is the required drop in separation.

## Constraints

- No imports from `visualization` or `pipeline`
- `RiskLevel` is imported from `data.schemas` — do not redefine it here
- `metrics.py` functions must remain pure (no I/O, no global state)
- `collision_threshold_m` (default 2.0 m) is distinct from the distance thresholds — it is the proximity radius used by `ttc()`, not a risk cut-off
