# Forecasting Module

## Responsibility

Predict future waypoints for a **single agent** given its observed trajectory history. Operates entirely on `AgentTrajectory → Prediction`. No knowledge of other agents, scene context, or risk.

## Key Files

| File | Role |
|------|------|
| `base.py` | `Forecaster` ABC — defines the one-method interface |
| `constant_velocity.py` | `ConstantVelocityForecaster` — rolls out last observed velocity for a fixed horizon |

## Interface

```python
class Forecaster(ABC):
    def predict(self, trajectory: AgentTrajectory) -> Prediction: ...
```

Every implementation must:
- Accept a trajectory with **any length ≥ 1**
- Return `Prediction.waypoints` of shape `(T, 2)` in the **global coordinate frame**
- Set `Prediction.horizon_s` to match the configured horizon (downstream TTC depends on this)

## Inputs

- `AgentTrajectory` — one agent, ≥ 1 observed `AgentState`

## Outputs

- `Prediction` — `agent_id`, `waypoints: np.ndarray (T, 2)`, `horizon_s: float`

## Constant Velocity Forecaster

Uses only `trajectory.states[-1].velocity` and `.position`. Earlier history is ignored.

Config (`configs/forecaster/constant_velocity.yaml`):
- `horizon_s: 3.0` — prediction horizon in seconds
- `dt: 0.5` — timestep (matches nuScenes 2 Hz; do not change without also updating risk thresholds)

Steps `T = max(1, round(horizon_s / dt))`.

## Adding a New Forecaster

1. Create a new file (e.g., `social_lstm.py`), subclass `Forecaster`, implement `predict()`.
2. Add `configs/forecaster/<name>.yaml`.
3. Wire into `scripts/run_pipeline.py` via Hydra switch.
4. Add unit tests using **synthetic** trajectories — no nuScenes data required.

## Constraints

- No imports from `risk`, `visualization`, or `pipeline`
- Waypoints must be in the global frame (same as `AgentState.position`)
- `Prediction.horizon_s` must equal the configured `horizon_s` value — not inferred from waypoint count
