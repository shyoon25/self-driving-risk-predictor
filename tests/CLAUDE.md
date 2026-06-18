# Tests

## Structure

```
tests/
  conftest.py          shared fixtures — synthetic trajectories, no nuScenes data
  unit/                tests that run without any dataset
  integration/         tests that require NUSCENES_DATAROOT env var
```

## Critical Rule

**Unit tests must never require nuScenes data.** All `tests/unit/` tests must run with `pytest tests/unit` on a machine with no dataset present.

Integration tests guard themselves:
```python
@pytest.mark.skipif(
    os.environ.get("NUSCENES_DATAROOT") is None,
    reason="NUSCENES_DATAROOT not set",
)
```

## Shared Fixtures (`conftest.py`)

| Fixture | Description |
|---------|-------------|
| `straight_trajectory` | `AgentTrajectory` at y=0, velocity=[10,0] m/s, 5 frames at 0.5s intervals |
| `stationary_ego` | `AgentTrajectory` stationary at origin, 5 frames |

Helper function `make_prediction(agent_id, waypoints, horizon_s)` is defined at module level in `conftest.py` and available for direct import in test files.

## Unit Test Conventions

- One test file per source module: `test_extractor.py`, `test_forecaster.py`, `test_risk.py`
- Test names describe the **scenario**, not the implementation: `test_ttc_head_on`, not `test_ttc_function`
- Numeric assertions use `np.testing.assert_allclose` (not `==`) for floating-point results
- Risk level assertions allow adjacent levels when the exact boundary depends on thresholds: `assert result in (RiskLevel.HIGH, RiskLevel.CRITICAL)` is acceptable for a head-on scenario

## What to Test

| Module | Key scenarios |
|--------|--------------|
| `extractor` | `_map_category` parametrize across all category prefixes |
| `forecaster` | straight line shape/values, stationary case, horizon stored correctly |
| `risk.metrics` | head-on collision, no-collision, already-overlapping, parallel paths |
| `risk.assessor` | critical scenario, low scenario, empty agent list, score in [0,1] |
| `visualization` | render returns non-empty uint8 array of correct shape (synthetic data only) |
| `pipeline` (integration) | first mini scene produces ≥ 1 PNG frame |
