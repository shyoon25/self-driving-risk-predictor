# Visualization Module

## Responsibility

Render a single bird's-eye view (BEV) frame for one nuScenes sample. Accepts fully computed `SceneRisk` and returns a raw OpenCV image array. No file I/O, no nuScenes API, no risk computation.

## Key Files

| File | Role |
|------|------|
| `bev_renderer.py` | `BEVRenderer` — world-frame → pixel mapping, draws agents, trajectories, HUD |

## Interface

```python
class BEVRenderer:
    def render(
        self,
        sample_token: str,
        current_states: list[AgentState],
        predictions: list[Prediction],
        scene_risk: SceneRisk,
        ego_state: AgentState,
    ) -> np.ndarray: ...   # (H, W, 3) BGR uint8
```

The **caller** (pipeline) writes the returned array to disk. The renderer never touches the filesystem.

## Inputs

- `current_states` — one `AgentState` per visible agent at this sample
- `predictions` — one `Prediction` per agent; matched to states by `agent_id`
- `scene_risk` — provides per-agent `RiskLevel` for color-coding
- `ego_state` — defines the BEV center; all positions are transformed to ego-relative frame

## Output

`np.ndarray` shape `(canvas_size, canvas_size, 3)`, dtype `uint8`, **BGR** (OpenCV convention).

## Coordinate Mapping

```
px = size/2 + rel_x * scale
py = size/2 - rel_y * scale   ← y-axis flipped (image coords increase downward)
```

Default: `canvas_size=800`, `range_m=50` → `scale = 8 px/m`. Ego is always at canvas center.

## Visual Encoding

| Element | Rendering |
|---------|-----------|
| Agent box | Oriented `cv2.boxPoints` rectangle filled with risk color |
| Predicted path | Line segments from current position forward, same color |
| Ego box | White rectangle at canvas center, heading locked to 0° |
| Range rings | Dark circles at 10, 20, 30, 40, 50 m |
| HUD (top-left) | Risk label + score, agent count |
| Footer | `sample_token[:12]` |

**Risk colors** (BGR, brighter = worse): `LOW=(0,200,0)`, `MEDIUM=(0,165,255)`, `HIGH=(0,90,255)`, `CRITICAL=(0,0,255)`

**Elevated agents** (`HIGH`/`CRITICAL`): drawn last (on top), floored to `_ELEVATED_MIN_PX` box size, and wrapped in a highlight ring so small objects (pedestrians, cones) stay visible. HUD reports the `high/crit` count.

## Constraints

- No imports from `risk`, `forecasting`, or `pipeline`
- Must not call `cv2.imwrite` or any file I/O — return the array only
- Agents outside canvas bounds are **skipped silently** (no exception, no clipping artifacts)
- Depends on `opencv-python-headless` — no display windows (`cv2.imshow` is never called)
