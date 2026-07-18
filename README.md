>  English | [中文](README_CN.md)

# Mouse Warm Spot Semi-Automatic Annotation System

A desktop GUI application for analyzing mouse warm-spot experiment videos. The system automatically detects periods when the warm spot is occluded, estimates the number of mice, and allows manual review with keyboard shortcuts for identity annotation before exporting statistical tables.

## Installation & Running

```bash
# 1. Enter the project directory
cd mouse_warm_spot

# 2. Install dependencies
pip install -r requirements.txt

# 3. (Optional) Install AI inference dependencies (ONNX CNN + AI vision model HTTP client)
pip install -r requirements-ai.txt

# 4. (Optional) Configure AI vision recognition (Kimi / MiniMax / OpenRouter)
# Export the following environment variables into the process environment, or create a .env file and source it:
#   MOUSE_COLOR_AI_PROVIDER=openrouter
#   OPENROUTER_API_KEY=<your-key>
#   OPENROUTER_API_BASE=https://openrouter.ai/api/v1
#   OPENROUTER_VISION_MODEL=minimax/minimax-m3
# See the "Optional AI Vision Recognition" section below for details

# 5. Launch (if using .env, source it before launching)
set -a && source .env && set +a && python main.py
```

### Optional AI Inference (Ear Tag Color CNN Classifier)

The system uses HSV rules for ear tag color classification by default, requiring no extra dependencies. For more accurate color recognition, you can enable the lightweight ONNX-based CNN classifier:

```bash
# Install optional AI inference dependencies
pip install -r requirements-ai.txt

# Place the trained ONNX model in the detection/models/ directory
# Expected filename: ear_tag_classifier.onnx
# The program auto-discovers this file and enables CNN inference
```

**Environment Variable Controls** (no code changes needed):

| Environment Variable | Description | Default |
|---|---|---|
| `MOUSE_COLOR_MODEL_PATH` | Custom ONNX model file path | Auto-discover `detection/models/ear_tag_classifier.onnx` |
| `MOUSE_COLOR_USE_CNN` | Enable CNN (`0`=disable, `1`=enable) | Auto-detect (enabled when dependency + file are both available) |

Fallback Behavior:

| Scenario | Behavior |
|---|---|
| `onnxruntime` not installed | Silent fallback to HSV rules; no impact on program function |
| Model file not found | Silent fallback to HSV rules |
| CNN inference failure (any exception) | That contour falls back to HSV; remaining contours processed normally |
| `MOUSE_COLOR_USE_CNN=0` | Forces HSV rules; no model loading attempted |

The `identity_method` field in recognition results indicates the actual classification method used:
- `ear_tag_color_vlm|exact_match` — AI vision model successful (Kimi / MiniMax / OpenRouter)
- `ear_tag_color_cnn|exact_match` — ONNX CNN successful
- `ear_tag_color_rule|exact_match` — HSV rules used

### Optional AI Vision Recognition (GPT)

The system can send large-context images (multi-frame) of a segment to a vision-language model for color recognition. Only one call is made per segment, with priority over CNN and HSV.

**How It Works**:
- Up to 9 context frames are selected per CountSegment: first three (start, start+5, start+10), middle three (mid-5, mid, mid+5), and last three (end-10, end-5, end), with a frame interval of 5
- Each frame is cropped to a large outer region: centered on the ROI core center, the search area is (Core × 2.8) × 1.5 rectangle; the sent image marks the ROI Core with a thin cyan ellipse without occluding ear tags
- All 9 frames are sent in a single request in chronological order, with front/middle/end context prompts; the model judges mouse count and ear tag colors based only on mice actually occupying or entering the Core; outer/corner wandering mice must be ignored
- Short segments with fewer frames are deduplicated and safely clamped to [start, end]

**Cost Control**:
- Each segment consumes only 1 `MOUSE_COLOR_AI_MAX_REQUESTS_PER_SEGMENT` budget
- Budget is deducted before the real API call; network/API/parse failures still consume the quota
- When budget is 0, AI vision is fully disabled (no network calls)

**Result Fusion**:
- When VLM returns legal `mouse_count` (1/2 only), a first known color, and confidence ≥ 0.3, it re-confirms the segment count and syncs `estimated_mouse_count`, `mouse_count` with the returned `target_count`
- VLM's ordered colors are used directly; a single mouse is not overwritten by rule-based candidates. For two mice, if the second color is `unknown` or duplicate, it is conservatively flagged for review — colors are never fabricated by rules
- VLM failure, unknown, low confidence, invalid JSON, or no valid crops → complete fallback to original CNN/HSV and local counting
- Segments already manually confirmed are never overwritten

**Environment Variable Configuration** (all required, no defaults; obtain correct endpoints and model names from provider documentation):

| Environment Variable | Description | Default |
|---|---|---|
| `MOUSE_COLOR_AI_PROVIDER` | Vision recognition provider: `kimi` / `minimax` / `openrouter` / `gpt` (or `hsv` to disable) | `hsv` (HSV rules only) |
| `MOUSE_COLOR_AI_TIMEOUT` | API request timeout in seconds | `30` |
| `MOUSE_COLOR_AI_MAX_TOKENS` | AI vision output token limit (must be positive integer; recommended ≥512 for reasoning models) | `512` |
| `MOUSE_COLOR_AI_MAX_REQUESTS_PER_SEGMENT` | Max cloud vision requests per segment (`0`=disable cloud vision) | `3` |
| `MOUSE_COLOR_BATCH_CONCURRENCY` | Max concurrent batch color recognition tasks (1=serial, 2=default, max 4; invalid values fall back to 2) | `2` |
| `MOUSE_COLOR_AI_SAVE_DEBUG` | `1`=save each VLM request's context crop / response / manifest to `debug_vision_output/<clip_id>/` | Off |

### Kimi (Moonshot) Required Environment Variables:

| Environment Variable | Description |
|---|---|
| `KIMI_API_KEY` | API key |
| `KIMI_API_BASE` | API endpoint URL |
| `KIMI_VISION_MODEL` | Vision model name |

### MiniMax Required Environment Variables:

| Environment Variable | Description |
|---|---|
| `MINIMAX_API_KEY` | API key |
| `MINIMAX_API_BASE` | API endpoint URL |
| `MINIMAX_VISION_MODEL` | Vision model name |

### OpenRouter Required Environment Variables:

| Environment Variable | Description |
|---|---|
| `OPENROUTER_API_KEY` | API key |
| `OPENROUTER_API_BASE` | API endpoint URL |
| `OPENROUTER_VISION_MODEL` | Vision model name |

**Configuration Method**:

These environment variables must be exported to the process environment or injected by the user's existing launcher. The project does **not** auto-load `.env` files.

If the user's launcher supports `.env` (e.g., `python-dotenv`, shell script `source .env`, IDE run configs, etc.), write the following variables into `.env` as a reference:

```bash
# Using Kimi vision model (example; replace with actual values)
MOUSE_COLOR_AI_PROVIDER=kimi
KIMI_API_KEY=<your-api-key>
KIMI_API_BASE=<provider-base-url>
KIMI_VISION_MODEL=<vision-capable-model-id>

# Or using MiniMax vision model
# MOUSE_COLOR_AI_PROVIDER=minimax
# MINIMAX_API_KEY=<your-api-key>
# MINIMAX_API_BASE=<provider-base-url>
# MINIMAX_VISION_MODEL=<vision-capable-model-id>

# Max 3 cloud requests per segment (beyond that, CNN/HSV only)
MOUSE_COLOR_AI_MAX_REQUESTS_PER_SEGMENT=3
```

For direct launch, use environment variable injection:
```bash
KIMI_API_KEY=<your-api-key> KIMI_API_BASE=<provider-base-url> KIMI_VISION_MODEL=<vision-capable-model-id> python main.py
```

**Response JSON Contract** (model must return the following structure):

```json
{
  "mouse_count": 1,
  "colors": ["red"],
  "confidence": 0.85,
  "thermometer_present": false
}
```

- `mouse_count` must be integer `1` or `2`; `colors` length must match, each item must be one of `red/yellow/blue/green/white/unknown`
- `confidence` must be 0.0–1.0; below 0.3 triggers automatic fallback. Only the final `content` JSON is parsed — reasoning text is never inferred from
- `thermometer_present` must be a JSON boolean; missing it is treated as `false` for compatibility; if present but not boolean, the entire VLM result is invalid and falls back
- When VLM is accepted and `thermometer_present=true`, the model's count/colors are not used to overwrite local results; `identity_confidence=0.0`, `identity_needs_review=true`, `identity_conflict=false`, and `identity_method` includes `thermometer_detected`; prompts manual review. Already confirmed events are still not overwritten
- Legacy final JSON `{ "color": "red", "confidence": 0.85 }` is compatible and treated as single mouse; new requests always use the new contract

**Fallback Behavior**:

| Scenario | Behavior |
|---|---|
| `MOUSE_COLOR_AI_PROVIDER=hsv` or unset | AI vision disabled; use CNN/HSV |
| `MOUSE_COLOR_AI_MAX_REQUESTS_PER_SEGMENT=0` | Cloud vision disabled; CNN/HSV only |
| API key missing | Warning at startup; fallback to CNN/HSV |
| API Base URL or model name missing | Report specific missing variables at startup (diagnosable error) |
| VLM context request failure (network/API/parse) | Consumes budget; fallback to CNN/HSV; not marked as VLM |
| Model returns invalid JSON | Parse failure; fallback to CNN/HSV |
| Output truncated (finish_reason=length, no parseable JSON) | Fallback to CNN/HSV; log suggests increasing `MOUSE_COLOR_AI_MAX_TOKENS`. Note: natural language in reasoning is not used as classification results |
| VLM returns unknown or confidence < 0.3 | Treated as failure; budget consumed; fallback to CNN/HSV |
| Segment has no valid context crop | Budget not consumed; directly use CNN/HSV |

**GUI Scope Selection & Concurrency**: After clicking "Color Recognition", choose "Current Event", "All Events", or "Cancel". When no event is selected, "Current Event" is unavailable. "All Events" skips zero-mouse segments and runs at **2** concurrent tasks by default (`MOUSE_COLOR_BATCH_CONCURRENCY=1` for serial; max **4**; invalid values fall back to 2). Each task has an independent VideoCapture/IdentityAssist and does not increase the per-segment VLM request budget. Progress shows real completed/total counts and current event ID; cancel stops unstarted tasks while running tasks complete safely and their results are preserved. Concurrency increases simultaneous provider load and cost peaks — keep the conservative default.

**Debug & Troubleshooting**:

Set `MOUSE_COLOR_AI_SAVE_DEBUG=1` to save each segment's VLM context request to `debug_vision_output/<clip_id>/` subdirectories:
- All context crops (filenames include ordering index and original video frame index)
- API response JSON (saved on both success and failure)
- Manifest JSON (provider/model without API key, clip_id, selected frame indices, crop algorithm/range, parse result or error info)

Each clip has its own subdirectory for per-segment troubleshooting.

**Provider Architecture**:
- `VisionProvider` (ABC): Abstract interface `classify(patch_bgr) → (color, confidence, method)` + `classify_frames(frames) → (color, confidence, method, raw_json)`
- `KimiVisionProvider` / `MinimaxVisionProvider` / `OpenRouterVisionProvider`: OpenAI-compatible implementations
- `VisionProviderFactory`: Factory creation, supports extending new providers via `register()`
- Located in `detection/models/vision_provider.py`

## Tech Stack

| Module | Technology |
|---|---|
| Language | Python 3.9+ |
| GUI | PySide6 |
| Video/Image | OpenCV + NumPy |
| AI Inference (optional) | ONNX Runtime (onnxruntime) / AI Vision Models (Kimi/Minimax/OpenRouter) |

## Core Design

```
No automatic mouse identity recognition.
No frame-by-frame full-video high-precision detection.

Sparse frame coarse screening → trigger backtracking → local frame-by-frame fine inspection → manual mouse annotation
```

Detection logic focuses on **whether the warm-spot core ROI is occluded by a dark mouse body**; no full-frame motion detection is performed.

---

## Project Structure

```
mouse_warm_spot/
├── main.py                      # Entry point + dark theme
├── requirements.txt             # PySide6 + OpenCV + NumPy
├── requirements-ai.txt          # Optional AI inference deps (onnxruntime + requests)
├── README.md
├── README_CN.md                 # Chinese README
├── 操作文档.md                   # Chinese operational guide
├── gui/
│   ├── main_window.py           # Main window: toolbar, menus, signal wiring, detection control
│   ├── video_widget.py          # Video display + ROI A/B/C overlay + loop playback
│   ├── zoom_widget.py           # Warm-spot ROI zoom window (2x)
│   ├── metrics_panel.py         # Occlusion metrics + mouse count + debug info
│   ├── event_list_widget.py     # Event list (CountSegment) + color status + context menu
│   ├── annotation_panel.py      # Annotation panel: mouse selection + count confirmation
│   └── calibration_store.py     # Calibration sample multi-frame manager
├── detection/
│   ├── metrics.py               # Single-frame occlusion detection metrics (occupancy)
│   ├── engine.py                # Full-video two-layer detection engine (OccupancyEpisode + CountSegment)
│   ├── counter.py               # Mouse count estimation engine (MouseCounter)
│   ├── identity_assist.py       # Ear tag color assist (HSV rules + optional ONNX CNN)
│   ├── color_mouse_mapping.py   # Persistent one-to-one color-to-mouse-id mapping
│   ├── batch_color.py           # Batch color recognition concurrency tool
│   └── models/                  # AI inference models
│       ├── __init__.py
│       ├── classifier.py        # EarTagClassifier: ONNX wrapper + HSV fallback
│       └── vision_provider.py   # AI vision provider (Kimi/MiniMax/OpenRouter)
├── export/
│   ├── __init__.py
│   └── csv_exporter.py          # CSV/Markdown export
└── tests/
    ├── __init__.py
    ├── test_color_classifier.py         # Color classifier unit tests
    ├── test_vision_provider.py          # AI vision provider unit tests
    ├── test_thermometer.py              # Thermometer detection unit tests
    ├── test_core_clip_splitting.py      # Core clip splitting regression tests
    └── test_core_connected_counting.py  # Core connected counting regression tests
```

---

## Workflow

### 1. Open Video

Menu `File → Open Video` (Ctrl+O) or click toolbar `Open Video`.

### 2. Draw Warm-Spot Core ROI

Click and drag on the video to outline the warm-spot disc area (ellipse).

Three ROIs are displayed on the video:
- **ROI A** (green solid line): Warm-spot core, used to determine occlusion
- **ROI B** (yellow dashed line): Buffer ROI (= ROI A × 1.8), used for coarse screening
- **ROI C** (blue dashed line): Counting ROI (= ROI A × 1.6), used for mouse count estimation

### 3. Mark Reference Frames (any order)

Calibration buttons are in the toolbar:

| Button | Purpose | Statistics Method |
|---|---|---|
| `[0]` Mark 0 | Empty background frame (at least one required) | Last frame used |
| `[1]` Mark 1 | Single-mouse area calibration | P80 across frames |
| `[2]` Mark 2 | Two-mouse reference area | Median across frames |
| `[3]` Mark 3 | Three-mouse reference area | Median across frames |
| `[4]` Mark 4 | Four-mouse reference area | Median across frames |

- **Left click** = append sample (unlimited)
- **Right click** = undo/clear menu
- Button shows current sample count, e.g. `[2] 2 mice x3/3`
- Order is not enforced: marking [1] before [0] works fine (previous frames are re-tested once [0] is available)
- Without a background, press `R` to use dark-only fallback mode (lower confidence)

### 4. Auto-Detect Full Video

Click `Auto-Detect Full Video` → runs two-layer detection in background:

- **Layer 1** — Coarse screening every N frames (ROI B) → fine local inspection (ROI A) → state machine → OccupancyEpisode
- **Layer 2** — Within each episode, estimate mouse count frame by frame → split by count changes → CountSegments

After detection completes, the event list populates automatically, awaiting manual review. Gap intervals that meet the threshold appear as gray "estimated count 0" rows; these rows are selectable and exported to CSV/Markdown like any other CountSegment.

### 5. Manual Review

| Action | Shortcut |
|---|---|
| Select mouse identity (multi-select toggle) | `1` `2` `3` `4` |
| Confirm segment mouse count | `Shift+0` `Shift+1` `Shift+2` |
| Refresh current frame count | `R` |
| Play/Pause | `Space` |
| Frame forward/back | `→` `←` |
| Fast forward/back 10 frames | `J` `L` |
| Previous/Next event | `↑` `↓` |
| Save & jump to next | `Enter` |
| Mark false detection | `X` |
| Split segment (at current frame) | `C` or `K` |

Annotation panel actions:
- `[v] Confirm` — Confirm mouse annotation for current segment
- `[x] False Detection` — Mark current segment as false detection
- `Save & Next` — Save and auto-jump to next pending segment

### 5.1 Color → Mouse ID Auto-Memory

When color recognition yields complete, non-conflicting known colors, the system automatically assigns the smallest available mouse ID (1–4). For example, green → 1 on first encounter, red → 2 next; subsequent same-color segments reuse the existing mapping. Different colors are not automatically assigned the same number. The mapping is saved in `~/.mouse_warm_spot/color_mouse_mapping.json` and contains no keys. Unknown, duplicate two-mouse colors, conflicts, thermometer interference, zero-mouse clips, and manually confirmed events are never auto-assigned/overwritten — they are conservatively flagged for review.

Use **View → Manage Color-Mouse Mapping...** to inspect the current mapping and "Reset All Mappings". After batch processing all events, a summary dialog shows total, successful, thermometer-interference, needs-review, failed/cancelled counts and the current mapping. The status bar shows the auto-assigned mouse ID for single events.

### 6. Export

Menu `File → Export CSV...` exports a simplified CSV (5 fields: segment_id | start_time_sec | end_time_sec | mouse_count | mouse_ids) based on manually marked mouse_ids. `File → Export Markdown Statistics...` generates a report including:
- Event detail table (start/end time, mouse IDs, count per segment)
- Per-mouse summary (total occupation duration, event count)
- Warm-spot summary (total occupation duration, multi-mouse event count)

---

## Detection Algorithm Summary

### Occupancy Detection (Layer 1)

1. Sample ROI B every 10 frames, calculate occlusion ratio
2. Occlusion ratio ≥ 20% → trigger fine scanning
3. Compute 5 metrics per frame for ROI A (warm color retention, dark pixel ratio, background difference, etc.)
4. State machine determines clip boundaries solely by **ROI Core (inner circle)** continuity: 4 consecutive frames ≥ 20% confirms entry; Core idle for `core_gap_tolerance_seconds` (default **0.3s**) ends/splits the clip. Idle is defined as `is_occupied=false` **or** composite `occlusion_area_ratio` below `core_empty_occupancy_threshold` (default **0.04**, strictly clamped to 0–1); the latter takes priority over potentially stale occupied flags. The threshold is deliberately below the 8% release threshold to avoid splitting on normal near-threshold noise.
5. Between two confirmed occupancy clips, any Core idle frames meeting the threshold produce an independent `estimated_mouse_count=0` CountSegment (`start_reason=core_empty_gap`); its debug fields include `core_empty_reason` and min/max/mean/threshold occupancy. No 0-clips are generated for video start/end background, and they are never merged across.
6. ROI Count (outer circle) is only used for coarse screening and count estimation; a mouse still active in the outer ring cannot sustain or merge a Core clip. To avoid overlapping coarse screening windows producing duplicates, post-processing only deduplicates **overlapping** events — it never merges two Core episodes across a time gap. Detection logs record the `core_gap` split rule.
7. Filter episodes shorter than 0.8s; for videos with different frame rates or noise characteristics, adjust `core_gap_tolerance_seconds` via the parameter dict passed to `DetectionEngine.detect` / `detect_with_counting`.

### Mouse Counting (Layer 2)

Conservative counting strategy. Core principle: **a single connected component is capped at 2 mice; 3 or 4 are never inferred**.

```
Foreground extraction (within ROI C):
  strong_dark (V < 55) | (dark_candidate (V < 95) & background diff > 25)
  → morphological denoising (open 3x3 + close 9x9)
  → connected component analysis
  → keep only components directly overlapping ROI Core (≥20px and ≥Core 2%) or touching Core within a clear 3px tolerance; outer ROI Count independent blobs are excluded
  → merge fragmented blobs (Union-Find nearest-neighbor clustering)
  → filter: area < 50px, aspect ratio > 5

Counting:
  count_by_blob = number of merged components touching ROI A (0–4)
  count_by_area = total area / single-mouse reference area → threshold mapping (< 1.7=1, 1.7–2.7=2, ...)

Composite decision:
  blob=0               → 0 mice, 0.9

Debug fields: `core_connected_blob_count`, `ignored_outer_blob_count`, `core_connected_area` show the number of Core-connected blobs used for counting, excluded outer independent blobs, and the area of Core-connected blobs only.
  blob=1, ratio<1.7    → 1 mouse, 0.85
  blob=1, 1.7≤r<2.7    → 2 mice, 0.35 (low confidence)
  blob=1, r≥2.7        → 2 mice, 0.25 (very low confidence)
  blob=1, reference area match → use match result (deviation <35%)
  blob≥2               → trust blob count, combined with area

Temporal stability: 1→2 requires 8 frames, 2→3 requires 10 frames, 1→3 requires 15 frames + additional evidence
```

---

## Keyboard Shortcuts Reference

| Key | Function |
|---|---|
| `1` `2` `3` `4` | Select/deselect mouse identity |
| `Shift+0` `Shift+1` `Shift+2` | Confirm segment mouse count (exclusive selection) |
| `R` | Refresh current frame count |
| `Space` | Play/Pause |
| `→` `←` | Frame forward/back |
| `J` `L` | Fast forward/back 10 frames |
| `↑` `↓` | Previous/Next event |
| `Enter` | Save & jump to next |
| `X` | Mark as false detection |
| `C` `K` | Split segment at current frame |
| `S` | Save current annotation |

---

## Menus

| Menu | Function |
|---|---|
| File → Open Video (Ctrl+O) | Open mp4/avi/mov etc. |
| File → Save/Load ROI | Save/Load warm-spot ROI coordinates (JSON) |
| File → Save/Load Background Frame | Save/Load empty background image (PNG) |
| File → Export Markdown Statistics... | Export annotation results |
| File → Exit (Ctrl+Q) | Exit application |
| View → Warm-spot Zoom Window | Show/Hide zoom window |
| View → Detection Metrics | Show/Hide metrics panel |
| View → Debug View | Show/Hide raw detection annotation overlay |
| View → Manage Color-Mouse Mapping... | View/Reset persistent color-to-mouse-id mapping |
