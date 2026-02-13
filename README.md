# Kido

**Hand-gesture controller for Autodesk Fusion 360**
Navigate your 3-D viewport with nothing but a webcam and your hand.

---

## Features

| Gesture | Action | How to perform |
|---------|--------|----------------|
| **Orbit** | Rotate the camera in the direction your hand moves (X / Y) | Hold an open hand in view and move it around |
| **Zoom In** | Continuously zoom into the model | Transition from a closed fist → open palm |
| **Zoom Out** | Continuously zoom out of the model | Transition from an open palm → closed fist |

- ~30 FPS processing on a modern laptop CPU (no GPU required)
- Lightweight MediaPipe Hands model (< 5 MB)
- Threaded camera capture – main loop never blocks on I/O
- Exponential-moving-average smoothing to reduce jitter
- Gesture hysteresis (confidence frames) to prevent false triggers

## Requirements

- Python 3.10+
- A webcam
- Autodesk Fusion 360 running in the foreground

## Quick start

```bash
# 1. Clone the repo
git clone https://github.com/NagusameCS/Kido.git
cd Kido

# 2. Create a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate      # Linux / macOS
# .venv\Scripts\activate       # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run
python -m src.main
```

A preview window will open showing your hand landmarks and the
current gesture.  Press **q** in the preview or **Ctrl-C** in the
terminal to quit.

## Configuration

All tuneable parameters live in [`src/config.py`](src/config.py):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `CAMERA_INDEX` | `0` | Webcam device index |
| `CAPTURE_WIDTH / HEIGHT` | `640 × 480` | Resolution fed to MediaPipe |
| `TARGET_FPS` | `30` | Processing frame-rate cap |
| `ORBIT_SENSITIVITY_X/Y` | `2.5` | Hand movement → orbit pixel multiplier |
| `ORBIT_DEAD_ZONE` | `0.015` | Minimum movement to register orbit |
| `ZOOM_IN_SCROLL / OUT` | `3 / -3` | Scroll ticks per frame during zoom |
| `ZOOM_SPEED_THRESHOLD` | `0.8` | Openness change-rate to trigger zoom |
| `EMA_ALPHA` | `0.45` | Smoothing factor (higher = more responsive) |
| `SHOW_PREVIEW` | `True` | Show / hide the debug webcam window |

## Architecture

```
┌─────────────┐   thread    ┌──────────────┐
│  Webcam I/O │ ──────────▸ │ HandTracker  │
└─────────────┘             │ (MediaPipe)  │
                            └──────┬───────┘
                                   │ HandData
                            ┌──────▼────────┐
                            │ GestureRecog. │  state machine
                            └──────┬────────┘
                                   │ Gesture + payload
                            ┌──────▼────────────┐
                            │ FusionController   │
                            │ (pynput → OS)      │
                            └───────────────────┘
```

## License

See [LICENSE](LICENSE).