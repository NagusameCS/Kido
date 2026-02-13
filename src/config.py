"""
Kido – Configuration
All tuneable knobs live here so nothing is scattered across modules.
"""

# ── Camera ───────────────────────────────────────────────────────────
CAMERA_INDEX = 0                # webcam device index
CAPTURE_WIDTH = 640             # resolution sent to MediaPipe (lower = faster)
CAPTURE_HEIGHT = 480
TARGET_FPS = 30                 # cap processing rate to save CPU

# ── MediaPipe Hands ──────────────────────────────────────────────────
MP_MAX_HANDS = 1                # track only the dominant hand
MP_DETECTION_CONFIDENCE = 0.7   # initial detection threshold
MP_TRACKING_CONFIDENCE = 0.6    # per-frame tracking threshold
MP_MODEL_COMPLEXITY = 0         # 0 = lite (fastest), 1 = full

# ── Gesture thresholds ───────────────────────────────────────────────
# Number of consecutive frames a gesture must be seen before acting.
GESTURE_CONFIDENCE_FRAMES = 3

# ── Rotation (orbit) ────────────────────────────────────────────────
# Sensitivity multiplier: hand-pixel-delta → Fusion 360 orbit pixels.
ORBIT_SENSITIVITY_X = 2.5
ORBIT_SENSITIVITY_Y = 2.5

# Minimum hand displacement (in normalised coords 0-1) before we
# register a rotation intent.  Prevents jitter while hand is still.
ORBIT_DEAD_ZONE = 0.015

# ── Zoom ─────────────────────────────────────────────────────────────
# Scroll ticks sent per frame while zoom gesture is active.
ZOOM_IN_SCROLL = 3              # positive = scroll up  = zoom in
ZOOM_OUT_SCROLL = -3            # negative = scroll down = zoom out

# Cooldown (seconds) between zoom scroll events to avoid flooding.
ZOOM_SCROLL_INTERVAL = 0.05

# Speed at which the fist must open/close (change in open-ratio per
# second) to trigger zoom.  Prevents accidental triggers.
ZOOM_SPEED_THRESHOLD = 0.8

# ── Smoothing ────────────────────────────────────────────────────────
# Exponential moving average factor for hand position (0-1).
# Higher = more responsive but noisier.
EMA_ALPHA = 0.45

# ── Debug / UI ───────────────────────────────────────────────────────
SHOW_PREVIEW = True             # show annotated webcam window
PREVIEW_SCALE = 0.6             # resize preview to save screen space
