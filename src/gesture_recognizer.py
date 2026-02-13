"""
Kido – Gesture Recogniser
Classifies raw hand landmarks into actionable gestures using a
lightweight state machine.  No ML model needed – pure geometry.

Gestures
--------
IDLE        – no actionable pose detected
ORBIT       – mostly-open hand moving; orbit the camera
ZOOM_IN     – fist opening to palm (transition triggers zoom-in)
ZOOM_OUT    – palm closing to fist (transition triggers zoom-out)
"""

from __future__ import annotations

import math
import time
from collections import deque
from enum import Enum, auto
from typing import Optional, Tuple

from . import config as cfg
from .hand_tracker import HandData, Landmark


class Gesture(Enum):
    IDLE = auto()
    ORBIT = auto()
    ZOOM_IN = auto()
    ZOOM_OUT = auto()


class GestureRecogniser:
    """
    Stateful recogniser that consumes ``HandData`` snapshots and emits
    a ``(Gesture, payload)`` pair each frame.

    *payload* depends on the gesture:
    - ORBIT  → (dx, dy) normalised displacement since last frame
    - ZOOM_* → None (the controller just fires a scroll tick)
    - IDLE   → None
    """

    def __init__(self) -> None:
        # ── EMA-smoothed position ────────────────────────────────────
        self._smooth_x: Optional[float] = None
        self._smooth_y: Optional[float] = None
        self._smooth_z: Optional[float] = None

        # Previous smoothed position (for delta / orbit)
        self._prev_x: Optional[float] = None
        self._prev_y: Optional[float] = None

        # ── Openness history for zoom detection ──────────────────────
        self._openness_history: deque = deque(maxlen=10)
        self._last_openness: Optional[float] = None
        self._last_openness_time: float = 0.0

        # ── State machine ────────────────────────────────────────────
        self._gesture_streak: int = 0
        self._current_gesture: Gesture = Gesture.IDLE
        self._candidate: Gesture = Gesture.IDLE

        self._last_zoom_time: float = 0.0

        # Cooldown: suppress orbit briefly after a zoom ends
        self._zoom_end_time: float = 0.0
        self._ORBIT_AFTER_ZOOM_COOLDOWN: float = 0.3  # seconds

    # ── public ───────────────────────────────────────────────────────
    def update(self, hand: Optional[HandData]) -> Tuple[Gesture, Optional[Tuple[float, float]]]:
        """
        Feed a new hand snapshot (or ``None`` if no hand detected).

        Returns ``(gesture, payload)``.
        """
        if hand is None:
            self._reset_smooth()
            return self._set_gesture(Gesture.IDLE), None

        # 1) Compute hand openness (0 = fist, 1 = fully open)
        openness = self._hand_openness(hand)

        # 2) Smooth the fingertip centre
        cx, cy, cz = hand.fingertip_center()
        sx, sy, sz = self._ema(cx, cy, cz)

        # 3) Decide gesture
        gesture, payload = self._classify(hand, openness, sx, sy)

        # 4) Store previous for next frame delta
        self._prev_x = sx
        self._prev_y = sy

        # 5) Track openness over time for zoom speed
        now = time.perf_counter()
        self._openness_history.append((now, openness))
        self._last_openness = openness
        self._last_openness_time = now

        return gesture, payload

    # ── internals ────────────────────────────────────────────────────
    def _classify(
        self,
        hand: HandData,
        openness: float,
        sx: float,
        sy: float,
    ) -> Tuple[Gesture, Optional[Tuple[float, float]]]:
        now = time.perf_counter()

        # ── Zoom detection (transition-based) ────────────────────────
        speed = self._openness_speed()

        if speed is not None:
            if speed > cfg.ZOOM_SPEED_THRESHOLD:
                # Opening fast → zoom in
                return self._set_gesture(Gesture.ZOOM_IN), None
            elif speed < -cfg.ZOOM_SPEED_THRESHOLD:
                # Closing fast → zoom out
                return self._set_gesture(Gesture.ZOOM_OUT), None

        # If hand is in a sustained zoom pose (open > 0.7 from a recent zoom-in,
        # or closed < 0.3 from a recent zoom-out), keep zooming
        if self._current_gesture == Gesture.ZOOM_IN and openness > 0.7:
            return self._set_gesture(Gesture.ZOOM_IN), None
        if self._current_gesture == Gesture.ZOOM_OUT and openness < 0.3:
            return self._set_gesture(Gesture.ZOOM_OUT), None

        # ── Orbit detection ──────────────────────────────────────────
        # Suppress orbit briefly after zoom ends to prevent accidental rotation
        if (self._current_gesture in (Gesture.ZOOM_IN, Gesture.ZOOM_OUT)
                and speed is not None and abs(speed) < cfg.ZOOM_SPEED_THRESHOLD):
            self._zoom_end_time = now

        if now - self._zoom_end_time < self._ORBIT_AFTER_ZOOM_COOLDOWN:
            return self._set_gesture(Gesture.IDLE), None

        # Open hand (openness > 0.55) moving → orbit
        if openness > 0.55 and self._prev_x is not None:
            dx = sx - self._prev_x
            dy = sy - self._prev_y
            dist = math.hypot(dx, dy)

            if dist > cfg.ORBIT_DEAD_ZONE:
                return self._set_gesture(Gesture.ORBIT), (dx, dy)

        # ── Nothing actionable ───────────────────────────────────────
        return self._set_gesture(Gesture.IDLE), None

    def _set_gesture(self, g: Gesture) -> Gesture:
        """Apply confidence-frame hysteresis before switching gesture."""
        if g == self._candidate:
            self._gesture_streak += 1
        else:
            self._candidate = g
            self._gesture_streak = 1

        if self._gesture_streak >= cfg.GESTURE_CONFIDENCE_FRAMES:
            self._current_gesture = g

        return self._current_gesture

    # ── hand openness ────────────────────────────────────────────────
    @staticmethod
    def _hand_openness(hand: HandData) -> float:
        """
        Return 0.0 (tight fist) → 1.0 (fully open palm).

        Measures average ratio of tip-to-wrist distance vs
        MCP-to-wrist distance for all five fingers.
        """
        wrist = hand.landmarks[0]
        tip_ids = (4, 8, 12, 16, 20)
        mcp_ids = (2, 5, 9, 13, 17)   # proximal joints

        ratios = []
        for tip_i, mcp_i in zip(tip_ids, mcp_ids):
            tip = hand.landmarks[tip_i]
            mcp = hand.landmarks[mcp_i]

            d_tip = _dist3(tip, wrist)
            d_mcp = _dist3(mcp, wrist)

            if d_mcp < 1e-6:
                continue
            ratios.append(d_tip / d_mcp)

        if not ratios:
            return 0.0

        avg = sum(ratios) / len(ratios)
        # Map ratio ~0.6 (fist) .. ~1.6 (open) → 0..1
        return max(0.0, min(1.0, (avg - 0.6) / 1.0))

    # ── openness speed ───────────────────────────────────────────────
    def _openness_speed(self) -> Optional[float]:
        """
        Rate of change of openness (units/sec).
        Positive = opening, Negative = closing.
        """
        if len(self._openness_history) < 4:
            return None

        t0, o0 = self._openness_history[0]
        t1, o1 = self._openness_history[-1]
        dt = t1 - t0
        if dt < 0.05:
            return None
        return (o1 - o0) / dt

    # ── EMA smoothing ────────────────────────────────────────────────
    def _ema(self, x: float, y: float, z: float) -> Tuple[float, float, float]:
        a = cfg.EMA_ALPHA
        if self._smooth_x is None:
            self._smooth_x, self._smooth_y, self._smooth_z = x, y, z
        else:
            self._smooth_x = a * x + (1 - a) * self._smooth_x
            self._smooth_y = a * y + (1 - a) * self._smooth_y
            self._smooth_z = a * z + (1 - a) * self._smooth_z
        return self._smooth_x, self._smooth_y, self._smooth_z

    def _reset_smooth(self) -> None:
        self._smooth_x = self._smooth_y = self._smooth_z = None
        self._prev_x = self._prev_y = None
        self._openness_history.clear()


# ── helpers ──────────────────────────────────────────────────────────
def _dist3(a: Landmark, b: Landmark) -> float:
    return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2)
