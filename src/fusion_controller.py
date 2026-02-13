"""
Kido – Fusion 360 Controller
Translates recognised gestures into OS-level input events that
Fusion 360 interprets as native navigation.

Fusion 360 navigation mapping
------------------------------
* **Orbit**   – Middle-mouse-button drag  (hold MMB + move mouse)
* **Zoom In** – Scroll wheel up
* **Zoom Out**– Scroll wheel down
* **Pan**     – Shift + Middle-mouse-button drag  (future)

We use ``pynput`` so the events land at the OS level and any
foreground app (Fusion 360) receives them.
"""

from __future__ import annotations

import time
from typing import Optional, Tuple

from pynput.mouse import Button, Controller as MouseController

from . import config as cfg
from .gesture_recognizer import Gesture


class FusionController:
    """Send mouse events that drive Fusion 360's viewport."""

    def __init__(self) -> None:
        self._mouse = MouseController()
        self._orbiting = False
        self._last_zoom_time: float = 0.0

    # ── public ───────────────────────────────────────────────────────
    def act(
        self,
        gesture: Gesture,
        payload: Optional[Tuple[float, float]],
    ) -> None:
        if gesture == Gesture.ORBIT:
            self._do_orbit(payload)
        elif gesture == Gesture.ZOOM_IN:
            self._end_orbit()          # release MMB before scrolling
            self._do_zoom(cfg.ZOOM_IN_SCROLL)
        elif gesture == Gesture.ZOOM_OUT:
            self._end_orbit()          # release MMB before scrolling
            self._do_zoom(cfg.ZOOM_OUT_SCROLL)
        else:
            self._end_orbit()

    def release_all(self) -> None:
        """Make sure nothing is stuck on shutdown."""
        self._end_orbit()

    # ── orbit ────────────────────────────────────────────────────────
    def _do_orbit(self, delta: Optional[Tuple[float, float]]) -> None:
        if delta is None:
            return

        dx, dy = delta

        # Convert normalised delta → pixel displacement
        px = int(dx * cfg.CAPTURE_WIDTH * cfg.ORBIT_SENSITIVITY_X)
        py = int(dy * cfg.CAPTURE_HEIGHT * cfg.ORBIT_SENSITIVITY_Y)

        if not self._orbiting:
            # Press middle mouse to start orbit
            self._mouse.press(Button.middle)
            self._orbiting = True

        # Move mouse relative
        self._mouse.move(px, py)

    def _end_orbit(self) -> None:
        if self._orbiting:
            self._mouse.release(Button.middle)
            self._orbiting = False

    # ── zoom ─────────────────────────────────────────────────────────
    def _do_zoom(self, ticks: int) -> None:
        now = time.perf_counter()
        if now - self._last_zoom_time < cfg.ZOOM_SCROLL_INTERVAL:
            return
        self._mouse.scroll(0, ticks)
        self._last_zoom_time = now
