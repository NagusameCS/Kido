#!/usr/bin/env python3
"""
Kido – hand-gesture controller for Fusion 360
Run with:  python -m src.main
"""

from __future__ import annotations

import time

import cv2

from src import config as cfg
from src.hand_tracker import HandTracker
from src.gesture_recognizer import Gesture, GestureRecogniser
from src.fusion_controller import FusionController

# ── colour palette for the HUD ──────────────────────────────────────
_COLOURS = {
    Gesture.IDLE: (180, 180, 180),
    Gesture.ORBIT: (0, 255, 200),
    Gesture.ZOOM_IN: (0, 200, 255),
    Gesture.ZOOM_OUT: (255, 100, 100),
}

_LABELS = {
    Gesture.IDLE: "IDLE",
    Gesture.ORBIT: "ORBIT",
    Gesture.ZOOM_IN: "ZOOM IN",
    Gesture.ZOOM_OUT: "ZOOM OUT",
}


def main() -> None:
    tracker = HandTracker()
    recogniser = GestureRecogniser()
    controller = FusionController()

    tracker.start()
    print("[Kido] Tracking started – press 'q' in preview or Ctrl-C to quit.")

    last_seq = -1  # track frame sequence to avoid re-processing

    try:
        while True:
            hand, frame, seq = tracker.latest()

            # Only process when a new frame is available
            if seq != last_seq:
                last_seq = seq
                gesture, payload = recogniser.update(hand)
                controller.act(gesture, payload)
            else:
                gesture = recogniser._current_gesture  # stale, for HUD only

            # ── optional preview window ──────────────────────────────
            if cfg.SHOW_PREVIEW and frame is not None:
                _draw_hud(frame, gesture, hand is not None)
                h, w = frame.shape[:2]
                small = cv2.resize(
                    frame,
                    (int(w * cfg.PREVIEW_SCALE), int(h * cfg.PREVIEW_SCALE)),
                )
                cv2.imshow("Kido", small)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            else:
                # Without preview, sleep briefly to yield CPU
                time.sleep(0.005)

    except KeyboardInterrupt:
        pass
    finally:
        controller.release_all()
        tracker.stop()
        cv2.destroyAllWindows()
        print("\n[Kido] Stopped.")


# ── HUD overlay ─────────────────────────────────────────────────────
def _draw_hud(frame, gesture: Gesture, hand_detected: bool) -> None:
    colour = _COLOURS.get(gesture, (200, 200, 200))
    label = _LABELS.get(gesture, "?")

    # Status bar
    cv2.rectangle(frame, (0, 0), (220, 50), (30, 30, 30), -1)
    cv2.putText(
        frame,
        f"Gesture: {label}",
        (10, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        colour,
        2,
    )

    # Hand indicator
    dot_colour = (0, 255, 0) if hand_detected else (0, 0, 255)
    cv2.circle(frame, (frame.shape[1] - 20, 20), 8, dot_colour, -1)


if __name__ == "__main__":
    main()
