"""
Kido – Hand Tracker
Threaded webcam capture + MediaPipe Hands inference.
Keeps the main loop free of I/O blocking.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np

from . import config as cfg


@dataclass(slots=True)
class Landmark:
    """Single 3-D landmark in *normalised* image coordinates."""
    x: float
    y: float
    z: float


@dataclass(slots=True)
class HandData:
    """Snapshot of one detected hand."""
    landmarks: List[Landmark]
    handedness: str  # "Left" or "Right"
    timestamp: float = 0.0

    # ── convenience accessors ────────────────────────────────────────
    @property
    def wrist(self) -> Landmark:
        return self.landmarks[0]

    @property
    def index_tip(self) -> Landmark:
        return self.landmarks[8]

    @property
    def middle_tip(self) -> Landmark:
        return self.landmarks[12]

    @property
    def ring_tip(self) -> Landmark:
        return self.landmarks[16]

    @property
    def pinky_tip(self) -> Landmark:
        return self.landmarks[20]

    @property
    def thumb_tip(self) -> Landmark:
        return self.landmarks[4]

    def fingertip_center(self) -> Tuple[float, float, float]:
        """Average position of all five fingertips (normalised)."""
        tips = [self.landmarks[i] for i in (4, 8, 12, 16, 20)]
        cx = sum(t.x for t in tips) / 5
        cy = sum(t.y for t in tips) / 5
        cz = sum(t.z for t in tips) / 5
        return cx, cy, cz

    def palm_center(self) -> Tuple[float, float, float]:
        """Average of wrist + MCP joints (0, 5, 9, 13, 17)."""
        ids = (0, 5, 9, 13, 17)
        pts = [self.landmarks[i] for i in ids]
        cx = sum(p.x for p in pts) / len(pts)
        cy = sum(p.y for p in pts) / len(pts)
        cz = sum(p.z for p in pts) / len(pts)
        return cx, cy, cz


class HandTracker:
    """
    Runs webcam capture in a background thread and exposes the latest
    hand data via :pymethod:`latest()`.
    """

    def __init__(self) -> None:
        self._mp_hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=cfg.MP_MAX_HANDS,
            model_complexity=cfg.MP_MODEL_COMPLEXITY,
            min_detection_confidence=cfg.MP_DETECTION_CONFIDENCE,
            min_tracking_confidence=cfg.MP_TRACKING_CONFIDENCE,
        )

        self._cap: Optional[cv2.VideoCapture] = None
        self._lock = threading.Lock()
        self._hand: Optional[HandData] = None
        self._frame: Optional[np.ndarray] = None
        self._frame_seq: int = 0          # bumped each new frame
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._frame_interval = 1.0 / cfg.TARGET_FPS

    # ── public API ───────────────────────────────────────────────────
    def start(self) -> None:
        """Open the webcam and begin the capture thread."""
        self._cap = cv2.VideoCapture(cfg.CAMERA_INDEX)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.CAPTURE_WIDTH)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.CAPTURE_HEIGHT)
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # drop stale frames

        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open camera {cfg.CAMERA_INDEX}")

        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2)
        if self._cap is not None:
            self._cap.release()
        self._mp_hands.close()

    def latest(self) -> Tuple[Optional[HandData], Optional[np.ndarray], int]:
        """Return the most recent (hand, frame, seq) snapshot."""
        with self._lock:
            return self._hand, self._frame, self._frame_seq

    # ── capture loop (runs in background thread) ─────────────────────
    def _loop(self) -> None:
        while self._running:
            t0 = time.perf_counter()

            ok, frame = self._cap.read()
            if not ok:
                continue

            # Flip horizontally for natural mirror view
            frame = cv2.flip(frame, 1)

            # MediaPipe expects RGB
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self._mp_hands.process(rgb)

            hand_data: Optional[HandData] = None

            if results.multi_hand_landmarks:
                hl = results.multi_hand_landmarks[0]
                handedness = "Right"
                if results.multi_handedness:
                    handedness = results.multi_handedness[0].classification[0].label

                landmarks = [
                    Landmark(lm.x, lm.y, lm.z)
                    for lm in hl.landmark
                ]
                hand_data = HandData(
                    landmarks=landmarks,
                    handedness=handedness,
                    timestamp=time.perf_counter(),
                )

                # Draw landmarks on frame for preview
                if cfg.SHOW_PREVIEW:
                    mp.solutions.drawing_utils.draw_landmarks(
                        frame,
                        hl,
                        mp.solutions.hands.HAND_CONNECTIONS,
                    )

            with self._lock:
                self._hand = hand_data
                self._frame = frame
                self._frame_seq += 1

            # Rate-limit to TARGET_FPS
            elapsed = time.perf_counter() - t0
            sleep_time = self._frame_interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
