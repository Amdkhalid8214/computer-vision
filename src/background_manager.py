"""Background acquisition and plate management for RealityFrame Ghost Mode.

Samples webcam frames over a configurable countdown period, calculates a noise-free
median background plate, supports live recapture without restart, and provides
informative status indicators.
"""

import logging
import time
from typing import List, Optional, Tuple
import cv2
import numpy as np

from config import BackgroundStatus

logger = logging.getLogger(__name__)


class BackgroundManager:
    """Coordinates background plate acquisition, median computation, and state tracking."""

    def __init__(
        self,
        capture_duration_sec: float = 2.5,
        target_sample_frames: int = 25,
    ) -> None:
        """Initialize the background manager.

        Args:
            capture_duration_sec: Duration in seconds to allow user to step out and sample frames.
            target_sample_frames: Number of frames to sample for median calculation.
        """
        self.capture_duration_sec = max(1.0, capture_duration_sec)
        self.target_sample_frames = max(5, target_sample_frames)

        self._status: BackgroundStatus = BackgroundStatus.NOT_CAPTURED
        self._background_plate: Optional[np.ndarray] = None
        self._previous_valid_plate: Optional[np.ndarray] = None

        self._frame_buffer: List[np.ndarray] = []
        self._capture_start_time: float = 0.0
        self._countdown_remaining: float = 0.0

    @property
    def status(self) -> BackgroundStatus:
        """Current lifecycle status of the background plate."""
        return self._status

    @property
    def is_ready(self) -> bool:
        """Return True if a valid background plate is stored and ready for Ghost Mode."""
        return self._status == BackgroundStatus.READY and self._background_plate is not None

    @property
    def background_plate(self) -> Optional[np.ndarray]:
        """The currently active confirmed background plate."""
        return self._background_plate

    @property
    def countdown_seconds(self) -> float:
        """Remaining countdown time in seconds during background acquisition."""
        return max(0.0, self._countdown_remaining)

    def trigger_capture(self) -> None:
        """Initiate background plate capture or recapture."""
        logger.info("Triggering background acquisition sequence...")
        if self._background_plate is not None:
            self._previous_valid_plate = self._background_plate.copy()
            self._status = BackgroundStatus.RECAPTURING
        else:
            self._status = BackgroundStatus.COUNTDOWN

        self._frame_buffer = []
        self._capture_start_time = time.perf_counter()
        self._countdown_remaining = self.capture_duration_sec

    def ingest_frame(self, frame: np.ndarray, dt: float = 0.033) -> None:
        """Ingest a live frame during capture sequence and update countdown/progress.

        Args:
            frame: 3-channel BGR image array from the webcam.
            dt: Time elapsed since last frame.
        """
        if self._status not in (BackgroundStatus.COUNTDOWN, BackgroundStatus.CAPTURING, BackgroundStatus.RECAPTURING):
            return

        if frame is None or frame.size == 0 or frame.ndim != 3:
            logger.warning("Invalid frame encountered during background acquisition.")
            return

        now = time.perf_counter()
        elapsed = now - self._capture_start_time
        self._countdown_remaining = max(0.0, self.capture_duration_sec - elapsed)

        # In the final 60% of the duration, accumulate frames for median plate
        if elapsed >= (self.capture_duration_sec * 0.35):
            self._status = BackgroundStatus.CAPTURING
            if len(self._frame_buffer) < self.target_sample_frames:
                self._frame_buffer.append(frame.copy())

        # If countdown expired or buffer full, finalize plate
        if elapsed >= self.capture_duration_sec or len(self._frame_buffer) >= self.target_sample_frames:
            self._finalize_background()

    def _finalize_background(self) -> None:
        """Compute the median image plate from accumulated frames."""
        if not self._frame_buffer:
            logger.error("Background acquisition failed: zero valid frames collected.")
            if self._previous_valid_plate is not None:
                logger.info("Restoring previous valid background plate.")
                self._background_plate = self._previous_valid_plate
                self._status = BackgroundStatus.READY
            else:
                self._status = BackgroundStatus.FAILED
            return

        try:
            logger.info("Computing median background plate from %d frames...", len(self._frame_buffer))
            # Stack along new axis 0: shape (N, H, W, 3)
            stacked = np.stack(self._frame_buffer, axis=0)
            median_plate = np.median(stacked, axis=0).astype(np.uint8)

            self._background_plate = median_plate
            self._status = BackgroundStatus.READY
            self._frame_buffer.clear()
            logger.info("RealityFrame background plate generated successfully. Dimensions: %s", median_plate.shape)

        except Exception as e:
            logger.exception("Failed to compute median background plate: %s", e)
            if self._previous_valid_plate is not None:
                self._background_plate = self._previous_valid_plate
                self._status = BackgroundStatus.READY
            else:
                self._status = BackgroundStatus.FAILED

    def render_overlay(self, frame: np.ndarray) -> np.ndarray:
        """Render high-visibility cyberpunk countdown graphics during background capture."""
        if self._status not in (BackgroundStatus.COUNTDOWN, BackgroundStatus.CAPTURING, BackgroundStatus.RECAPTURING):
            return frame

        h, w = frame.shape[:2]
        center_x, center_y = w // 2, h // 2

        # Semi-transparent backdrop overlay
        overlay = np.zeros_like(frame)
        cv2.rectangle(overlay, (w // 4, h // 3), (3 * w // 4, 2 * h // 3), (15, 15, 20), -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
        cv2.rectangle(frame, (w // 4, h // 3), (3 * w // 4, 2 * h // 3), (0, 215, 255), 2, cv2.LINE_AA)

        # Title
        title_text = "ACQUIRING REALITYFRAME BACKGROUND"
        (tw, th), _ = cv2.getTextSize(title_text, cv2.FONT_HERSHEY_DUPLEX, 0.65, 1)
        cv2.putText(
            frame,
            title_text,
            (center_x - tw // 2, center_y - 35),
            cv2.FONT_HERSHEY_DUPLEX,
            0.65,
            (0, 215, 255),
            1,
            cv2.LINE_AA,
        )

        # Instruction
        sub_text = "PLEASE STEP CLEAR OF THE CAMERA FEED"
        (stw, _), _ = cv2.getTextSize(sub_text, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
        cv2.putText(
            frame,
            sub_text,
            (center_x - stw // 2, center_y - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        # Countdown Timer Display
        cd_str = f"CAPTURING IN: {self.countdown_seconds:3.1f}s  (SAMPLES: {len(self._frame_buffer)}/{self.target_sample_frames})"
        (cw, _), _ = cv2.getTextSize(cd_str, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1)
        cv2.putText(
            frame,
            cd_str,
            (center_x - cw // 2, center_y + 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (0, 255, 128),
            1,
            cv2.LINE_AA,
        )

        return frame

