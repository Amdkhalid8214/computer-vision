"""Dynamic Region of Interest (ROI) geometry and smoothing controller.

Calculates bounding boxes from two hand control points, validates dimensions,
clamps to image boundaries, and applies low-latency exponential smoothing
with tracking discontinuity detection.
"""

from dataclasses import dataclass
import math
from typing import Optional, Tuple
import numpy as np


@dataclass(frozen=True)
class ROIResult:
    """Encapsulates the calculated ROI state for downstream rendering and filtering."""
    left: int
    top: int
    right: int
    bottom: int
    width: int
    height: int
    is_valid: bool
    status_message: str
    control_points: Tuple[Tuple[int, int], Tuple[int, int]]


class ROIController:
    """Computes, smooths, and validates the filter region from two control points."""

    def __init__(
        self,
        min_width: int = 30,
        min_height: int = 30,
        smoothing_enabled: bool = True,
        smoothing_factor: float = 0.35,
        discontinuity_threshold_pixels: float = 120.0,
    ) -> None:
        """Initialize the ROI controller.

        Args:
            min_width: Minimum allowable ROI width in pixels.
            min_height: Minimum allowable ROI height in pixels.
            smoothing_enabled: Toggle for EMA coordinate smoothing.
            smoothing_factor: Alpha in (0, 1]. Higher = faster response, lower = smoother.
            discontinuity_threshold_pixels: Maximum jump distance before resetting smoothing.
        """
        self.min_width = max(1, min_width)
        self.min_height = max(1, min_height)
        self.smoothing_enabled = smoothing_enabled
        self.smoothing_factor = max(0.01, min(1.0, smoothing_factor))
        self.discontinuity_threshold = discontinuity_threshold_pixels

        # Previous smoothed control points (stored as floats for accumulator precision)
        self._prev_pt1: Optional[Tuple[float, float]] = None
        self._prev_pt2: Optional[Tuple[float, float]] = None

    def reset(self) -> None:
        """Reset the internal smoothing state, e.g. after tracking loss or user request."""
        self._prev_pt1 = None
        self._prev_pt2 = None

    def update(
        self,
        pt1: Optional[Tuple[int, int]],
        pt2: Optional[Tuple[int, int]],
        frame_shape: Tuple[int, int],  # (height, width)
    ) -> ROIResult:
        """Calculate and validate the ROI from two control points.

        Args:
            pt1: (x, y) coordinates of the first hand control point.
            pt2: (x, y) coordinates of the second hand control point.
            frame_shape: (height, width) of the target image frame.

        Returns:
            ROIResult containing geometry, dimensions, validity flag, and status.
        """
        # Validate frame dimensions
        if not frame_shape or len(frame_shape) < 2:
            self.reset()
            return ROIResult(
                left=0, top=0, right=0, bottom=0, width=0, height=0,
                is_valid=False, status_message="INVALID FRAME DIMENSIONS",
                control_points=((0, 0), (0, 0)),
            )

        frame_h, frame_w = frame_shape[:2]
        if frame_h <= 0 or frame_w <= 0:
            self.reset()
            return ROIResult(
                left=0, top=0, right=0, bottom=0, width=0, height=0,
                is_valid=False, status_message="EMPTY FRAME",
                control_points=((0, 0), (0, 0)),
            )

        # Handle missing hands
        if pt1 is None and pt2 is None:
            self.reset()
            return ROIResult(
                left=0, top=0, right=0, bottom=0, width=0, height=0,
                is_valid=False, status_message="WAITING FOR HANDS",
                control_points=((0, 0), (0, 0)),
            )

        if pt1 is None or pt2 is None:
            self.reset()
            active_pt = pt1 if pt1 is not None else pt2
            # Clamp active point for display
            c_x = int(np.clip(active_pt[0], 0, frame_w - 1))
            c_y = int(np.clip(active_pt[1], 0, frame_h - 1))
            return ROIResult(
                left=0, top=0, right=0, bottom=0, width=0, height=0,
                is_valid=False, status_message="1 HAND DETECTED — SHOW YOUR OTHER HAND",
                control_points=((c_x, c_y), (c_x, c_y)),
            )

        # Clamp raw inputs to valid frame boundaries
        raw_x1 = float(np.clip(pt1[0], 0, frame_w - 1))
        raw_y1 = float(np.clip(pt1[1], 0, frame_h - 1))
        raw_x2 = float(np.clip(pt2[0], 0, frame_w - 1))
        raw_y2 = float(np.clip(pt2[1], 0, frame_h - 1))

        # Apply smoothing if enabled
        if self.smoothing_enabled:
            smooth_p1, smooth_p2 = self._apply_smoothing((raw_x1, raw_y1), (raw_x2, raw_y2))
        else:
            smooth_p1 = (raw_x1, raw_y1)
            smooth_p2 = (raw_x2, raw_y2)

        p1_x, p1_y = int(round(smooth_p1[0])), int(round(smooth_p1[1]))
        p2_x, p2_y = int(round(smooth_p2[0])), int(round(smooth_p2[1]))

        # Calculate bounding rectangle invariant to hand crossing or order
        left = max(0, min(p1_x, p2_x))
        top = max(0, min(p1_y, p2_y))
        right = min(frame_w, max(p1_x, p2_x))
        bottom = min(frame_h, max(p1_y, p2_y))

        width = right - left
        height = bottom - top

        # Validate minimum dimension criteria
        if width <= 0 or height <= 0:
            return ROIResult(
                left=left, top=top, right=right, bottom=bottom,
                width=width, height=height, is_valid=False,
                status_message="ROI ZERO DIMENSION",
                control_points=((p1_x, p1_y), (p2_x, p2_y)),
            )

        if width < self.min_width or height < self.min_height:
            return ROIResult(
                left=left, top=top, right=right, bottom=bottom,
                width=width, height=height, is_valid=False,
                status_message=f"ROI TOO SMALL (MIN {self.min_width}x{self.min_height})",
                control_points=((p1_x, p1_y), (p2_x, p2_y)),
            )

        return ROIResult(
            left=left,
            top=top,
            right=right,
            bottom=bottom,
            width=width,
            height=height,
            is_valid=True,
            status_message="2 HANDS DETECTED — EFFECT ACTIVE",
            control_points=((p1_x, p1_y), (p2_x, p2_y)),
        )

    def _apply_smoothing(
        self,
        target_p1: Tuple[float, float],
        target_p2: Tuple[float, float],
    ) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        """Apply exponential moving average with discontinuity detection."""
        if self._prev_pt1 is None or self._prev_pt2 is None:
            self._prev_pt1 = target_p1
            self._prev_pt2 = target_p2
            return target_p1, target_p2

        # Check distance jumps for discontinuity
        dist1 = math.hypot(target_p1[0] - self._prev_pt1[0], target_p1[1] - self._prev_pt1[1])
        dist2 = math.hypot(target_p2[0] - self._prev_pt2[0], target_p2[1] - self._prev_pt2[1])

        # If either hand jumped across the discontinuity threshold, reset directly to current
        if dist1 > self.discontinuity_threshold or dist2 > self.discontinuity_threshold:
            self._prev_pt1 = target_p1
            self._prev_pt2 = target_p2
            return target_p1, target_p2

        alpha = self.smoothing_factor
        s1_x = alpha * target_p1[0] + (1.0 - alpha) * self._prev_pt1[0]
        s1_y = alpha * target_p1[1] + (1.0 - alpha) * self._prev_pt1[1]
        s2_x = alpha * target_p2[0] + (1.0 - alpha) * self._prev_pt2[0]
        s2_y = alpha * target_p2[1] + (1.0 - alpha) * self._prev_pt2[1]

        self._prev_pt1 = (s1_x, s1_y)
        self._prev_pt2 = (s2_x, s2_y)

        return (s1_x, s1_y), (s2_x, s2_y)

