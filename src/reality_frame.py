"""RealityFrame Manager: Ghost Portal and Cybernetic ROI Portal Engine.

Coordinates two-hand portal tracking, background-replacement optical illusions (Ghost Mode),
cybernetic visual styling, technical grid lines, particle sparks, and stability verification.
"""

from dataclasses import dataclass
import logging
import math
import time
from typing import List, Optional, Tuple
import cv2
import numpy as np

from config import RealityFrameConfig
from src.roi_controller import ROIResult

logger = logging.getLogger(__name__)


@dataclass
class PortalState:
    """State descriptor for the active RealityFrame portal."""
    left: int
    top: int
    right: int
    bottom: int
    width: int
    height: int
    is_active: bool
    status_label: str
    control_points: Tuple[Tuple[int, int], Tuple[int, int]]


class RealityFrameManager:
    """Renders cybernetic Ghost and Filter Portals using two-hand geometric anchors."""

    def __init__(self, config: Optional[RealityFrameConfig] = None) -> None:
        self.config = config or RealityFrameConfig()
        self._stability_counter: int = 0
        self._prev_bounds: Optional[Tuple[int, int, int, int]] = None
        self._particle_phase: float = 0.0

    def reset_stability(self) -> None:
        """Reset portal stabilization counter."""
        self._stability_counter = 0
        self._prev_bounds = None

    def evaluate_portal(
        self,
        pt1: Optional[Tuple[int, int]],
        pt2: Optional[Tuple[int, int]],
        frame_shape: Tuple[int, int],
    ) -> PortalState:
        """Calculate and stabilize portal coordinates from two index fingertips.

        Args:
            pt1: Landmark 8 (Index fingertip) of Hand 1.
            pt2: Landmark 8 (Index fingertip) of Hand 2.
            frame_shape: (H, W) of the video frame.

        Returns:
            PortalState detailing geometry and active readiness.
        """
        h, w = frame_shape[:2]

        if pt1 is None or pt2 is None:
            self.reset_stability()
            return PortalState(
                left=0, top=0, right=0, bottom=0, width=0, height=0,
                is_active=False,
                status_label="WAITING FOR 2 HANDS TO FORM PORTAL",
                control_points=((0, 0), (0, 0)),
            )

        # Clamp fingertips to image bounds
        x1 = int(np.clip(pt1[0], 0, w - 1))
        y1 = int(np.clip(pt1[1], 0, h - 1))
        x2 = int(np.clip(pt2[0], 0, w - 1))
        y2 = int(np.clip(pt2[1], 0, h - 1))

        left = max(0, min(x1, x2))
        top = max(0, min(y1, y2))
        right = min(w, max(x1, x2))
        bottom = min(h, max(y1, y2))

        width = right - left
        height = bottom - top

        # Require minimum dimensions
        min_w = self.config.min_portal_width
        min_h = self.config.min_portal_height
        if width < min_w or height < min_h:
            self.reset_stability()
            return PortalState(
                left=left, top=top, right=right, bottom=bottom,
                width=width, height=height, is_active=False,
                status_label=f"PORTAL TOO SMALL (MIN {min_w}x{min_h})",
                control_points=((x1, y1), (x2, y2)),
            )

        # Stability check: ensure coordinates are not erratic jumps
        current_bounds = (left, top, right, bottom)
        if self._prev_bounds is not None:
            jump = max(abs(current_bounds[i] - self._prev_bounds[i]) for i in range(4))
            if jump < 80:
                self._stability_counter += 1
            else:
                self._stability_counter = 1
        else:
            self._stability_counter = 1

        self._prev_bounds = current_bounds
        is_stable = self._stability_counter >= 2

        return PortalState(
            left=left,
            top=top,
            right=right,
            bottom=bottom,
            width=width,
            height=height,
            is_active=is_stable,
            status_label="PORTAL LOCK ACQUIRED" if is_stable else "STABILIZING PORTAL...",
            control_points=((x1, y1), (x2, y2)),
        )

    def apply_ghost_mode(
        self,
        frame: np.ndarray,
        portal: PortalState,
        background_plate: Optional[np.ndarray],
    ) -> np.ndarray:
        """Composite background plate inside portal boundaries for the invisibility illusion.

        Args:
            frame: Live webcam frame.
            portal: Active PortalState.
            background_plate: Stored static median background plate.

        Returns:
            Frame with ghost optical illusion and cybernetic overlays applied.
        """
        h, w = frame.shape[:2]

        if not portal.is_active:
            return frame

        if background_plate is None:
            # Inform user background must be captured
            self._draw_portal_banner(
                frame, portal, "NO BACKGROUND — PRESS 'R' TO CAPTURE", (0, 165, 255)
            )
            return frame

        left, top, right, bottom = portal.left, portal.top, portal.right, portal.bottom

        # Verify dimension compatibility
        bg_h, bg_w = background_plate.shape[:2]
        if (bg_h, bg_w) != (h, w):
            bg_slice = cv2.resize(background_plate, (w, h))[top:bottom, left:right]
        else:
            bg_slice = background_plate[top:bottom, left:right]

        live_slice = frame[top:bottom, left:right]

        # Exact replacement: output[ROI] = background_plate[ROI]
        if bg_slice.shape == live_slice.shape and live_slice.size > 0:
            frame[top:bottom, left:right] = bg_slice

        # Render cybernetic portal styling
        self._render_cybernetic_portal(
            frame=frame,
            portal=portal,
            border_color=self.config.portal_border_color_ghost,  # Neon Gold
            mode_label="REALITYFRAME: GHOST PORTAL ACTIVE",
        )

        return frame

    def apply_roi_mode(
        self,
        frame: np.ndarray,
        portal: PortalState,
        processed_roi: Optional[np.ndarray],
        filter_name: str,
    ) -> np.ndarray:
        """Composite filtered image inside portal boundaries with cybernetic accents.

        Args:
            frame: Live webcam frame.
            portal: Active PortalState.
            processed_roi: Processed filter slice matching portal dimensions.
            filter_name: Name of the applied filter.

        Returns:
            Frame with filtered portal and cybernetic graphics applied.
        """
        if not portal.is_active:
            return frame

        left, top, right, bottom = portal.left, portal.top, portal.right, portal.bottom

        if processed_roi is not None:
            target_slice = frame[top:bottom, left:right]
            if processed_roi.shape == target_slice.shape:
                frame[top:bottom, left:right] = processed_roi

        # Render cybernetic portal styling
        self._render_cybernetic_portal(
            frame=frame,
            portal=portal,
            border_color=self.config.portal_border_color_roi,  # Neon Cyan/Green
            mode_label=f"REALITYFRAME: {filter_name.upper()}",
        )

        return frame

    def _render_cybernetic_portal(
        self,
        frame: np.ndarray,
        portal: PortalState,
        border_color: Tuple[int, int, int],
        mode_label: str,
    ) -> None:
        """Draw glowing cybernetic borders, technical grid details, and corner brackets."""
        left, top, right, bottom = portal.left, portal.top, portal.right, portal.bottom
        pw, ph = portal.width, portal.height

        # 1. Dual Glowing Border
        # Outer soft glow
        cv2.rectangle(frame, (left - 1, top - 1), (right + 1, bottom + 1), border_color, 1, cv2.LINE_AA)
        # Inner sharp border
        cv2.rectangle(frame, (left, top), (right, bottom), (255, 255, 255), 1, cv2.LINE_AA)

        # 2. Cybernetic Corner Accents
        b_len = min(28, pw // 4, ph // 4)
        thick = 3
        if b_len > 4:
            # Top-Left
            cv2.line(frame, (left, top), (left + b_len, top), border_color, thick)
            cv2.line(frame, (left, top), (left, top + b_len), border_color, thick)
            # Top-Right
            cv2.line(frame, (right, top), (right - b_len, top), border_color, thick)
            cv2.line(frame, (right, top), (right, top + b_len), border_color, thick)
            # Bottom-Left
            cv2.line(frame, (left, bottom), (left + b_len, bottom), border_color, thick)
            cv2.line(frame, (left, bottom), (left, bottom - b_len), border_color, thick)
            # Bottom-Right
            cv2.line(frame, (right, bottom), (right - b_len, bottom), border_color, thick)
            cv2.line(frame, (right, bottom), (right, bottom - b_len), border_color, thick)

        # 3. Subtle Cybernetic Crosshair Grid (only if enabled and portal is large enough)
        if self.config.enable_cyber_grid and pw > 120 and ph > 120:
            cx, cy = (left + right) // 2, (top + bottom) // 2
            grid_color = (border_color[0] // 3, border_color[1] // 3, border_color[2] // 3)
            # Center reticle ticks
            cv2.line(frame, (cx - 15, cy), (cx + 15, cy), grid_color, 1, cv2.LINE_AA)
            cv2.line(frame, (cx, cy - 15), (cx, cy + 15), grid_color, 1, cv2.LINE_AA)

        # 4. Animated Corner Particle Sparks (low-overhead sin-cos sparks)
        if self.config.enable_particles:
            self._particle_phase = (self._particle_phase + 0.18) % (2 * math.pi)
            pulse = int(5 * math.sin(self._particle_phase))
            p_radius = max(2, 3 + (pulse // 2))
            cv2.circle(frame, (left, top), p_radius, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(frame, (right, bottom), p_radius, (255, 255, 255), -1, cv2.LINE_AA)

        # 5. Technical Telemetry Badge
        badge_text = f"{mode_label} | {pw}x{ph} px"
        (tw, th), _ = cv2.getTextSize(badge_text, cv2.FONT_HERSHEY_SIMPLEX, 0.44, 1)
        bx = max(10, left)
        by = max(th + 14, top - 8)

        # Badge background
        cv2.rectangle(frame, (bx - 4, by - th - 4), (bx + tw + 4, by + 4), (12, 12, 16), -1)
        cv2.rectangle(frame, (bx - 4, by - th - 4), (bx + tw + 4, by + 4), border_color, 1)
        cv2.putText(frame, badge_text, (bx, by), cv2.FONT_HERSHEY_SIMPLEX, 0.44, border_color, 1, cv2.LINE_AA)

    def _draw_portal_banner(
        self,
        frame: np.ndarray,
        portal: PortalState,
        message: str,
        color: Tuple[int, int, int],
    ) -> None:
        """Render alert notification above the portal."""
        left, top, right, bottom = portal.left, portal.top, portal.right, portal.bottom
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2, cv2.LINE_AA)

        (tw, th), _ = cv2.getTextSize(message, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        bx = max(10, left)
        by = max(th + 14, top - 8)
        cv2.rectangle(frame, (bx - 4, by - th - 4), (bx + tw + 4, by + 4), (15, 15, 20), -1)
        cv2.putText(frame, message, (bx, by), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
