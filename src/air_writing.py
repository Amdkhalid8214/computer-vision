"""Air Writing Studio: Mid-Air Finger Drawing and Writing System.

Allows users to write, draw, sketch diagrams, and sign in the air using finger
tracking with smooth coordinate interpolation, undo/redo, color palettes,
neon glow effects, and PNG export.
"""

from dataclasses import dataclass, field
from datetime import datetime
import logging
import math
import os
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np

from config import AirWritingConfig

logger = logging.getLogger(__name__)


# Standard vibrant BGR color palette
PALETTE_BGR: Dict[str, Tuple[int, int, int]] = {
    "white": (255, 255, 255),
    "cyan": (255, 255, 0),
    "green": (0, 255, 128),
    "yellow": (0, 230, 255),
    "orange": (0, 140, 255),
    "red": (60, 60, 255),
    "purple": (230, 80, 210),
}


@dataclass
class Stroke:
    """Represents a single continuous drawn stroke."""
    points: List[Tuple[int, int]] = field(default_factory=list)
    color: Tuple[int, int, int] = (255, 255, 0)
    size: int = 6
    tool: str = "PEN"  # "PEN", "HIGHLIGHTER", "ERASER"
    glow: bool = False
    opacity: float = 1.0


class AirWritingManager:
    """Coordinates mid-air finger drawing, smoothing, stroke history, and rendering."""

    def __init__(self, config: Optional[AirWritingConfig] = None) -> None:
        self.config = config or AirWritingConfig()

        self.tool: str = self.config.default_tool
        self.color_name: str = self.config.default_color_name
        self.color_bgr: Tuple[int, int, int] = PALETTE_BGR.get(self.color_name, (255, 255, 0))
        self.brush_size: int = self.config.default_brush_size
        self.glow_enabled: bool = self.config.glow_enabled
        self.canvas_mode: str = self.config.canvas_mode  # "OVERLAY" or "BLACKBOARD"
        self.export_dir: str = self.config.export_dir

        # Stroke history for Undo / Redo
        self.strokes: List[Stroke] = []
        self.redo_stack: List[Stroke] = []
        self.current_stroke: Optional[Stroke] = None

        # Cursor and gesture tracking state
        self.smoothed_cursor: Optional[Tuple[float, float]] = None
        self.is_drawing: bool = False
        self.was_drawing: bool = False
        self.last_pinch_distance: float = 999.0

        # Status toast
        self.status_toast: str = "AIR WRITING READY: PINCH TO DRAW"
        self.status_toast_timer: float = 2.5

    def set_tool(self, tool_name: str) -> None:
        """Set drawing tool: 'PEN', 'HIGHLIGHTER', or 'ERASER'."""
        t = tool_name.upper()
        if t in ("PEN", "HIGHLIGHTER", "ERASER"):
            self.tool = t
            self.set_toast(f"TOOL: {t}")

    def set_color(self, color_name: str) -> None:
        """Select brush color from PALETTE_BGR."""
        c = color_name.lower()
        if c in PALETTE_BGR:
            self.color_name = c
            self.color_bgr = PALETTE_BGR[c]
            self.set_toast(f"COLOR: {c.upper()}")

    def cycle_color(self) -> str:
        """Cycle to the next color in the palette."""
        keys = list(PALETTE_BGR.keys())
        idx = (keys.index(self.color_name) + 1) % len(keys)
        self.set_color(keys[idx])
        return self.color_name

    def set_brush_size(self, size: int) -> None:
        """Set brush size clamped to configured limits."""
        self.brush_size = max(self.config.min_brush_size, min(self.config.max_brush_size, size))
        self.set_toast(f"BRUSH SIZE: {self.brush_size}px")

    def increase_brush_size(self, step: int = 2) -> int:
        """Increase brush size."""
        self.set_brush_size(self.brush_size + step)
        return self.brush_size

    def decrease_brush_size(self, step: int = 2) -> int:
        """Decrease brush size."""
        self.set_brush_size(self.brush_size - step)
        return self.brush_size

    def toggle_glow(self) -> bool:
        """Toggle neon glow (Magical-Energy Ink)."""
        self.glow_enabled = not self.glow_enabled
        self.set_toast(f"MAGIC GLOW: {'ON' if self.glow_enabled else 'OFF'}")
        return self.glow_enabled

    def toggle_canvas_mode(self) -> str:
        """Toggle between OVERLAY (camera feed) and BLACKBOARD (dark slate)."""
        if self.canvas_mode == "OVERLAY":
            self.canvas_mode = "BLACKBOARD"
        else:
            self.canvas_mode = "OVERLAY"
        self.set_toast(f"CANVAS: {self.canvas_mode}")
        return self.canvas_mode

    def undo(self) -> bool:
        """Undo the last drawn stroke."""
        if self.strokes:
            stroke = self.strokes.pop()
            self.redo_stack.append(stroke)
            self.set_toast("UNDO STROKE")
            return True
        self.set_toast("NOTHING TO UNDO")
        return False

    def redo(self) -> bool:
        """Redo the previously undone stroke."""
        if self.redo_stack:
            stroke = self.redo_stack.pop()
            self.strokes.append(stroke)
            self.set_toast("REDO STROKE")
            return True
        self.set_toast("NOTHING TO REDO")
        return False

    def clear_canvas(self) -> None:
        """Clear all strokes from the canvas."""
        if self.strokes:
            # Store in redo stack for recovery
            self.redo_stack.extend(reversed(self.strokes))
            self.strokes.clear()
            self.current_stroke = None
            self.set_toast("CANVAS CLEARED")

    def set_toast(self, message: str, duration: float = 1.8) -> None:
        """Display an informational notification toast."""
        self.status_toast = message
        self.status_toast_timer = duration

    def reset_gesture_state(self) -> None:
        """Reset cursor smoothing and drawing flags."""
        self.smoothed_cursor = None
        self.is_drawing = False
        self.was_drawing = False
        if self.current_stroke is not None:
            if len(self.current_stroke.points) > 1:
                self.strokes.append(self.current_stroke)
            self.current_stroke = None

    def export_png(
        self,
        filename: Optional[str] = None,
        transparent_bg: bool = False,
        current_frame: Optional[np.ndarray] = None,
        dimensions: Tuple[int, int] = (720, 1280),
    ) -> str:
        """Save current drawing to a PNG image file.

        Args:
            filename: Destination filepath. If None, auto-generates timestamped filename.
            transparent_bg: If True, exports transparent 4-channel RGBA image.
            current_frame: Optional live background frame to composite onto.
            dimensions: (H, W) fallback canvas dimensions.

        Returns:
            Absolute filepath of the saved PNG image.
        """
        os.makedirs(self.export_dir, exist_ok=True)
        if not filename:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(self.export_dir, f"air_writing_{ts}.png")

        h, w = dimensions
        if current_frame is not None:
            h, w = current_frame.shape[:2]

        if transparent_bg:
            # 4-channel BGRA canvas initialized to transparent
            canvas = np.zeros((h, w, 4), dtype=np.uint8)
            self._render_strokes_to_canvas(canvas, is_rgba=True)
            cv2.imwrite(filename, canvas)
        else:
            if current_frame is not None and self.canvas_mode == "OVERLAY":
                canvas = current_frame.copy()
            else:
                # Dark slate blackboard
                canvas = np.full((h, w, 3), (22, 22, 28), dtype=np.uint8)
            canvas = self.render(canvas, show_toolbar=False)
            cv2.imwrite(filename, canvas)

        logger.info("Air writing drawing saved successfully to: %s", filename)
        self.set_toast(f"SAVED: {os.path.basename(filename)}")
        return filename

    def update(
        self,
        index_tip: Optional[Tuple[int, int]],
        thumb_tip: Optional[Tuple[int, int]],
        dt: float = 0.033,
    ) -> None:
        """Process finger coordinates, detect pinch drawing gestures, and update stroke buffer."""
        if self.status_toast_timer > 0.0:
            self.status_toast_timer = max(0.0, self.status_toast_timer - dt)

        if index_tip is None:
            self.reset_gesture_state()
            return

        # Coordinate smoothing (Exponential Moving Average)
        raw_x, raw_y = float(index_tip[0]), float(index_tip[1])
        if self.smoothed_cursor is None:
            self.smoothed_cursor = (raw_x, raw_y)
        else:
            alpha = self.config.smoothing_factor
            sx = alpha * raw_x + (1.0 - alpha) * self.smoothed_cursor[0]
            sy = alpha * raw_y + (1.0 - alpha) * self.smoothed_cursor[1]
            self.smoothed_cursor = (sx, sy)

        curr_pt = (int(round(self.smoothed_cursor[0])), int(round(self.smoothed_cursor[1])))

        # Pinch detection (Thumb tip to Index tip distance)
        if thumb_tip is not None:
            self.last_pinch_distance = float(
                math.hypot(index_tip[0] - thumb_tip[0], index_tip[1] - thumb_tip[1])
            )
            self.is_drawing = self.last_pinch_distance <= self.config.pinch_threshold_px
        else:
            self.last_pinch_distance = 999.0
            self.is_drawing = False

        # Drawing lifecycle
        if self.is_drawing:
            if not self.was_drawing or self.current_stroke is None:
                # Start new stroke
                opacity = 0.45 if self.tool == "HIGHLIGHTER" else 1.0
                sz = self.brush_size * 2 if self.tool in ("HIGHLIGHTER", "ERASER") else self.brush_size
                self.current_stroke = Stroke(
                    points=[curr_pt],
                    color=self.color_bgr if self.tool != "ERASER" else (0, 0, 0),
                    size=sz,
                    tool=self.tool,
                    glow=self.glow_enabled,
                    opacity=opacity,
                )
                self.redo_stack.clear()
            else:
                # Append point with interpolation to eliminate gaps during fast motion
                prev_pt = self.current_stroke.points[-1]
                dist = math.hypot(curr_pt[0] - prev_pt[0], curr_pt[1] - prev_pt[1])
                if dist > 3.0:
                    steps = max(1, int(dist // 4.0))
                    for i in range(1, steps + 1):
                        interp_x = int(round(prev_pt[0] + (curr_pt[0] - prev_pt[0]) * (i / steps)))
                        interp_y = int(round(prev_pt[1] + (curr_pt[1] - prev_pt[1]) * (i / steps)))
                        self.current_stroke.points.append((interp_x, interp_y))
        else:
            if self.was_drawing and self.current_stroke is not None:
                # Finalize stroke
                if len(self.current_stroke.points) > 1:
                    if self.tool == "ERASER":
                        self._apply_eraser_stroke(self.current_stroke)
                    else:
                        self.strokes.append(self.current_stroke)
                self.current_stroke = None

        self.was_drawing = self.is_drawing

    def _apply_eraser_stroke(self, eraser_stroke: Stroke) -> None:
        """Erase points on existing strokes intersecting with the eraser stroke."""
        eraser_pts = set(eraser_stroke.points)
        erase_radius_sq = (eraser_stroke.size) ** 2

        new_strokes: List[Stroke] = []
        for s in self.strokes:
            remaining_pts = []
            for pt in s.points:
                # Check proximity to eraser
                hit = any((pt[0] - ep[0]) ** 2 + (pt[1] - ep[1]) ** 2 <= erase_radius_sq for ep in eraser_pts)
                if not hit:
                    remaining_pts.append(pt)
                else:
                    if len(remaining_pts) > 1:
                        new_strokes.append(
                            Stroke(
                                points=remaining_pts,
                                color=s.color,
                                size=s.size,
                                tool=s.tool,
                                glow=s.glow,
                                opacity=s.opacity,
                            )
                        )
                    remaining_pts = []
            if len(remaining_pts) > 1:
                new_strokes.append(
                    Stroke(
                        points=remaining_pts,
                        color=s.color,
                        size=s.size,
                        tool=s.tool,
                        glow=s.glow,
                        opacity=s.opacity,
                    )
                )

        self.strokes = new_strokes

    def _render_strokes_to_canvas(self, canvas: np.ndarray, is_rgba: bool = False) -> None:
        """Render all strokes into an existing canvas buffer."""
        all_strokes = list(self.strokes)
        if self.current_stroke is not None and self.tool != "ERASER":
            all_strokes.append(self.current_stroke)

        for stroke in all_strokes:
            if len(stroke.points) < 2:
                continue

            pts = np.array(stroke.points, dtype=np.int32)
            pts = pts.reshape((-1, 1, 2))

            col = (*stroke.color, int(255 * stroke.opacity)) if is_rgba else stroke.color

            if stroke.tool == "HIGHLIGHTER" and not is_rgba:
                # Blend highlighter smoothly onto canvas
                hl_overlay = canvas.copy()
                cv2.polylines(hl_overlay, [pts], False, stroke.color, stroke.size, cv2.LINE_AA)
                cv2.addWeighted(hl_overlay, 0.45, canvas, 0.55, 0, canvas)
            else:
                cv2.polylines(canvas, [pts], False, col, stroke.size, cv2.LINE_AA)

    def render(self, frame: np.ndarray, show_toolbar: bool = True) -> np.ndarray:
        """Render drawn strokes, optional magic glow, cursor reticle, and toolbar."""
        h, w = frame.shape[:2]

        # 1. Base canvas selection
        if self.canvas_mode == "BLACKBOARD":
            canvas = np.full((h, w, 3), (20, 20, 26), dtype=np.uint8)
        else:
            canvas = frame.copy()

        # 2. Render normal stroke ink
        all_strokes = list(self.strokes)
        if self.current_stroke is not None and self.tool != "ERASER":
            all_strokes.append(self.current_stroke)

        # Separate glowing strokes for luminous pass
        glow_layer = np.zeros_like(canvas) if self.glow_enabled else None

        for stroke in all_strokes:
            if len(stroke.points) < 2:
                continue

            pts = np.array(stroke.points, dtype=np.int32).reshape((-1, 1, 2))

            if stroke.tool == "HIGHLIGHTER":
                hl_overlay = canvas.copy()
                cv2.polylines(hl_overlay, [pts], False, stroke.color, stroke.size, cv2.LINE_AA)
                cv2.addWeighted(hl_overlay, 0.45, canvas, 0.55, 0, canvas)
            else:
                cv2.polylines(canvas, [pts], False, stroke.color, stroke.size, cv2.LINE_AA)

            # Accumulate on glow layer if glowing
            if self.glow_enabled or stroke.glow:
                if glow_layer is None:
                    glow_layer = np.zeros_like(canvas)
                cv2.polylines(glow_layer, [pts], False, stroke.color, stroke.size * 2, cv2.LINE_AA)

        # 3. Additive bloom pass for Magical-Energy Ink
        if glow_layer is not None and np.any(glow_layer):
            # 2-pass Gaussian blur creates vibrant atmospheric glow
            blurred_glow = cv2.GaussianBlur(glow_layer, (21, 21), 0)
            canvas = cv2.add(canvas, blurred_glow)

        # 4. Render finger drawing cursor reticle
        if self.smoothed_cursor is not None:
            cx = int(round(self.smoothed_cursor[0]))
            cy = int(round(self.smoothed_cursor[1]))

            ring_col = (0, 255, 0) if self.is_drawing else (0, 200, 255)
            cursor_r = max(4, self.brush_size // 2)

            # Circular reticle
            cv2.circle(canvas, (cx, cy), cursor_r, self.color_bgr, -1, cv2.LINE_AA)
            cv2.circle(canvas, (cx, cy), cursor_r + 4, ring_col, 2, cv2.LINE_AA)

            # Pinch indicator bar
            if self.last_pinch_distance < 80.0:
                p_text = "WRITING" if self.is_drawing else "PINCH TO WRITE"
                cv2.putText(
                    canvas,
                    p_text,
                    (cx + 12, cy - 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.42,
                    ring_col,
                    1,
                    cv2.LINE_AA,
                )

        # 5. Render Air Writing Toolbar & Notification Toast
        if show_toolbar:
            canvas = self._render_toolbar(canvas)

        return canvas

    def _render_toolbar(self, frame: np.ndarray) -> np.ndarray:
        """Render floating tool badges, color swatch, brush size, and action keys."""
        h, w = frame.shape[:2]

        bar_h = 36
        bar_y = h - bar_h - 10
        bar_w = min(w - 40, 780)
        bar_x = (w - bar_w) // 2

        # Semi-transparent dark toolbar pill
        overlay = frame.copy()
        cv2.rectangle(overlay, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (18, 18, 24), -1)
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (0, 215, 255), 1, cv2.LINE_AA)

        # Information text segments
        tool_info = f"TOOL: {self.tool}"
        size_info = f"SIZE: {self.brush_size}px"
        glow_info = f"GLOW: {'ON' if self.glow_enabled else 'OFF'}"
        mode_info = f"MODE: {self.canvas_mode}"
        keys_info = "[Z] Undo  [Y] Redo  [C] Clear  [S] Save  [G] Glow"

        curr_x = bar_x + 16

        # Color indicator circle
        cv2.circle(frame, (curr_x + 8, bar_y + bar_h // 2), 8, self.color_bgr, -1, cv2.LINE_AA)
        cv2.circle(frame, (curr_x + 8, bar_y + bar_h // 2), 9, (255, 255, 255), 1, cv2.LINE_AA)
        curr_x += 28

        # Labels
        cv2.putText(
            frame,
            f"{tool_info}  |  {size_info}  |  {glow_info}  |  {mode_info}  |  {keys_info}",
            (curr_x, bar_y + 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (240, 240, 240),
            1,
            cv2.LINE_AA,
        )

        # Render floating status toast
        if self.status_toast and self.status_toast_timer > 0.0:
            alpha = min(1.0, self.status_toast_timer * 2.0)
            (tw, th), _ = cv2.getTextSize(self.status_toast, cv2.FONT_HERSHEY_DUPLEX, 0.54, 1)
            toast_x = (w - tw) // 2
            toast_y = 68

            toast_overlay = frame.copy()
            cv2.rectangle(
                toast_overlay,
                (toast_x - 14, toast_y - th - 8),
                (toast_x + tw + 14, toast_y + 8),
                (15, 15, 20),
                -1,
            )
            cv2.addWeighted(toast_overlay, 0.8 * alpha, frame, 1.0 - 0.8 * alpha, 0, frame)
            cv2.putText(
                frame,
                self.status_toast,
                (toast_x, toast_y),
                cv2.FONT_HERSHEY_DUPLEX,
                0.54,
                (0, 220, 255),
                1,
                cv2.LINE_AA,
            )

        return frame

