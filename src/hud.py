"""Advanced Cyberpunk HUD and Telemetry Dashboard component.

Renders persistent top cyberpunk navigation buttons, live telemetry cards,
tracking status pills, viewfinder brackets, and handles mouse click interactions.
"""

from typing import Callable, Dict, List, Optional, Tuple
import cv2
import numpy as np

from config import AppMode, BackgroundStatus, HUDConfig
from src.roi_controller import ROIResult



class HUDButton:
    """Interactive mouse-clickable button on the top HUD dashboard."""

    def __init__(
        self,
        button_id: str,
        label: str,
        x: int,
        y: int,
        width: int,
        height: int,
        action_callback: Callable[[], None],
        color: Tuple[int, int, int] = (0, 220, 255),
    ) -> None:
        self.button_id = button_id
        self.label = label
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.callback = action_callback
        self.color = color

    def contains(self, px: int, py: int) -> bool:
        """Return True if (px, py) is inside button bounding box."""
        return self.x <= px <= (self.x + self.width) and self.y <= py <= (self.y + self.height)


class HUDManager:
    """Manages cyberpunk dashboard overlays, interactive top buttons, and performance telemetry."""

    def __init__(self, config: Optional[HUDConfig] = None) -> None:
        self.config = config or HUDConfig()
        self.show_debug_metrics: bool = self.config.show_debug_metrics
        self._fps_smoothed: float = 0.0
        self._min_fps: float = 999.0

        # Detailed timing telemetry (in milliseconds)
        self.telemetry: Dict[str, float] = {
            "inference_ms": 0.0,
            "processing_ms": 0.0,
            "render_ms": 0.0,
        }

        # Clickable top buttons list
        self.buttons: List[HUDButton] = []
        self._buttons_configured: bool = False

    def toggle_debug_metrics(self) -> bool:
        """Toggle detailed telemetry card visibility."""
        self.show_debug_metrics = not self.show_debug_metrics
        return self.show_debug_metrics

    def update_fps(self, raw_fps: float) -> float:
        """Update exponentially smoothed rolling FPS estimate and min FPS."""
        alpha = self.config.fps_smoothing_factor
        if self._fps_smoothed == 0.0:
            self._fps_smoothed = raw_fps
            self._min_fps = raw_fps
        else:
            self._fps_smoothed = alpha * raw_fps + (1.0 - alpha) * self._fps_smoothed
            if raw_fps > 5.0:  # Ignore freeze frames
                self._min_fps = min(self._min_fps, raw_fps)
        return self._fps_smoothed

    def setup_buttons(
        self,
        frame_width: int,
        callbacks: Dict[str, Callable[[], None]],
    ) -> None:
        """Configure top interactive dashboard buttons across frame width."""
        btn_defs = [
            ("mode_std", "1: Std Filter", callbacks.get("mode_std", lambda: None), (180, 180, 190)),
            ("mode_roi", "2: ROI Portal", callbacks.get("mode_roi", lambda: None), (0, 255, 128)),
            ("mode_ghost", "3: Ghost Portal", callbacks.get("mode_ghost", lambda: None), (0, 215, 255)),
            ("mode_kb", "4: Keyboard", callbacks.get("mode_kb", lambda: None), (255, 160, 0)),
            ("mode_write", "5: Air Write", callbacks.get("mode_write", lambda: None), (0, 255, 230)),
            ("mode_magic", "6: Magic Arts", callbacks.get("mode_magic", lambda: None), (0, 140, 255)),
            ("mode_universe", "7: Universe", callbacks.get("mode_universe", lambda: None), (210, 110, 255)),
            ("recapture", "R: BG Plate", callbacks.get("recapture", lambda: None), (0, 215, 255)),
            ("next_filter", "N: Next", callbacks.get("next_filter", lambda: None), (200, 220, 255)),
            ("hud", "F: Telemetry", callbacks.get("hud", lambda: None), (220, 220, 220)),
            ("exit", "Q: Exit", callbacks.get("exit", lambda: None), (80, 80, 240)),
        ]

        self.buttons = []
        btn_h = 24
        btn_y = 6
        spacing = 6
        num_btns = len(btn_defs)
        total_spacing = (num_btns + 1) * spacing
        btn_w = max(70, (frame_width - total_spacing) // num_btns)

        start_x = spacing
        for bid, label, cb, col in btn_defs:
            self.buttons.append(
                HUDButton(bid, label, start_x, btn_y, btn_w, btn_h, cb, col)
            )
            start_x += btn_w + spacing

        self._buttons_configured = True

    def handle_mouse_click(self, x: int, y: int) -> bool:
        """Check if a mouse click hit any dashboard button and invoke its callback."""
        for btn in self.buttons:
            if btn.contains(x, y):
                btn.callback()
                return True
        return False

    def render(
        self,
        frame: np.ndarray,
        roi_result: ROIResult,
        active_filter_name: str,
        num_hands: int,
        fps: float,
        control_points: List[Tuple[int, int]],
        current_mode: AppMode = AppMode.STANDARD_FILTER,
        bg_status: BackgroundStatus = BackgroundStatus.NOT_CAPTURED,
        tracking_confidence: float = 0.70,
    ) -> np.ndarray:
        """Render all interface layers on top of the composited frame."""
        h, w = frame.shape[:2]

        # 1. Render Top Cyberpunk Dashboard Bar & Buttons
        self._draw_dashboard(frame, current_mode, bg_status, active_filter_name, num_hands)

        # 2. Render ROI Geometry Overlays if valid (in Standard Filter Mode)
        if current_mode == AppMode.STANDARD_FILTER and roi_result.is_valid:
            self._draw_roi_box(frame, roi_result)

        # 3. Render Hand Control Point Indicators (in Standard Filter Mode)
        if current_mode == AppMode.STANDARD_FILTER:
            self._draw_control_points(frame, control_points)

        # 4. Render Bottom Controls Legend
        self._draw_bottom_bar(frame, current_mode)

        # 5. Render Detailed Performance Telemetry (if toggled on)
        if self.show_debug_metrics:
            self._draw_metrics_panel(
                frame=frame,
                fps=fps,
                num_hands=num_hands,
                roi=roi_result,
                active_filter_name=active_filter_name,
                current_mode=current_mode,
                bg_status=bg_status,
                tracking_confidence=tracking_confidence,
            )

        return frame

    def _draw_dashboard(
        self,
        frame: np.ndarray,
        current_mode: AppMode,
        bg_status: BackgroundStatus,
        active_filter: str,
        num_hands: int,
    ) -> None:
        """Draw persistent top cyberpunk control bar with clickable buttons and status pill."""
        h, w = frame.shape[:2]
        header_height = 56

        header_roi = frame[0:header_height, 0:w]
        overlay = np.zeros_like(header_roi)
        overlay[:] = self.config.header_bg_color
        cv2.addWeighted(overlay, self.config.header_opacity, header_roi, 1.0 - self.config.header_opacity, 0, header_roi)
        cv2.line(frame, (0, header_height), (w, header_height), (50, 50, 65), 1)

        # Render interactive buttons
        for btn in self.buttons:
            # Highlight button if it represents the currently active mode
            is_active_mode = (
                (btn.button_id == "mode_std" and current_mode == AppMode.STANDARD_FILTER)
                or (btn.button_id == "mode_roi" and current_mode == AppMode.REALITYFRAME_ROI)
                or (btn.button_id == "mode_ghost" and current_mode == AppMode.REALITYFRAME_GHOST)
                or (btn.button_id == "mode_kb" and current_mode == AppMode.VIRTUAL_KEYBOARD)
                or (btn.button_id == "mode_write" and current_mode == AppMode.AIR_WRITING)
                or (btn.button_id == "mode_magic" and current_mode == AppMode.MYSTIC_MAGIC)
                or (btn.button_id == "mode_universe" and current_mode == AppMode.UNIVERSE_PORTAL)
            )

            fill_col = (45, 55, 65) if is_active_mode else (26, 26, 32)
            border_col = (0, 255, 200) if is_active_mode else btn.color
            border_thick = 2 if is_active_mode else 1

            cv2.rectangle(frame, (btn.x, btn.y), (btn.x + btn.width, btn.y + btn.height), fill_col, -1)
            cv2.rectangle(frame, (btn.x, btn.y), (btn.x + btn.width, btn.y + btn.height), border_col, border_thick)

            (tw, th), _ = cv2.getTextSize(btn.label, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)
            tx = btn.x + max(2, (btn.width - tw) // 2)
            ty = btn.y + max(10, (btn.height + th) // 2)
            cv2.putText(frame, btn.label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (240, 240, 245), 1, cv2.LINE_AA)

        # Second row of dashboard: Title & Active Mode Status Badge
        cv2.putText(
            frame,
            "HANDVISION STUDIO — REALITYFRAME",
            (12, 47),
            cv2.FONT_HERSHEY_DUPLEX,
            0.50,
            (0, 215, 255),
            1,
            cv2.LINE_AA,
        )

        mode_badge = f"MODE: {current_mode.value.replace('_', ' ')}"
        cv2.putText(
            frame,
            mode_badge,
            (w // 2 - 120, 47),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.46,
            (0, 255, 128),
            1,
            cv2.LINE_AA,
        )

        bg_text = f"BG: {bg_status.value}"
        bg_col = (0, 255, 128) if bg_status == BackgroundStatus.READY else (0, 165, 255)
        cv2.putText(frame, bg_text, (w - 180, 47), cv2.FONT_HERSHEY_SIMPLEX, 0.44, bg_col, 1, cv2.LINE_AA)

    def _draw_roi_box(self, frame: np.ndarray, roi: ROIResult) -> None:
        """Draw bounding rectangle, corner viewfinders, and dimension tag."""
        left, top, right, bottom = roi.left, roi.top, roi.right, roi.bottom
        color = self.config.border_color
        thickness = self.config.border_thickness

        cv2.rectangle(frame, (left, top), (right, bottom), color, thickness, cv2.LINE_AA)

        b_len = min(self.config.bracket_length, roi.width // 3, roi.height // 3)
        b_thick = self.config.bracket_thickness

        if b_len > 4:
            cv2.line(frame, (left, top), (left + b_len, top), color, b_thick)
            cv2.line(frame, (left, top), (left, top + b_len), color, b_thick)
            cv2.line(frame, (right, top), (right - b_len, top), color, b_thick)
            cv2.line(frame, (right, top), (right, top + b_len), color, b_thick)
            cv2.line(frame, (left, bottom), (left + b_len, bottom), color, b_thick)
            cv2.line(frame, (left, bottom), (left, bottom - b_len), color, b_thick)
            cv2.line(frame, (right, bottom), (right - b_len, bottom), color, b_thick)
            cv2.line(frame, (right, bottom), (right, bottom - b_len), color, b_thick)

        dim_text = f"{roi.width} x {roi.height} px"
        tag_font_scale = 0.45
        (tw, th), _ = cv2.getTextSize(dim_text, cv2.FONT_HERSHEY_SIMPLEX, tag_font_scale, 1)

        tag_x = left + 6
        tag_y = max(th + 10, top - 6)
        cv2.rectangle(frame, (tag_x - 3, tag_y - th - 3), (tag_x + tw + 3, tag_y + 3), (15, 15, 15), -1)
        cv2.putText(frame, dim_text, (tag_x, tag_y), cv2.FONT_HERSHEY_SIMPLEX, tag_font_scale, color, 1, cv2.LINE_AA)

    def _draw_control_points(self, frame: np.ndarray, points: List[Tuple[int, int]]) -> None:
        """Draw target markers at hand control anchors."""
        colors = [self.config.hand1_marker_color, self.config.hand2_marker_color]
        labels = ["H1", "H2"]

        for i, pt in enumerate(points[:2]):
            c = colors[i % len(colors)]
            label = labels[i % len(labels)]
            x, y = pt

            cv2.circle(frame, (x, y), 12, c, 2, cv2.LINE_AA)
            cv2.circle(frame, (x, y), 4, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.line(frame, (x - 16, y), (x - 12, y), c, 1, cv2.LINE_AA)
            cv2.line(frame, (x + 12, y), (x + 16, y), c, 1, cv2.LINE_AA)
            cv2.line(frame, (x, y - 16), (x, y - 12), c, 1, cv2.LINE_AA)
            cv2.line(frame, (x, y + 12), (x, y + 16), c, 1, cv2.LINE_AA)
            cv2.putText(frame, label, (x + 14, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.42, c, 1, cv2.LINE_AA)

    def _draw_bottom_bar(self, frame: np.ndarray, current_mode: AppMode) -> None:
        """Draw bottom bar containing shortcut hints."""
        h, w = frame.shape[:2]
        bar_height = 28
        bottom_roi = frame[h - bar_height:h, 0:w]

        overlay = np.zeros_like(bottom_roi)
        overlay[:] = (12, 12, 16)
        cv2.addWeighted(overlay, 0.85, bottom_roi, 0.15, 0, bottom_roi)

        legend_text = "[1-7] Modes  [U] Universe Portal  [W] Air Write  [G] Spells  [R] BG Plate  [N/P] Filters  [F] HUD  [Q] Exit"
        cv2.putText(
            frame,
            legend_text,
            (14, h - 9),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (210, 210, 220),
            1,
            cv2.LINE_AA,
        )

    def _draw_metrics_panel(
        self,
        frame: np.ndarray,
        fps: float,
        num_hands: int,
        roi: ROIResult,
        active_filter_name: str,
        current_mode: AppMode,
        bg_status: BackgroundStatus,
        tracking_confidence: float,
    ) -> None:
        """Render cyberpunk performance telemetry card in the top-right corner."""
        h, w = frame.shape[:2]
        panel_w = 260
        panel_h = 168
        panel_x = w - panel_w - 12
        panel_y = 64

        if panel_x <= 0 or panel_y + panel_h >= h:
            return

        panel_roi = frame[panel_y:panel_y + panel_h, panel_x:panel_x + panel_w]
        overlay = np.zeros_like(panel_roi)
        overlay[:] = (18, 18, 24)
        cv2.addWeighted(overlay, 0.85, panel_roi, 0.15, 0, panel_roi)
        cv2.rectangle(frame, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), (0, 215, 255), 1)

        latency_ms = (1000.0 / fps) if fps > 0.1 else 0.0
        lines = [
            ("SYSTEM TELEMETRY", (0, 215, 255)),
            (f"FPS: {fps:4.1f} (Latency: {latency_ms:4.1f} ms)", (255, 255, 255)),
            (f"Resolution: {w}x{h} px", (210, 210, 220)),
            (f"Mode: {current_mode.name}", (0, 255, 128)),
            (f"Filter: {active_filter_name[:18]}", (200, 255, 200)),
            (f"Hands: {num_hands}/2 (Conf: {tracking_confidence:0.2f})", (255, 255, 255)),
            (f"Background: {bg_status.name}", (0, 215, 255) if bg_status == BackgroundStatus.READY else (0, 165, 255)),
            (f"Inference: {self.telemetry['inference_ms']:3.1f}ms | Render: {self.telemetry['render_ms']:3.1f}ms", (180, 180, 210)),
        ]

        text_y = panel_y + 18
        for i, (text, color) in enumerate(lines):
            font_scale = 0.40 if i > 0 else 0.44
            cv2.putText(
                frame,
                text,
                (panel_x + 8, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                color,
                1,
                cv2.LINE_AA,
            )
            text_y += 19
