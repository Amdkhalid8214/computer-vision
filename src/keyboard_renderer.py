"""Visual rendering component for the Virtual Finger Typing Air Keyboard.

Renders translucent keyboard frames, interactive QWERTY key buttons,
hover highlights, keypress pulse animations, live text input bar,
dwell timer progress rings, and fingertip cursor reticles.
"""

import time
from typing import Optional, Tuple
import cv2
import numpy as np

from src.keyboard_engine import KeyDefinition, KeyType, TypingMode, VirtualKeyboardEngine


class KeyboardRenderer:
    """Renders keyboard overlays directly on OpenCV video frames."""

    def __init__(self) -> None:
        self.color_bg = (22, 22, 26)
        self.color_border = (60, 60, 75)
        self.color_key_normal = (40, 40, 48)
        self.color_key_special = (52, 46, 58)
        self.color_key_hover = (0, 185, 255)       # Glowing Amber
        self.color_key_press = (0, 240, 90)        # Bright Neon Green
        self.color_key_active_toggle = (0, 215, 255)
        self.color_text_normal = (240, 240, 245)
        self.color_cursor_normal = (0, 220, 255)   # Cyan
        self.color_cursor_pinched = (0, 240, 90)   # Green

    def render(
        self,
        frame: np.ndarray,
        engine: VirtualKeyboardEngine,
    ) -> np.ndarray:
        """Render keyboard and typing indicators onto the provided frame."""
        if not engine.is_visible:
            return frame

        h, w = frame.shape[:2]

        if (w, h) != (engine.frame_width, engine.frame_height):
            engine.recalculate_layout(w, h)

        self._draw_keyboard_background(frame, engine)
        self._draw_text_bar(frame, engine)
        self._draw_keys(frame, engine)
        self._draw_cursor(frame, engine)

        return frame

    def _draw_keyboard_background(self, frame: np.ndarray, engine: VirtualKeyboardEngine) -> None:
        """Draw translucent background for the keyboard."""
        kx, ky, kw, kh = engine.keyboard_rect
        h, w = frame.shape[:2]

        kx = max(0, min(kx, w - 1))
        ky = max(0, min(ky, h - 1))
        kw = max(1, min(kw, w - kx))
        kh = max(1, min(kh, h - ky))

        panel_roi = frame[ky : ky + kh, kx : kx + kw]
        overlay = np.zeros_like(panel_roi)
        overlay[:] = self.color_bg
        opacity = 0.82
        cv2.addWeighted(overlay, opacity, panel_roi, 1.0 - opacity, 0, panel_roi)
        cv2.rectangle(frame, (kx, ky), (kx + kw, ky + kh), self.color_border, 1, cv2.LINE_AA)

    def _draw_text_bar(self, frame: np.ndarray, engine: VirtualKeyboardEngine) -> None:
        """Render the live text input area above the keyboard."""
        tx, ty, tw, th = engine.text_bar_rect
        h, w = frame.shape[:2]

        tx = max(0, min(tx, w - 1))
        ty = max(0, min(ty, h - 1))
        tw = max(1, min(tw, w - tx))
        th = max(1, min(th, h - ty))

        text_roi = frame[ty : ty + th, tx : tx + tw]
        overlay = np.zeros_like(text_roi)
        overlay[:] = (14, 14, 18)
        cv2.addWeighted(overlay, 0.88, text_roi, 0.12, 0, text_roi)
        cv2.rectangle(frame, (tx, ty), (tx + tw, ty + th), (75, 75, 90), 1, cv2.LINE_AA)

        show_cursor = int(time.time() * 2) % 2 == 0
        cursor_char = "|" if show_cursor else " "

        display_text = engine.input_text
        if len(display_text) > 42:
            display_text = "..." + display_text[-39:]

        full_display_str = f"TEXT: {display_text}{cursor_char}"

        cv2.putText(
            frame,
            full_display_str,
            (tx + 14, ty + 27),
            cv2.FONT_HERSHEY_DUPLEX,
            0.62,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        if engine.status_toast:
            toast_text = f"[{engine.status_toast}]"
            (ttw, _), _ = cv2.getTextSize(toast_text, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
            toast_x = tx + (tw - ttw) // 2
            cv2.rectangle(frame, (toast_x - 6, ty + 6), (toast_x + ttw + 6, ty + th - 6), (0, 160, 240), -1)
            cv2.putText(
                frame,
                toast_text,
                (toast_x, ty + 26),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (10, 10, 10),
                1,
                cv2.LINE_AA,
            )

        mode_str = f"[{engine.typing_mode.value}-TO-TYPE]"
        mode_color = (0, 220, 255) if engine.typing_mode == TypingMode.PINCH else (255, 180, 0)
        (mtw, _), _ = cv2.getTextSize(mode_str, cv2.FONT_HERSHEY_SIMPLEX, 0.44, 1)
        cv2.putText(
            frame,
            mode_str,
            (tx + tw - mtw - 12, ty + 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.44,
            mode_color,
            1,
            cv2.LINE_AA,
        )

    def _draw_keys(self, frame: np.ndarray, engine: VirtualKeyboardEngine) -> None:
        """Render all keys with hover, pressed pulse, and toggle indicators."""
        use_shift = engine.shift_active or engine.caps_lock

        for key in engine.keys:
            kx, ky, kw, kh = key.x, key.y, key.width, key.height
            is_hovered = engine.hovered_key == key
            is_pressed = key.key_id in engine.key_press_animations

            if is_pressed:
                fill_color = self.color_key_press
                border_color = (255, 255, 255)
                border_thickness = 2
            elif is_hovered:
                fill_color = (65, 55, 45)
                border_color = self.color_key_hover
                border_thickness = 2
            else:
                if key.key_type in (KeyType.CAPS, KeyType.SHIFT, KeyType.CLEAR, KeyType.COPY, KeyType.MODE, KeyType.CLOSE):
                    fill_color = self.color_key_special
                else:
                    fill_color = self.color_key_normal

                if key.key_type == KeyType.CAPS and engine.caps_lock:
                    border_color = self.color_key_active_toggle
                    border_thickness = 2
                elif key.key_type == KeyType.SHIFT and engine.shift_active:
                    border_color = self.color_key_active_toggle
                    border_thickness = 2
                else:
                    border_color = self.color_border
                    border_thickness = 1

            cv2.rectangle(frame, (kx, ky), (kx + kw, ky + kh), fill_color, -1)
            cv2.rectangle(frame, (kx, ky), (kx + kw, ky + kh), border_color, border_thickness, cv2.LINE_AA)

            label = key.label_shifted if use_shift else key.label_normal

            font_scale = 0.44 if len(label) > 3 else 0.52
            font_thickness = 1
            (ltw, lth), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
            lx = kx + max(4, (kw - ltw) // 2)
            ly = ky + max(12, (kh + lth) // 2)

            text_color = (0, 0, 0) if is_pressed else self.color_text_normal
            cv2.putText(frame, label, (lx, ly), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, font_thickness, cv2.LINE_AA)

            if is_hovered and engine.typing_mode == TypingMode.DWELL and engine.dwell_progress > 0.0:
                prog_w = int(kw * engine.dwell_progress)
                cv2.rectangle(frame, (kx, ky + kh - 4), (kx + prog_w, ky + kh), (0, 240, 80), -1)

    def _draw_cursor(self, frame: np.ndarray, engine: VirtualKeyboardEngine) -> None:
        """Render the user's index fingertip cursor reticle and pinch feedback."""
        if engine.smoothed_cursor is None:
            return

        cx = int(round(engine.smoothed_cursor[0]))
        cy = int(round(engine.smoothed_cursor[1]))

        if engine.is_pinching:
            c_color = self.color_cursor_pinched
            ring_radius = 16
            thickness = 3
        else:
            c_color = self.color_cursor_normal
            ring_radius = 12
            thickness = 2

        if engine.raw_thumb is not None and engine.raw_cursor is not None:
            tx, ty = engine.raw_thumb
            line_color = (0, 220, 80) if engine.is_pinching else (120, 120, 140)
            cv2.line(frame, (cx, cy), (tx, ty), line_color, 1, cv2.LINE_AA)
            cv2.circle(frame, (tx, ty), 4, line_color, -1, cv2.LINE_AA)

        cv2.circle(frame, (cx, cy), ring_radius, c_color, thickness, cv2.LINE_AA)
        cv2.circle(frame, (cx, cy), 3, (255, 255, 255), -1, cv2.LINE_AA)

        tick = 6
        cv2.line(frame, (cx - ring_radius - tick, cy), (cx - ring_radius + 2, cy), c_color, 1, cv2.LINE_AA)
        cv2.line(frame, (cx + ring_radius - 2, cy), (cx + ring_radius + tick, cy), c_color, 1, cv2.LINE_AA)
        cv2.line(frame, (cx, cy - ring_radius - tick), (cx, cy - ring_radius + 2), c_color, 1, cv2.LINE_AA)
        cv2.line(frame, (cx, cy + ring_radius - 2), (cx, cy + ring_radius + tick), c_color, 1, cv2.LINE_AA)

        if engine.typing_mode == TypingMode.DWELL and engine.dwell_progress > 0.0:
            arc_radius = ring_radius + 6
            start_angle = -90
            end_angle = int(-90 + 360 * engine.dwell_progress)
            cv2.ellipse(
                frame,
                (cx, cy),
                (arc_radius, arc_radius),
                0,
                start_angle,
                end_angle,
                (0, 240, 80),
                2,
                cv2.LINE_AA,
            )

        if engine.typing_mode == TypingMode.PINCH and engine.raw_thumb is not None:
            dist_label = f"{int(engine.current_pinch_distance)}px"
            cv2.putText(
                frame,
                dist_label,
                (cx + 18, cy + 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.36,
                c_color,
                1,
                cv2.LINE_AA,
            )

