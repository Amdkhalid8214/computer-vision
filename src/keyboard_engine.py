"""Virtual Finger Typing Air Keyboard engine and state coordinator.

Defines keyboard layouts, key geometries, cursor smoothing, pinch-to-type,
dwell-to-type, live text editing, clipboard synchronization, and audio feedback.
"""

from dataclasses import dataclass
from enum import Enum
import logging
import math
import subprocess
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

try:
    import winsound
    HAS_WINSOUND = True
except ImportError:
    HAS_WINSOUND = False


class KeyType(Enum):
    """Categorization of virtual keyboard keys."""
    CHAR = "char"
    SPACE = "space"
    BACKSPACE = "backspace"
    ENTER = "enter"
    CAPS = "caps"
    SHIFT = "shift"
    CLEAR = "clear"
    COPY = "copy"
    MODE = "mode"
    CLOSE = "close"


class TypingMode(Enum):
    """Supported finger interaction modes."""
    PINCH = "PINCH"
    DWELL = "DWELL"


@dataclass
class KeyDefinition:
    """Represents a single interactive key on the virtual keyboard."""
    key_id: str
    label_normal: str
    label_shifted: str
    char_normal: str
    char_shifted: str
    key_type: KeyType
    col_span: float = 1.0

    # Bounding box in frame pixel coordinates
    x: int = 0
    y: int = 0
    width: int = 0
    height: int = 0

    def contains(self, px: int, py: int) -> bool:
        """Return True if pixel coordinate (px, py) is inside key boundary."""
        return self.x <= px <= (self.x + self.width) and self.y <= py <= (self.y + self.height)


def copy_text_to_clipboard(text: str) -> bool:
    """Copy text string to Windows system clipboard using clip.exe."""
    if not text:
        return False
    try:
        create_no_window = 0x08000000
        subprocess.run(
            ["clip.exe"],
            input=text.encode("utf-16"),
            check=True,
            creationflags=create_no_window,
        )
        return True
    except Exception as e:
        logger.debug("Failed to copy to clipboard via clip.exe: %s", e)
        return False


def play_keystroke_sound() -> None:
    """Produce subtle audio click feedback on key press."""
    if HAS_WINSOUND:
        try:
            winsound.Beep(1200, 25)
        except Exception:
            try:
                winsound.MessageBeep(winsound.MB_OK)
            except Exception:
                pass


class VirtualKeyboardEngine:
    """Core logic engine for QWERTY air typing, gestures, and text management."""

    def __init__(
        self,
        enabled: bool = False,
        typing_mode: str = "PINCH",
        pinch_threshold_px: float = 40.0,
        dwell_duration_sec: float = 0.85,
        cooldown_sec: float = 0.35,
        cursor_smoothing_factor: float = 0.40,
        audio_feedback: bool = True,
        width_ratio: float = 0.90,
        height_ratio: float = 0.44,
    ) -> None:
        self.is_visible: bool = enabled
        self.typing_mode: TypingMode = TypingMode.PINCH if typing_mode.upper() == "PINCH" else TypingMode.DWELL
        self.pinch_threshold_px: float = max(10.0, pinch_threshold_px)
        self.dwell_duration_sec: float = max(0.2, dwell_duration_sec)
        self.cooldown_sec: float = max(0.1, cooldown_sec)
        self.cursor_smoothing_factor: float = max(0.05, min(1.0, cursor_smoothing_factor))
        self.audio_feedback: bool = audio_feedback
        self.width_ratio: float = width_ratio
        self.height_ratio: float = height_ratio

        # Text state
        self.input_text: str = ""
        self.caps_lock: bool = False
        self.shift_active: bool = False
        self.status_toast: str = ""
        self.status_toast_timer: float = 0.0

        # Cursor and gesture tracking state
        self.smoothed_cursor: Optional[Tuple[float, float]] = None
        self.raw_cursor: Optional[Tuple[int, int]] = None
        self.raw_thumb: Optional[Tuple[int, int]] = None
        self.current_pinch_distance: float = 999.0
        self.is_pinching: bool = False
        self.was_pinching: bool = False

        # Key selection state
        self.hovered_key: Optional[KeyDefinition] = None
        self.last_dwell_key: Optional[KeyDefinition] = None
        self.dwell_elapsed_sec: float = 0.0
        self.dwell_progress: float = 0.0
        self.cooldown_remaining_sec: float = 0.0
        self.key_press_animations: Dict[str, float] = {}

        # Layout boundaries
        self.frame_width: int = 1280
        self.frame_height: int = 720
        self.keyboard_rect: Tuple[int, int, int, int] = (0, 0, 0, 0)
        self.text_bar_rect: Tuple[int, int, int, int] = (0, 0, 0, 0)

        self.keys: List[KeyDefinition] = []
        self._build_key_definitions()
        self.recalculate_layout(self.frame_width, self.frame_height)

    def toggle_visibility(self) -> bool:
        """Toggle virtual keyboard visibility."""
        self.is_visible = not self.is_visible
        self.reset_gesture_state()
        return self.is_visible

    def toggle_mode(self) -> TypingMode:
        """Toggle between PINCH-to-type and DWELL-to-type."""
        if self.typing_mode == TypingMode.PINCH:
            self.typing_mode = TypingMode.DWELL
        else:
            self.typing_mode = TypingMode.PINCH
        self.reset_gesture_state()
        self.set_toast(f"MODE: {self.typing_mode.value}-TO-TYPE")
        return self.typing_mode

    def set_toast(self, message: str, duration: float = 1.8) -> None:
        """Display an informational notification toast above the keyboard."""
        self.status_toast = message
        self.status_toast_timer = duration

    def reset_gesture_state(self) -> None:
        """Reset gesture, dwell, and pinch detectors."""
        self.smoothed_cursor = None
        self.raw_cursor = None
        self.raw_thumb = None
        self.hovered_key = None
        self.last_dwell_key = None
        self.dwell_elapsed_sec = 0.0
        self.dwell_progress = 0.0
        self.is_pinching = False
        self.was_pinching = False

    def clear_text(self) -> None:
        """Erase all typed characters."""
        self.input_text = ""
        self.set_toast("TEXT CLEARED")

    def copy_text(self) -> bool:
        """Copy accumulated text to Windows system clipboard."""
        if not self.input_text:
            self.set_toast("NOTHING TO COPY")
            return False
        success = copy_text_to_clipboard(self.input_text)
        if success:
            self.set_toast("COPIED TO CLIPBOARD!")
        else:
            self.set_toast("CLIPBOARD ERROR")
        return success

    def _build_key_definitions(self) -> None:
        """Construct the QWERTY key map and special control buttons."""
        self.rows_raw: List[List[KeyDefinition]] = [
            # Row 0: Numbers & Symbols
            [
                KeyDefinition("1", "1", "!", "1", "!", KeyType.CHAR),
                KeyDefinition("2", "2", "@", "2", "@", KeyType.CHAR),
                KeyDefinition("3", "3", "#", "3", "#", KeyType.CHAR),
                KeyDefinition("4", "4", "$", "4", "$", KeyType.CHAR),
                KeyDefinition("5", "5", "%", "5", "%", KeyType.CHAR),
                KeyDefinition("6", "6", "^", "6", "^", KeyType.CHAR),
                KeyDefinition("7", "7", "&", "7", "&", KeyType.CHAR),
                KeyDefinition("8", "8", "*", "8", "*", KeyType.CHAR),
                KeyDefinition("9", "9", "(", "9", "(", KeyType.CHAR),
                KeyDefinition("0", "0", ")", "0", ")", KeyType.CHAR),
                KeyDefinition("-", "-", "_", "-", "_", KeyType.CHAR),
                KeyDefinition("=", "=", "+", "=", "+", KeyType.CHAR),
                KeyDefinition("BACKSPACE", "BACK", "BACK", "", "", KeyType.BACKSPACE, col_span=1.6),
            ],
            # Row 1: Top Alphabet Row
            [
                KeyDefinition("Q", "q", "Q", "q", "Q", KeyType.CHAR),
                KeyDefinition("W", "w", "W", "w", "W", KeyType.CHAR),
                KeyDefinition("E", "e", "E", "e", "E", KeyType.CHAR),
                KeyDefinition("R", "r", "R", "r", "R", KeyType.CHAR),
                KeyDefinition("T", "t", "T", "t", "T", KeyType.CHAR),
                KeyDefinition("Y", "y", "Y", "y", "Y", KeyType.CHAR),
                KeyDefinition("U", "u", "U", "u", "U", KeyType.CHAR),
                KeyDefinition("I", "i", "I", "i", "I", KeyType.CHAR),
                KeyDefinition("O", "o", "O", "o", "O", KeyType.CHAR),
                KeyDefinition("P", "p", "P", "p", "P", KeyType.CHAR),
                KeyDefinition("[", "[", "{", "[", "{", KeyType.CHAR),
                KeyDefinition("]", "]", "}", "]", "}", KeyType.CHAR),
            ],
            # Row 2: Home Alphabet Row
            [
                KeyDefinition("CAPS", "CAPS", "CAPS", "", "", KeyType.CAPS, col_span=1.5),
                KeyDefinition("A", "a", "A", "a", "A", KeyType.CHAR),
                KeyDefinition("S", "s", "S", "s", "S", KeyType.CHAR),
                KeyDefinition("D", "d", "D", "d", "D", KeyType.CHAR),
                KeyDefinition("F", "f", "F", "f", "F", KeyType.CHAR),
                KeyDefinition("G", "g", "G", "g", "G", KeyType.CHAR),
                KeyDefinition("H", "h", "H", "h", "H", KeyType.CHAR),
                KeyDefinition("J", "j", "J", "j", "J", KeyType.CHAR),
                KeyDefinition("K", "k", "K", "k", "K", KeyType.CHAR),
                KeyDefinition("L", "l", "L", "l", "L", KeyType.CHAR),
                KeyDefinition(";", ";", ":", ";", ":", KeyType.CHAR),
                KeyDefinition("'", "'", '"', "'", '"', KeyType.CHAR),
                KeyDefinition("ENTER", "ENTER", "ENTER", "\n", "\n", KeyType.ENTER, col_span=1.6),
            ],
            # Row 3: Bottom Alphabet Row
            [
                KeyDefinition("SHIFT", "SHIFT", "SHIFT", "", "", KeyType.SHIFT, col_span=1.5),
                KeyDefinition("Z", "z", "Z", "z", "Z", KeyType.CHAR),
                KeyDefinition("X", "x", "X", "x", "X", KeyType.CHAR),
                KeyDefinition("C", "c", "C", "c", "C", KeyType.CHAR),
                KeyDefinition("V", "v", "V", "v", "V", KeyType.CHAR),
                KeyDefinition("B", "b", "B", "b", "B", KeyType.CHAR),
                KeyDefinition("N", "n", "N", "n", "N", KeyType.CHAR),
                KeyDefinition("M", "m", "M", "m", "M", KeyType.CHAR),
                KeyDefinition(",", ",", "<", ",", "<", KeyType.CHAR),
                KeyDefinition(".", ".", ">", ".", ">", KeyType.CHAR),
                KeyDefinition("/", "/", "?", "/", "?", KeyType.CHAR),
                KeyDefinition("SHIFT_R", "SHIFT", "SHIFT", "", "", KeyType.SHIFT, col_span=1.5),
            ],
            # Row 4: Controls & Spacebar Row
            [
                KeyDefinition("MODE", "PINCH/DWELL", "PINCH/DWELL", "", "", KeyType.MODE, col_span=2.0),
                KeyDefinition("CLEAR", "CLEAR", "CLEAR", "", "", KeyType.CLEAR, col_span=1.6),
                KeyDefinition("SPACE", "SPACEBAR", "SPACEBAR", " ", " ", KeyType.SPACE, col_span=4.8),
                KeyDefinition("COPY", "COPY", "COPY", "", "", KeyType.COPY, col_span=1.6),
                KeyDefinition("CLOSE", "[HIDE]", "[HIDE]", "", "", KeyType.CLOSE, col_span=1.4),
            ],
        ]

    def recalculate_layout(self, frame_width: int, frame_height: int) -> None:
        """Compute pixel positions and dimensions for each key based on frame size."""
        self.frame_width = frame_width
        self.frame_height = frame_height

        kb_w = int(frame_width * self.width_ratio)
        kb_h = int(frame_height * self.height_ratio)
        kb_x = (frame_width - kb_w) // 2
        kb_y = frame_height - kb_h - 38

        self.keyboard_rect = (kb_x, kb_y, kb_w, kb_h)

        text_bar_h = 42
        text_bar_y = max(58, kb_y - text_bar_h - 8)
        self.text_bar_rect = (kb_x, text_bar_y, kb_w, text_bar_h)

        num_rows = len(self.rows_raw)
        row_gap = 5
        col_gap = 5
        available_h = kb_h - (num_rows - 1) * row_gap
        key_h = max(24, available_h // num_rows)

        self.keys = []

        for row_idx, row in enumerate(self.rows_raw):
            total_span = sum(k.col_span for k in row)
            available_w = kb_w - (len(row) - 1) * col_gap
            unit_w = available_w / max(1.0, total_span)

            curr_x = kb_x
            curr_y = kb_y + row_idx * (key_h + row_gap)

            for key in row:
                k_w = int(round(key.col_span * unit_w))
                key.x = curr_x
                key.y = curr_y
                key.width = k_w
                key.height = key_h
                curr_x += k_w + col_gap
                self.keys.append(key)

    def get_key_at(self, px: int, py: int) -> Optional[KeyDefinition]:
        """Find the key containing the given point, or None."""
        for key in self.keys:
            if key.contains(px, py):
                return key
        return None

    def execute_key(self, key: KeyDefinition) -> Optional[str]:
        """Execute the action or typing associated with a key."""
        self.key_press_animations[key.key_id] = 0.22

        if self.audio_feedback:
            play_keystroke_sound()

        char_output: Optional[str] = None

        if key.key_type == KeyType.CHAR:
            use_shift = self.shift_active or self.caps_lock
            char = key.char_shifted if use_shift else key.char_normal
            self.input_text += char
            char_output = char
            if self.shift_active:
                self.shift_active = False

        elif key.key_type == KeyType.SPACE:
            self.input_text += " "
            char_output = " "

        elif key.key_type == KeyType.BACKSPACE:
            if len(self.input_text) > 0:
                self.input_text = self.input_text[:-1]
                char_output = "[BACKSPACE]"

        elif key.key_type == KeyType.ENTER:
            self.input_text += "\n"
            char_output = "\n"

        elif key.key_type == KeyType.CAPS:
            self.caps_lock = not self.caps_lock
            self.set_toast(f"CAPS LOCK: {'ON' if self.caps_lock else 'OFF'}")

        elif key.key_type == KeyType.SHIFT:
            self.shift_active = not self.shift_active
            self.set_toast(f"SHIFT: {'ACTIVE' if self.shift_active else 'OFF'}")

        elif key.key_type == KeyType.CLEAR:
            self.clear_text()

        elif key.key_type == KeyType.COPY:
            self.copy_text()

        elif key.key_type == KeyType.MODE:
            self.toggle_mode()

        elif key.key_type == KeyType.CLOSE:
            self.toggle_visibility()

        return char_output

    def update(
        self,
        index_tip: Optional[Tuple[int, int]],
        thumb_tip: Optional[Tuple[int, int]],
        dt: float = 0.033,
    ) -> Optional[str]:
        """Process finger coordinates, update dwell / pinch state, and execute key if triggered."""
        if not self.is_visible:
            self.reset_gesture_state()
            return None

        if self.status_toast_timer > 0.0:
            self.status_toast_timer = max(0.0, self.status_toast_timer - dt)
            if self.status_toast_timer == 0.0:
                self.status_toast = ""

        if self.cooldown_remaining_sec > 0.0:
            self.cooldown_remaining_sec = max(0.0, self.cooldown_remaining_sec - dt)

        expired_anim = []
        for kid in self.key_press_animations:
            self.key_press_animations[kid] -= dt
            if self.key_press_animations[kid] <= 0.0:
                expired_anim.append(kid)
        for kid in expired_anim:
            del self.key_press_animations[kid]

        if index_tip is None:
            self.reset_gesture_state()
            return None

        self.raw_cursor = index_tip
        self.raw_thumb = thumb_tip

        raw_x, raw_y = float(index_tip[0]), float(index_tip[1])
        if self.smoothed_cursor is None:
            self.smoothed_cursor = (raw_x, raw_y)
        else:
            alpha = self.cursor_smoothing_factor
            s_x = alpha * raw_x + (1.0 - alpha) * self.smoothed_cursor[0]
            s_y = alpha * raw_y + (1.0 - alpha) * self.smoothed_cursor[1]
            self.smoothed_cursor = (s_x, s_y)

        cursor_ix = int(round(self.smoothed_cursor[0]))
        cursor_iy = int(round(self.smoothed_cursor[1]))

        self.hovered_key = self.get_key_at(cursor_ix, cursor_iy)

        if thumb_tip is not None:
            self.current_pinch_distance = float(
                math.hypot(index_tip[0] - thumb_tip[0], index_tip[1] - thumb_tip[1])
            )
            self.is_pinching = self.current_pinch_distance <= self.pinch_threshold_px
        else:
            self.current_pinch_distance = 999.0
            self.is_pinching = False

        triggered_output: Optional[str] = None

        if self.typing_mode == TypingMode.PINCH:
            if self.is_pinching and not self.was_pinching and self.cooldown_remaining_sec <= 0.0:
                if self.hovered_key is not None:
                    triggered_output = self.execute_key(self.hovered_key)
                    self.cooldown_remaining_sec = self.cooldown_sec
            self.was_pinching = self.is_pinching
            self.dwell_progress = 0.0

        elif self.typing_mode == TypingMode.DWELL:
            if self.hovered_key is not None:
                if self.last_dwell_key == self.hovered_key:
                    self.dwell_elapsed_sec += dt
                else:
                    self.last_dwell_key = self.hovered_key
                    self.dwell_elapsed_sec = dt

                self.dwell_progress = min(1.0, self.dwell_elapsed_sec / self.dwell_duration_sec)
                if self.dwell_progress >= 1.0 and self.cooldown_remaining_sec <= 0.0:
                    triggered_output = self.execute_key(self.hovered_key)
                    self.dwell_elapsed_sec = 0.0
                    self.dwell_progress = 0.0
                    self.cooldown_remaining_sec = self.cooldown_sec
            else:
                self.last_dwell_key = None
                self.dwell_elapsed_sec = 0.0
                self.dwell_progress = 0.0

        return triggered_output

