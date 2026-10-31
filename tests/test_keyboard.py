"""Automated unit tests for VirtualKeyboardEngine and KeyboardRenderer."""

import numpy as np
import pytest

from src.keyboard_engine import KeyType, TypingMode, VirtualKeyboardEngine
from src.keyboard_renderer import KeyboardRenderer


@pytest.fixture
def keyboard() -> VirtualKeyboardEngine:
    """Fixture providing an enabled VirtualKeyboardEngine in PINCH mode."""
    kb = VirtualKeyboardEngine(
        enabled=True,
        typing_mode="PINCH",
        pinch_threshold_px=40.0,
        dwell_duration_sec=0.5,
        cooldown_sec=0.2,
        cursor_smoothing_factor=1.0,  # 1.0 for instantaneous cursor response in tests
        audio_feedback=False,
    )
    kb.recalculate_layout(1280, 720)
    return kb


def test_keyboard_layout_and_key_geometry(keyboard: VirtualKeyboardEngine):
    """Verify keys are generated with positive bounding box coordinates."""
    assert len(keyboard.keys) > 40
    for key in keyboard.keys:
        assert key.width > 0
        assert key.height > 0
        assert key.x >= 0
        assert key.y >= 0


def test_key_hover_detection(keyboard: VirtualKeyboardEngine):
    """Verify get_key_at correctly detects key under fingertip."""
    target_key = keyboard.keys[10]
    center_x = target_key.x + target_key.width // 2
    center_y = target_key.y + target_key.height // 2

    found = keyboard.get_key_at(center_x, center_y)
    assert found is not None
    assert found.key_id == target_key.key_id


def test_pinch_to_type_rising_edge(keyboard: VirtualKeyboardEngine):
    """Verify pinch-to-type fires once on rising edge and avoids duplicate typing while held."""
    # Find key 'A'
    key_a = next(k for k in keyboard.keys if k.key_id == "A")
    cx = key_a.x + key_a.width // 2
    cy = key_a.y + key_a.height // 2

    # Hover over key with fingers apart (distance 100px)
    res1 = keyboard.update(index_tip=(cx, cy), thumb_tip=(cx + 100, cy), dt=0.033)
    assert res1 is None
    assert keyboard.hovered_key == key_a
    assert not keyboard.is_pinching
    assert keyboard.input_text == ""

    # Frame 2: Pinch fingers together (distance 20px <= threshold 40px)
    res2 = keyboard.update(index_tip=(cx, cy), thumb_tip=(cx + 20, cy), dt=0.033)
    assert res2 == "a"
    assert keyboard.input_text == "a"
    assert keyboard.is_pinching

    # Frame 3: Fingers remain pinched together -> must NOT trigger repeated key
    res3 = keyboard.update(index_tip=(cx, cy), thumb_tip=(cx + 20, cy), dt=0.033)
    assert res3 is None
    assert keyboard.input_text == "a"

    # Frame 4: Release pinch
    res4 = keyboard.update(index_tip=(cx, cy), thumb_tip=(cx + 100, cy), dt=0.033)
    assert res4 is None
    assert not keyboard.is_pinching
    assert keyboard.input_text == "a"


def test_dwell_to_type(keyboard: VirtualKeyboardEngine):
    """Verify dwell mode accumulates progress and fires keystroke upon completion."""
    keyboard.typing_mode = TypingMode.DWELL
    keyboard.dwell_duration_sec = 0.10  # 100ms dwell

    key_b = next(k for k in keyboard.keys if k.key_id == "B")
    cx = key_b.x + key_b.width // 2
    cy = key_b.y + key_b.height // 2

    # Frame 1: Hover start (dt = 0.04s)
    res1 = keyboard.update(index_tip=(cx, cy), thumb_tip=None, dt=0.04)
    assert res1 is None
    assert 0.0 < keyboard.dwell_progress < 1.0

    # Frame 2: Hover continues (dt = 0.04s, total 0.08s < 0.10s)
    res2 = keyboard.update(index_tip=(cx, cy), thumb_tip=None, dt=0.04)
    assert res2 is None
    assert 0.0 < keyboard.dwell_progress < 1.0

    # Frame 3: Hover reaches 0.12s (>= 0.10s dwell) -> trigger keystroke!
    res3 = keyboard.update(index_tip=(cx, cy), thumb_tip=None, dt=0.04)
    assert res3 == "b"
    assert keyboard.input_text == "b"
    assert keyboard.dwell_progress == 0.0


def test_text_editing_operations(keyboard: VirtualKeyboardEngine):
    """Verify caps lock, shift, space, backspace, and clear operations."""
    key_a = next(k for k in keyboard.keys if k.key_id == "A")
    key_caps = next(k for k in keyboard.keys if k.key_id == "CAPS")
    key_space = next(k for k in keyboard.keys if k.key_type == KeyType.SPACE)
    key_back = next(k for k in keyboard.keys if k.key_type == KeyType.BACKSPACE)
    key_clear = next(k for k in keyboard.keys if k.key_type == KeyType.CLEAR)

    # Type 'a'
    keyboard.execute_key(key_a)
    assert keyboard.input_text == "a"

    # Caps lock toggle ON
    keyboard.execute_key(key_caps)
    assert keyboard.caps_lock is True

    # Type 'A' with Caps Lock
    keyboard.execute_key(key_a)
    assert keyboard.input_text == "aA"

    # Caps lock toggle OFF
    keyboard.execute_key(key_caps)
    assert keyboard.caps_lock is False

    # Spacebar
    keyboard.execute_key(key_space)
    assert keyboard.input_text == "aA "

    # Backspace
    keyboard.execute_key(key_back)
    assert keyboard.input_text == "aA"

    # Clear
    keyboard.execute_key(key_clear)
    assert keyboard.input_text == ""


def test_hand_loss_resets_gesture_state(keyboard: VirtualKeyboardEngine):
    """Verify losing hand tracking resets cursor and dwell state cleanly."""
    key_a = next(k for k in keyboard.keys if k.key_id == "A")
    cx = key_a.x + key_a.width // 2
    cy = key_a.y + key_a.height // 2

    keyboard.update(index_tip=(cx, cy), thumb_tip=None, dt=0.033)
    assert keyboard.hovered_key is not None

    # Hand loss: index_tip is None
    keyboard.update(index_tip=None, thumb_tip=None, dt=0.033)
    assert keyboard.hovered_key is None
    assert keyboard.smoothed_cursor is None
    assert keyboard.raw_cursor is None


def test_keyboard_renderer():
    """Verify KeyboardRenderer draws keyboard overlay without mutating frame shape."""
    renderer = KeyboardRenderer()
    engine = VirtualKeyboardEngine(enabled=True)
    engine.recalculate_layout(640, 480)

    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    rendered = renderer.render(frame.copy(), engine)

    assert rendered.shape == frame.shape
    assert rendered.dtype == np.uint8

