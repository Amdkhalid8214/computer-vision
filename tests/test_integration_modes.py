"""Automated integration tests for HandVision Studio mode transitions, HUD buttons, and gesture isolation."""

import numpy as np
import pytest

from config import AppConfig, AppMode, BackgroundStatus
from src.application import HandFilterApp
from src.hand_tracker import HandDetection
from src.hud import HUDManager


@pytest.fixture
def test_app() -> HandFilterApp:
    """Fixture providing an initialized HandFilterApp without opening camera."""
    config = AppConfig()
    app = HandFilterApp(config)
    return app


def test_app_mode_transitions(test_app: HandFilterApp):
    """Verify application switches between all 6 modes and resets mode-specific states."""
    assert test_app.mode == AppMode.STANDARD_FILTER

    # Transition to Mode 2: REALITYFRAME_ROI
    test_app.set_mode(AppMode.REALITYFRAME_ROI)
    assert test_app.mode == AppMode.REALITYFRAME_ROI
    assert not test_app.keyboard_engine.is_visible

    # Transition to Mode 3: REALITYFRAME_GHOST
    test_app.set_mode(AppMode.REALITYFRAME_GHOST)
    assert test_app.mode == AppMode.REALITYFRAME_GHOST

    # Transition to Mode 4: VIRTUAL_KEYBOARD
    test_app.set_mode(AppMode.VIRTUAL_KEYBOARD)
    assert test_app.mode == AppMode.VIRTUAL_KEYBOARD
    assert test_app.keyboard_engine.is_visible

    # Transition to Mode 5: AIR_WRITING
    test_app.set_mode(AppMode.AIR_WRITING)
    assert test_app.mode == AppMode.AIR_WRITING
    assert not test_app.keyboard_engine.is_visible

    # Transition to Mode 6: MYSTIC_MAGIC
    test_app.set_mode(AppMode.MYSTIC_MAGIC)
    assert test_app.mode == AppMode.MYSTIC_MAGIC

    # Transition to Mode 7: UNIVERSE_PORTAL
    test_app.set_mode(AppMode.UNIVERSE_PORTAL)
    assert test_app.mode == AppMode.UNIVERSE_PORTAL
    assert not test_app.keyboard_engine.is_visible

    # Transition back to Mode 1: STANDARD_FILTER
    test_app.set_mode(AppMode.STANDARD_FILTER)
    assert test_app.mode == AppMode.STANDARD_FILTER
    assert not test_app.keyboard_engine.is_visible


def test_hud_button_callbacks_and_mouse_click(test_app: HandFilterApp):
    """Verify interactive dashboard buttons trigger correct application actions on mouse click."""
    hud = test_app.hud_manager
    test_app._setup_hud_buttons(1280)

    assert len(hud.buttons) == 11

    # Mode 2
    btn_roi = next(b for b in hud.buttons if b.button_id == "mode_roi")
    cx = btn_roi.x + btn_roi.width // 2
    cy = btn_roi.y + btn_roi.height // 2
    assert hud.handle_mouse_click(cx, cy) is True
    assert test_app.mode == AppMode.REALITYFRAME_ROI

    # Mode 3
    btn_ghost = next(b for b in hud.buttons if b.button_id == "mode_ghost")
    assert hud.handle_mouse_click(btn_ghost.x + 5, btn_ghost.y + 5) is True
    assert test_app.mode == AppMode.REALITYFRAME_GHOST

    # Mode 4
    btn_kb = next(b for b in hud.buttons if b.button_id == "mode_kb")
    assert hud.handle_mouse_click(btn_kb.x + 5, btn_kb.y + 5) is True
    assert test_app.mode == AppMode.VIRTUAL_KEYBOARD

    # Mode 5: Air Write
    btn_write = next(b for b in hud.buttons if b.button_id == "mode_write")
    assert hud.handle_mouse_click(btn_write.x + 5, btn_write.y + 5) is True
    assert test_app.mode == AppMode.AIR_WRITING

    # Mode 6: Mystic Magic
    btn_magic = next(b for b in hud.buttons if b.button_id == "mode_magic")
    assert hud.handle_mouse_click(btn_magic.x + 5, btn_magic.y + 5) is True
    assert test_app.mode == AppMode.MYSTIC_MAGIC

    # Mode 7: Universe Portal
    btn_univ = next(b for b in hud.buttons if b.button_id == "mode_universe")
    assert hud.handle_mouse_click(btn_univ.x + 5, btn_univ.y + 5) is True
    assert test_app.mode == AppMode.UNIVERSE_PORTAL

    # Outside buttons click should return False
    assert not hud.handle_mouse_click(10, 400)


def test_filter_cycling_via_hud_action(test_app: HandFilterApp):
    """Verify next_filter button click cycles active filter."""
    initial_filter = test_app.filter_engine.active_key
    test_app._setup_hud_buttons(1280)

    btn_filter = next(b for b in test_app.hud_manager.buttons if b.button_id == "next_filter")
    test_app.hud_manager.handle_mouse_click(btn_filter.x + 5, btn_filter.y + 5)

    assert test_app.filter_engine.active_key != initial_filter


def test_gesture_isolation_across_modes(test_app: HandFilterApp):
    """Verify that in AIR_WRITING mode and MYSTIC_MAGIC mode, modes operate independently."""
    h1 = HandDetection(
        landmarks_norm=[(0.1, 0.1, 0.0)] * 21,
        landmarks_pixel=[(100, 100)] * 21,
        control_point=(100, 100),
        handedness="Right",
    )
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)

    # In Air Writing mode
    test_app.set_mode(AppMode.AIR_WRITING)
    out_write, _ = test_app._process_single_frame(frame.copy(), [h1], dt=0.033)
    assert out_write.shape == frame.shape
    assert test_app.mode == AppMode.AIR_WRITING

    # In Mystic Magic mode
    test_app.set_mode(AppMode.MYSTIC_MAGIC)
    out_magic, _ = test_app._process_single_frame(frame.copy(), [h1], dt=0.033)
    assert out_magic.shape == frame.shape
    assert test_app.mode == AppMode.MYSTIC_MAGIC

    # In Universe Portal mode
    test_app.set_mode(AppMode.UNIVERSE_PORTAL)
    out_univ, _ = test_app._process_single_frame(frame.copy(), [h1], dt=0.033)
    assert out_univ.shape == frame.shape
    assert test_app.mode == AppMode.UNIVERSE_PORTAL
