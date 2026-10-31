"""Automated tests for frame processing, compositing, and state transitions with mocked inputs."""

from unittest.mock import MagicMock
import numpy as np
import pytest

from config import AppConfig
from src.application import HandFilterApp
from src.hand_tracker import HandDetection


def make_dummy_hand(x: int, y: int, handedness: str = "Right") -> HandDetection:
    """Helper to generate a mock HandDetection with anchor point at (x, y)."""
    norm_pts = [(x / 640.0, y / 480.0, 0.0)] * 21
    pixel_pts = [(x, y)] * 21
    return HandDetection(
        landmarks_norm=norm_pts,
        landmarks_pixel=pixel_pts,
        control_point=(x, y),
        handedness=handedness,
    )


@pytest.fixture
def app_instance() -> HandFilterApp:
    """Fixture providing a HandFilterApp configured for 640x480 testing."""
    config = AppConfig()
    app = HandFilterApp(config)
    # Mock the hand tracker so tests run deterministically without camera or ML models
    app.hand_tracker = MagicMock()
    return app


def test_frame_processing_zero_hands(app_instance: HandFilterApp):
    """Zero hands detected: filter is not applied; outside pixels remain unchanged."""
    app_instance.hand_tracker.process_frame.return_value = []
    app_instance.hud.show_debug_metrics = False

    # Solid blue frame
    frame = np.full((480, 640, 3), fill_value=120, dtype=np.uint8)
    initial_copy = frame.copy()

    processed_frame, roi_result = app_instance.process_frame(frame, elapsed_sec=0.033)

    assert roi_result.is_valid is False
    assert "WAITING FOR HANDS" in roi_result.status_message
    assert processed_frame.shape == (480, 640, 3)

    # In the middle (away from HUD top/bottom banners), pixels must be untouched
    # Header is top 54px, bottom bar is bottom 30px
    np.testing.assert_array_equal(processed_frame[100:380, 100:540], initial_copy[100:380, 100:540])


def test_frame_processing_one_hand(app_instance: HandFilterApp):
    """One hand detected: filter is not applied; status guides user to show other hand."""
    app_instance.hand_tracker.process_frame.return_value = [make_dummy_hand(200, 200)]
    app_instance.hud.show_debug_metrics = False

    frame = np.full((480, 640, 3), fill_value=80, dtype=np.uint8)
    initial_copy = frame.copy()

    processed_frame, roi_result = app_instance.process_frame(frame, elapsed_sec=0.033)

    assert roi_result.is_valid is False
    assert "1 HAND DETECTED" in roi_result.status_message

    # Pixels away from overlays and control marker must remain untouched
    np.testing.assert_array_equal(processed_frame[300:400, 300:500], initial_copy[300:400, 300:500])


def test_frame_processing_two_hands_active_roi(app_instance: HandFilterApp):
    """Two hands detected: ROI filter is applied inside box, outside remains untouched."""
    # Hand 1 at (150, 150), Hand 2 at (350, 300)
    app_instance.hand_tracker.process_frame.return_value = [
        make_dummy_hand(150, 150, "Left"),
        make_dummy_hand(350, 300, "Right"),
    ]
    app_instance.hud.show_debug_metrics = False

    # Default filter is negative (color inversion)
    app_instance.filter_engine.active_key = "negative"

    # Fill frame with constant gray value (100)
    frame = np.full((480, 640, 3), fill_value=100, dtype=np.uint8)
    initial_copy = frame.copy()

    processed_frame, roi_result = app_instance.process_frame(frame, elapsed_sec=0.033)

    assert roi_result.is_valid is True
    assert roi_result.left == 150
    assert roi_result.top == 150
    assert roi_result.right == 350
    assert roi_result.bottom == 300
    assert "2 HANDS DETECTED" in roi_result.status_message

    # Inside ROI (excluding border line of 2px):
    # Inverted value of 100 is 255 - 100 = 155
    inner_roi = processed_frame[160:290, 160:340]
    expected_val = 255 - 100
    assert np.all(inner_roi == expected_val)

    # Outside ROI (bottom-right region away from banner):
    # Must remain original 100
    outside_region = processed_frame[350:420, 450:600]
    assert np.all(outside_region == 100)


def test_frame_processing_undersized_roi(app_instance: HandFilterApp):
    """Two hands too close together (< min_width or min_height): filter is deactivated."""
    # Hands only 10px apart
    app_instance.hand_tracker.process_frame.return_value = [
        make_dummy_hand(200, 200),
        make_dummy_hand(210, 210),
    ]

    frame = np.full((480, 640, 3), fill_value=150, dtype=np.uint8)

    # Spy on filter engine to ensure apply_active is not called
    app_instance.filter_engine.apply_active = MagicMock()

    processed_frame, roi_result = app_instance.process_frame(frame, elapsed_sec=0.033)

    assert roi_result.is_valid is False
    assert "ROI TOO SMALL" in roi_result.status_message

    # Filter must NEVER be called when ROI is undersized or invalid
    app_instance.filter_engine.apply_active.assert_not_called()


def test_frame_processing_filter_switching(app_instance: HandFilterApp):
    """Cycling filter updates the applied effect inside the ROI."""
    app_instance.hand_tracker.process_frame.return_value = [
        make_dummy_hand(100, 100),
        make_dummy_hand(300, 300),
    ]

    # Test with Negative filter
    app_instance.filter_engine.active_key = "negative"
    frame1 = np.full((480, 640, 3), fill_value=50, dtype=np.uint8)
    out1, _ = app_instance.process_frame(frame1, elapsed_sec=0.033)
    val_neg = out1[200, 200, 0]
    assert val_neg == 255 - 50

    # Switch to Canny filter
    app_instance.filter_engine.active_key = "canny"
    frame2 = np.full((480, 640, 3), fill_value=50, dtype=np.uint8)
    out2, _ = app_instance.process_frame(frame2, elapsed_sec=0.033)
    val_canny = out2[200, 200, 0]
    # Solid constant image has zero gradients, so Canny output is 0
    assert val_canny == 0

    assert val_neg != val_canny


def test_frame_processing_tracking_loss_recovery(app_instance: HandFilterApp):
    """Test rapid transitions (2 hands -> 0 hands -> 2 hands) maintain stability."""
    frame = np.full((480, 640, 3), fill_value=128, dtype=np.uint8)

    # 1. Active tracking
    app_instance.hand_tracker.process_frame.return_value = [
        make_dummy_hand(100, 100),
        make_dummy_hand(250, 250),
    ]
    _, r1 = app_instance.process_frame(frame.copy(), elapsed_sec=0.033)
    assert r1.is_valid is True

    # 2. Tracking lost (0 hands)
    app_instance.hand_tracker.process_frame.return_value = []
    _, r2 = app_instance.process_frame(frame.copy(), elapsed_sec=0.033)
    assert r2.is_valid is False
    assert "WAITING FOR HANDS" in r2.status_message

    # 3. Tracking recovered
    app_instance.hand_tracker.process_frame.return_value = [
        make_dummy_hand(120, 120),
        make_dummy_hand(280, 280),
    ]
    _, r3 = app_instance.process_frame(frame.copy(), elapsed_sec=0.033)
    assert r3.is_valid is True
