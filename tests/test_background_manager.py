"""Automated unit tests for BackgroundManager in RealityFrame Ghost Mode."""

import time
from unittest.mock import patch
import numpy as np
import pytest

from config import BackgroundStatus
from src.background_manager import BackgroundManager


@pytest.fixture
def dummy_frame() -> np.ndarray:
    """Fixture providing a synthetic 720p frame."""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    frame[:, :] = (120, 100, 80)
    return frame


def test_background_manager_initial_state():
    """Verify initial default state of background manager."""
    bm = BackgroundManager(capture_duration_sec=2.0, target_sample_frames=10)
    assert bm.status == BackgroundStatus.NOT_CAPTURED
    assert not bm.is_ready
    assert bm.background_plate is None
    assert bm.countdown_seconds == 0.0


def test_background_manager_trigger_capture():
    """Verify trigger_capture initializes countdown state."""
    bm = BackgroundManager(capture_duration_sec=2.0, target_sample_frames=10)
    bm.trigger_capture()
    assert bm.status == BackgroundStatus.COUNTDOWN
    assert bm.countdown_seconds > 0.0
    assert not bm.is_ready


def test_background_manager_capture_flow(dummy_frame: np.ndarray):
    """Verify full capture flow from countdown -> capturing -> ready with median computation."""
    bm = BackgroundManager(capture_duration_sec=1.0, target_sample_frames=5)

    with patch("time.perf_counter") as mock_time:
        # Start at t = 0.0
        mock_time.return_value = 0.0
        bm.trigger_capture()
        assert bm.status == BackgroundStatus.COUNTDOWN

        # At t = 0.4s (40% elapsed), ingesting frames should enter CAPTURING phase
        mock_time.return_value = 0.4
        f1 = dummy_frame.copy()
        f1[:, :] = (100, 100, 100)
        bm.ingest_frame(f1)
        assert bm.status == BackgroundStatus.CAPTURING

        # Feed multiple frames with slight noise
        mock_time.return_value = 0.6
        f2 = dummy_frame.copy()
        f2[:, :] = (110, 110, 110)
        bm.ingest_frame(f2)

        mock_time.return_value = 0.8
        f3 = dummy_frame.copy()
        f3[:, :] = (120, 120, 120)
        bm.ingest_frame(f3)

        mock_time.return_value = 0.9
        f4 = dummy_frame.copy()
        f4[:, :] = (130, 130, 130)
        bm.ingest_frame(f4)

        # Final frame at t = 1.0s (countdown expired) triggers completion
        mock_time.return_value = 1.0
        f5 = dummy_frame.copy()
        f5[:, :] = (140, 140, 140)
        bm.ingest_frame(f5)

        assert bm.status == BackgroundStatus.READY
        assert bm.is_ready
        assert bm.background_plate is not None
        assert bm.background_plate.shape == dummy_frame.shape
        assert bm.background_plate.dtype == np.uint8
        # Median of [100, 110, 120, 130, 140] is 120
        assert bm.background_plate[0, 0, 0] == 120


def test_background_manager_recapture_fallback(dummy_frame: np.ndarray):
    """Verify that if recapture fails, previous confirmed plate is restored."""
    bm = BackgroundManager(capture_duration_sec=1.0, target_sample_frames=5)

    # Establish initial valid plate
    bm._background_plate = dummy_frame.copy()
    bm._status = BackgroundStatus.READY
    assert bm.is_ready

    # Trigger recapture
    bm.trigger_capture()
    assert bm.status == BackgroundStatus.RECAPTURING
    assert bm._previous_valid_plate is not None

    # Simulate failure: finalize with zero frames in buffer
    bm._frame_buffer.clear()
    bm._finalize_background()

    # Should restore previous plate
    assert bm.status == BackgroundStatus.READY
    assert bm.is_ready
    assert bm.background_plate is not None
    np.testing.assert_array_equal(bm.background_plate, dummy_frame)


def test_background_manager_render_overlay(dummy_frame: np.ndarray):
    """Verify render_overlay produces a decorated frame without altering dimensions."""
    bm = BackgroundManager(capture_duration_sec=2.0)
    bm.trigger_capture()

    frame_copy = dummy_frame.copy()
    rendered = bm.render_overlay(frame_copy)

    assert rendered.shape == dummy_frame.shape
    assert rendered.dtype == np.uint8

