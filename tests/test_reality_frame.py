"""Automated unit tests for RealityFrameManager, portal evaluation, and ghost rendering."""

import numpy as np
import pytest

from config import RealityFrameConfig
from src.reality_frame import PortalState, RealityFrameManager


@pytest.fixture
def frame_shape() -> tuple:
    return (480, 640)


@pytest.fixture
def dummy_frame(frame_shape: tuple) -> np.ndarray:
    frame = np.zeros((*frame_shape, 3), dtype=np.uint8)
    frame[:, :] = (50, 50, 50)
    return frame


@pytest.fixture
def dummy_bg(frame_shape: tuple) -> np.ndarray:
    bg = np.zeros((*frame_shape, 3), dtype=np.uint8)
    bg[:, :] = (200, 200, 200)
    return bg


def test_portal_evaluation_missing_hands(frame_shape: tuple):
    """Verify portal evaluation reports inactive when hands are missing."""
    rfm = RealityFrameManager()

    # Zero hands
    portal0 = rfm.evaluate_portal(None, None, frame_shape)
    assert not portal0.is_active
    assert "WAITING FOR 2 HANDS" in portal0.status_label

    # One hand
    portal1 = rfm.evaluate_portal((100, 100), None, frame_shape)
    assert not portal1.is_active


def test_portal_evaluation_undersized(frame_shape: tuple):
    """Verify portal is rejected when dimensions are smaller than minimum allowed."""
    rfm = RealityFrameManager(RealityFrameConfig(min_portal_width=40, min_portal_height=40))

    # Width: 15px (< 40)
    portal = rfm.evaluate_portal((100, 100), (115, 200), frame_shape)
    assert not portal.is_active
    assert "PORTAL TOO SMALL" in portal.status_label


def test_portal_evaluation_stability_and_activation(frame_shape: tuple):
    """Verify portal requires consecutive stable frames before becoming active."""
    rfm = RealityFrameManager()
    pt1 = (100, 100)
    pt2 = (300, 300)

    # Frame 1: Initial detection -> stabilizing
    p1 = rfm.evaluate_portal(pt1, pt2, frame_shape)
    assert not p1.is_active
    assert p1.width == 200
    assert p1.height == 200

    # Frame 2: Stable consecutive position -> active lock
    p2 = rfm.evaluate_portal((102, 101), (298, 301), frame_shape)
    assert p2.is_active
    assert "PORTAL LOCK ACQUIRED" in p2.status_label


def test_portal_evaluation_clamping(frame_shape: tuple):
    """Verify out-of-frame coordinates are clamped to video boundaries."""
    rfm = RealityFrameManager()
    # Coordinates exceeding frame width (640) and height (480)
    p = rfm.evaluate_portal((-50, -20), (700, 550), frame_shape)
    assert p.left == 0
    assert p.top == 0
    assert p.right == frame_shape[1] - 1
    assert p.bottom == frame_shape[0] - 1


def test_ghost_mode_without_background(dummy_frame: np.ndarray, frame_shape: tuple):
    """Verify ghost mode handles missing background plate gracefully without crashing."""
    rfm = RealityFrameManager()
    # Force active portal
    rfm.evaluate_portal((100, 100), (300, 300), frame_shape)
    portal = rfm.evaluate_portal((100, 100), (300, 300), frame_shape)
    assert portal.is_active

    out = rfm.apply_ghost_mode(dummy_frame.copy(), portal, background_plate=None)
    assert out.shape == dummy_frame.shape
    assert out.dtype == np.uint8


def test_ghost_mode_background_replacement(dummy_frame: np.ndarray, dummy_bg: np.ndarray, frame_shape: tuple):
    """Verify ghost mode replaces live frame ROI with stored background plate."""
    rfm = RealityFrameManager(RealityFrameConfig(enable_particles=False, enable_cyber_grid=False))
    rfm.evaluate_portal((100, 100), (300, 300), frame_shape)
    portal = rfm.evaluate_portal((100, 100), (300, 300), frame_shape)
    assert portal.is_active

    frame_copy = dummy_frame.copy()
    out = rfm.apply_ghost_mode(frame_copy, portal, background_plate=dummy_bg)

    assert out.shape == dummy_frame.shape
    # Center of portal must have background plate color (200, 200, 200), not dummy frame color (50, 50, 50)
    portal_center_pixel = out[200, 200, :].tolist()
    assert portal_center_pixel == [200, 200, 200]

    # Outside the portal, frame remains original (50, 50, 50)
    outside_pixel = out[20, 20, :].tolist()
    assert outside_pixel == [50, 50, 50]


def test_roi_mode_compositing(dummy_frame: np.ndarray, frame_shape: tuple):
    """Verify ROI portal mode composites processed ROI correctly."""
    rfm = RealityFrameManager(RealityFrameConfig(enable_particles=False, enable_cyber_grid=False))
    rfm.evaluate_portal((100, 100), (200, 200), frame_shape)
    portal = rfm.evaluate_portal((100, 100), (200, 200), frame_shape)
    assert portal.is_active

    # Create dummy processed ROI (100x100x3) colored bright red
    proc_roi = np.zeros((100, 100, 3), dtype=np.uint8)
    proc_roi[:, :] = (0, 0, 255)

    out = rfm.apply_roi_mode(dummy_frame.copy(), portal, proc_roi, filter_name="Test Red")
    assert out.shape == dummy_frame.shape
    # Inside portal
    assert out[150, 150, :].tolist() == [0, 0, 255]
