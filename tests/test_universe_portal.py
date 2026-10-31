"""Unit tests for RealityFrame Cinematic Universe Portal Manager and Masked Compositing."""

import math
import numpy as np
import pytest

from config import UniversePortalConfig
from src.hand_tracker import HandDetection
from src.universe_portal import UniversePortalManager, PortalLifecycle


def make_hand(
    index_tip=(640, 360),
    thumb_tip=(640, 360),
    pinch_dist=10.0,
    is_pointing=False,
    handedness="Right",
) -> HandDetection:
    """Helper to construct a mock HandDetection."""
    wrist = (index_tip[0], index_tip[1] + 120)
    pts = [wrist] * 21
    pts[0] = wrist
    pts[8] = index_tip
    if thumb_tip == index_tip and pinch_dist > 0:
        pts[4] = (int(index_tip[0] + pinch_dist), index_tip[1])
    else:
        pts[4] = thumb_tip

    if is_pointing:
        # Index extended
        pts[6] = (index_tip[0], index_tip[1] + 50)
        # Middle, ring, pinky curled
        pts[10] = (index_tip[0] + 20, index_tip[1] + 60)
        pts[12] = (index_tip[0] + 20, index_tip[1] + 75)
        pts[14] = (index_tip[0] + 40, index_tip[1] + 60)
        pts[16] = (index_tip[0] + 40, index_tip[1] + 75)
        pts[18] = (index_tip[0] + 60, index_tip[1] + 60)
        pts[20] = (index_tip[0] + 60, index_tip[1] + 75)

    return HandDetection(
        landmarks_norm=[(p[0] / 1280.0, p[1] / 720.0, 0.0) for p in pts],
        landmarks_pixel=pts,
        control_point=index_tip,
        handedness=handedness,
    )


def test_portal_initial_state():
    """Verify universe portal manager initializes in IDLE state with default parameters."""
    manager = UniversePortalManager()
    assert manager.state == PortalLifecycle.IDLE
    assert len(manager.drawn_stroke) == 0
    assert manager.universe_engine is not None


def test_portal_drawing_and_circle_activation():
    """Verify drawing a circular stroke transitions portal through CHARGING, DISTORTING, OPENING, to ACTIVE."""
    manager = UniversePortalManager()
    cx, cy, r = 600, 350, 100.0

    # Simulate drawing pinch trajectory
    num_pts = 40
    for i in range(num_pts):
        theta = i * (2.0 * math.pi / num_pts)
        px = int(cx + r * math.cos(theta))
        py = int(cy + r * math.sin(theta))
        hand_drawing = make_hand(index_tip=(px, py), thumb_tip=(px, py), pinch_dist=15.0)
        manager.update([hand_drawing], dt=0.033, frame_shape=(720, 1280))
        if i == 15:
            # Halfway through circle, actively drawing
            assert manager.state == PortalLifecycle.DRAWING
            assert len(manager.drawn_stroke) >= 10

    # Real-time circle detector activates automatically when circle is closed
    assert manager.state == PortalLifecycle.CHARGING
    assert abs(manager.portal_center[0] - cx) <= 5
    assert abs(manager.portal_center[1] - cy) <= 5
    assert abs(manager.portal_radius - r) <= 5.0

    # Advance time to transition through DISTORTING -> OPENING -> ACTIVE
    # (dt is clamped to 0.1 in update() for real-time video stability)
    for _ in range(int(manager.charge_duration_sec / 0.08) + 2):
        manager.update([], dt=0.08)
    assert manager.state == PortalLifecycle.DISTORTING

    for _ in range(int(manager.distortion_duration_sec / 0.08) + 2):
        manager.update([], dt=0.08)
    assert manager.state == PortalLifecycle.OPENING

    for _ in range(int(manager.opening_duration_sec / 0.08) + 2):
        manager.update([], dt=0.08)
    assert manager.state == PortalLifecycle.ACTIVE


def test_portal_masked_compositing_isolation():
    """Verify that inside the portal circle, the 3D universe is rendered, but outside pixels remain untouched."""
    manager = UniversePortalManager()
    manager.state = PortalLifecycle.ACTIVE
    manager.portal_center = (640, 360)
    manager.portal_radius = 120.0

    # Create dummy green frame
    orig_frame = np.full((720, 1280, 3), (0, 180, 0), dtype=np.uint8)
    rendered_frame = manager.render(orig_frame.copy())

    # Corner pixel outside portal must be completely identical to original frame
    assert np.array_equal(rendered_frame[10, 10], orig_frame[10, 10])
    assert np.array_equal(rendered_frame[700, 1200], orig_frame[700, 1200])

    # Center pixel inside portal must be modified by the 3D universe render
    assert not np.array_equal(rendered_frame[360, 640], orig_frame[360, 640])


def test_portal_closing_lifecycle():
    """Verify close_portal transitions from ACTIVE to CLOSING."""
    manager = UniversePortalManager()
    manager.state = PortalLifecycle.ACTIVE
    manager.close_portal()

    assert manager.state == PortalLifecycle.CLOSING

    # Reset
    manager.reset_to_idle()
    assert manager.state == PortalLifecycle.IDLE
    assert len(manager.drawn_stroke) == 0


def test_preset_and_scale_toggles():
    """Verify quality presets and scale mode switches."""
    manager = UniversePortalManager()

    manager.set_quality_preset("PERFORMANCE")
    assert manager.preset == "PERFORMANCE"
    assert manager.universe_engine.render_width == 384

    manager.set_quality_preset("CINEMATIC")
    assert manager.preset == "CINEMATIC"
    assert manager.universe_engine.render_width == 640

    # Scale mode toggle
    new_scale = manager.toggle_scale_mode()
    assert new_scale == "SCIENTIFIC"
    assert manager.universe_engine.scale_mode == "SCIENTIFIC"


def test_portal_activation_with_pointing_gesture():
    """Verify circle activation works using pointing index finger without pinch."""
    manager = UniversePortalManager()
    cx, cy, r = 500, 300, 90.0

    # Simulate pointing finger (pinch_dist = 90.0 is not pinching)
    num_pts = 40
    for i in range(num_pts):
        theta = i * (2.0 * math.pi / num_pts)
        px = int(cx + r * math.cos(theta))
        py = int(cy + r * math.sin(theta))
        hand_pointing = make_hand(
            index_tip=(px, py),
            thumb_tip=(px + 80, py + 80),
            pinch_dist=90.0,
            is_pointing=True,
        )
        manager.update([hand_pointing], dt=0.033, frame_shape=(720, 1280))

    # Circle completed via pointing
    assert manager.state == PortalLifecycle.CHARGING
    assert abs(manager.portal_center[0] - cx) <= 5
    assert abs(manager.portal_center[1] - cy) <= 5
