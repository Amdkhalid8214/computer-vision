"""Automated unit tests for ROI geometry, bounding boxes, clamping, and smoothing."""

import pytest
from src.roi_controller import ROIController, ROIResult


@pytest.fixture
def controller() -> ROIController:
    """Fixture providing a deterministic ROIController with smoothing disabled."""
    return ROIController(
        min_width=30,
        min_height=30,
        smoothing_enabled=False,
    )


@pytest.fixture
def smoothed_controller() -> ROIController:
    """Fixture providing an ROIController with smoothing enabled."""
    return ROIController(
        min_width=30,
        min_height=30,
        smoothing_enabled=True,
        smoothing_factor=0.5,
        discontinuity_threshold_pixels=100.0,
    )


def test_first_point_above_left_of_second_point(controller: ROIController):
    """Case 1: pt1 is at top-left (100, 100), pt2 is at bottom-right (300, 250)."""
    frame_shape = (480, 640)  # (H, W)
    res = controller.update(pt1=(100, 100), pt2=(300, 250), frame_shape=frame_shape)

    assert res.is_valid is True
    assert res.left == 100
    assert res.top == 100
    assert res.right == 300
    assert res.bottom == 250
    assert res.width == 200
    assert res.height == 150
    assert "2 HANDS DETECTED" in res.status_message


def test_first_point_below_right_of_second_point(controller: ROIController):
    """Case 2: pt1 is at bottom-right (400, 350), pt2 is at top-left (150, 100)."""
    frame_shape = (480, 640)
    res = controller.update(pt1=(400, 350), pt2=(150, 100), frame_shape=frame_shape)

    assert res.is_valid is True
    assert res.left == 150
    assert res.top == 100
    assert res.right == 400
    assert res.bottom == 350
    assert res.width == 250
    assert res.height == 250


def test_hands_cross_horizontally_and_vertically(controller: ROIController):
    """Case 3: Crossing hands (pt1 bottom-left, pt2 top-right) produces correct rectangle."""
    frame_shape = (720, 1280)
    # pt1 = (200, 500) [bottom-left], pt2 = (600, 150) [top-right]
    res = controller.update(pt1=(200, 500), pt2=(600, 150), frame_shape=frame_shape)

    assert res.is_valid is True
    assert res.left == 200
    assert res.top == 150
    assert res.right == 600
    assert res.bottom == 500
    assert res.width == 400
    assert res.height == 350


def test_coordinates_touch_frame_edges(controller: ROIController):
    """Case 4: Points exactly on image boundaries (0, 0) and (639, 479)."""
    frame_shape = (480, 640)
    res = controller.update(pt1=(0, 0), pt2=(639, 479), frame_shape=frame_shape)

    assert res.is_valid is True
    assert res.left == 0
    assert res.top == 0
    assert res.right == 639
    assert res.bottom == 479
    assert res.width == 639
    assert res.height == 479


def test_coordinates_exceed_frame_boundaries(controller: ROIController):
    """Case 5: Out-of-bounds coordinates are clamped to valid pixel boundaries."""
    frame_shape = (480, 640)  # W=640, H=480
    res = controller.update(pt1=(-50, -100), pt2=(800, 950), frame_shape=frame_shape)

    assert res.is_valid is True
    assert res.left == 0
    assert res.top == 0
    assert res.right == 639
    assert res.bottom == 479
    assert res.width == 639
    assert res.height == 479


def test_roi_width_is_zero(controller: ROIController):
    """Case 6: Identical x-coordinates produce zero width, rendering ROI invalid."""
    frame_shape = (480, 640)
    res = controller.update(pt1=(100, 50), pt2=(100, 200), frame_shape=frame_shape)

    assert res.is_valid is False
    assert res.width == 0
    assert res.status_message == "ROI ZERO DIMENSION"


def test_roi_height_is_zero(controller: ROIController):
    """Case 7: Identical y-coordinates produce zero height, rendering ROI invalid."""
    frame_shape = (480, 640)
    res = controller.update(pt1=(50, 100), pt2=(200, 100), frame_shape=frame_shape)

    assert res.is_valid is False
    assert res.height == 0
    assert res.status_message == "ROI ZERO DIMENSION"


def test_roi_smaller_than_configured_minimum(controller: ROIController):
    """Case 8: Box smaller than min_width (30) or min_height (30) is marked invalid."""
    frame_shape = (480, 640)
    # Width = 20 < 30, Height = 20 < 30
    res = controller.update(pt1=(100, 100), pt2=(120, 120), frame_shape=frame_shape)

    assert res.is_valid is False
    assert res.width == 20
    assert res.height == 20
    assert "ROI TOO SMALL" in res.status_message


def test_invalid_frame_dimensions(controller: ROIController):
    """Case 9: Zero or negative frame dimensions are handled safely without crashing."""
    res_zero = controller.update(pt1=(10, 10), pt2=(50, 50), frame_shape=(0, 0))
    assert res_zero.is_valid is False
    assert res_zero.status_message == "EMPTY FRAME"

    res_neg = controller.update(pt1=(10, 10), pt2=(50, 50), frame_shape=(-100, 640))
    assert res_neg.is_valid is False

    res_empty = controller.update(pt1=(10, 10), pt2=(50, 50), frame_shape=())
    assert res_empty.is_valid is False


def test_missing_hands(controller: ROIController):
    """Case 10: Zero or one detected hand."""
    frame_shape = (480, 640)

    # 0 hands
    res_none = controller.update(pt1=None, pt2=None, frame_shape=frame_shape)
    assert res_none.is_valid is False
    assert "WAITING FOR HANDS" in res_none.status_message

    # 1 hand (pt1 only)
    res_one1 = controller.update(pt1=(150, 150), pt2=None, frame_shape=frame_shape)
    assert res_one1.is_valid is False
    assert "1 HAND DETECTED" in res_one1.status_message

    # 1 hand (pt2 only)
    res_one2 = controller.update(pt1=None, pt2=(250, 250), frame_shape=frame_shape)
    assert res_one2.is_valid is False
    assert "1 HAND DETECTED" in res_one2.status_message


def test_smoothing_and_discontinuity_reset(smoothed_controller: ROIController):
    """Case 11: Smoothing averages consecutive positions and resets on big discontinuities."""
    frame_shape = (480, 640)

    # Frame 1: initial anchor at (100, 100) and (200, 200)
    r1 = smoothed_controller.update((100, 100), (200, 200), frame_shape)
    assert r1.left == 100
    assert r1.top == 100

    # Frame 2: minor movement to (110, 110) and (210, 210)
    # EMA with alpha=0.5: 0.5 * 110 + 0.5 * 100 = 105
    r2 = smoothed_controller.update((110, 110), (210, 210), frame_shape)
    assert r2.left == 105
    assert r2.top == 105

    # Frame 3: Discontinuity jump > 100px to (400, 300) and (550, 450)
    # Smoothing should reset directly to target to avoid lagging drag across screen
    r3 = smoothed_controller.update((400, 300), (550, 450), frame_shape)
    assert r3.left == 400
    assert r3.top == 300
    assert r3.right == 550
    assert r3.bottom == 450
