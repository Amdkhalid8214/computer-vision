"""Automated unit tests for AirWritingManager, stroke generation, undo/redo, and PNG export."""

import os
import numpy as np
import pytest

from src.air_writing import AirWritingManager, Stroke


@pytest.fixture
def manager(tmp_path) -> AirWritingManager:
    """Fixture providing an AirWritingManager pointing to a temporary export directory."""
    mgr = AirWritingManager()
    mgr.export_dir = str(tmp_path)
    return mgr


def test_air_writing_initial_state(manager: AirWritingManager):
    """Verify default tools, colors, and empty stroke stacks."""
    assert manager.tool == "PEN"
    assert manager.brush_size > 0
    assert len(manager.strokes) == 0
    assert len(manager.redo_stack) == 0
    assert not manager.is_drawing


def test_tool_and_color_selection(manager: AirWritingManager):
    """Verify tool switching, color setting, and palette cycling."""
    manager.set_tool("HIGHLIGHTER")
    assert manager.tool == "HIGHLIGHTER"

    manager.set_color("orange")
    assert manager.color_name == "orange"

    # Cycle color
    next_col = manager.cycle_color()
    assert next_col in ("white", "cyan", "green", "yellow", "orange", "red", "purple")

    # Brush size
    manager.set_brush_size(12)
    assert manager.brush_size == 12


def test_pinch_to_draw_creates_stroke(manager: AirWritingManager):
    """Verify pinch gesture creates continuous strokes and releasing pinch finalizes them."""
    # Frame 1: Index and thumb close together (distance 20px <= 40px threshold)
    manager.update(index_tip=(100, 100), thumb_tip=(110, 100), dt=0.033)
    assert manager.is_drawing is True
    assert manager.current_stroke is not None

    # Frame 2: Move index finger while still pinching
    manager.update(index_tip=(120, 100), thumb_tip=(130, 100), dt=0.033)
    assert manager.is_drawing is True
    assert len(manager.current_stroke.points) > 1

    # Frame 3: Release pinch (distance 100px)
    manager.update(index_tip=(150, 100), thumb_tip=(250, 100), dt=0.033)
    assert manager.is_drawing is False
    assert manager.current_stroke is None
    assert len(manager.strokes) == 1
    assert len(manager.strokes[0].points) > 1


def test_undo_and_redo(manager: AirWritingManager):
    """Verify undo and redo operations on drawn strokes."""
    # Add 2 dummy strokes
    s1 = Stroke(points=[(10, 10), (20, 20)], color=(255, 0, 0))
    s2 = Stroke(points=[(30, 30), (40, 40)], color=(0, 255, 0))
    manager.strokes.extend([s1, s2])

    assert len(manager.strokes) == 2

    # Undo
    assert manager.undo() is True
    assert len(manager.strokes) == 1
    assert len(manager.redo_stack) == 1

    # Redo
    assert manager.redo() is True
    assert len(manager.strokes) == 2
    assert len(manager.redo_stack) == 0


def test_clear_canvas(manager: AirWritingManager):
    """Verify clear canvas removes all strokes."""
    s1 = Stroke(points=[(10, 10), (20, 20)])
    manager.strokes.append(s1)
    assert len(manager.strokes) == 1

    manager.clear_canvas()
    assert len(manager.strokes) == 0


def test_eraser_stroke_removes_intersecting_points(manager: AirWritingManager):
    """Verify eraser tool splits or removes intersecting stroke points."""
    s = Stroke(points=[(10, 10), (12, 10), (14, 10), (50, 50), (52, 50)])
    manager.strokes.append(s)

    eraser = Stroke(points=[(11, 10)], size=10, tool="ERASER")
    manager._apply_eraser_stroke(eraser)

    # Stroke points around (11, 10) must be removed while (50, 50) points survive
    remaining_pts = [pt for stroke in manager.strokes for pt in stroke.points]
    assert (10, 10) not in remaining_pts
    assert (50, 50) in remaining_pts


def test_export_png_file(manager: AirWritingManager):
    """Verify exporting drawing to standard PNG and transparent PNG."""
    s = Stroke(points=[(100, 100), (200, 200)], color=(0, 255, 0), size=4)
    manager.strokes.append(s)

    # 1. Standard composite export
    file1 = manager.export_png()
    assert os.path.exists(file1)
    assert file1.endswith(".png")

    # 2. Transparent background export (4 channels RGBA)
    file2 = manager.export_png(transparent_bg=True)
    assert os.path.exists(file2)


def test_render_blackboard_and_overlay(manager: AirWritingManager):
    """Verify rendering in OVERLAY and BLACKBOARD modes returns valid 3-channel frame."""
    frame = np.full((480, 640, 3), 100, dtype=np.uint8)

    # Overlay mode
    out_overlay = manager.render(frame)
    assert out_overlay.shape == frame.shape
    assert out_overlay.dtype == np.uint8

    # Blackboard mode
    manager.toggle_canvas_mode()
    out_bb = manager.render(frame)
    assert out_bb.shape == frame.shape
    assert out_bb.dtype == np.uint8
    # Background in blackboard mode is dark slate
    assert out_bb[100, 100, 0] < 50

