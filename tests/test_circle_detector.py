"""Unit tests for geometric circle recognition and air-drawn gesture validation."""

import math
from typing import List, Tuple
import numpy as np
import pytest

from src.circle_detector import CircleDetector, CircleDetectionResult


def generate_circle_points(
    cx: int = 640,
    cy: int = 360,
    radius: float = 120.0,
    num_points: int = 60,
    start_angle: float = 0.0,
    sweep_angle: float = 2.0 * math.pi,
    noise_sigma: float = 0.0,
) -> List[Tuple[int, int]]:
    """Helper to synthesize a sequence of 2D coordinates for testing."""
    pts = []
    angles = np.linspace(start_angle, start_angle + sweep_angle, num_points)
    for theta in angles:
        r = radius + (np.random.normal(0, noise_sigma) if noise_sigma > 0 else 0.0)
        x = int(round(cx + r * math.cos(theta)))
        y = int(round(cy + r * math.sin(theta)))
        pts.append((x, y))
    return pts


def test_perfect_circle_detected():
    """Verify that a synthetic smooth closed circle is detected successfully."""
    detector = CircleDetector()
    pts = generate_circle_points(cx=500, cy=300, radius=100.0, num_points=60)
    result = detector.detect(pts, frame_shape=(720, 1280))

    assert result.is_circle is True
    assert abs(result.center[0] - 500) <= 2
    assert abs(result.center[1] - 300) <= 2
    assert abs(result.radius - 100.0) <= 3.0
    assert result.circularity_score > 0.95
    assert result.closure_gap_px <= 15.0
    assert result.angular_coverage_rad >= 5.2


def test_noisy_hand_drawn_circle_detected():
    """Verify that a circle with realistic hand tremor/noise is still detected."""
    detector = CircleDetector()
    pts = generate_circle_points(cx=400, cy=350, radius=90.0, num_points=50, noise_sigma=4.0)
    result = detector.detect(pts, frame_shape=(720, 1280))

    assert result.is_circle is True
    assert abs(result.center[0] - 400) <= 10
    assert abs(result.center[1] - 350) <= 10
    assert abs(result.radius - 90.0) <= 10.0


def test_reject_too_few_points():
    """Verify rejection when the stroke has fewer than min_points."""
    detector = CircleDetector(min_points=15)
    pts = [(100 + i * 5, 200) for i in range(10)]
    result = detector.detect(pts)

    assert result.is_circle is False
    assert "Stroke too short" in result.error_message


def test_reject_straight_line():
    """Verify rejection when the drawn stroke is a straight diagonal or horizontal line."""
    detector = CircleDetector()
    pts = [(100 + i * 10, 200 + i * 10) for i in range(30)]
    result = detector.detect(pts)

    assert result.is_circle is False
    assert "straight line" in result.error_message.lower()


def test_reject_too_small_circle():
    """Verify rejection when the circle radius is below min_radius_px."""
    detector = CircleDetector(min_radius_px=35.0)
    pts = generate_circle_points(cx=640, cy=360, radius=20.0, num_points=40)
    result = detector.detect(pts)

    assert result.is_circle is False
    assert "too small" in result.error_message.lower()


def test_reject_too_large_circle():
    """Verify rejection when the circle radius exceeds max allowed frame ratio."""
    detector = CircleDetector(max_radius_ratio=0.48)
    # 720 * 0.48 = ~345px
    pts = generate_circle_points(cx=640, cy=360, radius=400.0, num_points=80)
    result = detector.detect(pts, frame_shape=(720, 1280))

    assert result.is_circle is False
    assert "too large" in result.error_message.lower()


def test_reject_incomplete_half_circle():
    """Verify rejection of a half-circle arc (< 300 degrees)."""
    detector = CircleDetector()
    pts = generate_circle_points(cx=640, cy=360, radius=100.0, num_points=35, sweep_angle=math.pi)
    result = detector.detect(pts)

    assert result.is_circle is False
    # Will fail closure or angular coverage
    assert ("not closed" in result.error_message.lower()) or ("incomplete arc" in result.error_message.lower())


def test_reject_irregular_scribble():
    """Verify rejection of an irregular polygon / scribble with high radial variance."""
    detector = CircleDetector()
    pts = []
    # Make a sharp star-like or zig-zag pattern
    for i in range(40):
        r = 100.0 if (i % 2 == 0) else 25.0
        angle = i * (2.0 * math.pi / 40.0)
        pts.append((int(640 + r * math.cos(angle)), int(360 + r * math.sin(angle))))
    result = detector.detect(pts)

    assert result.is_circle is False
    assert "irregular shape" in result.error_message.lower()

