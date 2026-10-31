"""Automated unit tests for the visual filter engine, strategy classes, and registry."""

import numpy as np
import pytest

from src.filters import (
    BaseFilter,
    CannyEdgeFilter,
    CustomPipelineFilter,
    FilterEngine,
    FilterError,
    GrayscaleFilter,
    NegativeFilter,
    PixelatedFilter,
    SketchFilter,
    ThermalFilter,
)


@pytest.fixture
def sample_roi() -> np.ndarray:
    """Fixture providing a synthetic 3-channel BGR image ROI (100x120x3 uint8)."""
    np.random.seed(42)
    return np.random.randint(0, 256, size=(100, 120, 3), dtype=np.uint8)


def test_negative_filter_output_and_immutability(sample_roi: np.ndarray):
    """Verify NegativeFilter inverts colors, preserves shape/dtype, and preserves input array."""
    filter_inst = NegativeFilter()
    roi_clone = sample_roi.copy()

    output = filter_inst.apply(sample_roi)

    assert output.shape == sample_roi.shape
    assert output.dtype == np.uint8
    expected = 255 - sample_roi
    np.testing.assert_array_equal(output, expected)
    np.testing.assert_array_equal(sample_roi, roi_clone)


def test_canny_edge_filter_output_dimensions(sample_roi: np.ndarray):
    """Verify Canny filter returns 3 channels BGR with exact identical dimensions."""
    canny = CannyEdgeFilter(low_threshold=50, high_threshold=150)
    roi_clone = sample_roi.copy()

    output = canny.apply(sample_roi)

    assert output.shape == sample_roi.shape
    assert output.dtype == np.uint8
    assert output.ndim == 3
    assert output.shape[2] == 3
    unique_vals = set(np.unique(output))
    assert unique_vals.issubset({0, 255})
    np.testing.assert_array_equal(sample_roi, roi_clone)


def test_cyber_canny_colored_edges(sample_roi: np.ndarray):
    """Verify Canny filter with neon colored edge styling."""
    neon_color = (0, 255, 128)
    canny = CannyEdgeFilter(edge_color=neon_color)
    output = canny.apply(sample_roi)

    assert output.shape == sample_roi.shape
    assert output.dtype == np.uint8
    # Background must be 0, edges should contain neon_color components
    assert output[0, 0, :].tolist() in ([0, 0, 0], list(neon_color))


def test_thermal_filter_variants(sample_roi: np.ndarray):
    """Verify Thermal filters (inferno, jet, ocean) return 3-channel images."""
    for colormap in ["inferno", "jet", "ocean"]:
        thermal = ThermalFilter(colormap)
        roi_clone = sample_roi.copy()
        output = thermal.apply(sample_roi)

        assert output.shape == sample_roi.shape
        assert output.dtype == np.uint8
        assert output.shape[2] == 3
        np.testing.assert_array_equal(sample_roi, roi_clone)


def test_sketch_filter(sample_roi: np.ndarray):
    """Verify Pencil Sketch filter converts ROI to shaded grayscale in 3-channel BGR."""
    sketch = SketchFilter()
    roi_clone = sample_roi.copy()
    output = sketch.apply(sample_roi)

    assert output.shape == sample_roi.shape
    assert output.dtype == np.uint8
    assert output.shape[2] == 3
    # In BGR grayscale output, B == G == R
    np.testing.assert_array_equal(output[:, :, 0], output[:, :, 1])
    np.testing.assert_array_equal(output[:, :, 1], output[:, :, 2])
    np.testing.assert_array_equal(sample_roi, roi_clone)


def test_grayscale_filter(sample_roi: np.ndarray):
    """Verify GrayscaleFilter produces balanced 3-channel luminance."""
    gray = GrayscaleFilter()
    output = gray.apply(sample_roi)

    assert output.shape == sample_roi.shape
    assert output.dtype == np.uint8
    np.testing.assert_array_equal(output[:, :, 0], output[:, :, 1])
    np.testing.assert_array_equal(output[:, :, 1], output[:, :, 2])


def test_pixelated_filter(sample_roi: np.ndarray):
    """Verify PixelatedFilter produces mosaic blocks with preserved outer dimensions."""
    pix = PixelatedFilter(block_size=10)
    output = pix.apply(sample_roi)

    assert output.shape == sample_roi.shape
    assert output.dtype == np.uint8
    # Within a 10x10 block, pixels should be uniform
    block = output[0:10, 0:10, :]
    for c in range(3):
        assert np.all(block[:, :, c] == block[0, 0, c])


def test_custom_pipeline_filter(sample_roi: np.ndarray):
    """Verify CustomPipelineFilter chains multiple filters sequentially."""
    pipeline = CustomPipelineFilter(
        pipeline_key="custom_test",
        name="Test Pipeline",
        filters=[GrayscaleFilter(), NegativeFilter()],
    )
    output = pipeline.apply(sample_roi)

    assert output.shape == sample_roi.shape
    assert output.dtype == np.uint8
    # Inverted grayscale
    np.testing.assert_array_equal(output[:, :, 0], output[:, :, 1])


@pytest.mark.parametrize(
    "filter_cls",
    [NegativeFilter, CannyEdgeFilter, ThermalFilter, SketchFilter, GrayscaleFilter, PixelatedFilter],
)
def test_filter_invalid_inputs(filter_cls):
    """Verify all filters raise FilterError on invalid input types or shapes."""
    f = filter_cls()

    # None
    with pytest.raises(FilterError):
        f.apply(None)

    # Empty array
    with pytest.raises(FilterError):
        f.apply(np.empty((0, 0, 3), dtype=np.uint8))

    # Single-channel grayscale (2D instead of 3D)
    with pytest.raises(FilterError):
        f.apply(np.zeros((50, 50), dtype=np.uint8))

    # 4-channel BGRA (wrong channel count)
    with pytest.raises(FilterError):
        f.apply(np.zeros((50, 50, 4), dtype=np.uint8))

    # Zero height or width
    with pytest.raises(FilterError):
        f.apply(np.zeros((0, 50, 3), dtype=np.uint8))


def test_filter_engine_cycling_and_dispatch(sample_roi: np.ndarray):
    """Verify FilterEngine registry, bi-directional cycling order, and active filter application."""
    engine = FilterEngine()

    filters = engine.available_filters
    assert len(filters) >= 6
    assert "negative" in filters
    assert "canny" in filters
    assert "thermal" in filters
    assert "sketch" in filters
    assert "grayscale" in filters
    assert "pixelated" in filters

    # Initial active filter is negative
    assert engine.active_key == "negative"
    out1 = engine.apply_active(sample_roi)
    assert out1.shape == sample_roi.shape

    # Forward cycling through all filters
    visited = [engine.active_key]
    for _ in range(len(filters)):
        nxt = engine.cycle_next()
        visited.append(nxt)
        assert engine.active_key == nxt
        out = engine.apply_active(sample_roi)
        assert out.shape == sample_roi.shape

    # First cycle after full round wraps back to starting filter
    assert visited[-1] == "negative"

    # Backward cycling
    prev_key = engine.cycle_prev()
    assert prev_key == filters[-1]
    assert engine.active_key == filters[-1]
