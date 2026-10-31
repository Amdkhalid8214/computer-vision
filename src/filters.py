"""Filter engine component providing distinct real-time visual effects.

Implements an extensible registry/strategy architecture for Region of Interest (ROI)
filtering, including:
1. Color Inversion (Negative)
2. Thermal Vision (Inferno, Jet, Ocean false-color maps)
3. Sketch Vision (Pencil sketch via color dodge)
4. Cybernetic Edge Detection (Canny with neon accents)
5. Grayscale Vision
6. Pixelated / Mosaic Vision
7. Custom Composite Filter Pipeline
"""

from abc import ABC, abstractmethod
import logging
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np

logger = logging.getLogger(__name__)


class FilterError(Exception):
    """Raised when an image filter receives incompatible or invalid input."""
    pass


class BaseFilter(ABC):
    """Abstract base class representing an image processing filter strategy."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Display name of the filter."""
        pass

    @property
    @abstractmethod
    def key(self) -> str:
        """Unique identifier key for registry lookups."""
        pass

    @abstractmethod
    def apply(self, roi: np.ndarray) -> np.ndarray:
        """Apply the effect to the input ROI image and return the processed result.

        Args:
            roi: Valid 3-channel BGR uint8 NumPy image array.

        Returns:
            Processed 3-channel BGR uint8 NumPy image array with identical dimensions.
        """
        pass

    def validate_roi(self, roi: np.ndarray) -> None:
        """Validate input array dimensions and properties before processing."""
        if roi is None:
            raise FilterError("ROI array is None.")
        if not isinstance(roi, np.ndarray):
            raise FilterError(f"ROI must be a numpy ndarray, got {type(roi)}.")
        if roi.size == 0:
            raise FilterError("ROI has size 0.")
        if roi.ndim != 3 or roi.shape[2] != 3:
            raise FilterError(f"ROI must have shape (H, W, 3), got {roi.shape}.")
        if roi.shape[0] <= 0 or roi.shape[1] <= 0:
            raise FilterError(f"ROI dimensions must be positive, got {roi.shape[:2]}.")


class NegativeFilter(BaseFilter):
    """Color Inversion (Negative) filter.

    Inverts RGB color values using bitwise NOT operation: pixel' = 255 - pixel.
    """

    @property
    def name(self) -> str:
        return "Color Inversion (Negative)"

    @property
    def key(self) -> str:
        return "negative"

    def apply(self, roi: np.ndarray) -> np.ndarray:
        self.validate_roi(roi)
        return cv2.bitwise_not(roi)


class ThermalFilter(BaseFilter):
    """Thermal False-Color Color Map filter.

    Supports multiple thermal colormaps: INFERNO (default), JET, and OCEAN.
    Note: Produces a false-color intensity visualization, not physical temperature.
    """

    def __init__(self, colormap_name: str = "inferno") -> None:
        self.colormap_name = colormap_name.lower()
        self._colormaps = {
            "inferno": cv2.COLORMAP_INFERNO,
            "jet": cv2.COLORMAP_JET,
            "ocean": cv2.COLORMAP_OCEAN,
        }

    @property
    def name(self) -> str:
        return f"Thermal Vision ({self.colormap_name.capitalize()})"

    @property
    def key(self) -> str:
        if self.colormap_name == "inferno":
            return "thermal"
        return f"thermal_{self.colormap_name}"

    def apply(self, roi: np.ndarray) -> np.ndarray:
        self.validate_roi(roi)
        cmap = self._colormaps.get(self.colormap_name, cv2.COLORMAP_INFERNO)
        return cv2.applyColorMap(roi, cmap)


class SketchFilter(BaseFilter):
    """Pencil-Sketch effect filter.

    Converts ROI to grayscale, inverts it, applies Gaussian blur, and applies
    color dodge image division: result = min(255, (gray * 256) / (255 - blur + 1)).
    """

    @property
    def name(self) -> str:
        return "Pencil Sketch"

    @property
    def key(self) -> str:
        return "sketch"

    def apply(self, roi: np.ndarray) -> np.ndarray:
        self.validate_roi(roi)
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        inverted = cv2.bitwise_not(gray)
        # Large kernel blur creates smooth pencil shading
        blurred = cv2.GaussianBlur(inverted, (21, 21), sigmaX=0, sigmaY=0)
        # Color dodge division
        sketch_gray = cv2.divide(gray, 255 - blurred, scale=256)
        # Convert back to 3-channel BGR
        return cv2.cvtColor(sketch_gray, cv2.COLOR_GRAY2BGR)


class CannyEdgeFilter(BaseFilter):
    """Cybernetic Edge Detection filter.

    Applies Gaussian noise filtering, gradient edge extraction with cv2.Canny,
    and colors detected edges with neon cyan/green against a dark backdrop.
    """

    def __init__(
        self,
        low_threshold: int = 50,
        high_threshold: int = 150,
        aperture_size: int = 3,
        blur_kernel: tuple = (3, 3),
        edge_color: Tuple[int, int, int] = (255, 255, 255),
    ) -> None:
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold
        self.aperture_size = aperture_size
        self.blur_kernel = blur_kernel
        self.edge_color = edge_color

    @property
    def name(self) -> str:
        return "Cyber Edge Detection"

    @property
    def key(self) -> str:
        return "canny"

    def apply(self, roi: np.ndarray) -> np.ndarray:
        self.validate_roi(roi)
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        if self.blur_kernel[0] > 0 and self.blur_kernel[1] > 0:
            gray = cv2.GaussianBlur(gray, self.blur_kernel, 0)
        edges = cv2.Canny(
            gray,
            self.low_threshold,
            self.high_threshold,
            apertureSize=self.aperture_size,
        )

        # Create dark background
        h, w = roi.shape[:2]
        output = np.zeros((h, w, 3), dtype=np.uint8)
        # Colorize the edges using edge_color mask
        edge_mask = edges > 0
        output[edge_mask] = self.edge_color
        return output


class GrayscaleFilter(BaseFilter):
    """Grayscale Vision filter.

    Converts ROI to grayscale and maps it back to 3-channel BGR format.
    """

    @property
    def name(self) -> str:
        return "Grayscale Vision"

    @property
    def key(self) -> str:
        return "grayscale"

    def apply(self, roi: np.ndarray) -> np.ndarray:
        self.validate_roi(roi)
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


class PixelatedFilter(BaseFilter):
    """Pixelated / Digital Mosaic filter.

    Downsamples the ROI and scales it back up using nearest-neighbor interpolation.
    """

    def __init__(self, block_size: int = 14) -> None:
        self.block_size = max(2, block_size)

    @property
    def name(self) -> str:
        return f"Pixelated Vision ({self.block_size}px)"

    @property
    def key(self) -> str:
        return "pixelated"

    def apply(self, roi: np.ndarray) -> np.ndarray:
        self.validate_roi(roi)
        h, w = roi.shape[:2]
        # Calculate downsampled dimensions
        small_w = max(1, w // self.block_size)
        small_h = max(1, h // self.block_size)
        # Downsample
        small = cv2.resize(roi, (small_w, small_h), interpolation=cv2.INTER_LINEAR)
        # Upscale back to original ROI dimensions
        return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)


class CustomPipelineFilter(BaseFilter):
    """Composite filter executing a sequence of filters in order."""

    def __init__(self, pipeline_key: str, name: str, filters: List[BaseFilter]) -> None:
        self._key = pipeline_key
        self._name = name
        self.pipeline = filters

    @property
    def name(self) -> str:
        return self._name

    @property
    def key(self) -> str:
        return self._key

    def apply(self, roi: np.ndarray) -> np.ndarray:
        self.validate_roi(roi)
        current = roi
        for f in self.pipeline:
            current = f.apply(current)
        return current


class FilterEngine:
    """Registry and dispatcher for image processing filters with bi-directional cycling."""

    def __init__(self) -> None:
        self._filters: Dict[str, BaseFilter] = {}
        self._keys_order: List[str] = []
        self._active_key: str = ""

        # Register standard default filters
        self.register(NegativeFilter())
        self.register(CannyEdgeFilter())
        self.register(ThermalFilter("inferno"))
        self.register(SketchFilter())
        self.register(ThermalFilter("jet"))
        self.register(ThermalFilter("ocean"))
        self.register(GrayscaleFilter())
        self.register(PixelatedFilter())

        # Set default active filter
        if self._keys_order:
            self._active_key = self._keys_order[0]

    def register(self, filter_instance: BaseFilter) -> None:
        """Register a new filter into the registry."""
        key = filter_instance.key.lower()
        if key not in self._filters:
            self._keys_order.append(key)
        self._filters[key] = filter_instance

    @property
    def available_filters(self) -> List[str]:
        """List of available filter keys."""
        return list(self._keys_order)

    @property
    def filter_items(self) -> List[Tuple[str, str]]:
        """List of (key, display_name) tuples for UI menus."""
        return [(k, self._filters[k].name) for k in self._keys_order]

    @property
    def active_key(self) -> str:
        """Current active filter key."""
        return self._active_key

    @active_key.setter
    def active_key(self, key: str) -> None:
        """Set the active filter key with validation."""
        normalized_key = key.lower()
        # Support aliases
        if normalized_key == "thermal":
            normalized_key = "thermal_inferno"

        if normalized_key in self._filters:
            self._active_key = normalized_key
        else:
            logger.warning("Filter key '%s' not recognized. Remaining on '%s'.", key, self._active_key)

    @property
    def active_name(self) -> str:
        """Display name of the active filter."""
        active_filter = self._filters.get(self._active_key)
        return active_filter.name if active_filter else "None"

    def cycle_next(self) -> str:
        """Cycle to the next available filter and return its key."""
        if not self._keys_order:
            return ""
        try:
            curr_idx = self._keys_order.index(self._active_key)
            next_idx = (curr_idx + 1) % len(self._keys_order)
        except ValueError:
            next_idx = 0
        self._active_key = self._keys_order[next_idx]
        return self._active_key

    def cycle_prev(self) -> str:
        """Cycle to the previous available filter and return its key."""
        if not self._keys_order:
            return ""
        try:
            curr_idx = self._keys_order.index(self._active_key)
            prev_idx = (curr_idx - 1) % len(self._keys_order)
        except ValueError:
            prev_idx = 0
        self._active_key = self._keys_order[prev_idx]
        return self._active_key

    def apply_active(self, roi: np.ndarray) -> np.ndarray:
        """Apply the currently active filter to the provided ROI image array."""
        current_filter = self._filters.get(self._active_key)
        if current_filter is None:
            raise FilterError(f"No active filter registered for key '{self._active_key}'.")
        return current_filter.apply(roi)
