"""Webcam capture and frame management component.

Handles camera initialization, resolution negotiation, frame acquisition,
horizontal mirroring, error diagnostics, and resource release.
"""

import logging
from typing import Optional, Tuple
import cv2
import numpy as np

logger = logging.getLogger(__name__)


class CameraInitializationError(Exception):
    """Raised when the camera device cannot be opened or accessed."""
    pass


class CameraManager:
    """Manages OpenCV VideoCapture lifecycle and frame ingestion."""

    def __init__(
        self,
        camera_index: int = 0,
        requested_width: int = 1280,
        requested_height: int = 720,
        mirror_horizontal: bool = True,
    ) -> None:
        """Initialize camera settings without opening hardware immediately.

        Args:
            camera_index: Integer device index (typically 0 for built-in or default camera).
            requested_width: Desired capture width in pixels.
            requested_height: Desired capture height in pixels.
            mirror_horizontal: Whether to mirror frames horizontally for natural webcam interaction.
        """
        self.camera_index = camera_index
        self.requested_width = requested_width
        self.requested_height = requested_height
        self.mirror_horizontal = mirror_horizontal

        self._cap: Optional[cv2.VideoCapture] = None
        self._actual_width: int = 0
        self._actual_height: int = 0

    @property
    def is_opened(self) -> bool:
        """Return True if the camera is currently opened and available."""
        return self._cap is not None and self._cap.isOpened()

    @property
    def actual_dimensions(self) -> Tuple[int, int]:
        """Return the confirmed (width, height) of captured frames."""
        return self._actual_width, self._actual_height

    def open(self) -> None:
        """Open the video capture device and configure dimensions.

        Raises:
            CameraInitializationError: If the device fails to open or is inaccessible.
        """
        if self.is_opened:
            logger.info("Camera device %d is already open.", self.camera_index)
            return

        logger.info("Attempting to open camera device index %d...", self.camera_index)
        
        # On Windows, cv2.CAP_DSHOW or default backend
        cap = cv2.VideoCapture(self.camera_index)

        if not cap.isOpened():
            msg = (
                f"Failed to open webcam at device index {self.camera_index}.\n"
                "Actionable Troubleshooting Steps:\n"
                "1. Verify that your webcam is physically connected.\n"
                "2. Ensure no other application (Zoom, Teams, Discord, browser) is currently using the camera.\n"
                "3. Check Windows Settings -> Privacy & Security -> Camera to ensure desktop app access is granted.\n"
                "4. If using an external USB webcam, try specifying a different camera index (e.g., 1 or 2)."
            )
            logger.error(msg)
            raise CameraInitializationError(msg)

        # Request preferred resolution
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(self.requested_width))
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(self.requested_height))

        # Inspect actual hardware dimensions rather than assuming requested resolution
        actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Fallback inspection by reading one initial test frame if properties return 0
        if actual_w <= 0 or actual_h <= 0:
            ret, test_frame = cap.read()
            if ret and test_frame is not None:
                actual_h, actual_w = test_frame.shape[:2]
            else:
                cap.release()
                raise CameraInitializationError(
                    f"Camera device {self.camera_index} opened but failed to deliver an initial test frame."
                )

        self._cap = cap
        self._actual_width = actual_w
        self._actual_height = actual_h

        logger.info(
            "Camera %d initialized successfully. Requested: %dx%d, Negotiated: %dx%d.",
            self.camera_index,
            self.requested_width,
            self.requested_height,
            self._actual_width,
            self._actual_height,
        )

    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Read and validate the next frame from the camera.

        Returns:
            Tuple of (success_flag, frame_array). If read fails, returns (False, None).
        """
        if not self.is_opened:
            return False, None

        ret, frame = self._cap.read()

        if not ret or frame is None or frame.size == 0:
            logger.warning("Failed to retrieve valid frame from camera %d.", self.camera_index)
            return False, None

        # Verify frame dimensions match expected actuals
        h, w = frame.shape[:2]
        if w != self._actual_width or h != self._actual_height:
            self._actual_width = w
            self._actual_height = h

        # Mirror horizontally so user movement mirrors natural reflection
        if self.mirror_horizontal:
            frame = cv2.flip(frame, 1)

        return True, frame

    def release(self) -> None:
        """Safely release the camera device."""
        if self._cap is not None:
            logger.info("Releasing camera device %d.", self.camera_index)
            try:
                self._cap.release()
            except Exception as e:
                logger.warning("Exception during camera release: %s", e)
            finally:
                self._cap = None

    def __enter__(self) -> "CameraManager":
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.release()

