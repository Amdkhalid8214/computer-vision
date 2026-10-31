"""Automated unit tests for CameraManager, error handling, and resource release."""

from unittest.mock import MagicMock, patch
import numpy as np
import pytest

from src.camera import CameraInitializationError, CameraManager


def test_camera_initialization_failure():
    """Verify CameraInitializationError is raised when device fails to open."""
    with patch("cv2.VideoCapture") as mock_cap_cls:
        mock_instance = MagicMock()
        mock_instance.isOpened.return_value = False
        mock_cap_cls.return_value = mock_instance

        cam = CameraManager(camera_index=99)
        with pytest.raises(CameraInitializationError) as exc_info:
            cam.open()

        assert "Failed to open webcam at device index 99" in str(exc_info.value)
        assert "Troubleshooting Steps" in str(exc_info.value)


def test_camera_successful_open_and_dimension_negotiation():
    """Verify camera negotiates and reads actual dimensions."""
    with patch("cv2.VideoCapture") as mock_cap_cls:
        mock_instance = MagicMock()
        mock_instance.isOpened.return_value = True
        mock_instance.get.side_effect = lambda prop: 1280.0 if prop == 3 else 720.0
        mock_cap_cls.return_value = mock_instance

        cam = CameraManager(camera_index=0, requested_width=1280, requested_height=720)
        cam.open()

        assert cam.is_opened is True
        assert cam.actual_dimensions == (1280, 720)


def test_camera_frame_read_success_and_mirroring():
    """Verify successful frame read applies horizontal flip."""
    with patch("cv2.VideoCapture") as mock_cap_cls:
        mock_instance = MagicMock()
        mock_instance.isOpened.return_value = True
        mock_instance.get.side_effect = lambda prop: 640.0 if prop == 3 else 480.0

        # Create an asymmetrical synthetic frame: column 0 is white (255), column 639 is black (0)
        fake_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        fake_frame[:, 0] = 255
        mock_instance.read.return_value = (True, fake_frame)
        mock_cap_cls.return_value = mock_instance

        cam = CameraManager(camera_index=0, mirror_horizontal=True)
        cam.open()

        success, frame = cam.read_frame()
        assert success is True
        assert frame is not None
        # With horizontal mirror (cv2.flip(frame, 1)), the white line at col 0 moves to col 639
        assert np.all(frame[:, 639] == 255)
        assert np.all(frame[:, 0] == 0)


def test_camera_frame_read_failure_handling():
    """Verify failed frame read returns (False, None) gracefully without throwing."""
    with patch("cv2.VideoCapture") as mock_cap_cls:
        mock_instance = MagicMock()
        mock_instance.isOpened.return_value = True
        mock_instance.get.side_effect = lambda prop: 640.0 if prop == 3 else 480.0
        mock_instance.read.return_value = (False, None)
        mock_cap_cls.return_value = mock_instance

        cam = CameraManager(camera_index=0)
        cam.open()

        success, frame = cam.read_frame()
        assert success is False
        assert frame is None


def test_camera_context_manager_cleanup_on_normal_exit():
    """Verify context manager releases the camera device on normal exit."""
    with patch("cv2.VideoCapture") as mock_cap_cls:
        mock_instance = MagicMock()
        mock_instance.isOpened.return_value = True
        mock_instance.get.side_effect = lambda prop: 640.0 if prop == 3 else 480.0
        mock_cap_cls.return_value = mock_instance

        cam = CameraManager(camera_index=0)
        with cam:
            assert cam.is_opened is True

        mock_instance.release.assert_called_once()
        assert cam.is_opened is False


def test_camera_context_manager_cleanup_on_exception():
    """Verify context manager guarantees release even if an exception is raised."""
    with patch("cv2.VideoCapture") as mock_cap_cls:
        mock_instance = MagicMock()
        mock_instance.isOpened.return_value = True
        mock_instance.get.side_effect = lambda prop: 640.0 if prop == 3 else 480.0
        mock_cap_cls.return_value = mock_instance

        cam = CameraManager(camera_index=0)
        with pytest.raises(RuntimeError):
            with cam:
                raise RuntimeError("Simulated mid-loop exception")

        mock_instance.release.assert_called_once()
        assert cam.is_opened is False

