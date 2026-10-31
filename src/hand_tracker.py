"""Hand detection and landmark processing component using MediaPipe Hands.

Extracts up to 21 landmarks per detected hand, converts normalized coordinates
to clamped pixel coordinates, determines control anchor points, and handles
detection lifecycle safely.
"""

from dataclasses import dataclass
import logging
from typing import List, Optional, Tuple
import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Try importing mediapipe safely
try:
    import mediapipe as mp
    HAS_MEDIAPIPE = True
except ImportError:
    mp = None
    HAS_MEDIAPIPE = False


@dataclass
class HandDetection:
    """Represents a single detected hand with normalized and pixel landmarks."""
    landmarks_norm: List[Tuple[float, float, float]]  # (x, y, z) normalized [0, 1]
    landmarks_pixel: List[Tuple[int, int]]           # (x, y) clamped to frame pixel coordinates
    control_point: Tuple[int, int]                  # (x, y) designated anchor point
    handedness: Optional[str] = None                # "Left", "Right", or None

    @property
    def index_tip(self) -> Tuple[int, int]:
        """Return (x, y) coordinates of the index fingertip (landmark 8)."""
        if len(self.landmarks_pixel) > 8:
            return self.landmarks_pixel[8]
        return self.control_point

    @property
    def thumb_tip(self) -> Tuple[int, int]:
        """Return (x, y) coordinates of the thumb tip (landmark 4)."""
        if len(self.landmarks_pixel) > 4:
            return self.landmarks_pixel[4]
        return self.control_point

    @property
    def pinch_distance(self) -> float:
        """Euclidean distance in pixels between thumb tip and index fingertip."""
        ix, iy = self.index_tip
        tx, ty = self.thumb_tip
        return float(np.hypot(ix - tx, iy - ty))

    @property
    def wrist(self) -> Tuple[int, int]:
        """Return (x, y) coordinates of the wrist (landmark 0)."""
        if len(self.landmarks_pixel) > 0:
            return self.landmarks_pixel[0]
        return self.control_point

    @property
    def palm_center(self) -> Tuple[int, int]:
        """Return (x, y) coordinates of the center of the palm."""
        if len(self.landmarks_pixel) >= 18:
            pts = [self.landmarks_pixel[i] for i in (0, 5, 9, 17)]
            cx = int(sum(p[0] for p in pts) / len(pts))
            cy = int(sum(p[1] for p in pts) / len(pts))
            return (cx, cy)
        return self.control_point

    @property
    def middle_tip(self) -> Tuple[int, int]:
        """Return (x, y) of middle fingertip (landmark 12)."""
        if len(self.landmarks_pixel) > 12:
            return self.landmarks_pixel[12]
        return self.control_point

    @property
    def ring_tip(self) -> Tuple[int, int]:
        """Return (x, y) of ring fingertip (landmark 16)."""
        if len(self.landmarks_pixel) > 16:
            return self.landmarks_pixel[16]
        return self.control_point

    @property
    def pinky_tip(self) -> Tuple[int, int]:
        """Return (x, y) of pinky fingertip (landmark 20)."""
        if len(self.landmarks_pixel) > 20:
            return self.landmarks_pixel[20]
        return self.control_point

    @property
    def hand_size(self) -> float:
        """Approximate hand scale in pixels (distance from wrist to middle MCP)."""
        if len(self.landmarks_pixel) > 9:
            w = self.landmarks_pixel[0]
            m = self.landmarks_pixel[9]
            return float(np.hypot(w[0] - m[0], w[1] - m[1]))
        return 80.0

    def is_finger_extended(self, tip_idx: int, pip_idx: int) -> bool:
        """Check if a finger is extended relative to the palm/wrist."""
        if len(self.landmarks_pixel) <= max(tip_idx, pip_idx):
            return False
        w = self.wrist
        tip = self.landmarks_pixel[tip_idx]
        pip = self.landmarks_pixel[pip_idx]
        d_tip = np.hypot(tip[0] - w[0], tip[1] - w[1])
        d_pip = np.hypot(pip[0] - w[0], pip[1] - w[1])
        return bool(d_tip > d_pip * 1.15)

    @property
    def is_open_palm(self) -> bool:
        """True if all 4 fingers (index, middle, ring, pinky) are extended."""
        return (
            self.is_finger_extended(8, 6)
            and self.is_finger_extended(12, 10)
            and self.is_finger_extended(16, 14)
            and self.is_finger_extended(20, 18)
        )

    @property
    def is_pointing(self) -> bool:
        """True if index finger is extended while middle, ring, and pinky are curled."""
        return (
            self.is_finger_extended(8, 6)
            and not self.is_finger_extended(12, 10)
            and not self.is_finger_extended(16, 14)
            and not self.is_finger_extended(20, 18)
        )


class HandTracker:
    """Manages MediaPipe Hands inference and coordinates conversion."""

    def __init__(
        self,
        max_num_hands: int = 2,
        model_complexity: int = 0,
        min_detection_confidence: float = 0.7,
        min_tracking_confidence: float = 0.7,
        landmark_anchor: str = "INDEX_FINGERTIP",
        anchor_indices: Optional[dict] = None,
    ) -> None:
        """Initialize the hand tracker.

        Args:
            max_num_hands: Maximum number of hands to track simultaneously (default 2).
            model_complexity: MediaPipe model complexity (0 for lite, 1 for full).
            min_detection_confidence: Detection confidence threshold.
            min_tracking_confidence: Tracking confidence threshold.
            landmark_anchor: Name of the landmark used for corner control.
            anchor_indices: Mapping from anchor name to landmark index.
        """
        self.max_num_hands = max_num_hands
        self.model_complexity = model_complexity
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.landmark_anchor = landmark_anchor
        self.anchor_indices = anchor_indices or {
            "INDEX_FINGERTIP": 8,
            "INDEX_MCP": 5,
            "WRIST": 0,
        }

        self.anchor_idx = self.anchor_indices.get(self.landmark_anchor, 8)
        self._hands_solution = None
        self._init_mediapipe()

    def _init_mediapipe(self) -> None:
        """Initialize the underlying MediaPipe Hands solution."""
        if not HAS_MEDIAPIPE:
            logger.warning("MediaPipe library is not installed or failed to import.")
            return

        try:
            self._mp_hands = mp.solutions.hands
            self._hands_solution = self._mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=self.max_num_hands,
                model_complexity=self.model_complexity,
                min_detection_confidence=self.min_detection_confidence,
                min_tracking_confidence=self.min_tracking_confidence,
            )
            logger.info("MediaPipe Hands initialized (model_complexity=%d).", self.model_complexity)
        except Exception as e:
            logger.error("Failed to initialize MediaPipe Hands: %s", e)
            self._hands_solution = None

    def process_frame(self, frame: np.ndarray) -> List[HandDetection]:
        """Detect hands in an image and return structured detection results.

        Args:
            frame: BGR image array from the camera.

        Returns:
            List of HandDetection objects (0, 1, or 2 hands).
            The list is sorted deterministically from left to right (by control point x)
            to ensure stability across frames regardless of detection order.
        """
        if self._hands_solution is None or frame is None or frame.size == 0:
            return []

        h, w = frame.shape[:2]
        if h <= 0 or w <= 0:
            return []

        # MediaPipe expects RGB images. Avoid full copy if possible by converting BGR to RGB.
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Performance optimization: set writeable to False for inference
        rgb_frame.flags.writeable = False

        try:
            results = self._hands_solution.process(rgb_frame)
        except Exception as e:
            logger.error("Exception during MediaPipe frame processing: %s", e)
            return []

        detections: List[HandDetection] = []

        if not results.multi_hand_landmarks:
            return detections

        # Extract handedness if available
        handedness_list: List[Optional[str]] = []
        if results.multi_handedness:
            for classification_list in results.multi_handedness:
                if classification_list.classification:
                    handedness_list.append(classification_list.classification[0].label)
                else:
                    handedness_list.append(None)
        else:
            handedness_list = [None] * len(results.multi_hand_landmarks)

        for hand_idx, landmarks in enumerate(results.multi_hand_landmarks):
            norm_pts: List[Tuple[float, float, float]] = []
            pixel_pts: List[Tuple[int, int]] = []

            for lm in landmarks.landmark:
                norm_pts.append((lm.x, lm.y, lm.z))
                # Convert normalized coords to pixel coords and clamp to frame boundaries
                px = int(np.clip(round(lm.x * w), 0, w - 1))
                py = int(np.clip(round(lm.y * h), 0, h - 1))
                pixel_pts.append((px, py))

            # Validate that 21 landmarks exist
            if len(pixel_pts) < 21:
                logger.debug("Incomplete landmarks detected (%d/21), skipping hand.", len(pixel_pts))
                continue

            # Determine control point from configured anchor index
            anchor_pos = self.anchor_idx if 0 <= self.anchor_idx < len(pixel_pts) else 8
            ctrl_pt = pixel_pts[anchor_pos]

            handedness = handedness_list[hand_idx] if hand_idx < len(handedness_list) else None

            detections.append(
                HandDetection(
                    landmarks_norm=norm_pts,
                    landmarks_pixel=pixel_pts,
                    control_point=ctrl_pt,
                    handedness=handedness,
                )
            )

        # Enforce deterministic order: sort detected hands by control point X coordinate (left to right)
        detections.sort(key=lambda d: d.control_point[0])

        return detections[: self.max_num_hands]

    def close(self) -> None:
        """Release MediaPipe resources."""
        if self._hands_solution is not None:
            try:
                self._hands_solution.close()
            except Exception as e:
                logger.warning("Error releasing MediaPipe Hands: %s", e)
            finally:
                self._hands_solution = None

