"""Air-Drawn Circle Recognition and Portal Geometry Stabilization.

Detects when a user draws a substantially closed circle in the air using finger
landmarks, validates circularity, angular coverage, closure gap, and size,
rejecting straight lines, scribbles, and accidental gestures.
"""

from dataclasses import dataclass
import math
from typing import List, Optional, Tuple
import numpy as np


@dataclass
class CircleDetectionResult:
    """Detailed result descriptor for circle gesture recognition."""
    is_circle: bool
    center: Tuple[int, int]
    radius: float
    circularity_score: float  # 0.0 to 1.0 (1.0 = perfect circle)
    angular_coverage_rad: float
    closure_gap_px: float
    closure_ratio: float
    error_message: str = ""


class CircleDetector:
    """Robust geometric and temporal circle recognition engine."""

    def __init__(
        self,
        min_points: int = 14,
        min_radius_px: float = 30.0,
        max_radius_ratio: float = 0.50,
        circularity_threshold: float = 0.42,
        closure_threshold_ratio: float = 0.95,
        min_angular_coverage_rad: float = 4.6,  # ~264 degrees
    ) -> None:
        self.min_points = min_points
        self.min_radius_px = min_radius_px
        self.max_radius_ratio = max_radius_ratio
        self.circularity_threshold = circularity_threshold
        self.closure_threshold_ratio = closure_threshold_ratio
        self.min_angular_coverage_rad = min_angular_coverage_rad

    def detect(
        self,
        points: List[Tuple[int, int]],
        frame_shape: Tuple[int, int] = (720, 1280),
    ) -> CircleDetectionResult:
        """Analyze a 2D stroke trajectory and determine if it represents a valid circle.

        Args:
            points: Sequence of (x, y) coordinates forming the drawn stroke.
            frame_shape: (H, W) of the video feed.

        Returns:
            CircleDetectionResult detailing geometric parameters and validity.
        """
        h, w = frame_shape[:2]

        # 1. Point count check
        if len(points) < self.min_points:
            return CircleDetectionResult(
                is_circle=False,
                center=(0, 0),
                radius=0.0,
                circularity_score=0.0,
                angular_coverage_rad=0.0,
                closure_gap_px=999.0,
                closure_ratio=999.0,
                error_message=f"Stroke too short ({len(points)} < {self.min_points} points)",
            )

        # Spatial decimation: remove consecutive points that are too close (< 3px apart)
        # to ensure uniform angular density during slow hand movements
        filtered_points = [points[0]]
        for pt in points[1:]:
            last = filtered_points[-1]
            if math.hypot(pt[0] - last[0], pt[1] - last[1]) >= 3.0:
                filtered_points.append(pt)

        if len(filtered_points) < max(10, self.min_points - 3):
            filtered_points = points

        pts = np.array(filtered_points, dtype=np.float64)

        # 2. Straight line rejection using 2D PCA / Covariance analysis
        cov = np.cov(pts, rowvar=False)
        if cov.shape == (2, 2):
            eigvals = np.linalg.eigvalsh(cov)
            if eigvals[1] > 1e-5:
                aspect_ratio = eigvals[0] / eigvals[1]
                if aspect_ratio < 0.08:
                    return CircleDetectionResult(
                        is_circle=False,
                        center=(int(np.mean(pts[:, 0])), int(np.mean(pts[:, 1]))),
                        radius=0.0,
                        circularity_score=float(aspect_ratio),
                        angular_coverage_rad=0.0,
                        closure_gap_px=999.0,
                        closure_ratio=999.0,
                        error_message="Rejected: stroke is a straight line, not a circle",
                    )

        # 3. Algebraic Circle Fit (Kåsa least squares)
        # (x - cx)^2 + (y - cy)^2 = r^2  =>  2*x*cx + 2*y*cy + C = x^2 + y^2  where C = r^2 - cx^2 - cy^2
        try:
            A = np.column_stack([2.0 * pts[:, 0], 2.0 * pts[:, 1], np.ones(len(pts))])
            B = pts[:, 0] ** 2 + pts[:, 1] ** 2
            sol, _, _, _ = np.linalg.lstsq(A, B, rcond=None)
            cx, cy, c_val = sol[0], sol[1], sol[2]
            r_sq = c_val + cx**2 + cy**2
            if r_sq <= 0:
                raise ValueError("Negative radius squared in circle fit")
            fitted_radius = math.sqrt(r_sq)
        except Exception:
            # Fallback to centroid and mean radius
            cx = float(np.mean(pts[:, 0]))
            cy = float(np.mean(pts[:, 1]))
            fitted_radius = float(np.mean(np.hypot(pts[:, 0] - cx, pts[:, 1] - cy)))

        # 4. Radius range verification
        max_allowed_r = min(h, w) * self.max_radius_ratio
        if fitted_radius < self.min_radius_px:
            return CircleDetectionResult(
                is_circle=False,
                center=(int(round(cx)), int(round(cy))),
                radius=fitted_radius,
                circularity_score=0.0,
                angular_coverage_rad=0.0,
                closure_gap_px=0.0,
                closure_ratio=0.0,
                error_message=f"Circle too small ({fitted_radius:.1f}px < {self.min_radius_px}px)",
            )

        if fitted_radius > max_allowed_r:
            return CircleDetectionResult(
                is_circle=False,
                center=(int(round(cx)), int(round(cy))),
                radius=fitted_radius,
                circularity_score=0.0,
                angular_coverage_rad=0.0,
                closure_gap_px=0.0,
                closure_ratio=0.0,
                error_message=f"Circle too large ({fitted_radius:.1f}px > {max_allowed_r:.1f}px)",
            )

        # 5. Radial Consistency (Circularity Error)
        radii = np.hypot(pts[:, 0] - cx, pts[:, 1] - cy)
        mean_r = float(np.mean(radii))
        std_r = float(np.std(radii))
        radial_error = std_r / max(1.0, mean_r)
        circularity_score = max(0.0, min(1.0, 1.0 - radial_error))

        if radial_error > self.circularity_threshold:
            return CircleDetectionResult(
                is_circle=False,
                center=(int(round(cx)), int(round(cy))),
                radius=mean_r,
                circularity_score=circularity_score,
                angular_coverage_rad=0.0,
                closure_gap_px=0.0,
                closure_ratio=0.0,
                error_message=f"Irregular shape (circularity error {radial_error:.2f} > {self.circularity_threshold:.2f})",
            )

        # 6. Angular Coverage Check (Unwrapped rotation around center)
        angles = np.arctan2(pts[:, 1] - cy, pts[:, 0] - cx)
        unwrapped = np.unwrap(angles)
        angular_coverage = float(abs(unwrapped[-1] - unwrapped[0]))

        # 7. Robust Loop Closure Check (Handles continuous loops and overshoots)
        direct_gap = float(np.hypot(pts[0, 0] - pts[-1, 0], pts[0, 1] - pts[-1, 1]))
        early_count = max(4, min(len(pts) // 3, 20))
        early_pts = pts[:early_count]
        early_dists = np.hypot(early_pts[:, 0] - pts[-1, 0], early_pts[:, 1] - pts[-1, 1])
        loop_gap = float(np.min(early_dists))
        closure_gap = min(direct_gap, loop_gap)
        closure_ratio = closure_gap / max(1.0, mean_r)

        # Forgive slightly larger gap when user has traversed a full or wide arc (>= 280 deg)
        effective_max_ratio = self.closure_threshold_ratio
        if angular_coverage >= 4.9:
            effective_max_ratio = max(effective_max_ratio, 1.05)

        if closure_ratio > effective_max_ratio and angular_coverage < 5.8:
            return CircleDetectionResult(
                is_circle=False,
                center=(int(round(cx)), int(round(cy))),
                radius=mean_r,
                circularity_score=circularity_score,
                angular_coverage_rad=angular_coverage,
                closure_gap_px=closure_gap,
                closure_ratio=closure_ratio,
                error_message=f"Circle not closed (gap {closure_gap:.1f}px, ratio {closure_ratio:.2f})",
            )

        if angular_coverage < self.min_angular_coverage_rad:
            return CircleDetectionResult(
                is_circle=False,
                center=(int(round(cx)), int(round(cy))),
                radius=mean_r,
                circularity_score=circularity_score,
                angular_coverage_rad=angular_coverage,
                closure_gap_px=closure_gap,
                closure_ratio=closure_ratio,
                error_message=f"Incomplete arc (coverage {math.degrees(angular_coverage):.1f}° < {math.degrees(self.min_angular_coverage_rad):.1f}°)",
            )

        # Success! Valid circle confirmed
        return CircleDetectionResult(
            is_circle=True,
            center=(int(round(cx)), int(round(cy))),
            radius=mean_r,
            circularity_score=circularity_score,
            angular_coverage_rad=angular_coverage,
            closure_gap_px=closure_gap,
            closure_ratio=closure_ratio,
            error_message="Valid circle recognized",
        )

