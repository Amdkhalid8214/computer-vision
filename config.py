"""Centralized configuration module for HANDVISION STUDIO — REALITYFRAME EDITION.

Defines all tunable parameters for camera capture, hand tracking, ROI calculation,
filters, RealityFrame Ghost Portal, Virtual Finger Typing Keyboard, and Cyberpunk HUD.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Tuple


class AppMode(Enum):
    """Primary operational modes of HandVision Studio."""
    STANDARD_FILTER = "STANDARD_FILTER"      # Mode 1: Classic two-hand filter box
    REALITYFRAME_ROI = "REALITYFRAME_ROI"    # Mode 2: Cybernetic ROI Filter Portal
    REALITYFRAME_GHOST = "REALITYFRAME_GHOST"# Mode 3: Optical Illusion Ghost Portal (Invisible background)
    VIRTUAL_KEYBOARD = "VIRTUAL_KEYBOARD"    # Mode 4: Mid-air finger typing QWERTY keyboard
    AIR_WRITING = "AIR_WRITING"              # Mode 5: Air Writing Studio (draw & write in the air)
    MYSTIC_MAGIC = "MYSTIC_MAGIC"            # Mode 6: Mystic Magic Engine (Doctor Strange spells & portals)
    UNIVERSE_PORTAL = "UNIVERSE_PORTAL"      # Mode 7: Cinematic 3D Universe Portal (Hand-drawn cosmic window)


class BackgroundStatus(Enum):
    """Lifecycle states of the RealityFrame background plate."""
    NOT_CAPTURED = "NOT_CAPTURED"
    COUNTDOWN = "COUNTDOWN"
    CAPTURING = "CAPTURING"
    READY = "READY"
    RECAPTURING = "RECAPTURING"
    FAILED = "FAILED"


@dataclass(frozen=True)
class CameraConfig:
    """Settings for webcam hardware capture and frame dimensions."""
    camera_index: int = 0
    requested_width: int = 1280
    requested_height: int = 720
    fps_limit: int = 60
    mirror_horizontal: bool = True


@dataclass(frozen=True)
class HandTrackerConfig:
    """Settings for MediaPipe hand tracking and landmark anchor points."""
    max_num_hands: int = 2
    model_complexity: int = 0  # 0: Lite (fastest for real-time video), 1: Full
    min_detection_confidence: float = 0.7
    min_tracking_confidence: float = 0.7
    landmark_anchor: str = "INDEX_FINGERTIP"
    anchor_indices: dict = field(
        default_factory=lambda: {
            "INDEX_FINGERTIP": 8,
            "INDEX_MCP": 5,
            "WRIST": 0,
        }
    )


@dataclass(frozen=True)
class ROIConfig:
    """Settings for dynamic Region of Interest (ROI) geometry and smoothing."""
    min_width: int = 40
    min_height: int = 40
    smoothing_enabled: bool = True
    smoothing_factor: float = 0.35
    discontinuity_threshold_pixels: float = 120.0
    stability_required_frames: int = 3  # Frames required before activating portal


@dataclass(frozen=True)
class RealityFrameConfig:
    """Settings for RealityFrame Ghost Mode and Portal optical illusions."""
    background_capture_duration_sec: float = 2.5
    background_capture_frames: int = 25  # Number of frames sampled for median plate
    min_portal_width: int = 40
    min_portal_height: int = 40
    portal_border_color_ghost: Tuple[int, int, int] = (0, 215, 255)  # Neon Gold
    portal_border_color_roi: Tuple[int, int, int] = (0, 255, 128)    # Neon Cyan/Green
    enable_particles: bool = True
    enable_cyber_grid: bool = True


@dataclass(frozen=True)
class FilterConfig:
    """Settings for image processing filters."""
    default_filter: str = "negative"
    canny_low_threshold: int = 50
    canny_high_threshold: int = 150
    canny_aperture_size: int = 3
    canny_blur_kernel: Tuple[int, int] = (3, 3)
    pixelate_block_size: int = 14
    thermal_colormap: str = "inferno"  # choices: inferno, jet, ocean


@dataclass(frozen=True)
class KeyboardConfig:
    """Settings for Virtual Finger Typing Air Keyboard."""
    enabled: bool = False
    typing_mode: str = "PINCH"  # "PINCH" or "DWELL"
    pinch_threshold_px: float = 40.0
    dwell_duration_sec: float = 0.85
    cooldown_sec: float = 0.35
    cursor_smoothing_factor: float = 0.40
    audio_feedback: bool = True
    keyboard_bg_opacity: float = 0.82
    keyboard_width_ratio: float = 0.90
    keyboard_height_ratio: float = 0.44


@dataclass(frozen=True)
class AirWritingConfig:
    """Settings for Air Writing Studio."""
    pinch_threshold_px: float = 40.0
    smoothing_factor: float = 0.45
    default_brush_size: int = 6
    min_brush_size: int = 2
    max_brush_size: int = 36
    default_color_name: str = "cyan"
    default_tool: str = "PEN"  # PEN, HIGHLIGHTER, ERASER
    glow_enabled: bool = False
    canvas_mode: str = "OVERLAY"  # "OVERLAY" (on camera) or "BLACKBOARD"
    export_dir: str = "recordings"


@dataclass(frozen=True)
class MysticMagicConfig:
    """Settings for Mystic Magic Engine & Doctor Strange Spell Casting."""
    primary_color_bgr: Tuple[int, int, int] = (0, 140, 255)   # Fiery Amber Orange
    secondary_color_bgr: Tuple[int, int, int] = (0, 220, 255) # Bright Mystic Gold
    core_color_bgr: Tuple[int, int, int] = (220, 245, 255)    # White-hot Energy Core
    portal_base_radius: int = 110
    max_particles: int = 350
    rotation_speed_outer: float = 1.2    # rad/sec
    rotation_speed_inner: float = -1.8   # counter-clockwise rad/sec
    enable_glow: bool = True
    enable_particles: bool = True
    default_spell: str = "AUTO"  # AUTO, SHIELD, BLAST, BEAM, PORTAL, TRAIL


@dataclass(frozen=True)
class UniversePortalConfig:
    """Settings for RealityFrame: Cinematic Universe Portal."""
    min_circle_radius_px: float = 30.0
    max_circle_radius_ratio: float = 0.50
    min_stroke_points: int = 14
    circularity_threshold: float = 0.42  # Forgiving tolerance for natural air gestures
    closure_threshold_ratio: float = 0.95
    min_angular_coverage_rad: float = 4.6  # ~264 degrees
    portal_border_color_bgr: Tuple[int, int, int] = (0, 140, 255)  # Fiery Amber Orange
    border_accent_color_bgr: Tuple[int, int, int] = (0, 220, 255)  # Mystic Gold
    enable_portal_sparks: bool = True
    enable_nebula: bool = True
    default_preset: str = "BALANCED"       # "PERFORMANCE", "BALANCED", "CINEMATIC"
    default_scale_mode: str = "CINEMATIC"   # "CINEMATIC" or "SCIENTIFIC"


@dataclass(frozen=True)
class HUDConfig:
    """Settings for on-screen interface, colors, and diagnostic dashboard."""
    window_name: str = "HANDVISION STUDIO — REALITYFRAME EDITION"
    show_debug_metrics: bool = True
    border_color: Tuple[int, int, int] = (0, 255, 255)  # BGR: Yellow
    border_thickness: int = 2
    bracket_length: int = 20
    bracket_thickness: int = 3
    hand1_marker_color: Tuple[int, int, int] = (0, 165, 255)  # BGR: Orange
    hand2_marker_color: Tuple[int, int, int] = (255, 0, 255)  # BGR: Magenta
    marker_radius: int = 8
    marker_thickness: int = -1  # Solid fill
    text_color: Tuple[int, int, int] = (255, 255, 255)  # BGR: White
    header_bg_color: Tuple[int, int, int] = (18, 18, 22)  # Dark cyberpunk overlay
    header_opacity: float = 0.80
    fps_smoothing_factor: float = 0.1  # Smoothing for rolling FPS display


@dataclass(frozen=True)
class KeyBindingsConfig:
    """Key bindings mapping for rapid mode and parameter control."""
    key_exit_q: int = ord("q")
    key_exit_esc: int = 27
    key_mode_standard_1: int = ord("1")
    key_mode_roi_portal_2: int = ord("2")
    key_mode_ghost_portal_3: int = ord("3")
    key_mode_keyboard_4: int = ord("4")
    key_mode_air_writing_5: int = ord("5")
    key_mode_mystic_magic_6: int = ord("6")
    key_mode_universe_7: int = ord("7")
    key_toggle_keyboard_k: int = ord("k")
    key_toggle_writing_w: int = ord("w")
    key_toggle_magic_g: int = ord("g")
    key_toggle_universe_u: int = ord("u")
    key_toggle_scale_x: int = ord("x")
    key_undo_z: int = ord("z")
    key_redo_y: int = ord("y")
    key_save_s: int = ord("s")
    key_recapture_bg_r: int = ord("r")
    key_next_filter_n: int = ord("n")
    key_prev_filter_p: int = ord("p")
    key_cycle_filter_m: int = ord("m")
    key_cycle_filter_space: int = 32
    key_toggle_hud_h: int = ord("h")
    key_toggle_hud_f: int = ord("f")
    key_toggle_typing_mode_t: int = ord("t")
    key_clear_text_c: int = ord("c")


@dataclass(frozen=True)
class AppConfig:
    """Root configuration holding all sub-system configurations."""
    camera: CameraConfig = field(default_factory=CameraConfig)
    hand_tracker: HandTrackerConfig = field(default_factory=HandTrackerConfig)
    roi: ROIConfig = field(default_factory=ROIConfig)
    reality_frame: RealityFrameConfig = field(default_factory=RealityFrameConfig)
    filters: FilterConfig = field(default_factory=FilterConfig)
    keyboard: KeyboardConfig = field(default_factory=KeyboardConfig)
    air_writing: AirWritingConfig = field(default_factory=AirWritingConfig)
    mystic_magic: MysticMagicConfig = field(default_factory=MysticMagicConfig)
    universe_portal: UniversePortalConfig = field(default_factory=UniversePortalConfig)
    hud: HUDConfig = field(default_factory=HUDConfig)
    keys: KeyBindingsConfig = field(default_factory=KeyBindingsConfig)
    initial_mode: AppMode = AppMode.STANDARD_FILTER

    def validate(self) -> None:
        """Validate configuration parameters to catch misconfigurations early."""
        if self.camera.camera_index < 0:
            raise ValueError(f"Invalid camera_index: {self.camera.camera_index}")
        if self.camera.requested_width <= 0 or self.camera.requested_height <= 0:
            raise ValueError(f"Invalid capture resolution: {self.camera.requested_width}x{self.camera.requested_height}")
        if self.hand_tracker.max_num_hands < 2:
            raise ValueError("max_num_hands must be at least 2 for two-hand corner control")
        if not (0.0 <= self.hand_tracker.min_detection_confidence <= 1.0):
            raise ValueError("min_detection_confidence must be between 0.0 and 1.0")
        if not (0.0 <= self.hand_tracker.min_tracking_confidence <= 1.0):
            raise ValueError("min_tracking_confidence must be between 0.0 and 1.0")
        if self.hand_tracker.landmark_anchor not in self.hand_tracker.anchor_indices:
            raise ValueError(
                f"Unknown landmark anchor: {self.hand_tracker.landmark_anchor}. "
                f"Valid choices: {list(self.hand_tracker.anchor_indices.keys())}"
            )
        if self.roi.min_width <= 0 or self.roi.min_height <= 0:
            raise ValueError("ROI minimum dimensions must be positive integers")
        if not (0.0 < self.roi.smoothing_factor <= 1.0):
            raise ValueError("smoothing_factor must be in range (0.0, 1.0]")
        if self.filters.canny_low_threshold >= self.filters.canny_high_threshold:
            raise ValueError("canny_low_threshold must be strictly less than canny_high_threshold")
        if self.reality_frame.background_capture_duration_sec <= 0.5:
            raise ValueError("background_capture_duration_sec must be > 0.5 seconds")
        if self.keyboard.typing_mode not in ("PINCH", "DWELL"):
            raise ValueError(f"Invalid typing_mode: {self.keyboard.typing_mode}. Must be 'PINCH' or 'DWELL'.")


# Default global configuration instance
DEFAULT_CONFIG = AppConfig()
DEFAULT_CONFIG.validate()
