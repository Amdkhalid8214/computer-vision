"""Application orchestration and main loop lifecycle management for HANDVISION STUDIO.

Coordinates camera ingestion, single-pass hand tracking, RealityFrame Ghost Mode,
Cybernetic ROI Portals, Virtual Finger Typing Air Keyboard, Cyberpunk Dashboard,
and mouse/keyboard event dispatch with guaranteed resource cleanup.
"""

import logging
import time
from typing import List, Optional, Tuple
import cv2
import numpy as np

from config import AppConfig, AppMode, BackgroundStatus, DEFAULT_CONFIG
from src.background_manager import BackgroundManager
from src.camera import CameraManager, CameraInitializationError
from src.filters import FilterEngine
from src.hand_tracker import HandTracker, HandDetection
from src.hud import HUDManager
from src.keyboard_engine import VirtualKeyboardEngine
from src.keyboard_renderer import KeyboardRenderer
from src.air_writing import AirWritingManager
from src.mystic_magic import MysticMagicManager
from src.universe_portal import UniversePortalManager, PortalLifecycle
from src.reality_frame import RealityFrameManager, PortalState
from src.roi_controller import ROIController, ROIResult

logger = logging.getLogger(__name__)


class HandFilterApp:
    """Master application coordinator for HandVision Studio — RealityFrame Edition."""

    def __init__(self, config: Optional[AppConfig] = None) -> None:
        """Initialize all subsystems with the specified or default configuration."""
        self.config = config or DEFAULT_CONFIG
        self.config.validate()

        self.current_mode: AppMode = self.config.initial_mode

        # 1. Camera Manager
        self.camera = CameraManager(
            camera_index=self.config.camera.camera_index,
            requested_width=self.config.camera.requested_width,
            requested_height=self.config.camera.requested_height,
            mirror_horizontal=self.config.camera.mirror_horizontal,
        )

        # 2. Hand Tracker (single shared instance across all features)
        self.hand_tracker = HandTracker(
            max_num_hands=self.config.hand_tracker.max_num_hands,
            model_complexity=self.config.hand_tracker.model_complexity,
            min_detection_confidence=self.config.hand_tracker.min_detection_confidence,
            min_tracking_confidence=self.config.hand_tracker.min_tracking_confidence,
            landmark_anchor=self.config.hand_tracker.landmark_anchor,
            anchor_indices=self.config.hand_tracker.anchor_indices,
        )

        # 3. Dynamic ROI Controller (Standard Filter Mode)
        self.roi_controller = ROIController(
            min_width=self.config.roi.min_width,
            min_height=self.config.roi.min_height,
            smoothing_enabled=self.config.roi.smoothing_enabled,
            smoothing_factor=self.config.roi.smoothing_factor,
            discontinuity_threshold_pixels=self.config.roi.discontinuity_threshold_pixels,
        )

        # 4. Filter Engine (6+ filters and custom pipeline)
        self.filter_engine = FilterEngine()
        if self.config.filters.default_filter in self.filter_engine.available_filters:
            self.filter_engine.active_key = self.config.filters.default_filter

        # 5. RealityFrame Background Acquisition Manager
        self.background_manager = BackgroundManager(
            capture_duration_sec=self.config.reality_frame.background_capture_duration_sec,
            target_sample_frames=self.config.reality_frame.background_capture_frames,
        )

        # 6. RealityFrame Portal Manager (Ghost Mode & Cybernetic ROI Portal)
        self.reality_frame = RealityFrameManager(self.config.reality_frame)

        # 7. Virtual Finger Typing Air Keyboard
        self.keyboard_engine = VirtualKeyboardEngine(
            enabled=self.config.keyboard.enabled,
            typing_mode=self.config.keyboard.typing_mode,
            pinch_threshold_px=self.config.keyboard.pinch_threshold_px,
            dwell_duration_sec=self.config.keyboard.dwell_duration_sec,
            cooldown_sec=self.config.keyboard.cooldown_sec,
            cursor_smoothing_factor=self.config.keyboard.cursor_smoothing_factor,
            audio_feedback=self.config.keyboard.audio_feedback,
            width_ratio=self.config.keyboard.keyboard_width_ratio,
            height_ratio=self.config.keyboard.keyboard_height_ratio,
        )
        self.keyboard_renderer = KeyboardRenderer()

        # 8. Air Writing Studio
        self.air_writing = AirWritingManager(self.config.air_writing)

        # 9. Mystic Magic Engine (Doctor Strange effects & spells)
        self.mystic_magic = MysticMagicManager(self.config.mystic_magic)

        # 10. Cinematic Universe Portal Engine
        self.universe_portal = UniversePortalManager(self.config.universe_portal)

        # 11. Cyberpunk HUD & Dashboard Manager
        self.hud = HUDManager(self.config.hud)

        self._is_running = False
        self._prev_frame_time: float = 0.0

        # Set up interactive top dashboard button callbacks
        self._setup_dashboard_callbacks()

    @property
    def mode(self) -> AppMode:
        """Current operational mode."""
        return self.current_mode

    @property
    def hud_manager(self) -> HUDManager:
        """Dashboard HUD manager."""
        return self.hud

    def _setup_hud_buttons(self, width: int) -> None:
        """Reconfigure dashboard buttons with specified frame width."""
        callbacks = {
            "mode_std": lambda: self.set_mode(AppMode.STANDARD_FILTER),
            "mode_roi": lambda: self.set_mode(AppMode.REALITYFRAME_ROI),
            "mode_ghost": lambda: self.set_mode(AppMode.REALITYFRAME_GHOST),
            "mode_kb": lambda: self.set_mode(AppMode.VIRTUAL_KEYBOARD),
            "mode_write": lambda: self.set_mode(AppMode.AIR_WRITING),
            "mode_magic": lambda: self.set_mode(AppMode.MYSTIC_MAGIC),
            "mode_universe": lambda: self.set_mode(AppMode.UNIVERSE_PORTAL),
            "recapture": self.trigger_background_recapture,
            "next_filter": lambda: self.filter_engine.cycle_next(),
            "hud": lambda: self.hud.toggle_debug_metrics(),
            "exit": lambda: setattr(self, "_is_running", False),
        }
        self.hud.setup_buttons(width, callbacks)

    def _setup_dashboard_callbacks(self) -> None:
        """Bind interactive HUD button actions to application methods."""
        self._setup_hud_buttons(self.config.camera.requested_width)

    def set_mode(self, mode: AppMode) -> None:
        """Switch application mode smoothly while resetting temporary gesture states."""
        if self.current_mode == mode:
            return
        logger.info("Switching application mode: %s -> %s", self.current_mode.name, mode.name)
        self.current_mode = mode

        # Reset gesture states
        self.roi_controller.reset()
        self.reality_frame.reset_stability()
        self.keyboard_engine.reset_gesture_state()
        self.air_writing.reset_gesture_state()
        self.mystic_magic.clear_effects()
        self.universe_portal.reset_to_idle()

        # Synchronize keyboard visibility
        if mode == AppMode.VIRTUAL_KEYBOARD:
            self.keyboard_engine.is_visible = True
        else:
            self.keyboard_engine.is_visible = False

        # If entering Ghost Mode and background plate is missing, trigger capture sequence automatically
        if mode == AppMode.REALITYFRAME_GHOST and not self.background_manager.is_ready:
            self.trigger_background_recapture()

    def trigger_background_recapture(self) -> None:
        """Initiate background acquisition sequence."""
        self.background_manager.trigger_capture()

    def process_frame(self, frame: np.ndarray, elapsed_sec: float = 0.033) -> Tuple[np.ndarray, ROIResult]:
        """Process a single video frame end-to-end.

        Supports both interactive loop and automated unit testing.
        """
        if frame is None or frame.size == 0:
            raise ValueError("Input frame is empty or None.")

        # Update FPS
        instant_fps = (1.0 / elapsed_sec) if elapsed_sec > 1e-5 else 30.0
        self.hud.update_fps(instant_fps)

        # 1. Background Manager Sequence Ingestion (if active)
        if self.background_manager.status in (
            BackgroundStatus.COUNTDOWN,
            BackgroundStatus.CAPTURING,
            BackgroundStatus.RECAPTURING,
        ):
            self.background_manager.ingest_frame(frame, dt=elapsed_sec)

        # 2. Single-pass Hand Landmark Inference
        t_inf_start = time.perf_counter()
        detections: List[HandDetection] = self.hand_tracker.process_frame(frame)
        self.hud.telemetry["inference_ms"] = (time.perf_counter() - t_inf_start) * 1000.0

        return self._process_single_frame(frame, detections, dt=elapsed_sec)

    def _process_single_frame(
        self,
        frame: np.ndarray,
        detections: List[HandDetection],
        dt: float = 0.033,
    ) -> Tuple[np.ndarray, ROIResult]:
        """Process a single frame with pre-computed or mock hand detections."""
        num_hands = len(detections)

        # Extract landmarks
        idx_pt1 = detections[0].index_tip if num_hands >= 1 else None
        idx_pt2 = detections[1].index_tip if num_hands >= 2 else None
        ctrl_pt1 = detections[0].control_point if num_hands >= 1 else None
        ctrl_pt2 = detections[1].control_point if num_hands >= 2 else None

        # 3. Dynamic ROI calculation (for Standard Filter Mode)
        roi_result = self.roi_controller.update(ctrl_pt1, ctrl_pt2, frame.shape[:2])

        # 4. Mode-Specific Interaction & Visual Compositing
        t_proc_start = time.perf_counter()

        # MODE 1: STANDARD FILTER BOX
        if self.current_mode == AppMode.STANDARD_FILTER:
            if roi_result.is_valid and num_hands >= 2:
                left, top, right, bottom = roi_result.left, roi_result.top, roi_result.right, roi_result.bottom
                roi_slice = frame[top:bottom, left:right]
                try:
                    filtered_slice = self.filter_engine.apply_active(roi_slice)
                    if filtered_slice.shape == roi_slice.shape:
                        frame[top:bottom, left:right] = filtered_slice
                except Exception as e:
                    logger.error("Error applying filter '%s': %s", self.filter_engine.active_key, e)

        # MODE 2: REALITYFRAME ROI FILTER PORTAL
        elif self.current_mode == AppMode.REALITYFRAME_ROI:
            portal = self.reality_frame.evaluate_portal(idx_pt1, idx_pt2, frame.shape[:2])
            processed_slice: Optional[np.ndarray] = None
            if portal.is_active:
                p_slice = frame[portal.top:portal.bottom, portal.left:portal.right]
                try:
                    processed_slice = self.filter_engine.apply_active(p_slice)
                except Exception as e:
                    logger.error("Error in ROI Portal filter: %s", e)
            frame = self.reality_frame.apply_roi_mode(
                frame=frame,
                portal=portal,
                processed_roi=processed_slice,
                filter_name=self.filter_engine.active_name,
            )

        # MODE 3: REALITYFRAME GHOST PORTAL (Invisible Background Illusion)
        elif self.current_mode == AppMode.REALITYFRAME_GHOST:
            portal = self.reality_frame.evaluate_portal(idx_pt1, idx_pt2, frame.shape[:2])
            frame = self.reality_frame.apply_ghost_mode(
                frame=frame,
                portal=portal,
                background_plate=self.background_manager.background_plate,
            )

        # MODE 4: VIRTUAL FINGER TYPING AIR KEYBOARD
        elif self.current_mode == AppMode.VIRTUAL_KEYBOARD:
            k_idx = detections[0].index_tip if num_hands >= 1 else None
            k_thb = detections[0].thumb_tip if num_hands >= 1 else None
            self.keyboard_engine.update(k_idx, k_thb, dt=dt)
            frame = self.keyboard_renderer.render(frame, self.keyboard_engine)

        # MODE 5: AIR WRITING STUDIO (Mid-Air Finger Drawing)
        elif self.current_mode == AppMode.AIR_WRITING:
            k_idx = detections[0].index_tip if num_hands >= 1 else None
            k_thb = detections[0].thumb_tip if num_hands >= 1 else None
            self.air_writing.update(k_idx, k_thb, dt=dt)
            frame = self.air_writing.render(frame)

        # MODE 6: MYSTIC MAGIC ENGINE (Doctor Strange Spells & Portals)
        elif self.current_mode == AppMode.MYSTIC_MAGIC:
            self.mystic_magic.update(detections, dt=dt)
            frame = self.mystic_magic.render(frame, detections)

        # MODE 7: REALITYFRAME CINEMATIC UNIVERSE PORTAL
        elif self.current_mode == AppMode.UNIVERSE_PORTAL:
            self.universe_portal.update(detections, dt=dt, frame_shape=frame.shape[:2])
            frame = self.universe_portal.render(frame)

        self.hud.telemetry["processing_ms"] = (time.perf_counter() - t_proc_start) * 1000.0

        # 5. Render Background Countdown Overlay (if actively capturing)
        frame = self.background_manager.render_overlay(frame)

        # 6. Render Cyberpunk Dashboard, Status Banners, and Telemetry
        t_rend_start = time.perf_counter()
        control_points = [d.control_point for d in detections]
        annotated_frame = self.hud.render(
            frame=frame,
            roi_result=roi_result,
            active_filter_name=self.filter_engine.active_name,
            num_hands=num_hands,
            fps=self.hud._fps_smoothed,
            control_points=control_points,
            current_mode=self.current_mode,
            bg_status=self.background_manager.status,
            tracking_confidence=self.config.hand_tracker.min_tracking_confidence,
        )
        self.hud.telemetry["render_ms"] = (time.perf_counter() - t_rend_start) * 1000.0

        return annotated_frame, roi_result

    def _mouse_callback(self, event: int, x: int, y: int, flags: int, param: object) -> None:
        """Handle mouse clicks on OpenCV window dashboard buttons."""
        if event == cv2.EVENT_LBUTTONDOWN:
            hit = self.hud.handle_mouse_click(x, y)
            if hit:
                logger.debug("Dashboard button clicked at (%d, %d)", x, y)

    def run(self) -> None:
        """Run the main interactive application loop."""
        logger.info("Starting HandVision Studio...")

        try:
            self.camera.open()
        except CameraInitializationError as e:
            print("\n=======================================================")
            print("ERROR: CAMERA INITIALIZATION FAILED")
            print("=======================================================")
            print(str(e))
            print("=======================================================\n")
            return

        window_name = self.config.hud.window_name
        cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(window_name, self._mouse_callback)

        actual_w, actual_h = self.camera.actual_dimensions
        if actual_w > 0:
            self._setup_dashboard_callbacks()

        self._is_running = True
        self._prev_frame_time = time.perf_counter()

        logger.info("Entering main video loop. Press 'q' or 'ESC' to exit.")

        try:
            while self._is_running:
                loop_start = time.perf_counter()
                elapsed = max(1e-5, loop_start - self._prev_frame_time)
                self._prev_frame_time = loop_start

                success, frame = self.camera.read_frame()
                if not success or frame is None:
                    logger.warning("Failed to read frame from webcam.")
                    time.sleep(0.01)
                    continue

                output_frame, _ = self.process_frame(frame, elapsed_sec=elapsed)
                cv2.imshow(window_name, output_frame)

                key = cv2.waitKey(1) & 0xFF
                if key != 255:
                    self._handle_key_event(key)

        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received. Shutting down gracefully.")
        except Exception as e:
            logger.exception("Unexpected error in main application loop: %s", e)
        finally:
            self.shutdown()

    def _handle_key_event(self, key: int) -> None:
        """Process keyboard shortcut inputs."""
        keys = self.config.keys

        # Exit: 'q', 'Q', ESC
        if key in (keys.key_exit_q, ord("Q"), keys.key_exit_esc):
            logger.info("Exit requested via keypress.")
            self._is_running = False

        # Modes: 1, 2, 3, 4, 5, 6
        elif key in (keys.key_mode_standard_1, ord("1")):
            self.set_mode(AppMode.STANDARD_FILTER)

        elif key in (keys.key_mode_roi_portal_2, ord("2")):
            self.set_mode(AppMode.REALITYFRAME_ROI)

        elif key in (keys.key_mode_ghost_portal_3, ord("3")):
            self.set_mode(AppMode.REALITYFRAME_GHOST)

        elif key in (keys.key_mode_keyboard_4, ord("4")):
            self.set_mode(AppMode.VIRTUAL_KEYBOARD)

        elif key in (keys.key_mode_air_writing_5, ord("5")):
            self.set_mode(AppMode.AIR_WRITING)

        elif key in (keys.key_mode_mystic_magic_6, ord("6")):
            self.set_mode(AppMode.MYSTIC_MAGIC)

        elif key in (keys.key_mode_universe_7, ord("7")):
            self.set_mode(AppMode.UNIVERSE_PORTAL)

        # Toggle Universe Portal: 'u', 'U'
        elif key in (keys.key_toggle_universe_u, ord("U")):
            if self.current_mode == AppMode.UNIVERSE_PORTAL:
                self.set_mode(AppMode.STANDARD_FILTER)
            else:
                self.set_mode(AppMode.UNIVERSE_PORTAL)

        # Toggle Planetary Scale Mode: 'x', 'X'
        elif key in (keys.key_toggle_scale_x, ord("X")):
            if self.current_mode == AppMode.UNIVERSE_PORTAL:
                self.universe_portal.toggle_scale_mode()

        # Toggle Air Writing: 'w', 'W'
        elif key in (keys.key_toggle_writing_w, ord("W")):
            if self.current_mode == AppMode.AIR_WRITING:
                self.set_mode(AppMode.STANDARD_FILTER)
            else:
                self.set_mode(AppMode.AIR_WRITING)

        # Toggle Magic / Glow: 'g', 'G'
        elif key in (keys.key_toggle_magic_g, ord("G")):
            if self.current_mode == AppMode.AIR_WRITING:
                self.air_writing.toggle_glow()
            elif self.current_mode == AppMode.MYSTIC_MAGIC:
                self.mystic_magic.toggle_color_scheme()
            else:
                self.set_mode(AppMode.MYSTIC_MAGIC)

        # Undo: 'z', 'Z'
        elif key in (keys.key_undo_z, ord("Z")):
            if self.current_mode == AppMode.AIR_WRITING:
                self.air_writing.undo()

        # Redo: 'y', 'Y'
        elif key in (keys.key_redo_y, ord("Y")):
            if self.current_mode == AppMode.AIR_WRITING:
                self.air_writing.redo()

        # Save Drawing: 's', 'S'
        elif key in (keys.key_save_s, ord("S")):
            if self.current_mode == AppMode.AIR_WRITING:
                self.air_writing.export_png()

        # Toggle Virtual Keyboard: 'k', 'K'
        elif key in (keys.key_toggle_keyboard_k, ord("K")):
            if self.current_mode == AppMode.VIRTUAL_KEYBOARD:
                self.set_mode(AppMode.STANDARD_FILTER)
            else:
                self.set_mode(AppMode.VIRTUAL_KEYBOARD)

        # Recapture Background: 'r', 'R'
        elif key in (keys.key_recapture_bg_r, ord("R")):
            self.trigger_background_recapture()

        # Next filter: 'n', 'N', 'm', 'M'
        elif key in (keys.key_next_filter_n, ord("N"), keys.key_cycle_filter_m, ord("M")):
            new_filter = self.filter_engine.cycle_next()
            logger.info("Filter switched to: %s (%s)", self.filter_engine.active_name, new_filter)

        # Space key: Pause universe portal if in Universe mode, else cycle next filter
        elif key == keys.key_cycle_filter_space:
            if self.current_mode == AppMode.UNIVERSE_PORTAL:
                self.universe_portal.toggle_pause()
            else:
                new_filter = self.filter_engine.cycle_next()
                logger.info("Filter switched to: %s (%s)", self.filter_engine.active_name, new_filter)

        # Previous filter: 'p', 'P'
        elif key in (keys.key_prev_filter_p, ord("P")):
            prev_filter = self.filter_engine.cycle_prev()
            logger.info("Filter switched to: %s (%s)", self.filter_engine.active_name, prev_filter)

        # Toggle Typing Mode / Cycle Color: 't', 'T'
        elif key in (keys.key_toggle_typing_mode_t, ord("T")):
            if self.current_mode == AppMode.AIR_WRITING:
                c = self.air_writing.cycle_color()
                logger.info("Air writing color cycled to: %s", c)
            else:
                new_mode = self.keyboard_engine.toggle_mode()
                logger.info("Typing mode toggled to: %s", new_mode.value)

        # Clear Canvas, Typed Text, or Close Universe Portal: 'c', 'C'
        elif key in (keys.key_clear_text_c, ord("C")):
            if self.current_mode == AppMode.AIR_WRITING:
                self.air_writing.clear_canvas()
                logger.info("Air writing canvas cleared.")
            elif self.current_mode == AppMode.VIRTUAL_KEYBOARD:
                self.keyboard_engine.clear_text()
                logger.info("Virtual keyboard text cleared.")
            elif self.current_mode == AppMode.UNIVERSE_PORTAL:
                self.universe_portal.close_portal()
                logger.info("Universe portal close requested.")

        # Brush size adjustment: '+', '=', '-'
        elif key in (ord("+"), ord("=")):
            if self.current_mode == AppMode.AIR_WRITING:
                self.air_writing.increase_brush_size()
        elif key in (ord("-"), ord("_")):
            if self.current_mode == AppMode.AIR_WRITING:
                self.air_writing.decrease_brush_size()

        # Toggle performance / debug HUD: 'f', 'F', 'h', 'H'
        elif key in (keys.key_toggle_hud_f, ord("F"), keys.key_toggle_hud_h, ord("H")):
            metrics_visible = self.hud.toggle_debug_metrics()
            logger.info("Performance telemetry toggled: %s", "Visible" if metrics_visible else "Hidden")

    def shutdown(self) -> None:
        """Clean up all hardware and GUI resources reliably."""
        logger.info("Releasing all application resources...")
        self._is_running = False

        if self.camera is not None:
            self.camera.release()

        if self.hand_tracker is not None:
            self.hand_tracker.close()

        try:
            cv2.destroyAllWindows()
        except Exception as e:
            logger.debug("cv2.destroyAllWindows exception: %s", e)

        logger.info("Application shutdown complete.")
