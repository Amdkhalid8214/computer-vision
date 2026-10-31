"""RealityFrame Cinematic Universe Portal Manager.

Coordinates circle gesture recognition, magical portal opening transitions,
anti-aliased portal masking, 3D Solar System compositing, hand-controlled
cosmic exploration, and educational planetary telemetry.
"""

from dataclasses import dataclass
from enum import Enum
import logging
import math
import time
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np

from config import UniversePortalConfig
from src.circle_detector import CircleDetector, CircleDetectionResult
from src.hand_tracker import HandDetection
from src.particle_system import ParticleSystem
from src.universe_engine import Universe3DEngine, CELESTIAL_SPECS, PlanetSpec

logger = logging.getLogger(__name__)


class PortalLifecycle(Enum):
    """Lifecycle stages of the Cinematic Universe Portal."""
    IDLE = "IDLE"              # Waiting for circle drawing gesture
    DRAWING = "DRAWING"        # Tracking index finger trajectory
    CHARGING = "CHARGING"      # Circle validated: ignition of energy ring & runes
    DISTORTING = "DISTORTING"  # Space-time localized cosmic distortion
    OPENING = "OPENING"        # Progressive reveal of the 3D universe
    ACTIVE = "ACTIVE"          # Interactive 3D Solar System active inside portal
    CLOSING = "CLOSING"        # Dissolving portal with fading spark embers


class UniversePortalManager:
    """Orchestrates gesture-drawn circle portal activation, 3D compositing, and universe exploration."""

    def __init__(self, config: Optional[UniversePortalConfig] = None) -> None:
        self.config = config or UniversePortalConfig()

        self.circle_detector = CircleDetector(
            min_points=self.config.min_stroke_points,
            min_radius_px=self.config.min_circle_radius_px,
            max_radius_ratio=self.config.max_circle_radius_ratio,
            circularity_threshold=self.config.circularity_threshold,
            closure_threshold_ratio=self.config.closure_threshold_ratio,
            min_angular_coverage_rad=self.config.min_angular_coverage_rad,
        )

        self.universe_engine = Universe3DEngine(render_width=512, render_height=512)
        self.particles = ParticleSystem(max_particles=300)

        # State Machine
        self.state: PortalLifecycle = PortalLifecycle.IDLE
        self.state_time: float = 0.0

        # Circle Drawing State
        self.drawn_stroke: List[Tuple[int, int]] = []
        self.is_pinching_draw: bool = False
        self.was_pinching_draw: bool = False
        self.current_cursor: Optional[Tuple[int, int]] = None
        self.smoothed_idx: Optional[Tuple[float, float]] = None
        self.idle_timeout: float = 0.0

        # Stabilized Portal Geometry
        self.portal_center: Tuple[int, int] = (640, 360)
        self.portal_radius: float = 180.0
        self.current_anim_radius: float = 0.0

        # Animation Timing Parameters
        self.charge_duration_sec: float = 0.8
        self.distortion_duration_sec: float = 0.6
        self.opening_duration_sec: float = 1.0
        self.closing_duration_sec: float = 0.6

        # Hand Navigation State
        self.prev_nav_pos: Optional[Tuple[int, int]] = None
        self.is_zooming: bool = False
        self.nav_cooldown: float = 0.0

        # Quality Preset & Scale
        self.preset: str = self.config.default_preset
        self.border_color = self.config.portal_border_color_bgr
        self.accent_color = self.config.border_accent_color_bgr

        # Status Toast
        self.status_toast: str = "DRAW A CIRCLE IN THE AIR TO OPEN PORTAL"
        self.status_toast_timer: float = 3.5

    def set_toast(self, message: str, duration: float = 2.0) -> None:
        """Display floating informative notification."""
        self.status_toast = message
        self.status_toast_timer = duration

    def set_quality_preset(self, preset_name: str) -> None:
        """Set visual preset: 'PERFORMANCE', 'BALANCED', 'CINEMATIC'."""
        p = preset_name.upper()
        if p in ("PERFORMANCE", "BALANCED", "CINEMATIC"):
            self.preset = p
            if p == "PERFORMANCE":
                self.universe_engine.render_width = 384
                self.universe_engine.render_height = 384
                self.particles.max_particles = 150
            elif p == "BALANCED":
                self.universe_engine.render_width = 512
                self.universe_engine.render_height = 512
                self.particles.max_particles = 300
            elif p == "CINEMATIC":
                self.universe_engine.render_width = 640
                self.universe_engine.render_height = 640
                self.particles.max_particles = 450
            self.set_toast(f"PRESET: {p}")

    def close_portal(self) -> None:
        """Initiate portal closing animation."""
        if self.state in (PortalLifecycle.ACTIVE, PortalLifecycle.OPENING, PortalLifecycle.CHARGING):
            self.state = PortalLifecycle.CLOSING
            self.state_time = 0.0
            self.particles.emit_burst(
                self.portal_center[0], self.portal_center[1], count=80, color=self.border_color
            )
            self.set_toast("CLOSING UNIVERSE PORTAL...")

    def reset_to_idle(self) -> None:
        """Reset state cleanly to IDLE."""
        self.state = PortalLifecycle.IDLE
        self.state_time = 0.0
        self.drawn_stroke.clear()
        self.particles.clear()
        self.prev_nav_pos = None
        self.current_cursor = None
        self.smoothed_idx = None
        self.idle_timeout = 0.0
        self.universe_engine.reset_view()
        self.set_toast("DRAW A CIRCLE IN THE AIR TO OPEN PORTAL")

    def toggle_scale_mode(self) -> str:
        """Toggle planetary scaling between CINEMATIC and SCIENTIFIC."""
        new_mode = self.universe_engine.toggle_scale_mode()
        self.set_toast(f"SCALE MODE: {new_mode}")
        return new_mode

    def toggle_pause(self) -> bool:
        """Pause or resume orbital simulation."""
        paused = self.universe_engine.toggle_pause()
        self.set_toast("SIMULATION PAUSED" if paused else "SIMULATION RESUMED")
        return paused

    def update(
        self,
        detections: List[HandDetection],
        dt: float = 0.033,
        frame_shape: Tuple[int, int] = (720, 1280),
    ) -> None:
        """Update portal state machine, particle physics, 3D simulation, and hand gestures."""
        dt = max(0.001, min(0.1, dt))
        self.state_time += dt

        if self.status_toast_timer > 0.0:
            self.status_toast_timer = max(0.0, self.status_toast_timer - dt)

        if self.nav_cooldown > 0.0:
            self.nav_cooldown = max(0.0, self.nav_cooldown - dt)

        # Update 3D simulation & particle physics
        self.universe_engine.update(dt)
        self.particles.update(dt)

        num_hands = len(detections)
        primary_hand = detections[0] if num_hands >= 1 else None

        # Track cursor position whenever a hand is visible
        if primary_hand is not None:
            self.current_cursor = primary_hand.index_tip
        else:
            self.current_cursor = None

        # -------------------------------------------------------------
        # STATE 1 & 2: IDLE & DRAWING CIRCLE GESTURE
        # -------------------------------------------------------------
        if self.state in (PortalLifecycle.IDLE, PortalLifecycle.DRAWING):
            if primary_hand is not None:
                idx_tip = primary_hand.index_tip
                pinch_dist = primary_hand.pinch_distance

                # Active drawing can be triggered by:
                # 1) Pinch gesture (thumb + index pinch <= 50px)
                # 2) Pointing index finger gesture (is_pointing)
                # 3) Index extended motion (while palm is not fully open)
                is_pinch = pinch_dist <= 50.0
                is_pointing = getattr(primary_hand, "is_pointing", False)
                is_extended = primary_hand.is_finger_extended(8, 6) if hasattr(primary_hand, "is_finger_extended") else True
                is_open_palm = getattr(primary_hand, "is_open_palm", False)

                is_drawing = is_pinch or is_pointing or (is_extended and not is_open_palm)
                self.is_pinching_draw = is_drawing

                if is_drawing:
                    # Exponential smoothing on incoming coordinates to filter camera tremor
                    if self.smoothed_idx is None:
                        self.smoothed_idx = (float(idx_tip[0]), float(idx_tip[1]))
                    else:
                        sx = 0.55 * idx_tip[0] + 0.45 * self.smoothed_idx[0]
                        sy = 0.55 * idx_tip[1] + 0.45 * self.smoothed_idx[1]
                        self.smoothed_idx = (sx, sy)

                    pt = (int(round(self.smoothed_idx[0])), int(round(self.smoothed_idx[1])))

                    # Filter micro-movements (< 4px) to ensure uniform point density
                    if not self.drawn_stroke or math.hypot(pt[0] - self.drawn_stroke[-1][0], pt[1] - self.drawn_stroke[-1][1]) >= 4.0:
                        self.drawn_stroke.append(pt)
                        self.state = PortalLifecycle.DRAWING
                        self.idle_timeout = 0.0

                        # Keep trajectory manageable (max 100 points)
                        if len(self.drawn_stroke) > 100:
                            self.drawn_stroke.pop(0)

                        # Emit trailing guided spark
                        self.particles.emit_spark(
                            pt[0], pt[1], 0, 0, color=self.accent_color, size=3.0, life=0.3
                        )

                        # LIVE ROLLING CIRCLE DETECTION:
                        # As soon as the user completes a closed circular trajectory in the air, ignite immediately!
                        if len(self.drawn_stroke) >= 16:
                            res: CircleDetectionResult = self.circle_detector.detect(
                                self.drawn_stroke, frame_shape=frame_shape
                            )
                            if res.is_circle:
                                self.portal_center = res.center
                                self.portal_radius = res.radius
                                self.state = PortalLifecycle.CHARGING
                                self.state_time = 0.0
                                self.particles.emit_ring_sparks(
                                    res.center[0], res.center[1], res.radius, count=30, color=self.accent_color
                                )
                                self.set_toast(f"PORTAL IGNITED! CIRCULARITY: {res.circularity_score * 100:.0f}%")
                                self.drawn_stroke.clear()
                                self.smoothed_idx = None

                else:
                    # Gesture paused or released: evaluate stroke if sufficiently formed
                    if self.state == PortalLifecycle.DRAWING and len(self.drawn_stroke) >= 14:
                        res: CircleDetectionResult = self.circle_detector.detect(
                            self.drawn_stroke, frame_shape=frame_shape
                        )
                        if res.is_circle:
                            self.portal_center = res.center
                            self.portal_radius = res.radius
                            self.state = PortalLifecycle.CHARGING
                            self.state_time = 0.0
                            self.particles.emit_ring_sparks(
                                res.center[0], res.center[1], res.radius, count=30, color=self.accent_color
                            )
                            self.set_toast(f"PORTAL IGNITED! CIRCULARITY: {res.circularity_score * 100:.0f}%")
                            self.drawn_stroke.clear()
                            self.smoothed_idx = None
                        else:
                            # Grace timeout before resetting so momentary pauses don't immediately wipe stroke
                            self.idle_timeout += dt
                            if self.idle_timeout > 0.8:
                                self.set_toast(f"TRY AGAIN: {res.error_message}")
                                self.drawn_stroke.clear()
                                self.state = PortalLifecycle.IDLE
                                self.idle_timeout = 0.0
                                self.smoothed_idx = None
                    elif not is_drawing and len(self.drawn_stroke) < 14:
                        self.idle_timeout += dt
                        if self.idle_timeout > 0.5:
                            self.drawn_stroke.clear()
                            self.state = PortalLifecycle.IDLE
                            self.idle_timeout = 0.0

                self.was_pinching_draw = is_drawing
            else:
                self.idle_timeout += dt
                if self.idle_timeout > 0.8:
                    self.drawn_stroke.clear()
                    self.state = PortalLifecycle.IDLE
                    self.was_pinching_draw = False
                    self.smoothed_idx = None

        # -------------------------------------------------------------
        # STATE 3: CHARGING (Energy Ring & Sparks Ignition)
        # -------------------------------------------------------------
        elif self.state == PortalLifecycle.CHARGING:
            cx, cy = self.portal_center
            self.particles.emit_ring_sparks(cx, cy, self.portal_radius, count=8, color=self.border_color)
            if self.state_time >= self.charge_duration_sec:
                self.state = PortalLifecycle.DISTORTING
                self.state_time = 0.0

        # -------------------------------------------------------------
        # STATE 4: DISTORTING (Space-Time Localized Warp)
        # -------------------------------------------------------------
        elif self.state == PortalLifecycle.DISTORTING:
            cx, cy = self.portal_center
            self.particles.emit_ring_sparks(cx, cy, self.portal_radius, count=12, color=self.accent_color)
            if self.state_time >= self.distortion_duration_sec:
                self.state = PortalLifecycle.OPENING
                self.state_time = 0.0

        # -------------------------------------------------------------
        # STATE 5: OPENING (Cosmic Iris Opening to Universe)
        # -------------------------------------------------------------
        elif self.state == PortalLifecycle.OPENING:
            cx, cy = self.portal_center
            self.particles.emit_ring_sparks(cx, cy, self.portal_radius, count=10, color=self.accent_color)
            if self.state_time >= self.opening_duration_sec:
                self.state = PortalLifecycle.ACTIVE
                self.state_time = 0.0
                self.set_toast("UNIVERSE PORTAL OPEN: PINCH TO ZOOM, MOVE TO ORBIT")

        # -------------------------------------------------------------
        # STATE 6: ACTIVE (Interactive 3D Universe Exploration)
        # -------------------------------------------------------------
        elif self.state == PortalLifecycle.ACTIVE:
            cx, cy = self.portal_center
            # Keep subtle sparks on portal border
            if self.config.enable_portal_sparks:
                self.particles.emit_ring_sparks(cx, cy, self.portal_radius, count=3, color=self.border_color)

            # Hand Navigation in Universe
            if primary_hand is not None:
                idx = primary_hand.index_tip
                pinch_dist = primary_hand.pinch_distance

                if self.prev_nav_pos is not None:
                    dx = idx[0] - self.prev_nav_pos[0]
                    dy = idx[1] - self.prev_nav_pos[1]

                    if pinch_dist <= 38.0:
                        # Pinch + vertical drag: Zoom in/out
                        self.universe_engine.zoom(dy * 2.2)
                    elif primary_hand.is_open_palm:
                        # Open palm movement: Rotate/orbit camera
                        self.universe_engine.rotate(d_yaw=-dx * 0.012, d_pitch=dy * 0.012)
                    elif primary_hand.is_pointing:
                        # Pointing: check planet selection
                        if self.nav_cooldown <= 0.0:
                            self._check_planet_hit_test(idx)

                self.prev_nav_pos = idx
            else:
                self.prev_nav_pos = None

        # -------------------------------------------------------------
        # STATE 7: CLOSING (Shrinking with Dissolving Embers)
        # -------------------------------------------------------------
        elif self.state == PortalLifecycle.CLOSING:
            if self.state_time >= self.closing_duration_sec:
                self.reset_to_idle()

    def _check_planet_hit_test(self, cursor: Tuple[int, int]) -> None:
        """Check if index fingertip is pointing at a planet's screen position."""
        cx, cy = self.portal_center
        # Relative cursor in portal coordinates
        rel_x = cursor[0] - (cx - int(self.portal_radius))
        rel_y = cursor[1] - (cy - int(self.portal_radius))
        p_diam = int(self.portal_radius * 2)

        if not (0 <= rel_x < p_diam and 0 <= rel_y < p_diam):
            return

        # Scale relative position to 3D universe render canvas
        ren_w = self.universe_engine.render_width
        ren_h = self.universe_engine.render_height
        norm_x = (rel_x / p_diam) * ren_w
        norm_y = (rel_y / p_diam) * ren_h

        for spec in CELESTIAL_SPECS:
            pos = self.universe_engine._get_planet_world_pos(spec.name)
            # Find closest planet
            eye_dist = math.hypot(pos[0], math.hypot(pos[1], pos[2]))
            if eye_dist > 0:
                pass

    def render(self, frame: np.ndarray) -> np.ndarray:
        """Composite the 3D Universe strictly inside the circular portal mask."""
        h, w = frame.shape[:2]
        cx, cy = self.portal_center
        r = int(round(self.portal_radius))

        # 1. Render Circle Drawing Feedback while in IDLE or DRAWING
        if self.state in (PortalLifecycle.IDLE, PortalLifecycle.DRAWING):
            # Render multi-layer neon energy trajectory with zero frame blur
            if len(self.drawn_stroke) > 1:
                pts = np.array(self.drawn_stroke, dtype=np.int32).reshape((-1, 1, 2))
                # Outer fiery glow
                cv2.polylines(frame, [pts], False, self.border_color, 6, cv2.LINE_AA)
                # Inner mystic gold core
                cv2.polylines(frame, [pts], False, self.accent_color, 3, cv2.LINE_AA)
                # White-hot energy centerline
                cv2.polylines(frame, [pts], False, (255, 255, 255), 1, cv2.LINE_AA)

            # Render Doctor Strange Targeting Cursor at index fingertip
            if self.current_cursor is not None:
                tx, ty = self.current_cursor
                cursor_col = (0, 255, 120) if self.is_pinching_draw else self.accent_color
                reticle_r = 14
                cv2.circle(frame, (tx, ty), reticle_r, cursor_col, 1, cv2.LINE_AA)
                cv2.circle(frame, (tx, ty), 3, (255, 255, 255), -1, cv2.LINE_AA)
                # Compass crosshair ticks
                cv2.line(frame, (tx - reticle_r - 4, ty), (tx - reticle_r + 2, ty), cursor_col, 1, cv2.LINE_AA)
                cv2.line(frame, (tx + reticle_r - 2, ty), (tx + reticle_r + 4, ty), cursor_col, 1, cv2.LINE_AA)
                cv2.line(frame, (tx, ty - reticle_r - 4), (tx, ty - reticle_r + 2), cursor_col, 1, cv2.LINE_AA)
                cv2.line(frame, (tx, ty + reticle_r - 2), (tx, ty + reticle_r + 4), cursor_col, 1, cv2.LINE_AA)

                label = "DRAWING..." if self.is_pinching_draw else "DRAW CIRCLE"
                cv2.putText(
                    frame,
                    label,
                    (tx + 18, ty + 4),
                    cv2.FONT_HERSHEY_DUPLEX,
                    0.42,
                    cursor_col,
                    1,
                    cv2.LINE_AA,
                )

        # 2. Render Portal Opening / Active / Closing Stages
        elif self.state in (
            PortalLifecycle.CHARGING,
            PortalLifecycle.DISTORTING,
            PortalLifecycle.OPENING,
            PortalLifecycle.ACTIVE,
            PortalLifecycle.CLOSING,
        ):
            # Calculate current animated radius
            if self.state == PortalLifecycle.CHARGING:
                anim_progress = min(1.0, self.state_time / self.charge_duration_sec)
                effective_r = r * 0.95 + 4.0 * math.sin(anim_progress * math.pi * 3.0)
            elif self.state == PortalLifecycle.OPENING:
                anim_progress = min(1.0, self.state_time / self.opening_duration_sec)
                effective_r = r * anim_progress
            elif self.state == PortalLifecycle.CLOSING:
                anim_progress = max(0.0, 1.0 - (self.state_time / self.closing_duration_sec))
                effective_r = r * anim_progress
            else:
                effective_r = float(r)

            eff_r = int(round(max(4.0, effective_r)))

            # Render 3D Universe Canvas
            universe_img = self.universe_engine.render(output_size=(eff_r * 2, eff_r * 2))

            # Composite 3D Universe inside circular portal mask
            frame = self._composite_portal_masked(frame, universe_img, (cx, cy), eff_r)

            # Render Doctor Strange Glowing Rune Ring Border
            self._render_portal_border(frame, (cx, cy), eff_r)

        # 3. Render Particles
        frame = self.particles.render(frame)

        # 4. Render Educational Planet Telemetry Card (if active and planet selected)
        if self.state == PortalLifecycle.ACTIVE:
            frame = self._render_planet_info_card(frame)

        # 5. Render Status Toast & Controls Header
        frame = self._render_portal_hud(frame)

        return frame

    def _composite_portal_masked(
        self,
        frame: np.ndarray,
        universe_img: np.ndarray,
        center: Tuple[int, int],
        radius: int,
    ) -> np.ndarray:
        """Composite 3D universe image onto frame using a feathered circular alpha mask."""
        fh, fw = frame.shape[:2]
        cx, cy = center
        r = radius

        # Bounding box of circular portal clipped to frame
        x1 = max(0, cx - r)
        y1 = max(0, cy - r)
        x2 = min(fw, cx + r)
        y2 = min(fh, cy + r)

        if x2 <= x1 or y2 <= y1:
            return frame

        # Sub-region dimensions
        box_w = x2 - x1
        box_h = y2 - y1

        # Resize universe image to match exactly 2r x 2r
        u_full = cv2.resize(universe_img, (2 * r, 2 * r))

        # Crop universe image to clipped sub-region
        u_offset_x = x1 - (cx - r)
        u_offset_y = y1 - (cy - r)
        u_crop = u_full[u_offset_y : u_offset_y + box_h, u_offset_x : u_offset_x + box_w]

        # Circular feathered mask
        mask = np.zeros((box_h, box_w), dtype=np.float32)
        local_cx = cx - x1
        local_cy = cy - y1
        cv2.circle(mask, (local_cx, local_cy), r, 1.0, -1, cv2.LINE_AA)
        # Soft 2-pixel anti-aliased edge feathering
        mask = cv2.GaussianBlur(mask, (5, 5), 0)
        mask = np.clip(mask * 1.10, 0.0, 1.0)[:, :, np.newaxis]

        # Blend: output = frame * (1 - mask) + universe * mask
        frame_slice = frame[y1:y2, x1:x2].astype(np.float32)
        u_crop_f = u_crop.astype(np.float32)

        blended = frame_slice * (1.0 - mask) + u_crop_f * mask
        frame[y1:y2, x1:x2] = np.clip(blended, 0, 255).astype(np.uint8)

        return frame

    def _render_portal_border(self, frame: np.ndarray, center: Tuple[int, int], radius: int) -> None:
        """Render multi-layer rotating Doctor Strange golden-orange magic border with rune ticks."""
        cx, cy = center
        r = radius
        if r < 10:
            return

        # Dual rotating angles
        t = self.universe_engine.sim_time
        angle1 = (t * 1.5) % (2.0 * math.pi)
        angle2 = (-t * 2.2) % (2.0 * math.pi)

        # Concentric glowing rings
        cv2.circle(frame, (cx, cy), r, self.border_color, 2, cv2.LINE_AA)
        cv2.circle(frame, (cx, cy), max(2, r - 5), self.accent_color, 1, cv2.LINE_AA)
        cv2.circle(frame, (cx, cy), r + 5, self.border_color, 1, cv2.LINE_AA)

        # Rotating Rune Ticks
        num_ticks = 24
        for i in range(num_ticks):
            theta = angle1 + (i * 2.0 * math.pi / num_ticks)
            x1 = int(cx + math.cos(theta) * (r - 6))
            y1 = int(cy + math.sin(theta) * (r - 6))
            x2 = int(cx + math.cos(theta) * (r + 4))
            y2 = int(cy + math.sin(theta) * (r + 4))
            cv2.line(frame, (x1, y1), (x2, y2), self.accent_color, 1, cv2.LINE_AA)

        # Counter-rotating energy arcs
        for a_start in (0.0, 1.2, 2.4, 3.6, 4.8):
            deg_start = int(math.degrees(angle2 + a_start))
            cv2.ellipse(frame, (cx, cy), (r + 2, r + 2), 0, deg_start, deg_start + 35, self.accent_color, 2, cv2.LINE_AA)

    def _render_planet_info_card(self, frame: np.ndarray) -> np.ndarray:
        """Render transparent educational information card for focused planet."""
        spec = self.universe_engine.get_selected_planet_spec()
        if not spec:
            return frame

        fh, fw = frame.shape[:2]
        card_w = 340
        card_h = 210
        card_x = fw - card_w - 18
        card_y = 65

        # Glassmorphic dark card overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (card_x, card_y), (card_x + card_w, card_y + card_h), (18, 20, 26), -1)
        cv2.addWeighted(overlay, 0.82, frame, 0.18, 0, frame)
        cv2.rectangle(frame, (card_x, card_y), (card_x + card_w, card_y + card_h), (0, 220, 255), 1, cv2.LINE_AA)

        # Title
        cv2.putText(
            frame,
            f"{spec.name.upper()}  [{spec.category}]",
            (card_x + 14, card_y + 26),
            cv2.FONT_HERSHEY_DUPLEX,
            0.52,
            (0, 255, 220),
            1,
            cv2.LINE_AA,
        )
        cv2.line(frame, (card_x + 14, card_y + 34), (card_x + card_w - 14, card_y + 34), (60, 75, 90), 1)

        # Factual details
        lines = [
            f"Diameter:      {spec.real_diameter_km:,} km",
            f"Distance:      {spec.real_distance_au:.2f} AU",
            f"Orbital Period: {spec.orbital_period_days:.1f} Earth days",
            f"Rotation:       {spec.rotation_period_hours:.1f} hours",
            f"Surface Temp:   {spec.surface_temp_c}",
        ]

        curr_y = card_y + 58
        for line in lines:
            cv2.putText(frame, line, (card_x + 14, curr_y), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (220, 225, 230), 1, cv2.LINE_AA)
            curr_y += 20

        # Notable fact wrap
        fact_text = f"Fact: {spec.notable_fact}"
        cv2.putText(frame, fact_text[:44], (card_x + 14, curr_y + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 210, 255), 1, cv2.LINE_AA)
        if len(fact_text) > 44:
            cv2.putText(frame, fact_text[44:88], (card_x + 14, curr_y + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 210, 255), 1, cv2.LINE_AA)

        return frame

    def _render_portal_hud(self, frame: np.ndarray) -> np.ndarray:
        """Render Universe Portal status pill, planet selector menu, and toasts."""
        fh, fw = frame.shape[:2]

        # Planet quick selector bar at bottom if portal is active
        if self.state == PortalLifecycle.ACTIVE:
            bar_h = 32
            bar_y = fh - bar_h - 10
            bar_w = min(fw - 40, 820)
            bar_x = (fw - bar_w) // 2

            overlay = frame.copy()
            cv2.rectangle(overlay, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (16, 18, 24), -1)
            cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (0, 200, 255), 1, cv2.LINE_AA)

            planets_txt = "PLANETS: [Sun] [Mer] [Ven] [Ear] [Mar] [Jup] [Sat] [Ura] [Nep]  |  [X] Scale  [Space] Pause  [U] Close"
            cv2.putText(
                frame,
                planets_txt,
                (bar_x + 16, bar_y + 21),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.40,
                (230, 235, 240),
                1,
                cv2.LINE_AA,
            )

        # Status toast
        if self.status_toast and self.status_toast_timer > 0.0:
            alpha = min(1.0, self.status_toast_timer * 2.0)
            (tw, th), _ = cv2.getTextSize(self.status_toast, cv2.FONT_HERSHEY_DUPLEX, 0.52, 1)
            tx = (fw - tw) // 2
            ty = 68

            toast_overlay = frame.copy()
            cv2.rectangle(toast_overlay, (tx - 14, ty - th - 8), (tx + tw + 14, ty + 8), (14, 16, 22), -1)
            cv2.addWeighted(toast_overlay, 0.85 * alpha, frame, 1.0 - 0.85 * alpha, 0, frame)
            cv2.putText(frame, self.status_toast, (tx, ty), cv2.FONT_HERSHEY_DUPLEX, 0.52, (0, 225, 255), 1, cv2.LINE_AA)

        return frame
