"""Mystic Magic Engine: Doctor Strange-Inspired Circular Mandalas, Spells & Particle Systems.

Features procedural concentric rotating rune rings, sacred geometry octagrams,
fiery amber/gold glowing energy shields, two-hand portal scaling/rotation,
energy beams, and gesture-controlled spell casting.
"""

from dataclasses import dataclass
from enum import Enum
import logging
import math
import time
from typing import List, Optional, Tuple
import cv2
import numpy as np

from config import MysticMagicConfig
from src.hand_tracker import HandDetection
from src.particle_system import ParticleSystem

logger = logging.getLogger(__name__)


class SpellType(Enum):
    """Categorization of active mystic spells."""
    NONE = "NONE"
    SHIELD = "MYSTIC SHIELD"
    BLAST = "ENERGY BLAST"
    BEAM = "ENERGY BEAM"
    PORTAL = "MYSTIC PORTAL"
    TRAIL = "ENERGY TRAIL"


@dataclass
class BlastEffect:
    """State of an expanding radial energy blast shockwave."""
    cx: float
    cy: float
    current_radius: float
    max_radius: float
    life: float
    max_life: float
    color: Tuple[int, int, int]


class MysticMagicManager:
    """Coordinates gesture-based spell activation, procedural mandala generation, and effects compositing."""

    def __init__(self, config: Optional[MysticMagicConfig] = None) -> None:
        self.config = config or MysticMagicConfig()

        self.primary_color: Tuple[int, int, int] = self.config.primary_color_bgr      # Fiery Amber Orange
        self.secondary_color: Tuple[int, int, int] = self.config.secondary_color_bgr  # Mystic Gold
        self.core_color: Tuple[int, int, int] = self.config.core_color_bgr            # Core White-Hot

        self.particles = ParticleSystem(max_particles=self.config.max_particles)
        self.active_spell: SpellType = SpellType.NONE
        self.spell_mode: str = self.config.default_spell  # "AUTO", "SHIELD", "BLAST", "BEAM", "PORTAL", "TRAIL"

        # Continuous animation angles (framerate-independent)
        self.angle_outer: float = 0.0
        self.angle_inner: float = 0.0
        self.anim_time: float = 0.0

        # Gesture tracking and cooldowns
        self.prev_index_pos: Optional[Tuple[int, int]] = None
        self.prev_pinch: bool = False
        self.blast_cooldown: float = 0.0
        self.active_blasts: List[BlastEffect] = []

        # Smoothed portal state for two-hand interaction
        self.smoothed_portal_center: Optional[Tuple[float, float]] = None
        self.smoothed_portal_radius: float = float(self.config.portal_base_radius)
        self.smoothed_portal_angle: float = 0.0

        # Status toast
        self.status_toast: str = "MYSTIC MAGIC ENGINE ACTIVE"
        self.status_toast_timer: float = 2.0

    def set_spell_mode(self, mode: str) -> None:
        """Set spell mode: 'AUTO', 'SHIELD', 'BLAST', 'BEAM', 'PORTAL', 'TRAIL'."""
        m = mode.upper()
        if m in ("AUTO", "SHIELD", "BLAST", "BEAM", "PORTAL", "TRAIL"):
            self.spell_mode = m
            self.status_toast = f"SPELL MODE: {m}"
            self.status_toast_timer = 1.8

    def toggle_color_scheme(self) -> None:
        """Cycle between Fiery Amber, Eldritch Cyan, and Amethyst Purple."""
        if self.primary_color == (0, 140, 255):
            # Switch to Eldritch Cyan
            self.primary_color = (255, 200, 0)
            self.secondary_color = (255, 255, 100)
            self.core_color = (255, 255, 255)
            self.status_toast = "THEME: ELDRITCH CYAN"
        elif self.primary_color == (255, 200, 0):
            # Switch to Amethyst Purple
            self.primary_color = (230, 40, 180)
            self.secondary_color = (255, 120, 240)
            self.core_color = (255, 230, 255)
            self.status_toast = "THEME: AMETHYST PURPLE"
        else:
            # Switch back to Fiery Amber Gold
            self.primary_color = self.config.primary_color_bgr
            self.secondary_color = self.config.secondary_color_bgr
            self.core_color = self.config.core_color_bgr
            self.status_toast = "THEME: FIERY AMBER GOLD"
        self.status_toast_timer = 1.8

    def clear_effects(self) -> None:
        """Clear all active spells, blasts, and particles."""
        self.particles.clear()
        self.active_blasts.clear()
        self.active_spell = SpellType.NONE

    def update(
        self,
        detections: List[HandDetection],
        dt: float = 0.033,
    ) -> None:
        """Process gestures, advance rotations, update particles and blast shockwaves."""
        dt = max(0.001, min(0.1, dt))
        self.anim_time += dt

        # Continuous counter-rotating angles
        self.angle_outer = (self.angle_outer + self.config.rotation_speed_outer * dt) % (2.0 * math.pi)
        self.angle_inner = (self.angle_inner + self.config.rotation_speed_inner * dt) % (2.0 * math.pi)

        # Update toast timer
        if self.status_toast_timer > 0.0:
            self.status_toast_timer = max(0.0, self.status_toast_timer - dt)

        # Update blast cooldown
        if self.blast_cooldown > 0.0:
            self.blast_cooldown = max(0.0, self.blast_cooldown - dt)

        # Update active blasts
        alive_blasts: List[BlastEffect] = []
        for b in self.active_blasts:
            b.life -= dt
            if b.life > 0.0:
                prog = 1.0 - (b.life / b.max_life)
                b.current_radius = b.max_radius * (1.0 - math.pow(1.0 - prog, 2))
                alive_blasts.append(b)
        self.active_blasts = alive_blasts

        # Update particle physics
        self.particles.update(dt)

        # Determine active spell from gestures
        num_hands = len(detections)
        if num_hands == 0:
            self.active_spell = SpellType.NONE
            self.prev_index_pos = None
            return

        h1 = detections[0]
        h2 = detections[1] if num_hands >= 2 else None

        # Check for Blast gesture: pinch release with rapid motion
        is_pinching = h1.pinch_distance <= 40.0
        if self.prev_pinch and not is_pinching and self.blast_cooldown <= 0.0:
            # Trigger Energy Blast!
            self._trigger_energy_blast(h1.index_tip)
            self.blast_cooldown = 0.6
            self.active_spell = SpellType.BLAST
        self.prev_pinch = is_pinching

        # Manual override mode vs automatic gesture detection
        if self.spell_mode != "AUTO":
            self.active_spell = SpellType[self.spell_mode]
        elif len(self.active_blasts) > 0:
            self.active_spell = SpellType.BLAST
        else:
            if num_hands >= 2:
                # Two hands -> Mystic Portal
                self.active_spell = SpellType.PORTAL
            elif h1.is_open_palm:
                # Open palm -> Mystic Shield
                self.active_spell = SpellType.SHIELD
            elif h1.is_pointing:
                # Pointing finger -> Energy Beam
                self.active_spell = SpellType.BEAM
            else:
                # Check for rapid finger movement -> Energy Trail
                speed = 0.0
                if self.prev_index_pos is not None:
                    dx = h1.index_tip[0] - self.prev_index_pos[0]
                    dy = h1.index_tip[1] - self.prev_index_pos[1]
                    speed = math.hypot(dx, dy) / dt

                if speed > 350.0:
                    self.active_spell = SpellType.TRAIL
                else:
                    self.active_spell = SpellType.NONE

        self.prev_index_pos = h1.index_tip

        # Emit particles based on active spell
        if self.active_spell == SpellType.SHIELD:
            cx, cy = h1.palm_center
            radius = max(60.0, h1.hand_size * 1.35)
            self.particles.emit_ring_sparks(cx, cy, radius, count=6, color=self.secondary_color)

        elif self.active_spell == SpellType.PORTAL and h2 is not None:
            # Two-hand midpoint and radius
            p1 = np.array(h1.palm_center, dtype=float)
            p2 = np.array(h2.palm_center, dtype=float)
            raw_center = ((p1[0] + p2[0]) * 0.5, (p1[1] + p2[1]) * 0.5)
            raw_radius = float(np.hypot(p1[0] - p2[0], p1[1] - p2[1]) * 0.65)
            raw_angle = float(math.atan2(p2[1] - p1[1], p2[0] - p1[0]))

            # Temporal smoothing for portal stability
            if self.smoothed_portal_center is None:
                self.smoothed_portal_center = raw_center
                self.smoothed_portal_radius = max(80.0, raw_radius)
                self.smoothed_portal_angle = raw_angle
            else:
                alpha = 0.35
                self.smoothed_portal_center = (
                    alpha * raw_center[0] + (1.0 - alpha) * self.smoothed_portal_center[0],
                    alpha * raw_center[1] + (1.0 - alpha) * self.smoothed_portal_center[1],
                )
                self.smoothed_portal_radius = alpha * raw_radius + (1.0 - alpha) * self.smoothed_portal_radius
                self.smoothed_portal_angle = raw_angle

            cx, cy = self.smoothed_portal_center
            self.particles.emit_ring_sparks(cx, cy, self.smoothed_portal_radius, count=10, color=self.secondary_color)

        elif self.active_spell == SpellType.BEAM:
            start_pt = h1.index_tip
            # Vector from middle palm or landmark 6 to 8 gives pointing direction
            w = h1.landmarks_pixel[6] if len(h1.landmarks_pixel) > 6 else h1.wrist
            dx = float(start_pt[0] - w[0])
            dy = float(start_pt[1] - w[1])
            length = math.hypot(dx, dy)
            if length > 5.0:
                ux = dx / length
                uy = dy / length
                end_pt = (start_pt[0] + ux * 600.0, start_pt[1] + uy * 600.0)
                self.particles.emit_beam_particles(start_pt, end_pt, count=6, color=self.core_color)

        elif self.active_spell == SpellType.TRAIL:
            self.particles.emit_trail_particle(
                h1.index_tip[0], h1.index_tip[1], color=self.primary_color, count=3
            )

    def _trigger_energy_blast(self, center: Tuple[int, int]) -> None:
        """Trigger an explosive radial shockwave and spark burst."""
        cx, cy = center
        self.active_blasts.append(
            BlastEffect(
                cx=float(cx),
                cy=float(cy),
                current_radius=10.0,
                max_radius=220.0,
                life=0.55,
                max_life=0.55,
                color=self.secondary_color,
            )
        )
        self.particles.emit_burst(cx, cy, count=75, color=self.primary_color, life=0.85)

    def render(self, frame: np.ndarray, detections: List[HandDetection]) -> np.ndarray:
        """Render all mystic mandalas, energy shields, beams, and particles onto the frame."""
        h, w = frame.shape[:2]
        num_hands = len(detections)

        # Create overlay layer for procedural glowing mandalas
        mandala_layer = np.zeros_like(frame)

        # 1. Render Active Spell Geometry onto mandala layer
        if num_hands >= 1:
            h1 = detections[0]
            h2 = detections[1] if num_hands >= 2 else None

            if self.active_spell == SpellType.SHIELD:
                cx, cy = h1.palm_center
                radius = int(max(60, h1.hand_size * 1.35))
                self._draw_mystic_mandala(mandala_layer, cx, cy, radius)

            elif self.active_spell == SpellType.PORTAL and self.smoothed_portal_center is not None:
                cx = int(round(self.smoothed_portal_center[0]))
                cy = int(round(self.smoothed_portal_center[1]))
                radius = int(round(self.smoothed_portal_radius))
                self._draw_mystic_mandala(mandala_layer, cx, cy, radius, rotation_offset=self.smoothed_portal_angle)

            elif self.active_spell == SpellType.BEAM:
                self._draw_energy_beam(mandala_layer, h1, w, h)

        # 2. Render Active Blast Shockwaves
        for b in self.active_blasts:
            alpha = max(0.0, min(1.0, b.life / b.max_life))
            rad = int(round(b.current_radius))
            thick = max(1, int(round(6.0 * alpha)))
            col = (int(b.color[0] * alpha), int(b.color[1] * alpha), int(b.color[2] * alpha))
            cv2.circle(mandala_layer, (int(b.cx), int(b.cy)), rad, col, thick, cv2.LINE_AA)
            cv2.circle(mandala_layer, (int(b.cx), int(b.cy)), max(1, rad - 8), (int(255 * alpha), int(255 * alpha), int(255 * alpha)), 1, cv2.LINE_AA)

        # 3. Additive Bloom & Compositing for Mandala Layer
        if np.any(mandala_layer):
            glow = cv2.GaussianBlur(mandala_layer, (25, 25), 0)
            frame = cv2.add(frame, glow)
            frame = cv2.add(frame, mandala_layer)

        # 4. Render Particle System
        frame = self.particles.render(frame)

        # 5. Render Mystic HUD Banner
        frame = self._render_mystic_hud(frame)

        return frame

    def _draw_mystic_mandala(
        self,
        canvas: np.ndarray,
        cx: int,
        cy: int,
        radius: int,
        rotation_offset: float = 0.0,
    ) -> None:
        """Render multi-layer rotating Doctor Strange mystic shield mandala.

        Includes concentric outer rings, radial rune dashes, interlaced octagrams,
        and pulsing center core.
        """
        h, w = canvas.shape[:2]
        if not (0 <= cx < w and 0 <= cy < h) or radius < 15:
            return

        r = float(radius)

        # 1. Outer Concentric Rings with Fine Energy Ticks (Clockwise rotation)
        cv2.circle(canvas, (cx, cy), int(r), self.primary_color, 2, cv2.LINE_AA)
        cv2.circle(canvas, (cx, cy), int(r * 0.94), self.secondary_color, 1, cv2.LINE_AA)
        cv2.circle(canvas, (cx, cy), int(r * 0.88), self.primary_color, 2, cv2.LINE_AA)

        # Outer Rune Ticks
        num_ticks = 24
        tick_r_in = r * 0.88
        tick_r_out = r * 0.94
        for i in range(num_ticks):
            theta = self.angle_outer + rotation_offset + (i * 2.0 * math.pi / num_ticks)
            x1 = int(round(cx + math.cos(theta) * tick_r_in))
            y1 = int(round(cy + math.sin(theta) * tick_r_in))
            x2 = int(round(cx + math.cos(theta) * tick_r_out))
            y2 = int(round(cy + math.sin(theta) * tick_r_out))
            cv2.line(canvas, (x1, y1), (x2, y2), self.secondary_color, 1, cv2.LINE_AA)

        # 2. Interlaced Geometric Octagram (Two overlapping rotated squares, Counter-Clockwise rotation)
        oct_r = r * 0.76
        for offset_angle in (0.0, math.pi / 4.0):
            pts = []
            for k in range(4):
                theta = self.angle_inner + rotation_offset + offset_angle + (k * math.pi / 2.0)
                px = int(round(cx + math.cos(theta) * oct_r))
                py = int(round(cy + math.sin(theta) * oct_r))
                pts.append([px, py])
            pts_arr = np.array(pts, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(canvas, [pts_arr], True, self.secondary_color, 2, cv2.LINE_AA)

        # 3. Middle Segmented Dashed Ring
        mid_r = int(r * 0.58)
        cv2.circle(canvas, (cx, cy), mid_r, self.primary_color, 1, cv2.LINE_AA)

        # 8 Radial Spokes connecting Octagram corners to center
        for k in range(8):
            theta = self.angle_inner + rotation_offset + (k * math.pi / 4.0)
            sx = int(round(cx + math.cos(theta) * (r * 0.30)))
            sy = int(round(cy + math.sin(theta) * (r * 0.30)))
            ex = int(round(cx + math.cos(theta) * (r * 0.58)))
            ey = int(round(cy + math.sin(theta) * (r * 0.58)))
            cv2.line(canvas, (sx, sy), (ex, ey), self.secondary_color, 1, cv2.LINE_AA)

        # 4. Pulsing Center Energy Core
        pulse = 1.0 + 0.08 * math.sin(self.anim_time * 7.0)
        core_r = max(4, int(round(r * 0.24 * pulse)))
        cv2.circle(canvas, (cx, cy), core_r, self.secondary_color, 2, cv2.LINE_AA)
        cv2.circle(canvas, (cx, cy), max(2, int(round(r * 0.12 * pulse))), self.core_color, -1, cv2.LINE_AA)

    def _draw_energy_beam(
        self,
        canvas: np.ndarray,
        hand: HandDetection,
        frame_w: int,
        frame_h: int,
    ) -> None:
        """Render high-intensity energy beam projecting from index fingertip."""
        start_pt = hand.index_tip
        w = hand.landmarks_pixel[6] if len(hand.landmarks_pixel) > 6 else hand.wrist
        dx = float(start_pt[0] - w[0])
        dy = float(start_pt[1] - w[1])
        length = math.hypot(dx, dy)
        if length < 5.0:
            return

        ux = dx / length
        uy = dy / length

        # Project beam to screen edge
        end_pt = (
            int(round(start_pt[0] + ux * 800.0)),
            int(round(start_pt[1] + uy * 800.0)),
        )

        # Outer amber glow line
        cv2.line(canvas, start_pt, end_pt, self.primary_color, 12, cv2.LINE_AA)
        # Secondary golden cylinder
        cv2.line(canvas, start_pt, end_pt, self.secondary_color, 6, cv2.LINE_AA)
        # Inner white-hot laser core
        cv2.line(canvas, start_pt, end_pt, self.core_color, 2, cv2.LINE_AA)

        # Glowing emitter flare at fingertip
        cv2.circle(canvas, start_pt, 10, self.secondary_color, -1, cv2.LINE_AA)
        cv2.circle(canvas, start_pt, 5, self.core_color, -1, cv2.LINE_AA)

    def _render_mystic_hud(self, frame: np.ndarray) -> np.ndarray:
        """Render active spell indicator pill and notification toasts."""
        h, w = frame.shape[:2]

        # Active Spell Pill (Bottom Center)
        pill_text = f"MYSTIC ARTS: {self.active_spell.value}"
        (tw, th), _ = cv2.getTextSize(pill_text, cv2.FONT_HERSHEY_DUPLEX, 0.52, 1)

        pill_w = tw + 28
        pill_h = 28
        pill_x = (w - pill_w) // 2
        pill_y = h - pill_h - 12

        overlay = frame.copy()
        cv2.rectangle(overlay, (pill_x, pill_y), (pill_x + pill_w, pill_y + pill_h), (16, 16, 22), -1)
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
        cv2.rectangle(frame, (pill_x, pill_y), (pill_x + pill_w, pill_y + pill_h), self.primary_color, 1, cv2.LINE_AA)

        cv2.putText(
            frame,
            pill_text,
            (pill_x + 14, pill_y + 19),
            cv2.FONT_HERSHEY_DUPLEX,
            0.52,
            self.secondary_color,
            1,
            cv2.LINE_AA,
        )

        # Floating toast notification
        if self.status_toast and self.status_toast_timer > 0.0:
            alpha = min(1.0, self.status_toast_timer * 2.0)
            (stw, sth), _ = cv2.getTextSize(self.status_toast, cv2.FONT_HERSHEY_DUPLEX, 0.54, 1)
            toast_x = (w - stw) // 2
            toast_y = 68

            toast_overlay = frame.copy()
            cv2.rectangle(
                toast_overlay,
                (toast_x - 14, toast_y - sth - 8),
                (toast_x + stw + 14, toast_y + 8),
                (15, 15, 20),
                -1,
            )
            cv2.addWeighted(toast_overlay, 0.8 * alpha, frame, 1.0 - 0.8 * alpha, 0, frame)
            cv2.putText(
                frame,
                self.status_toast,
                (toast_x, toast_y),
                cv2.FONT_HERSHEY_DUPLEX,
                0.54,
                self.secondary_color,
                1,
                cv2.LINE_AA,
            )

        return frame
