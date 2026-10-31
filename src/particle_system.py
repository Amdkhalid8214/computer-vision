"""High-performance 2D particle simulation engine for mystic sparks, energy bursts, and luminous trails.

Provides GPU-friendly additive compositing, lifespans, drag, tangential ring emissions,
and Doctor Strange-inspired fiery ember visual effects.
"""

from dataclasses import dataclass
import math
import random
from typing import List, Optional, Tuple
import cv2
import numpy as np


@dataclass
class Particle:
    """Single dynamic particle instance."""
    x: float
    y: float
    vx: float
    vy: float
    color: Tuple[int, int, int]  # BGR
    size: float
    life: float
    max_life: float
    spark: bool = True


class ParticleSystem:
    """Manages emission, spatial physics update, and additive rendering of particles."""

    def __init__(self, max_particles: int = 400) -> None:
        self.max_particles: int = max(50, max_particles)
        self.particles: List[Particle] = []
        self._overlay_cache: Optional[np.ndarray] = None

    def clear(self) -> None:
        """Remove all active particles."""
        self.particles.clear()

    @property
    def count(self) -> int:
        """Return number of currently active particles."""
        return len(self.particles)

    def emit_spark(
        self,
        x: float,
        y: float,
        vx: float,
        vy: float,
        color: Tuple[int, int, int] = (0, 140, 255),
        size: float = 2.5,
        life: float = 0.5,
        spark: bool = True,
    ) -> None:
        """Emit a single spark particle."""
        if len(self.particles) >= self.max_particles:
            return
        self.particles.append(
            Particle(
                x=x,
                y=y,
                vx=vx,
                vy=vy,
                color=color,
                size=max(1.0, size),
                life=life,
                max_life=life,
                spark=spark,
            )
        )

    def emit_burst(
        self,
        cx: float,
        cy: float,
        count: int = 50,
        speed_min: float = 60.0,
        speed_max: float = 280.0,
        color: Tuple[int, int, int] = (0, 140, 255),
        size: float = 3.0,
        life: float = 0.7,
    ) -> None:
        """Emit an explosive omnidirectional radial burst of sparks."""
        count = min(count, self.max_particles - len(self.particles))
        for _ in range(count):
            angle = random.uniform(0.0, 2.0 * math.pi)
            speed = random.uniform(speed_min, speed_max)
            vx = math.cos(angle) * speed
            vy = math.sin(angle) * speed
            p_life = life * random.uniform(0.6, 1.2)
            p_size = size * random.uniform(0.7, 1.3)
            # Add subtle color variance (amber, orange, gold)
            r_boost = random.randint(-15, 25)
            c = (
                max(0, min(255, color[0])),
                max(0, min(255, color[1] + r_boost)),
                max(0, min(255, color[2] + r_boost // 2)),
            )
            self.emit_spark(cx, cy, vx, vy, color=c, size=p_size, life=p_life, spark=True)

    def emit_ring_sparks(
        self,
        cx: float,
        cy: float,
        radius: float,
        count: int = 8,
        color: Tuple[int, int, int] = (0, 200, 255),
        life: float = 0.45,
    ) -> None:
        """Emit sparks around the perimeter of a circular mystic shield.

        Sparks fly off with tangential and slight outward velocity, creating the
        classic Doctor Strange shield spinning ember shower.
        """
        count = min(count, self.max_particles - len(self.particles))
        for _ in range(count):
            angle = random.uniform(0.0, 2.0 * math.pi)
            px = cx + math.cos(angle) * radius
            py = cy + math.sin(angle) * radius

            # Tangential velocity + outward flare
            tangent_speed = random.uniform(40.0, 140.0)
            radial_speed = random.uniform(10.0, 50.0)

            # Tangent vector (-sin, cos)
            vx = -math.sin(angle) * tangent_speed + math.cos(angle) * radial_speed
            vy = math.cos(angle) * tangent_speed + math.sin(angle) * radial_speed

            p_life = life * random.uniform(0.5, 1.1)
            p_size = random.uniform(1.5, 3.2)
            self.emit_spark(px, py, vx, vy, color=color, size=p_size, life=p_life, spark=True)

    def emit_beam_particles(
        self,
        start_pt: Tuple[float, float],
        end_pt: Tuple[float, float],
        count: int = 6,
        color: Tuple[int, int, int] = (200, 240, 255),
    ) -> None:
        """Emit luminous particles traveling along an energy beam line."""
        dx = end_pt[0] - start_pt[0]
        dy = end_pt[1] - start_pt[1]
        dist = math.hypot(dx, dy)
        if dist < 1.0:
            return

        ux = dx / dist
        uy = dy / dist

        count = min(count, self.max_particles - len(self.particles))
        for _ in range(count):
            t = random.uniform(0.0, 1.0)
            px = start_pt[0] + dx * t + random.uniform(-4.0, 4.0)
            py = start_pt[1] + dy * t + random.uniform(-4.0, 4.0)

            speed = random.uniform(120.0, 320.0)
            vx = ux * speed + random.uniform(-20.0, 20.0)
            vy = uy * speed + random.uniform(-20.0, 20.0)

            self.emit_spark(px, py, vx, vy, color=color, size=2.5, life=0.35, spark=True)

    def emit_trail_particle(
        self,
        x: float,
        y: float,
        vx: float = 0.0,
        vy: float = 0.0,
        color: Tuple[int, int, int] = (0, 180, 255),
        count: int = 2,
    ) -> None:
        """Emit trailing spark particles at a moving fingertip."""
        count = min(count, self.max_particles - len(self.particles))
        for _ in range(count):
            rx = x + random.uniform(-3.0, 3.0)
            ry = y + random.uniform(-3.0, 3.0)
            evx = vx * 0.2 + random.uniform(-25.0, 25.0)
            evy = vy * 0.2 + random.uniform(-25.0, 25.0)
            self.emit_spark(rx, ry, evx, evy, color=color, size=random.uniform(2.0, 4.0), life=0.4, spark=True)

    def update(self, dt: float) -> None:
        """Advance physics simulation by time delta dt."""
        dt = max(0.001, min(0.1, dt))
        drag = math.pow(0.96, dt * 30.0)
        updraft = 15.0 * dt

        alive: List[Particle] = []
        for p in self.particles:
            p.life -= dt
            if p.life <= 0.0:
                continue

            p.vx *= drag
            p.vy *= drag
            p.vy -= updraft  # Heat updraft

            p.x += p.vx * dt
            p.y += p.vy * dt
            alive.append(p)

        self.particles = alive

    def render(self, frame: np.ndarray) -> np.ndarray:
        """Render all alive particles onto the video frame using additive blending."""
        if not self.particles:
            return frame

        h, w = frame.shape[:2]
        overlay = np.zeros_like(frame)

        for p in self.particles:
            ix = int(round(p.x))
            iy = int(round(p.y))
            if not (0 <= ix < w and 0 <= iy < h):
                continue

            alpha = max(0.0, min(1.0, p.life / p.max_life))
            radius = max(1, int(round(p.size * alpha)))

            # Luminous colored outer halo
            b = int(p.color[0] * alpha)
            g = int(p.color[1] * alpha)
            r = int(p.color[2] * alpha)
            col = (b, g, r)

            cv2.circle(overlay, (ix, iy), radius, col, -1, cv2.LINE_AA)

            # Bright white core spark
            if p.spark and radius >= 2 and alpha > 0.4:
                core_alpha = (alpha - 0.4) / 0.6
                cw = int(255 * core_alpha)
                cv2.circle(overlay, (ix, iy), max(1, radius // 2), (cw, cw, cw), -1, cv2.LINE_AA)

        # Fast additive blend to achieve brilliant light bloom
        return cv2.add(frame, overlay)

