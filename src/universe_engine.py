"""3D Universe and Solar System Engine for RealityFrame Universe Portal.

Features real 3D perspective projection, orbital physics, Lambertian sphere
shading with day/night terminators, procedural planetary textures, 3D tilted
Saturn rings, asteroid belt, starfield parallax, nebulae, and interactive
camera navigation.
"""

from dataclasses import dataclass, field
import math
import random
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np


@dataclass
class PlanetSpec:
    """Astronomical specifications and visual properties for a celestial body."""
    name: str
    category: str  # Star, Terrestrial, Gas Giant, Ice Giant
    real_diameter_km: float
    real_distance_au: float
    orbital_period_days: float
    rotation_period_hours: float
    surface_temp_c: str
    notable_fact: str

    # Rendering properties
    visual_radius_cinematic: float
    orbit_distance_cinematic: float
    base_color_bgr: Tuple[int, int, int]
    secondary_color_bgr: Tuple[int, int, int]
    has_rings: bool = False
    ring_inner_r: float = 0.0
    ring_outer_r: float = 0.0
    ring_color_bgr: Tuple[int, int, int] = (180, 200, 210)
    axial_tilt_deg: float = 0.0


# Comprehensive factual Solar System dataset
CELESTIAL_SPECS: List[PlanetSpec] = [
    PlanetSpec(
        name="Sun",
        category="Yellow Dwarf Star (G2V)",
        real_diameter_km=1392700,
        real_distance_au=0.0,
        orbital_period_days=0.0,
        rotation_period_hours=600.0,
        surface_temp_c="5,500 °C (Core: 15M °C)",
        notable_fact="Contains 99.86% of all mass in the entire Solar System.",
        visual_radius_cinematic=28.0,
        orbit_distance_cinematic=0.0,
        base_color_bgr=(0, 215, 255),
        secondary_color_bgr=(255, 255, 255),
    ),
    PlanetSpec(
        name="Mercury",
        category="Terrestrial Planet",
        real_diameter_km=4879,
        real_distance_au=0.39,
        orbital_period_days=88.0,
        rotation_period_hours=1407.6,
        surface_temp_c="-180 °C to 430 °C",
        notable_fact="Smallest planet; has no atmosphere and experiences extreme temperature swings.",
        visual_radius_cinematic=6.0,
        orbit_distance_cinematic=48.0,
        base_color_bgr=(140, 145, 150),
        secondary_color_bgr=(100, 105, 110),
        axial_tilt_deg=0.03,
    ),
    PlanetSpec(
        name="Venus",
        category="Terrestrial Planet",
        real_diameter_km=12104,
        real_distance_au=0.72,
        orbital_period_days=224.7,
        rotation_period_hours=-5832.5,
        surface_temp_c="465 °C (Hottest planet)",
        notable_fact="Spins backwards (retrograde) and is covered in runaway greenhouse sulfuric acid clouds.",
        visual_radius_cinematic=9.0,
        orbit_distance_cinematic=72.0,
        base_color_bgr=(160, 205, 230),
        secondary_color_bgr=(120, 160, 190),
        axial_tilt_deg=177.3,
    ),
    PlanetSpec(
        name="Earth",
        category="Terrestrial Planet (Habitable)",
        real_diameter_km=12742,
        real_distance_au=1.00,
        orbital_period_days=365.25,
        rotation_period_hours=24.0,
        surface_temp_c="-88 °C to 58 °C (Avg: 15 °C)",
        notable_fact="The only known celestial body harboring life and vast liquid water oceans.",
        visual_radius_cinematic=10.0,
        orbit_distance_cinematic=102.0,
        base_color_bgr=(210, 140, 30),  # Blue ocean base
        secondary_color_bgr=(60, 180, 80), # Green/brown continents
        axial_tilt_deg=23.44,
    ),
    PlanetSpec(
        name="Mars",
        category="Terrestrial Planet",
        real_diameter_km=6779,
        real_distance_au=1.52,
        orbital_period_days=687.0,
        rotation_period_hours=24.6,
        surface_temp_c="-140 °C to 20 °C",
        notable_fact="Home to Olympus Mons, the largest volcano and mountain in the Solar System (21.9 km).",
        visual_radius_cinematic=7.5,
        orbit_distance_cinematic=136.0,
        base_color_bgr=(45, 65, 190),   # Rust-red iron oxide
        secondary_color_bgr=(230, 240, 255), # White polar caps
        axial_tilt_deg=25.19,
    ),
    PlanetSpec(
        name="Jupiter",
        category="Gas Giant",
        real_diameter_km=139820,
        real_distance_au=5.20,
        orbital_period_days=4332.6,
        rotation_period_hours=9.9,
        surface_temp_c="-110 °C",
        notable_fact="Has a persistent anticyclonic storm (Great Red Spot) larger than Earth itself.",
        visual_radius_cinematic=19.0,
        orbit_distance_cinematic=185.0,
        base_color_bgr=(140, 175, 210), # Banded ochre/cream
        secondary_color_bgr=(60, 85, 145),
        axial_tilt_deg=3.13,
    ),
    PlanetSpec(
        name="Saturn",
        category="Gas Giant",
        real_diameter_km=116460,
        real_distance_au=9.58,
        orbital_period_days=10759.2,
        rotation_period_hours=10.7,
        surface_temp_c="-140 °C",
        notable_fact="Has an extensive, magnificent ring system made primarily of water ice particles and dust.",
        visual_radius_cinematic=16.0,
        orbit_distance_cinematic=240.0,
        base_color_bgr=(160, 205, 225), # Pale gold
        secondary_color_bgr=(130, 165, 195),
        has_rings=True,
        ring_inner_r=22.0,
        ring_outer_r=38.0,
        ring_color_bgr=(180, 210, 230),
        axial_tilt_deg=26.73,
    ),
    PlanetSpec(
        name="Uranus",
        category="Ice Giant",
        real_diameter_km=50724,
        real_distance_au=19.22,
        orbital_period_days=30685.4,
        rotation_period_hours=-17.2,
        surface_temp_c="-195 °C",
        notable_fact="Rotates completely on its side with a dramatic 97.8° axial tilt.",
        visual_radius_cinematic=12.0,
        orbit_distance_cinematic=295.0,
        base_color_bgr=(225, 220, 160), # Pale Cyan / Aquamarine
        secondary_color_bgr=(190, 190, 130),
        axial_tilt_deg=97.77,
    ),
    PlanetSpec(
        name="Neptune",
        category="Ice Giant",
        real_diameter_km=49244,
        real_distance_au=30.05,
        orbital_period_days=60189.0,
        rotation_period_hours=16.1,
        surface_temp_c="-200 °C",
        notable_fact="Endures the fastest supersonic winds in the Solar System, reaching up to 2,100 km/h.",
        visual_radius_cinematic=11.5,
        orbit_distance_cinematic=345.0,
        base_color_bgr=(235, 145, 45),  # Deep Azure Blue
        secondary_color_bgr=(250, 190, 100),
        axial_tilt_deg=28.32,
    ),
]


@dataclass
class Star:
    """3D background star with celestial coordinates and color."""
    x: float
    y: float
    z: float
    size: float
    color: Tuple[int, int, int]
    twinkle_phase: float


@dataclass
class Asteroid:
    """Single rocky asteroid in the main asteroid belt."""
    orbit_r: float
    angle: float
    speed: float
    y_offset: float
    size: float
    color: Tuple[int, int, int]


class Universe3DEngine:
    """High-performance 3D Solar System and Deep-Space Rendering Engine."""

    def __init__(self, render_width: int = 640, render_height: int = 640) -> None:
        self.render_width = render_width
        self.render_height = render_height

        # Simulation Time and Planetary Motion
        self.sim_time: float = 0.0
        self.time_speed: float = 1.0
        self.is_paused: bool = False
        self.scale_mode: str = "CINEMATIC"  # "CINEMATIC" or "SCIENTIFIC"

        # 3D Camera Controls
        self.camera_yaw: float = 0.45       # Horizontal angle in radians
        self.camera_pitch: float = 0.55     # Elevation angle in radians
        self.camera_dist: float = 620.0     # Distance from camera target
        self.min_dist: float = 60.0
        self.max_dist: float = 1400.0

        # Camera Target & Smooth Interpolation
        self.current_target: np.ndarray = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        self.desired_target: np.ndarray = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        self.selected_planet_name: Optional[str] = None

        # Stars and Asteroids
        self.stars: List[Star] = []
        self.asteroids: List[Asteroid] = []
        self._init_starfield(count=550)
        self._init_asteroid_belt(count=220)

        # Pre-computed procedural nebula canvas
        self._nebula_layer = self._generate_procedural_nebula(render_width, render_height)

    def _init_starfield(self, count: int = 550) -> None:
        """Initialize spherical 3D star coordinates with realistic stellar classifications."""
        random.seed(1337)
        self.stars.clear()

        # Stellar colors based on spectral class (BGR)
        star_palette = [
            (255, 230, 200),  # Class O/B (Blue-white)
            (255, 255, 255),  # Class A (Pure White)
            (180, 240, 255),  # Class F/G (Yellowish-white)
            (80, 160, 255),   # Class K (Orange)
            (80, 80, 255),    # Class M (Red Giant / Dwarf)
        ]

        sphere_radius = 2400.0
        for _ in range(count):
            # Uniform distribution on sphere
            u = random.uniform(-1.0, 1.0)
            theta = random.uniform(0.0, 2.0 * math.pi)
            sqrt_term = math.sqrt(1.0 - u * u)
            x = sphere_radius * sqrt_term * math.cos(theta)
            y = sphere_radius * u
            z = sphere_radius * sqrt_term * math.sin(theta)

            sz = random.choices([1.0, 1.5, 2.0, 2.5], weights=[0.6, 0.25, 0.10, 0.05])[0]
            col = random.choice(star_palette)
            self.stars.append(Star(x, y, z, sz, col, random.uniform(0.0, 2.0 * math.pi)))

    def _init_asteroid_belt(self, count: int = 220) -> None:
        """Initialize main asteroid belt particles between Mars and Jupiter."""
        random.seed(42)
        self.asteroids.clear()
        for _ in range(count):
            r = random.uniform(150.0, 172.0)  # Between Mars (136) and Jupiter (185)
            ang = random.uniform(0.0, 2.0 * math.pi)
            spd = 0.25 * math.sqrt(100.0 / r) * random.uniform(0.9, 1.1)
            y_off = random.uniform(-6.0, 6.0)
            sz = random.uniform(1.0, 2.2)
            shade = random.randint(110, 170)
            self.asteroids.append(Asteroid(r, ang, spd, y_off, sz, (shade, shade, shade)))

    def _generate_procedural_nebula(self, w: int, h: int) -> np.ndarray:
        """Create a soft cosmic nebula backdrop with glowing color gradients and deep space contrast."""
        # Deep space backdrop: near-pitch black with subtle celestial tint
        canvas = np.zeros((h, w, 3), dtype=np.uint8)
        canvas[:, :] = (5, 5, 8)

        # Subtle cosmic dust clouds
        nebula = np.zeros((h, w, 3), dtype=np.uint8)
        cv2.circle(nebula, (int(w * 0.35), int(h * 0.40)), int(w * 0.32), (50, 18, 35), -1)
        cv2.circle(nebula, (int(w * 0.70), int(h * 0.65)), int(w * 0.28), (40, 22, 55), -1)
        cv2.circle(nebula, (int(w * 0.50), int(h * 0.28)), int(w * 0.20), (18, 32, 50), -1)
        nebula_blur = cv2.GaussianBlur(nebula, (51, 51), 0)
        canvas = cv2.add(canvas, nebula_blur)
        return canvas

    def zoom(self, delta: float) -> None:
        """Zoom camera distance closer or further."""
        self.camera_dist = max(self.min_dist, min(self.max_dist, self.camera_dist + delta))

    def rotate(self, d_yaw: float, d_pitch: float) -> None:
        """Orbit camera angle around focus target."""
        self.camera_yaw = (self.camera_yaw + d_yaw) % (2.0 * math.pi)
        # Clamp pitch to avoid gimbal inversion
        self.camera_pitch = max(-1.42, min(1.42, self.camera_pitch + d_pitch))

    def orbit_camera(self, d_yaw: float, d_pitch: float) -> None:
        """Alias for rotate."""
        self.rotate(d_yaw, d_pitch)

    def select_planet(self, planet_name: str) -> bool:
        """Set focus to specified planet by name."""
        for spec in CELESTIAL_SPECS:
            if spec.name.lower() == planet_name.lower():
                self.selected_planet_name = spec.name
                if spec.name == "Sun":
                    self.zoom_to_dist(180.0)
                else:
                    self.zoom_to_dist(max(75.0, spec.visual_radius_cinematic * 4.2))
                return True
        return False

    def reset_view(self) -> None:
        """Return to whole Solar System overview."""
        self.selected_planet_name = None
        self.desired_target = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        self.camera_dist = 620.0
        self.camera_pitch = 0.55

    def zoom_to_dist(self, target_dist: float) -> None:
        """Smoothly adjust zoom distance towards target."""
        self.camera_dist = target_dist

    def toggle_scale_mode(self) -> str:
        """Toggle between CINEMATIC and SCIENTIFIC scaling."""
        if self.scale_mode == "CINEMATIC":
            self.scale_mode = "SCIENTIFIC"
        else:
            self.scale_mode = "CINEMATIC"
        return self.scale_mode

    def toggle_pause(self) -> bool:
        """Pause or resume orbital simulation."""
        self.is_paused = not self.is_paused
        return self.is_paused

    def get_selected_planet_spec(self) -> Optional[PlanetSpec]:
        """Return PlanetSpec of currently focused planet."""
        if not self.selected_planet_name:
            return None
        return next((p for p in CELESTIAL_SPECS if p.name == self.selected_planet_name), None)

    def update(self, dt: float) -> None:
        """Advance planetary positions, rotations, asteroid orbits, and camera interpolation."""
        dt = max(0.001, min(0.1, dt))
        if not self.is_paused:
            self.sim_time += dt * self.time_speed

        # Update Asteroids
        for a in self.asteroids:
            a.angle = (a.angle + a.speed * dt * self.time_speed) % (2.0 * math.pi)

        # Smooth camera target tracking
        if self.selected_planet_name:
            coords = self._get_planet_world_pos(self.selected_planet_name)
            self.desired_target = np.array(coords, dtype=np.float64)
        else:
            self.desired_target = np.array([0.0, 0.0, 0.0], dtype=np.float64)

        # Exponential smoothing on camera target
        alpha = 0.12
        self.current_target = alpha * self.desired_target + (1.0 - alpha) * self.current_target

    def _get_planet_world_pos(self, name: str) -> Tuple[float, float, float]:
        """Calculate current 3D world coordinates of a planet."""
        spec = next((p for p in CELESTIAL_SPECS if p.name == name), None)
        if not spec or spec.orbit_distance_cinematic == 0.0:
            return (0.0, 0.0, 0.0)

        # Base orbital period in simulation time
        # Period scaled by Kepler's 3rd Law ratio T ~ r^(1.5)
        period = max(1.5, math.sqrt(spec.orbit_distance_cinematic) * 1.8)
        theta = (self.sim_time * (2.0 * math.pi / period)) % (2.0 * math.pi)

        # Orbital plane is in (X, Z) with slight realistic inclination in Y
        orbit_r = spec.orbit_distance_cinematic
        x = orbit_r * math.cos(theta)
        y = math.sin(theta * 2.0) * (spec.axial_tilt_deg * 0.08)
        z = orbit_r * math.sin(theta)
        return (x, y, z)

    def render(self, output_size: Optional[Tuple[int, int]] = None) -> np.ndarray:
        """Render the complete 3D Universe to an RGB/BGR image canvas."""
        w = output_size[0] if output_size else self.render_width
        h = output_size[1] if output_size else self.render_height

        # 1. Base Deep-Space Canvas (Resized nebula + dark space)
        if (w, h) != (self.render_width, self.render_height):
            canvas = cv2.resize(self._nebula_layer, (w, h))
        else:
            canvas = self._nebula_layer.copy()

        # 2. Camera Setup (Eye position in spherical coordinates around target)
        cx, cy = w // 2, h // 2
        focal_length = w * 0.85  # Field of view scaling

        eye_x = self.current_target[0] + self.camera_dist * math.cos(self.camera_pitch) * math.sin(self.camera_yaw)
        eye_y = self.current_target[1] + self.camera_dist * math.sin(self.camera_pitch)
        eye_z = self.current_target[2] + self.camera_dist * math.cos(self.camera_pitch) * math.cos(self.camera_yaw)

        # Forward, Right, Up camera coordinate basis vectors
        forward = self.current_target - np.array([eye_x, eye_y, eye_z])
        f_norm = np.linalg.norm(forward)
        forward = forward / f_norm if f_norm > 1e-5 else np.array([0.0, 0.0, 1.0])

        up_world = np.array([0.0, 1.0, 0.0])
        right = np.cross(forward, up_world)
        r_norm = np.linalg.norm(right)
        right = right / r_norm if r_norm > 1e-5 else np.array([1.0, 0.0, 0.0])

        up = np.cross(right, forward)

        def project_3d_point(P: Tuple[float, float, float]) -> Optional[Tuple[int, int, float]]:
            """Transform 3D world coordinate into 2D camera viewport pixel coordinate."""
            V = np.array([P[0] - eye_x, P[1] - eye_y, P[2] - eye_z])
            z_cam = np.dot(V, forward)
            if z_cam <= 2.0:  # Behind camera
                return None
            x_cam = np.dot(V, right)
            y_cam = np.dot(V, up)

            screen_x = int(round(cx + (x_cam / z_cam) * focal_length))
            screen_y = int(round(cy - (y_cam / z_cam) * focal_length))
            return (screen_x, screen_y, z_cam)

        # 3. Render 3D Parallax Starfield
        for star in self.stars:
            # Stars infinite depth: project with eye translation only
            V = np.array([star.x, star.y, star.z])
            z_cam = np.dot(V, forward)
            if z_cam <= 10.0:
                continue
            x_cam = np.dot(V, right)
            y_cam = np.dot(V, up)

            sx = int(round(cx + (x_cam / z_cam) * focal_length))
            sy = int(round(cy - (y_cam / z_cam) * focal_length))

            if 0 <= sx < w and 0 <= sy < h:
                twinkle = 0.85 + 0.15 * math.sin(self.sim_time * 3.0 + star.twinkle_phase)
                col = (
                    int(star.color[0] * twinkle),
                    int(star.color[1] * twinkle),
                    int(star.color[2] * twinkle),
                )
                r_px = int(round(star.size))
                cv2.circle(canvas, (sx, sy), r_px, col, -1, cv2.LINE_AA)

        # 4. Render Orbital Paths (Faint glowing 3D ellipses)
        for spec in CELESTIAL_SPECS[1:]:
            orbit_r = spec.orbit_distance_cinematic
            orbit_pts = []
            for deg in range(0, 360, 8):
                rad = math.radians(deg)
                ox = orbit_r * math.cos(rad)
                oy = 0.0
                oz = orbit_r * math.sin(rad)
                p = project_3d_point((ox, oy, oz))
                if p is not None:
                    orbit_pts.append((p[0], p[1]))

            if len(orbit_pts) > 8:
                pts_arr = np.array(orbit_pts, dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(canvas, [pts_arr], True, (40, 48, 55), 1, cv2.LINE_AA)

        # 5. Render Asteroid Belt Particles
        for a in self.asteroids:
            ax = a.orbit_r * math.cos(a.angle)
            ay = a.y_offset
            az = a.orbit_r * math.sin(a.angle)
            proj = project_3d_point((ax, ay, az))
            if proj is not None and (0 <= proj[0] < w and 0 <= proj[1] < h):
                cv2.circle(canvas, (proj[0], proj[1]), max(1, int(round(a.size))), a.color, -1, cv2.LINE_AA)

        # 6. Collect Celestial Bodies for Z-Sorting (Back to Front)
        render_queue: List[Tuple[float, str, Tuple[float, float, float], PlanetSpec]] = []
        for spec in CELESTIAL_SPECS:
            pos = self._get_planet_world_pos(spec.name)
            dist_to_eye = float(np.hypot(pos[0] - eye_x, np.hypot(pos[1] - eye_y, pos[2] - eye_z)))
            render_queue.append((dist_to_eye, spec.name, pos, spec))

        # Sort descending by distance so furthest objects render first
        render_queue.sort(key=lambda item: item[0], reverse=True)

        # 7. Render Sun & Planets with 3D Lambertian Shading
        for dist_to_eye, name, world_pos, spec in render_queue:
            proj = project_3d_point(world_pos)
            if proj is None:
                continue

            sx, sy, z_cam = proj
            r_screen = max(2.0, (spec.visual_radius_cinematic / z_cam) * focal_length)

            if not (-r_screen < sx < w + r_screen and -r_screen < sy < h + r_screen):
                continue

            ir = int(round(r_screen))

            if name == "Sun":
                # Emissive Sun with Corona Flares and Layered Bloom
                sun_glow = np.zeros_like(canvas)
                # Outer pulsating corona
                pulse = 1.0 + 0.05 * math.sin(self.sim_time * 4.0)
                cv2.circle(sun_glow, (sx, sy), int(ir * 1.8 * pulse), (0, 100, 255), -1, cv2.LINE_AA)
                cv2.circle(sun_glow, (sx, sy), int(ir * 1.3 * pulse), (0, 180, 255), -1, cv2.LINE_AA)
                # Core white-hot sun
                cv2.circle(sun_glow, (sx, sy), ir, (160, 240, 255), -1, cv2.LINE_AA)
                cv2.circle(sun_glow, (sx, sy), max(2, int(ir * 0.7)), (255, 255, 255), -1, cv2.LINE_AA)

                # Soft blur corona bloom
                glow_blur = cv2.GaussianBlur(sun_glow, (35, 35), 0)
                canvas = cv2.add(canvas, glow_blur)
                cv2.circle(canvas, (sx, sy), ir, (220, 245, 255), -1, cv2.LINE_AA)

            else:
                # 3D Shaded Planet Sphere
                # Light vector L points from planet to Sun at (0, 0, 0)
                L_world = np.array([-world_pos[0], -world_pos[1], -world_pos[2]])
                L_norm = np.linalg.norm(L_world)
                L_world = L_world / L_norm if L_norm > 1e-5 else np.array([0.0, 0.0, 1.0])

                # 2D screen direction of light
                sun_proj = project_3d_point((0.0, 0.0, 0.0))
                if sun_proj:
                    dx_light = sun_proj[0] - sx
                    dy_light = sun_proj[1] - sy
                    light_angle = math.atan2(dy_light, dx_light)
                else:
                    light_angle = 0.0

                # Render base planet sphere
                self._render_shaded_planet(
                    canvas=canvas,
                    center=(sx, sy),
                    radius=ir,
                    base_color=spec.base_color_bgr,
                    secondary_color=spec.secondary_color_bgr,
                    light_angle=light_angle,
                    name=name,
                )

                # Render Saturn's 3D Tilted Concentric Rings
                if spec.has_rings:
                    self._render_saturn_rings(
                        canvas=canvas,
                        center=(sx, sy),
                        z_cam=z_cam,
                        focal_length=focal_length,
                        inner_r_world=spec.ring_inner_r,
                        outer_r_world=spec.ring_outer_r,
                        ring_color=spec.ring_color_bgr,
                        axial_tilt_deg=spec.axial_tilt_deg,
                    )

            # Planet selection ring indicator
            if self.selected_planet_name == name:
                cv2.circle(canvas, (sx, sy), ir + 8, (0, 255, 200), 2, cv2.LINE_AA)
                cv2.putText(
                    canvas,
                    name.upper(),
                    (sx - 20, sy - ir - 12),
                    cv2.FONT_HERSHEY_DUPLEX,
                    0.46,
                    (0, 255, 220),
                    1,
                    cv2.LINE_AA,
                )

        return canvas

    def _render_shaded_planet(
        self,
        canvas: np.ndarray,
        center: Tuple[int, int],
        radius: int,
        base_color: Tuple[int, int, int],
        secondary_color: Tuple[int, int, int],
        light_angle: float,
        name: str,
    ) -> None:
        """Render a shaded sphere with 3D illuminated crescent, shadow terminator, and surface bands."""
        cx, cy = center
        r = max(2, radius)
        h, w = canvas.shape[:2]

        if r <= 3:
            cv2.circle(canvas, (cx, cy), r, base_color, -1, cv2.LINE_AA)
            return

        # Base illuminated body
        cv2.circle(canvas, (cx, cy), r, base_color, -1, cv2.LINE_AA)

        # Procedural surface details
        if name == "Jupiter":
            # Cloud bands
            for offset in (-0.5, -0.2, 0.1, 0.4):
                by = int(cy + r * offset)
                bw = int(r * math.sqrt(max(0.1, 1.0 - offset**2)))
                cv2.line(canvas, (cx - bw, by), (cx + bw, by), secondary_color, max(1, r // 7), cv2.LINE_AA)
            # Great Red Spot
            cv2.circle(canvas, (int(cx + r * 0.25), int(cy + r * 0.3)), max(2, r // 5), (45, 60, 160), -1, cv2.LINE_AA)

        elif name == "Earth":
            # Green continents
            cv2.circle(canvas, (int(cx - r * 0.2), int(cy - r * 0.1)), max(2, r // 3), secondary_color, -1, cv2.LINE_AA)
            cv2.circle(canvas, (int(cx + r * 0.3), int(cy + r * 0.2)), max(2, r // 4), secondary_color, -1, cv2.LINE_AA)
            # Swirling white clouds
            cv2.ellipse(canvas, (cx, cy), (int(r * 0.8), int(r * 0.3)), 25, 0, 360, (255, 255, 255), 1, cv2.LINE_AA)

        elif name == "Mars":
            # Dark volcanic lowlands + white polar cap
            cv2.circle(canvas, (int(cx + r * 0.1), int(cy + r * 0.1)), max(2, r // 3), (30, 45, 120), -1, cv2.LINE_AA)
            cv2.circle(canvas, (cx, int(cy - r * 0.8)), max(2, r // 4), secondary_color, -1, cv2.LINE_AA)

        # 3D Shadow Terminator (Dark crescent on side opposite the Sun)
        shadow_angle = light_angle + math.pi
        shadow_ox = int(cx + math.cos(shadow_angle) * (r * 0.45))
        shadow_oy = int(cy + math.sin(shadow_angle) * (r * 0.45))

        # Crescent shadow mask overlay (applied strictly to this planet's pixels only)
        planet_roi_x1 = max(0, cx - r)
        planet_roi_y1 = max(0, cy - r)
        planet_roi_x2 = min(w, cx + r)
        planet_roi_y2 = min(h, cy + r)

        if planet_roi_x2 > planet_roi_x1 and planet_roi_y2 > planet_roi_y1:
            box_w = planet_roi_x2 - planet_roi_x1
            box_h = planet_roi_y2 - planet_roi_y1

            # Planet sphere mask in local ROI
            p_mask = np.zeros((box_h, box_w), dtype=np.uint8)
            cv2.circle(p_mask, (cx - planet_roi_x1, cy - planet_roi_y1), r, 255, -1, cv2.LINE_AA)

            # Shadow circle in local ROI
            s_mask = np.zeros((box_h, box_w), dtype=np.uint8)
            cv2.circle(
                s_mask,
                (shadow_ox - planet_roi_x1, shadow_oy - planet_roi_y1),
                int(r * 0.95),
                255,
                -1,
                cv2.LINE_AA,
            )

            # Darken only the intersecting shadow region on this planet
            shadow_intersect = cv2.bitwise_and(p_mask, s_mask)
            roi = canvas[planet_roi_y1:planet_roi_y2, planet_roi_x1:planet_roi_x2]
            darkened = (roi.astype(np.float32) * 0.42).astype(np.uint8)
            np.copyto(roi, darkened, where=(shadow_intersect[:, :, None] > 0))

        # Subtle specular highlight facing Sun
        hl_x = int(cx + math.cos(light_angle) * (r * 0.45))
        hl_y = int(cy + math.sin(light_angle) * (r * 0.45))
        cv2.circle(canvas, (hl_x, hl_y), max(1, r // 4), (255, 255, 255), -1, cv2.LINE_AA)

    def _render_saturn_rings(
        self,
        canvas: np.ndarray,
        center: Tuple[int, int],
        z_cam: float,
        focal_length: float,
        inner_r_world: float,
        outer_r_world: float,
        ring_color: Tuple[int, int, int],
        axial_tilt_deg: float,
    ) -> None:
        """Render 3D tilted elliptical ring system of Saturn with Cassini Division."""
        cx, cy = center
        in_r = max(4.0, (inner_r_world / z_cam) * focal_length)
        out_r = max(in_r + 4.0, (outer_r_world / z_cam) * focal_length)

        # Tilted ellipse aspect ratio
        aspect = 0.36
        tilt_angle = axial_tilt_deg + math.degrees(self.camera_yaw * 0.4)

        # Outer Ring A
        cv2.ellipse(
            canvas,
            (cx, cy),
            (int(out_r), int(out_r * aspect)),
            tilt_angle,
            0,
            360,
            ring_color,
            max(2, int((out_r - in_r) * 0.45)),
            cv2.LINE_AA,
        )

        # Cassini Division (dark ring gap)
        mid_r = (in_r + out_r) * 0.5
        cv2.ellipse(
            canvas,
            (cx, cy),
            (int(mid_r), int(mid_r * aspect)),
            tilt_angle,
            0,
            360,
            (25, 25, 30),
            max(1, int(round((out_r - in_r) * 0.12))),
            cv2.LINE_AA,
        )

        # Inner Ring B
        cv2.ellipse(
            canvas,
            (cx, cy),
            (int(in_r), int(in_r * aspect)),
            tilt_angle,
            0,
            360,
            (int(ring_color[0] * 0.85), int(ring_color[1] * 0.85), int(ring_color[2] * 0.85)),
            max(2, int((out_r - in_r) * 0.35)),
            cv2.LINE_AA,
        )
