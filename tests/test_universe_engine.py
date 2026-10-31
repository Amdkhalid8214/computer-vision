"""Unit tests for the 3D Universe and Solar System rendering engine."""

import numpy as np
import pytest

from src.universe_engine import Universe3DEngine, CELESTIAL_SPECS, PlanetSpec


def test_universe_engine_initialization():
    """Verify universe engine initializes stars, asteroids, nebulae, and celestial specs."""
    engine = Universe3DEngine(render_width=320, render_height=320)

    assert len(CELESTIAL_SPECS) == 9  # Sun + 8 planets
    planet_names = [p.name for p in CELESTIAL_SPECS]
    for required in ["Sun", "Mercury", "Venus", "Earth", "Mars", "Jupiter", "Saturn", "Uranus", "Neptune"]:
        assert required in planet_names

    assert len(engine.stars) >= 150
    assert len(engine.asteroids) >= 50
    assert engine.selected_planet_name is None
    assert engine.is_paused is False


def test_planet_selection_and_focus():
    """Verify selecting celestial bodies updates focus, target, and camera distance."""
    engine = Universe3DEngine()

    assert engine.select_planet("Earth") is True
    assert engine.selected_planet_name == "Earth"
    spec = engine.get_selected_planet_spec()
    assert spec is not None
    assert spec.name == "Earth"
    assert spec.category.startswith("Terrestrial")

    # Select another valid planet
    assert engine.select_planet("Saturn") is True
    assert engine.selected_planet_name == "Saturn"
    assert engine.get_selected_planet_spec().has_rings is True

    # Invalid celestial body
    assert engine.select_planet("UnknownPlanetX") is False
    assert engine.selected_planet_name == "Saturn"

    # Reset view
    engine.reset_view()
    assert engine.selected_planet_name is None
    assert engine.get_selected_planet_spec() is None


def test_camera_orbit_and_zoom():
    """Verify camera orbit angles and zoom distance adjustments with limits."""
    engine = Universe3DEngine()
    initial_yaw = engine.camera_yaw
    initial_pitch = engine.camera_pitch
    initial_dist = engine.camera_dist

    # Orbit
    engine.orbit_camera(d_yaw=0.5, d_pitch=0.2)
    assert abs(engine.camera_yaw - (initial_yaw + 0.5)) < 1e-4
    assert abs(engine.camera_pitch - (initial_pitch + 0.2)) < 1e-4

    # Pitch clamping
    engine.orbit_camera(d_yaw=0.0, d_pitch=5.0)
    assert engine.camera_pitch <= 1.45

    # Zoom in / out
    engine.zoom(100.0)
    assert engine.camera_dist > initial_dist
    engine.zoom(-500.0)
    assert engine.camera_dist >= 45.0  # Min zoom clamp


def test_scale_mode_and_pause_toggle():
    """Verify scale mode switching and simulation pause."""
    engine = Universe3DEngine()

    # Scale mode
    assert engine.scale_mode == "CINEMATIC"
    mode1 = engine.toggle_scale_mode()
    assert mode1 == "SCIENTIFIC"
    assert engine.scale_mode == "SCIENTIFIC"
    mode2 = engine.toggle_scale_mode()
    assert mode2 == "CINEMATIC"

    # Pause
    t0 = engine.sim_time
    engine.update(dt=0.1)
    assert engine.sim_time > t0

    assert engine.toggle_pause() is True
    t_paused = engine.sim_time
    engine.update(dt=0.1)
    assert engine.sim_time == t_paused


def test_render_output_canvas():
    """Verify rendering generates an RGB/BGR canvas of requested dimensions."""
    engine = Universe3DEngine(render_width=256, render_height=256)
    engine.update(dt=0.033)

    canvas = engine.render(output_size=(200, 200))
    assert canvas.shape == (200, 200, 3)
    assert canvas.dtype == np.uint8
    # Starfield/Sun/Nebula ensures canvas is not completely black
    assert np.count_nonzero(canvas) > 500

