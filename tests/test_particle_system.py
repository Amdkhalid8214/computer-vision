"""Automated unit tests for ParticleSystem 2D simulation and additive rendering."""

import numpy as np
import pytest

from src.particle_system import ParticleSystem


def test_particle_system_initialization_and_clear():
    """Verify particle system starts empty and clears properly."""
    ps = ParticleSystem(max_particles=100)
    assert ps.count == 0

    ps.emit_spark(50, 50, 10, 10)
    assert ps.count == 1

    ps.clear()
    assert ps.count == 0


def test_particle_burst_and_max_limit():
    """Verify burst emission respects max particle limits."""
    ps = ParticleSystem(max_particles=50)
    ps.emit_burst(cx=100, cy=100, count=120)
    assert ps.count <= 50


def test_particle_ring_sparks():
    """Verify ring sparks are emitted along circle circumference."""
    ps = ParticleSystem(max_particles=100)
    ps.emit_ring_sparks(cx=200, cy=200, radius=50, count=10)
    assert ps.count == 10
    # Every particle should be near radius 50
    for p in ps.particles:
        dist = np.hypot(p.x - 200, p.y - 200)
        assert abs(dist - 50) < 5.0


def test_particle_beam_particles():
    """Verify beam particles are emitted along line between start and end points."""
    ps = ParticleSystem(max_particles=100)
    ps.emit_beam_particles((0, 0), (100, 0), count=5)
    assert ps.count == 5
    for p in ps.particles:
        assert -5.0 <= p.x <= 105.0


def test_particle_update_physics_and_expiration():
    """Verify particles expire after their lifetime finishes."""
    ps = ParticleSystem(max_particles=50)
    ps.emit_spark(100, 100, 0, 0, life=0.1)
    assert ps.count == 1

    # Update by 0.05s -> still alive
    ps.update(dt=0.05)
    assert ps.count == 1

    # Update by another 0.06s -> expired
    ps.update(dt=0.06)
    assert ps.count == 0


def test_particle_render_additive_blend():
    """Verify particle rendering applies additive blending without changing frame dimensions."""
    ps = ParticleSystem(max_particles=50)
    ps.emit_spark(50, 50, 0, 0, color=(0, 200, 255), size=4.0, life=1.0)

    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    rendered = ps.render(frame)

    assert rendered.shape == (100, 100, 3)
    assert rendered.dtype == np.uint8
    # Center pixel (50, 50) must have been brightened
    assert rendered[50, 50, :].sum() > 0

