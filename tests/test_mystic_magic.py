"""Automated unit tests for MysticMagicManager, Doctor Strange mandala rendering, and spell casting."""

import numpy as np
import pytest

from src.hand_tracker import HandDetection
from src.mystic_magic import MysticMagicManager, SpellType


def make_dummy_hand(open_palm: bool = False, pointing: bool = False, pinch_dist: float = 80.0) -> HandDetection:
    """Helper to synthesize a mock HandDetection with specific finger postures."""
    wrist = (300, 400)
    # 21 landmarks
    pts = [wrist] * 21

    if open_palm:
        # All fingers extended away from wrist
        pts[0] = (300, 400)  # wrist
        pts[5] = (260, 320)  # index MCP
        pts[6] = (250, 260)  # index PIP
        pts[8] = (240, 180)  # index tip (far)
        pts[9] = (300, 310)  # middle MCP
        pts[10] = (300, 240) # middle PIP
        pts[12] = (300, 160) # middle tip (far)
        pts[13] = (340, 320) # ring MCP
        pts[14] = (350, 260) # ring PIP
        pts[16] = (360, 180) # ring tip (far)
        pts[17] = (380, 330) # pinky MCP
        pts[18] = (390, 280) # pinky PIP
        pts[20] = (400, 210) # pinky tip (far)
    elif pointing:
        # Index extended, others curled
        pts[0] = (300, 400)
        pts[6] = (250, 300)
        pts[8] = (240, 180)  # index extended
        pts[10] = (300, 320)
        pts[12] = (300, 340) # middle curled
        pts[14] = (340, 320)
        pts[16] = (340, 340) # ring curled
        pts[18] = (380, 320)
        pts[20] = (380, 340) # pinky curled
    else:
        # Neutral hand
        pts[8] = (int(300 + pinch_dist), 200)
        pts[4] = (300, 200)

    norm_pts = [(p[0] / 640.0, p[1] / 480.0, 0.0) for p in pts]
    return HandDetection(
        landmarks_norm=norm_pts,
        landmarks_pixel=pts,
        control_point=wrist,
        handedness="Right",
    )


@pytest.fixture
def magic_mgr() -> MysticMagicManager:
    return MysticMagicManager()


def test_mystic_magic_initialization(magic_mgr: MysticMagicManager):
    """Verify default initial state of mystic magic engine."""
    assert magic_mgr.active_spell == SpellType.NONE
    assert magic_mgr.particles.count == 0


def test_spell_mode_and_theme_toggle(magic_mgr: MysticMagicManager):
    """Verify manual spell selection and theme palette cycling."""
    magic_mgr.set_spell_mode("SHIELD")
    assert magic_mgr.spell_mode == "SHIELD"

    # Theme toggle
    init_col = magic_mgr.primary_color
    magic_mgr.toggle_color_scheme()
    assert magic_mgr.primary_color != init_col


def test_open_palm_triggers_mystic_shield(magic_mgr: MysticMagicManager):
    """Verify open palm gesture activates Mystic Shield."""
    hand = make_dummy_hand(open_palm=True)
    assert hand.is_open_palm is True

    magic_mgr.update([hand], dt=0.033)
    assert magic_mgr.active_spell == SpellType.SHIELD
    assert magic_mgr.particles.count > 0


def test_pointing_finger_triggers_energy_beam(magic_mgr: MysticMagicManager):
    """Verify pointing index finger gesture activates Energy Beam."""
    hand = make_dummy_hand(pointing=True)
    assert hand.is_pointing is True

    magic_mgr.update([hand], dt=0.033)
    assert magic_mgr.active_spell == SpellType.BEAM


def test_two_hands_trigger_mystic_portal(magic_mgr: MysticMagicManager):
    """Verify detecting two hands activates Mystic Portal with smoothed center."""
    h1 = make_dummy_hand()
    h2 = make_dummy_hand()
    h1.landmarks_pixel[0] = (150, 200)
    h2.landmarks_pixel[0] = (350, 200)

    magic_mgr.update([h1, h2], dt=0.033)
    assert magic_mgr.active_spell == SpellType.PORTAL
    assert magic_mgr.smoothed_portal_center is not None
    assert magic_mgr.smoothed_portal_radius > 50.0


def test_pinch_release_triggers_energy_blast(magic_mgr: MysticMagicManager):
    """Verify pinch followed by release triggers Energy Blast shockwave."""
    # Frame 1: Pinch active (dist 10px <= 40px)
    h_pinch = make_dummy_hand(pinch_dist=10.0)
    magic_mgr.update([h_pinch], dt=0.033)
    assert magic_mgr.prev_pinch is True

    # Frame 2: Release pinch (dist 80px > 40px)
    h_release = make_dummy_hand(pinch_dist=80.0)
    magic_mgr.update([h_release], dt=0.033)

    assert len(magic_mgr.active_blasts) > 0
    assert magic_mgr.active_spell == SpellType.BLAST


def test_render_mandala_and_particles(magic_mgr: MysticMagicManager):
    """Verify rendering applies effects onto frame without altering shape or dtype."""
    hand = make_dummy_hand(open_palm=True)
    magic_mgr.update([hand], dt=0.033)

    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    rendered = magic_mgr.render(frame, [hand])

    assert rendered.shape == (480, 640, 3)
    assert rendered.dtype == np.uint8
    # Shield mandala and particles brighten pixels on the frame
    assert np.any(rendered > 0)

