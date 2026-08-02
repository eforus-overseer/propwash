"""ANGLE and ACRO control laws, ported from index.html `const FlightCtrl`."""

import numpy as np
import pytest

from propwash_gym.core.flight_ctrl import MAX_TILT, RATE_RP, RATE_Y, FlightController
from propwash_gym.core.rotations import quat_to_euler
from propwash_gym.core.state import DroneState


def _level_state():
    return DroneState([0, 0, 10], [0, 0, 0], [1, 0, 0, 0], [0, 0, 0], 0.0)


def test_constants_match_the_browser_sim():
    assert pytest.approx(np.deg2rad(38.0), abs=1e-12) == MAX_TILT
    assert pytest.approx(np.deg2rad(480.0), abs=1e-12) == RATE_RP
    assert pytest.approx(np.deg2rad(300.0), abs=1e-12) == RATE_Y


def test_angle_mode_holds_level_with_no_input():
    fc = FlightController(mode="angle")
    s = _level_state()
    for _ in range(50):
        fc.apply(s, roll=0.0, pitch=0.0, yaw=0.0, dt=0.02)
    roll, pitch, _ = quat_to_euler(s.quaternion)
    assert pytest.approx(0.0, abs=1e-6) == roll
    assert pytest.approx(0.0, abs=1e-6) == pitch


def test_angle_mode_converges_toward_commanded_tilt():
    fc = FlightController(mode="angle")
    s = _level_state()
    for _ in range(400):
        fc.apply(s, roll=0.0, pitch=1.0, yaw=0.0, dt=0.02)
    _, pitch, _ = quat_to_euler(s.quaternion)
    assert pytest.approx(MAX_TILT, rel=0.02) == pitch


def test_angle_mode_never_exceeds_max_tilt():
    fc = FlightController(mode="angle")
    s = _level_state()
    for _ in range(400):
        fc.apply(s, roll=1.0, pitch=1.0, yaw=0.0, dt=0.02)
    roll, pitch, _ = quat_to_euler(s.quaternion)
    assert abs(roll) <= MAX_TILT + 1e-6
    assert abs(pitch) <= MAX_TILT + 1e-6


def test_angle_mode_yaw_integrates_at_2_6_rad_per_second():
    fc = FlightController(mode="angle")
    s = _level_state()
    for _ in range(50):          # 1 second at dt=0.02
        fc.apply(s, roll=0.0, pitch=0.0, yaw=1.0, dt=0.02)
    _, _, yaw = quat_to_euler(s.quaternion)
    assert pytest.approx(2.6, rel=0.02) == abs(yaw)


def test_acro_mode_keeps_rotating_while_stick_is_held():
    fc = FlightController(mode="acro")
    s = _level_state()
    for _ in range(60):
        fc.apply(s, roll=1.0, pitch=0.0, yaw=0.0, dt=0.02)
    # A held roll command in ACRO produces continuous rotation, not a fixed tilt.
    assert float(np.linalg.norm(s.angular_velocity)) > 1.0


def test_acro_expo_softens_small_stick_inputs():
    fc = FlightController(mode="acro")
    small = fc.expo(0.2)
    assert small < 0.2, "expo must reduce small inputs"
    assert pytest.approx(1.0, abs=1e-12) == fc.expo(1.0)


def test_unknown_mode_is_rejected():
    with pytest.raises(ValueError, match="mode"):
        FlightController(mode="sport")
