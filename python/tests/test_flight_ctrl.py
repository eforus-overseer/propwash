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


def test_angle_mode_reports_stick_derived_rates_matching_the_browser():
    """ANGLE rates come from stick deflection, not from differencing Euler angles.

    index.html:503 sets `Drone.omega.set(-inp.p*2, -inp.y*2.6, -inp.r*2)`. A
    finite-difference alternative reports 4.62 rad/s where the browser reports
    2.0, and varies with dt — defeating the timestep independence the
    exponential smoothing exists to provide.
    """
    for dt in (0.005, 0.02, 0.1):
        fc = FlightController(mode="angle")
        s = _level_state()
        fc.apply(s, roll=0.0, pitch=1.0, yaw=0.0, dt=dt)
        assert pytest.approx(2.0, abs=1e-9) == float(s.angular_velocity[1])
        assert pytest.approx(0.0, abs=1e-9) == float(s.angular_velocity[0])


def test_angle_rates_scale_linearly_with_stick_and_are_signed():
    fc = FlightController(mode="angle")
    s = _level_state()
    fc.apply(s, roll=-0.5, pitch=0.25, yaw=1.0, dt=0.02)
    assert pytest.approx(-1.0, abs=1e-9) == float(s.angular_velocity[0])
    assert pytest.approx(0.5, abs=1e-9) == float(s.angular_velocity[1])
    assert pytest.approx(2.6, abs=1e-9) == float(s.angular_velocity[2])


@pytest.mark.parametrize("bad_dt", [0.0, -0.02, -1.0])
@pytest.mark.parametrize("mode", ["angle", "acro"])
def test_non_positive_dt_is_rejected_in_both_modes(mode, bad_dt):
    """A negative dt would rotate the attitude backwards, silently.

    Guarding only the rate calculation left the quaternion still updating, so
    dt=-0.02 tilted the drone the wrong way while reporting zero angular
    velocity — motion with no diagnostic.
    """
    fc = FlightController(mode=mode)
    s = _level_state()
    with pytest.raises(ValueError, match="dt must be positive"):
        fc.apply(s, roll=0.0, pitch=1.0, yaw=0.0, dt=bad_dt)
    # Attitude must be untouched by the rejected call.
    assert pytest.approx(0.0, abs=1e-12) == float(quat_to_euler(s.quaternion)[1])


def test_controller_yaw_leaks_across_episodes_unless_reset():
    """ANGLE rebuilds attitude from the controller's own integrated yaw.

    A controller reused without `reset()` therefore imposes its stale heading on
    a freshly spawned drone, snapping it on the first step. This is a tripwire
    for the env: `reset()` must call `controller.reset(yaw=...)`, and nothing
    else in the codebase would catch it being missed.
    """
    fc = FlightController(mode="angle")
    flown = _level_state()
    for _ in range(50):                       # yaw away from zero
        fc.apply(flown, roll=0.0, pitch=0.0, yaw=1.0, dt=0.02)
    stale = quat_to_euler(flown.quaternion)[2]
    assert abs(stale) > 1.0, "precondition: controller should hold a nonzero yaw"

    # A new episode's drone, spawned level, reusing the same controller.
    fresh = _level_state()
    fc.apply(fresh, roll=0.0, pitch=0.0, yaw=0.0, dt=0.02)
    assert quat_to_euler(fresh.quaternion)[2] == pytest.approx(stale, abs=1e-9), (
        "stale controller yaw should leak into the new episode — if this now "
        "fails, the leak was fixed and the env's reset contract can relax"
    )

    # ...and reset() is what prevents it.
    fc.reset(yaw=0.0)
    fixed = _level_state()
    fc.apply(fixed, roll=0.0, pitch=0.0, yaw=0.0, dt=0.02)
    assert quat_to_euler(fixed.quaternion)[2] == pytest.approx(0.0, abs=1e-9)
