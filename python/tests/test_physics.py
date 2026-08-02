"""Rigid-body integration, drag, wind, and crash detection."""

import numpy as np
import pytest

from propwash_gym.core.battery import Battery
from propwash_gym.core.flight_ctrl import FlightController
from propwash_gym.core.physics import (
    DRAG_K,
    GRAVITY,
    MASS,
    MAX_THRUST,
    CrashReason,
    Physics,
)
from propwash_gym.core.state import DroneState
from propwash_gym.world.terrain import Heightfield


def _flat(height=0.0, extent=400.0):
    return Heightfield(heights=np.full((16, 16), height), extent_m=extent)


def _phys(terrain=None, wind=0.0):
    return Physics(
        terrain=terrain if terrain is not None else _flat(),
        controller=FlightController(mode="angle"),
        battery=Battery(1300.0),
        wind_scale=wind,
    )


def _airborne(z=50.0):
    return DroneState([0, 0, z], [0, 0, 0], [1, 0, 0, 0], [0, 0, 0], 0.0)


def test_constants_match_the_browser_sim():
    assert pytest.approx(0.62, abs=1e-12) == MASS
    assert pytest.approx(24.0, abs=1e-12) == MAX_THRUST
    assert pytest.approx(0.045, abs=1e-12) == DRAG_K
    assert pytest.approx(9.81, abs=1e-12) == GRAVITY


def test_zero_throttle_falls_under_gravity():
    p = _phys()
    s = _airborne()
    p.step(s, throttle=0.0, roll=0.0, pitch=0.0, yaw=0.0, dt=0.02)
    assert s.velocity[2] < 0.0


def test_hover_throttle_roughly_cancels_gravity():
    p = _phys()
    s = _airborne()
    hover = MASS * GRAVITY / MAX_THRUST     # ~0.2534 at full pack scale
    for _ in range(10):
        p.step(s, throttle=hover, roll=0.0, pitch=0.0, yaw=0.0, dt=0.02)
    assert abs(float(s.velocity[2])) < 0.5, f"vz={s.velocity[2]}"


def test_full_throttle_climbs():
    p = _phys()
    s = _airborne()
    for _ in range(25):
        p.step(s, throttle=1.0, roll=0.0, pitch=0.0, yaw=0.0, dt=0.02)
    assert s.velocity[2] > 1.0


def test_drag_bounds_terminal_velocity():
    p = _phys()
    s = _airborne(z=5000.0)
    for _ in range(4000):
        p.step(s, throttle=0.0, roll=0.0, pitch=0.0, yaw=0.0, dt=0.02)
    # Terminal speed where drag balances weight: sqrt(m*g/k)
    expected = np.sqrt(MASS * GRAVITY / DRAG_K)
    assert float(np.linalg.norm(s.velocity)) == pytest.approx(expected, rel=0.1)


def test_wind_perturbs_a_hovering_drone_laterally():
    calm = _phys(wind=0.0)
    windy = _phys(wind=2.0)
    hover = MASS * GRAVITY / MAX_THRUST
    a, b = _airborne(), _airborne()
    for _ in range(100):
        calm.step(a, hover, 0.0, 0.0, 0.0, 0.02)
        windy.step(b, hover, 0.0, 0.0, 0.0, 0.02)
    assert float(np.linalg.norm(b.position[:2])) > float(np.linalg.norm(a.position[:2]))


def test_gentle_ground_contact_does_not_crash():
    p = _phys(_flat(0.0))
    s = DroneState([0, 0, 0.12], [0, 0, -0.5], [1, 0, 0, 0], [0, 0, 0], 0.0)
    reason = p.step(s, throttle=0.0, roll=0.0, pitch=0.0, yaw=0.0, dt=0.02)
    assert reason is None
    assert s.position[2] >= 0.09 - 1e-9


def test_fast_descent_into_ground_crashes():
    p = _phys(_flat(0.0))
    # Start low enough that one 0.02 s step actually reaches contact: at 8 m/s
    # the drone falls ~0.16 m, and contact needs z < ground + GROUND_OFFSET.
    s = DroneState([0, 0, 0.2], [0, 0, -8.0], [1, 0, 0, 0], [0, 0, 0], 0.0)
    reason = p.step(s, throttle=0.0, roll=0.0, pitch=0.0, yaw=0.0, dt=0.02)
    assert reason is CrashReason.GROUND


def test_water_contact_crashes_at_any_speed():
    hf = Heightfield(
        heights=np.where(np.arange(256).reshape(16, 16) < 128, -1000.0, 200.0),
        extent_m=400.0,
        has_water=True,
    )
    p = _phys(hf)
    assert hf.water_z is not None
    s = DroneState([0, 0, hf.water_z - 0.5], [0, 0, -0.01], [1, 0, 0, 0], [0, 0, 0], 0.0)
    assert p.step(s, 0.0, 0.0, 0.0, 0.0, 0.02) is CrashReason.WATER


def test_leaving_the_arena_crashes():
    p = _phys(_flat(0.0, extent=400.0))
    s = DroneState([10_000.0, 0.0, 50.0], [0, 0, 0], [1, 0, 0, 0], [0, 0, 0], 0.0)
    assert p.step(s, 0.0, 0.0, 0.0, 0.0, 0.02) is CrashReason.OUT_OF_BOUNDS


def test_non_ground_crashes_report_positions_outside_the_world():
    """Faithful to the browser: velocity is zeroed but position is not restored.

    GROUND snaps to the contact point, but WATER reports the drone below the
    surface and OUT_OF_BOUNDS reports it past the wall. Anything reading
    terminal position must expect an out-of-world value.
    """
    hf = Heightfield(
        heights=np.where(np.arange(256).reshape(16, 16) < 128, -1000.0, 200.0),
        extent_m=400.0,
        has_water=True,
    )
    p = _phys(hf)
    s = DroneState([0, 0, hf.water_z - 0.5], [0, 0, -0.01], [1, 0, 0, 0], [0, 0, 0], 0.0)
    assert p.step(s, 0.0, 0.0, 0.0, 0.0, 0.02) is CrashReason.WATER
    assert float(s.position[2]) < hf.water_z, "water crash reports sub-surface position"

    p2 = _phys(_flat(0.0, extent=400.0))
    s2 = DroneState([10_000.0, 0.0, 50.0], [0, 0, 0], [1, 0, 0, 0], [0, 0, 0], 0.0)
    assert p2.step(s2, 0.0, 0.0, 0.0, 0.0, 0.02) is CrashReason.OUT_OF_BOUNDS
    assert abs(float(s2.position[0])) > 400.0 / 2


def test_physics_does_not_update_the_battery():
    """The env owns battery updates, once per step with the full dt.

    index.html calls Battery.update() from loop(), outside the substep loop.
    Draining it here would double-count against the substeps.
    """
    p = _phys()
    s = _airborne()
    for _ in range(50):
        p.step(s, throttle=1.0, roll=0.0, pitch=0.0, yaw=0.0, dt=0.02)
    assert p.battery.drawn_mah == 0.0
    assert p.battery.soc == 1.0


def test_time_advances_by_dt():
    p = _phys()
    s = _airborne()
    p.step(s, 0.3, 0.0, 0.0, 0.0, 0.02)
    assert pytest.approx(0.02, abs=1e-12) == s.time


def test_stepping_is_deterministic():
    a, b = _phys(wind=1.5), _phys(wind=1.5)
    sa, sb = _airborne(), _airborne()
    for _ in range(200):
        a.step(sa, 0.4, 0.1, 0.2, 0.05, 0.02)
        b.step(sb, 0.4, 0.1, 0.2, 0.05, 0.02)
    np.testing.assert_array_equal(sa.position, sb.position)
