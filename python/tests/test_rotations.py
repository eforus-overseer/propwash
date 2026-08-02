"""Frame conventions: Z-up world, scalar-first quaternions [w, x, y, z]."""

import numpy as np
import pytest

from propwash_gym.core.rotations import (
    body_up_axis,
    euler_to_quat,
    quat_multiply,
    quat_normalize,
    quat_to_euler,
)
from propwash_gym.core.state import DroneState


def test_identity_quaternion_has_body_up_along_world_z():
    q = np.array([1.0, 0.0, 0.0, 0.0])
    np.testing.assert_allclose(body_up_axis(q), [0.0, 0.0, 1.0], atol=1e-12)


def test_euler_quat_roundtrip():
    for roll, pitch, yaw in [(0.0, 0.0, 0.0), (0.3, -0.2, 1.1), (-0.5, 0.4, -2.0)]:
        q = euler_to_quat(roll, pitch, yaw)
        r2, p2, y2 = quat_to_euler(q)
        assert pytest.approx(roll, abs=1e-9) == r2
        assert pytest.approx(pitch, abs=1e-9) == p2
        assert pytest.approx(yaw, abs=1e-9) == y2


def test_pitching_forward_tilts_body_up_toward_minus_x():
    # 30 deg nose-down pitch: body-up leans toward +x in this convention.
    q = euler_to_quat(0.0, np.deg2rad(30.0), 0.0)
    up = body_up_axis(q)
    assert up[0] > 0.4          # leaned over
    assert up[2] == pytest.approx(np.cos(np.deg2rad(30.0)), abs=1e-9)


def test_quat_multiply_by_identity_is_noop():
    q = euler_to_quat(0.2, 0.3, 0.4)
    ident = np.array([1.0, 0.0, 0.0, 0.0])
    np.testing.assert_allclose(quat_multiply(q, ident), q, atol=1e-12)


def test_quat_normalize_returns_unit_length():
    q = quat_normalize(np.array([2.0, 0.0, 0.0, 0.0]))
    assert pytest.approx(1.0, abs=1e-12) == float(np.linalg.norm(q))


def test_drone_state_coerces_shapes_and_normalizes_quat():
    s = DroneState(
        position=[1, 2, 3],
        velocity=[0, 0, 0],
        quaternion=[2.0, 0.0, 0.0, 0.0],
        angular_velocity=[0, 0, 0],
        time=0.0,
    )
    assert s.position.shape == (3,)
    assert s.position.dtype == np.float64
    assert s.quaternion.shape == (4,)
    assert pytest.approx(1.0, abs=1e-12) == float(np.linalg.norm(s.quaternion))


def test_drone_state_copy_is_independent():
    s = DroneState([0, 0, 1], [0, 0, 0], [1, 0, 0, 0], [0, 0, 0], 0.0)
    c = s.copy()
    c.position[0] = 99.0
    assert s.position[0] == 0.0
