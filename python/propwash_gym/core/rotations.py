"""The single frame convention for this package.

World frame is **Z-up** (x forward, y left, z up), matching robotics practice
and the sibling ``rotorenv`` project. Quaternions are **scalar-first**
``[w, x, y, z]`` and are always unit length.

The browser simulator is Y-up (Three.js). Conversion happens only where
trajectories are exported for browser replay — never inside the sim loop. Do
not inline ad-hoc rotation maths anywhere else in the package.
"""

from __future__ import annotations

import numpy as np


def quat_normalize(q: np.ndarray) -> np.ndarray:
    """Return ``q`` scaled to unit length, falling back to identity if degenerate."""
    q = np.asarray(q, dtype=np.float64).reshape(4)
    n = float(np.linalg.norm(q))
    if n < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0])
    return q / n


def quat_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Return the Hamilton product ``a ⊗ b`` for scalar-first quaternions."""
    aw, ax, ay, az = (float(v) for v in a)
    bw, bx, by, bz = (float(v) for v in b)
    return np.array(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ]
    )


def euler_to_quat(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Convert ZYX Euler angles (radians) to a scalar-first unit quaternion."""
    cr, sr = np.cos(roll * 0.5), np.sin(roll * 0.5)
    cp, sp = np.cos(pitch * 0.5), np.sin(pitch * 0.5)
    cy, sy = np.cos(yaw * 0.5), np.sin(yaw * 0.5)
    return quat_normalize(
        np.array(
            [
                cr * cp * cy + sr * sp * sy,
                sr * cp * cy - cr * sp * sy,
                cr * sp * cy + sr * cp * sy,
                cr * cp * sy - sr * sp * cy,
            ]
        )
    )


def quat_to_euler(q: np.ndarray) -> tuple[float, float, float]:
    """Convert a scalar-first quaternion to ZYX Euler angles ``(roll, pitch, yaw)``."""
    w, x, y, z = (float(v) for v in quat_normalize(q))
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = float(np.arctan2(sinr_cosp, cosr_cosp))

    sinp = 2.0 * (w * y - z * x)
    sinp = float(np.clip(sinp, -1.0, 1.0))
    pitch = float(np.arcsin(sinp))

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = float(np.arctan2(siny_cosp, cosy_cosp))
    return roll, pitch, yaw


def rotate_vector(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate world-frame vector ``v`` by quaternion ``q``."""
    w, x, y, z = (float(c) for c in quat_normalize(q))
    v = np.asarray(v, dtype=np.float64).reshape(3)
    u = np.array([x, y, z])
    return v + 2.0 * np.cross(u, np.cross(u, v) + w * v)


def body_up_axis(q: np.ndarray) -> np.ndarray:
    """Return the drone's body-up axis in world coordinates (thrust direction)."""
    return rotate_vector(q, np.array([0.0, 0.0, 1.0]))
