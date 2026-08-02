"""Kinematic state of the drone at one instant.

A plain dataclass of numpy arrays, so physics backends, reward terms, and
observation builders all read one well-defined structure without coupling to
each other.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from propwash_gym.core.rotations import quat_normalize


@dataclass
class DroneState:
    """Full kinematic state at a single timestep.

    Attributes:
        position: World-frame ``(x, y, z)`` in metres, Z-up.
        velocity: World-frame linear velocity in m/s.
        quaternion: Attitude, scalar-first ``[w, x, y, z]``, unit length.
        angular_velocity: Body angular rates in rad/s.
        time: Elapsed simulation time in seconds.
    """

    position: np.ndarray
    velocity: np.ndarray
    quaternion: np.ndarray
    angular_velocity: np.ndarray
    time: float

    def __post_init__(self) -> None:
        """Coerce arrays to float64 and normalise the quaternion."""
        self.position = np.asarray(self.position, dtype=np.float64).reshape(3)
        self.velocity = np.asarray(self.velocity, dtype=np.float64).reshape(3)
        self.quaternion = quat_normalize(self.quaternion)
        self.angular_velocity = np.asarray(
            self.angular_velocity, dtype=np.float64
        ).reshape(3)
        self.time = float(self.time)

    def copy(self) -> "DroneState":
        """Return a copy with independent array buffers."""
        return DroneState(
            position=self.position.copy(),
            velocity=self.velocity.copy(),
            quaternion=self.quaternion.copy(),
            angular_velocity=self.angular_velocity.copy(),
            time=self.time,
        )
