"""Stick-to-attitude control laws.

Ported from ``index.html`` ``const FlightCtrl``:

* **ANGLE** — self-levelling. Sticks command a tilt angle, capped at 38°, and
  attitude eases toward it. Yaw integrates at a fixed rate.
* **ACRO** — sticks command body rates directly (480°/s roll and pitch,
  300°/s yaw) through an expo curve, integrated on the quaternion.

Smoothing uses the sim's exponential form ``1 - exp(-dt * k)``, which is
timestep-independent — do not replace it with a plain lerp constant.
"""

from __future__ import annotations

import math

import numpy as np

from propwash_gym.core.rotations import (
    euler_to_quat,
    quat_multiply,
    quat_normalize,
    quat_to_euler,
)
from propwash_gym.core.state import DroneState

MAX_TILT = math.radians(38.0)
RATE_RP = math.radians(480.0)
RATE_Y = math.radians(300.0)
ANGLE_SMOOTH_K = 7.5
ANGLE_YAW_RATE = 2.6
ACRO_SMOOTH_K = 18.0
MODES = ("angle", "acro")


class FlightController:
    """Maps normalised stick inputs onto the drone's attitude.

    Args:
        mode: ``"angle"`` (self-levelling) or ``"acro"`` (rate mode).
    """

    def __init__(self, mode: str = "angle") -> None:
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
        self.mode = mode
        self._yaw = 0.0

    def reset(self, yaw: float = 0.0) -> None:
        """Clear the integrated heading held for ANGLE mode."""
        self._yaw = float(yaw)

    @staticmethod
    def expo(v: float) -> float:
        """Apply the sim's expo curve: ``v * (0.35 + 0.65 * v²)``."""
        return float(v) * (0.35 + 0.65 * float(v) * float(v))

    def apply(
        self, state: DroneState, roll: float, pitch: float, yaw: float, dt: float
    ) -> None:
        """Advance ``state``'s attitude by one control step, in place."""
        if self.mode == "angle":
            self._apply_angle(state, roll, pitch, yaw, dt)
        else:
            self._apply_acro(state, roll, pitch, yaw, dt)

    def _apply_angle(
        self, state: DroneState, roll: float, pitch: float, yaw: float, dt: float
    ) -> None:
        cur_roll, cur_pitch, _ = quat_to_euler(state.quaternion)
        smooth = 1.0 - math.exp(-dt * ANGLE_SMOOTH_K)
        target_pitch = float(pitch) * MAX_TILT
        target_roll = float(roll) * MAX_TILT
        new_pitch = cur_pitch + (target_pitch - cur_pitch) * smooth
        new_roll = cur_roll + (target_roll - cur_roll) * smooth
        self._yaw += float(yaw) * ANGLE_YAW_RATE * dt
        state.quaternion = euler_to_quat(new_roll, new_pitch, self._yaw)
        state.angular_velocity = np.array(
            [
                (new_roll - cur_roll) / dt if dt > 0 else 0.0,
                (new_pitch - cur_pitch) / dt if dt > 0 else 0.0,
                float(yaw) * ANGLE_YAW_RATE,
            ]
        )

    def _apply_acro(
        self, state: DroneState, roll: float, pitch: float, yaw: float, dt: float
    ) -> None:
        target = np.array(
            [
                self.expo(float(roll)) * RATE_RP,
                self.expo(float(pitch)) * RATE_RP,
                self.expo(float(yaw)) * RATE_Y,
            ]
        )
        smooth = 1.0 - math.exp(-dt * ACRO_SMOOTH_K)
        state.angular_velocity = state.angular_velocity + (
            target - state.angular_velocity
        ) * smooth
        wx, wy, wz = (float(v) for v in state.angular_velocity)
        omega_q = np.array([0.0, wx * 0.5 * dt, wy * 0.5 * dt, wz * 0.5 * dt])
        state.quaternion = quat_normalize(
            state.quaternion + quat_multiply(state.quaternion, omega_q)
        )
        self._yaw = quat_to_euler(state.quaternion)[2]
