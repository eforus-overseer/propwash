"""Rigid-body flight physics and terrain collision.

Ported from ``index.html`` ``const Physics`` and the substepping in ``loop()``:
each env step runs ``SUBSTEPS`` integration passes of ``dt / SUBSTEPS``, which
is what keeps the simulation stable at aggressive attitudes.

Wind is deterministic — a fixed sum of sinusoids in simulation time, exactly as
the browser computes it. It is not random, so it never needs seeding.
"""

from __future__ import annotations

import enum
import math

import numpy as np

from propwash_gym.core.battery import Battery
from propwash_gym.core.flight_ctrl import FlightController
from propwash_gym.core.rotations import body_up_axis
from propwash_gym.core.state import DroneState
from propwash_gym.world.terrain import Heightfield

MASS = 0.62
MAX_THRUST = 24.0
DRAG_K = 0.045
GRAVITY = 9.81
SUBSTEPS = 2
#: Ground clearance the airframe rests at, and the contact threshold.
GROUND_OFFSET = 0.09
#: Descent rate (m/s) above which ground contact is a crash.
CRASH_DESCENT_MS = 3.2
#: Speed (m/s) above which ground contact is a crash regardless of direction.
CRASH_SPEED_MS = 9.0
#: Velocity retained after a survivable ground scrape.
GROUND_BOUNCE_DAMPING = 0.6
#: Distance from the arena edge that terminates an episode.
ARENA_MARGIN_M = 10.0
#: Height above the spawn terrain beyond which the episode ends out of bounds.
#:
#: The browser has no ceiling because a human pilot wants to come back. An RL
#: agent has no such preference: a random policy averages roughly twice hover
#: throttle, so the default behaviour is to climb away and never crash. Measured
#: unbounded, full throttle reaches 1843 m and terminates only when the battery
#: dies — a degenerate strategy that dodges every terrain penalty. 400 m is well
#: clear of the tallest real relief in the flyable patches (Grindelwald's valley
#: walls span ~1400 m of DEM, but only ~400 m above a valley-floor spawn).
CEILING_AGL_M = 400.0


class CrashReason(enum.Enum):
    """Why an episode ended in a crash."""

    GROUND = "ground"
    WATER = "water"
    OUT_OF_BOUNDS = "out_of_bounds"


class Physics:
    """Integrates drone motion and detects terminal contacts.

    Args:
        terrain: Height source for collision and ground contact.
        controller: Attitude control law.
        battery: Pack whose sag scales available thrust.
        wind_scale: Mission wind strength (0 = calm).
    """

    def __init__(
        self,
        terrain: Heightfield,
        controller: FlightController,
        battery: Battery,
        wind_scale: float = 0.0,
    ) -> None:
        self.terrain = terrain
        self.controller = controller
        self.battery = battery
        self.wind_scale = float(wind_scale)
        # Ceiling reference: terrain height at the pad, so the limit is an
        # altitude above the launch point rather than above sea level.
        self.spawn_ground = terrain.height(0.0, 0.0)

    def wind(self, t: float) -> np.ndarray:
        """Return the deterministic wind vector at simulation time ``t``."""
        if self.wind_scale == 0.0:
            return np.zeros(3)
        return (
            np.array(
                [
                    math.sin(t * 0.31) + math.sin(t * 0.83) * 0.5,
                    math.cos(t * 0.27) + math.sin(t * 0.71) * 0.5,
                    math.sin(t * 0.57) * 0.25,
                ]
            )
            * self.wind_scale
        )

    def step(
        self,
        state: DroneState,
        throttle: float,
        roll: float,
        pitch: float,
        yaw: float,
        dt: float,
    ) -> CrashReason | None:
        """Advance ``state`` by ``dt`` in place; return a crash reason or ``None``."""
        h = dt / SUBSTEPS
        for _ in range(SUBSTEPS):
            self.controller.apply(state, roll=roll, pitch=pitch, yaw=yaw, dt=h)
            thrust = body_up_axis(state.quaternion) * (
                float(np.clip(throttle, 0.0, 1.0))
                * MAX_THRUST
                * self.battery.thrust_scale
            )
            speed = float(np.linalg.norm(state.velocity))
            drag = -DRAG_K * speed * state.velocity
            gravity = np.array([0.0, 0.0, -GRAVITY * MASS])
            force = thrust + gravity + drag + self.wind(state.time)
            state.velocity = state.velocity + force / MASS * h
            state.position = state.position + state.velocity * h
            state.time += h
            reason = self._collide(state)
            if reason is not None:
                state.velocity = np.zeros(3)
                return reason
        return None

    def _collide(self, state: DroneState) -> CrashReason | None:
        """Resolve terrain contact, returning a crash reason when terminal."""
        x, y, z = (float(v) for v in state.position)

        half = self.terrain.extent_m / 2.0 - ARENA_MARGIN_M
        if abs(x) > half or abs(y) > half:
            return CrashReason.OUT_OF_BOUNDS

        if z > self.spawn_ground + CEILING_AGL_M:
            return CrashReason.OUT_OF_BOUNDS

        if self.terrain.water_z is not None and z < self.terrain.water_z + 0.05:
            return CrashReason.WATER

        ground = self.terrain.height(x, y)
        if z < ground + GROUND_OFFSET:
            descent = -float(state.velocity[2])
            speed = float(np.linalg.norm(state.velocity))
            state.position[2] = ground + GROUND_OFFSET
            if descent > CRASH_DESCENT_MS or speed > CRASH_SPEED_MS:
                return CrashReason.GROUND
            state.velocity[2] = max(float(state.velocity[2]), 0.0)
            state.velocity = state.velocity * GROUND_BOUNCE_DAMPING
        return None
