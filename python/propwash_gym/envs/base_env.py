"""Gymnasium base environment for PROP//WASH.

Subclasses supply the task through hooks — ``_make_target``, ``_build_reward``,
``_task_terminated`` — so this class never knows what a "target" means. That
keeps physics, terrain, reward, and observation independent.

**All randomness must come from ``self.np_random``.** The browser sim uses 29
unseeded ``Math.random()`` calls; reproducing episodes from ``reset(seed=n)``
requires routing every draw through Gymnasium's generator.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from gymnasium import Env, spaces

from propwash_gym.core.battery import Battery
from propwash_gym.core.flight_ctrl import FlightController
from propwash_gym.core.physics import Physics
from propwash_gym.core.reward import CompositeReward, RewardContext
from propwash_gym.core.rotations import euler_to_quat, quat_to_euler
from propwash_gym.core.state import DroneState
from propwash_gym.world.terrain import Heightfield, build_heightfield
from propwash_gym.world.tiles import TileCache

#: Width of the ``state`` observation vector. See the spec for the layout.
OBS_DIM = 23
PERCEPTIONS = ("state", "rgb", "depth")
#: Camera image side length for pixel perception modes.
IMAGE_SIZE = 64
DT = 0.02
#: Distances (m) ahead of the drone at which terrain height is probed.
PROBE_DISTANCES = (5.0, 10.0, 20.0, 40.0)


class PropwashEnv(Env):
    """Base flight environment. Not directly useful — subclass it with a task.

    Args:
        location: Which real-world location to fly.
        perception: ``"state"``, ``"rgb"``, or ``"depth"``.
        flight_mode: ``"angle"`` or ``"acro"``.
        spawn_height: Metres above ground at spawn.
        wind_scale: Mission wind strength.
        battery_mah: Pack capacity.
        max_time: Episode time limit in seconds.
        offline: Never fetch tiles (procedural terrain).
        tile_cache: Injectable cache, mainly for tests.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        location: str = "negev",
        perception: str = "state",
        flight_mode: str = "angle",
        spawn_height: float = 2.0,
        wind_scale: float = 0.0,
        battery_mah: float = 1300.0,
        max_time: float = 60.0,
        offline: bool = False,
        tile_cache: TileCache | None = None,
    ) -> None:
        if perception not in PERCEPTIONS:
            raise ValueError(
                f"perception must be one of {PERCEPTIONS}, got {perception!r}"
            )
        self.location = location
        self.perception = perception
        self.flight_mode = flight_mode
        self.spawn_height = float(spawn_height)
        self.wind_scale = float(wind_scale)
        self.battery_mah = float(battery_mah)
        self.max_time = float(max_time)
        self.offline = bool(offline)
        self._tile_cache = tile_cache or TileCache(offline=offline)

        #: Curriculum difficulty, driven by ``reset(options={"difficulty": d})``.
        self.difficulty: float = 1.0

        self.terrain: Heightfield = self._build_terrain()
        self.battery = Battery(self.battery_mah)
        self.controller = FlightController(mode=flight_mode)
        self.physics = Physics(
            terrain=self.terrain,
            controller=self.controller,
            battery=self.battery,
            wind_scale=self.wind_scale,
        )

        self.action_space = spaces.Box(-1.0, 1.0, shape=(4,), dtype=np.float32)
        self.observation_space = self._build_observation_space()

        self.state: DroneState | None = None
        self.target: np.ndarray = np.zeros(3)
        self.reward_fn: CompositeReward | None = None
        self._crashed = False

    # ---- hooks for subclasses ------------------------------------------------

    def _build_terrain(self) -> Heightfield:
        """Return the heightfield for this episode's location."""
        return build_heightfield(self.location, cache=self._tile_cache)

    def _make_target(self) -> np.ndarray:
        """Return the active target position. Subclasses must override."""
        raise NotImplementedError

    def _build_reward(self) -> CompositeReward:
        """Return this task's reward. Subclasses must override."""
        raise NotImplementedError

    def _task_terminated(self, state: DroneState) -> bool:
        """Return True when the task's own success condition is met."""
        raise NotImplementedError

    def _task_step(self, state: DroneState) -> tuple[bool, bool]:
        """Advance task bookkeeping. Return ``(target_reached, all_cleared)``."""
        return False, False

    def _task_reset(self) -> None:
        """Reset task state. Called after terrain and before the first target."""

    # ---- spaces and observations -------------------------------------------

    def _build_observation_space(self) -> spaces.Space:
        if self.perception == "state":
            return spaces.Box(-np.inf, np.inf, shape=(OBS_DIM,), dtype=np.float32)
        if self.perception == "rgb":
            return spaces.Box(
                0, 255, shape=(3, IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8
            )
        return spaces.Box(
            0.0, 1.0, shape=(1, IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32
        )

    def _state_obs(self, state: DroneState) -> np.ndarray:
        """Build the 23-D kinematic observation."""
        _, _, yaw = quat_to_euler(state.quaternion)
        to_target = self.target - state.position
        distance = float(np.linalg.norm(to_target))
        bearing = math.atan2(float(to_target[1]), float(to_target[0])) - yaw
        ground = self.terrain.height(float(state.position[0]), float(state.position[1]))
        probes = self.terrain.probe_ahead(
            float(state.position[0]),
            float(state.position[1]),
            yaw,
            PROBE_DISTANCES,
        )
        obs = np.concatenate(
            [
                state.position,                              # 0:3
                state.velocity,                              # 3:6
                state.quaternion,                            # 6:10
                state.angular_velocity,                       # 10:13
                [math.sin(bearing), math.cos(bearing), math.log1p(distance)],  # 13:16
                [self.battery.soc, self.battery.voltage / 16.8],               # 16:18
                [float(state.position[2]) - ground],          # 18:19
                probes - float(state.position[2]),            # 19:23
            ]
        )
        return obs.astype(np.float32)

    def _pixel_obs(self, state: DroneState) -> np.ndarray:
        """Sample terrain along camera rays into an image observation.

        A NumPy ray-march over the cached heightfield — no GPU, no browser. This
        approximates the browser's render; the browser itself is the visual
        check, not the training signal.
        """
        _, _, yaw = quat_to_euler(state.quaternion)
        fov = math.radians(100.0)
        angles = np.linspace(-fov / 2, fov / 2, IMAGE_SIZE)
        pitches = np.linspace(fov / 4, -fov / 2, IMAGE_SIZE)
        depth = np.ones((IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)
        max_range = 400.0
        px, py, pz = (float(v) for v in state.position)
        for row, elev in enumerate(pitches):
            for col, az in enumerate(angles):
                a = yaw + az
                dx, dy = math.cos(a), math.sin(a)
                dz = math.tan(elev)
                hit = max_range
                for d in np.linspace(4.0, max_range, 24):
                    if pz + dz * d <= self.terrain.height(px + dx * d, py + dy * d):
                        hit = float(d)
                        break
                depth[row, col] = hit / max_range
        if self.perception == "depth":
            return depth[None, :, :]
        shade = (depth * 255.0).astype(np.uint8)
        return np.repeat(shade[None, :, :], 3, axis=0)

    def _get_obs(self, state: DroneState) -> np.ndarray:
        """Return the observation for the configured perception mode."""
        if self.perception == "state":
            return self._state_obs(state)
        return self._pixel_obs(state)

    def _initial_state(self) -> DroneState:
        """Spawn at the field centre, ``spawn_height`` above the ground."""
        ground = self.terrain.height(0.0, 0.0)
        z = ground + self.spawn_height
        if self.terrain.water_z is not None:
            z = max(z, self.terrain.water_z + self.spawn_height)
        return DroneState(
            position=[0.0, 0.0, z],
            velocity=[0.0, 0.0, 0.0],
            quaternion=euler_to_quat(0.0, 0.0, 0.0),
            angular_velocity=[0.0, 0.0, 0.0],
            time=0.0,
        )

    def _get_info(self, state: DroneState) -> dict[str, Any]:
        """Diagnostics returned alongside every observation."""
        return {
            "battery_soc": self.battery.soc,
            "battery_voltage": self.battery.voltage,
            "crashed": self._crashed,
            "distance_to_target": float(
                np.linalg.norm(self.target - state.position)
            ),
            "terrain_source": self.terrain.source,
            "difficulty": self.difficulty,
        }

    # ---- Gymnasium API -----------------------------------------------------

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Start a new episode.

        Args:
            seed: Seeds ``self.np_random``; the same seed reproduces the episode.
            options: ``{"difficulty": float}`` in ``[0, 1]`` for the curriculum.
        """
        super().reset(seed=seed)
        if options is not None and "difficulty" in options:
            self.difficulty = float(np.clip(options["difficulty"], 0.0, 1.0))

        self._crashed = False
        self.battery.reset(self.battery_mah)
        # Required, not optional: ANGLE mode rebuilds attitude from the
        # controller's own integrated yaw, so a stale value snaps the heading on
        # the first step (verified: a drone spawned at 1.0 rad jumps to 0.0).
        self.controller.reset(yaw=0.0)
        self._task_reset()
        self.state = self._initial_state()
        self.target = self._make_target()
        self.reward_fn = self._build_reward()
        self.reward_fn.reset(self.state.position, self.target)
        return self._get_obs(self.state), self._get_info(self.state)

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Advance one control step of ``DT`` seconds."""
        if self.state is None or self.reward_fn is None:
            raise RuntimeError("reset() must be called before step()")

        a = np.clip(np.asarray(action, dtype=np.float64).reshape(4), -1.0, 1.0)
        throttle = float((a[0] + 1.0) / 2.0)      # [-1,1] -> [0,1]
        roll, pitch, yaw = (float(v) for v in a[1:])

        crash = self.physics.step(
            self.state, throttle=throttle, roll=roll, pitch=pitch, yaw=yaw, dt=DT
        )
        self.battery.update(throttle=throttle, dt=DT)

        self._crashed = crash is not None
        battery_empty = self.battery.is_empty
        target_reached, all_cleared = (False, False)
        if not self._crashed:
            target_reached, all_cleared = self._task_step(self.state)
            if target_reached and not all_cleared:
                self.target = self._make_target()
                self.reward_fn.retarget(self.state.position, self.target)

        reward = self.reward_fn(
            RewardContext(
                state=self.state,
                throttle=throttle,
                target=self.target,
                crashed=self._crashed,
                target_reached=target_reached,
                all_targets_cleared=all_cleared,
                battery_empty=battery_empty,
            )
        )

        terminated = bool(
            self._crashed
            or battery_empty
            or all_cleared
            or self._task_terminated(self.state)
        )
        truncated = bool(not terminated and self.state.time >= self.max_time)

        info = self._get_info(self.state)
        info["crash_reason"] = crash.value if crash is not None else None
        info["battery_empty"] = battery_empty
        return self._get_obs(self.state), float(reward), terminated, truncated, info
