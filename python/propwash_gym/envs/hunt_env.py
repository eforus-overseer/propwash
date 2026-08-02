"""TARGET HUNT — collect randomised targets scattered across real terrain.

Ported from ``index.html`` ``Race.layoutHunt()``. A single ``roll`` per target
selects its perch:

* ``roll < 0.20`` → rooftop (only when buildings exist)
* ``0.20 ≤ roll < 0.55`` → hilltop, best of 7 samples by height
* ``roll ≥ 0.55`` → open ground, one sample

Two faithful quirks from the original, both easy to port wrong:

1. The rooftop branch is guarded by "are there buildings". With none, a
   ``roll < 0.20`` **falls through to the hilltop branch** rather than being
   discarded.
2. The hill sampler skips candidates within 25 m of the pad **without
   retrying**, so all 7 samples can be rejected. The original then falls back to
   a fixed ``(60, 60)`` position.

All randomness goes through ``self.np_random`` so layouts are reproducible.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from propwash_gym.core.reward import (
    AllTargetsBonus,
    BatteryDepletedPenalty,
    CompositeReward,
    CrashPenalty,
    EnergyPenalty,
    ProgressReward,
    TargetReachedBonus,
)
from propwash_gym.core.state import DroneState
from propwash_gym.envs.base_env import PropwashEnv

MAX_TARGETS = 12
MIN_TARGETS = 1
MAX_SPREAD_M = 330.0
MIN_SPREAD_M = 60.0
MIN_ORIGIN_DISTANCE = 25.0
HILL_SAMPLES = 7
HILL_PERCH_MIN_HEIGHT = 6.0
#: Added to half the target's size to give the collection radius.
COLLECT_PAD = 1.7
SIZE_MIN = 1.3
SIZE_RANGE = 2.0
FALLBACK_XY = (60.0, 60.0)
MAX_WIND = 0.7


@dataclass
class Target:
    """One hunt target.

    Attributes:
        position: World position of the target's centre.
        size: Edge length in metres.
        perch: ``"GND"``, ``"HILL"``, ``"ROOF"``, or ``"BUOY"``.
    """

    position: np.ndarray
    size: float
    perch: str

    @property
    def collect_radius(self) -> float:
        """Proximity radius that counts as collecting this target."""
        return self.size / 2.0 + COLLECT_PAD


class HuntEnv(PropwashEnv):
    """Collect every target in a randomised layout without crashing.

    Args:
        spread_override: Force the placement radius, for tests.
        **kwargs: Forwarded to :class:`PropwashEnv`.
    """

    def __init__(self, spread_override: float | None = None, **kwargs) -> None:
        self.targets: list[Target] = []
        self.next_index = 0
        self._all_cleared = False
        self._spread_override = spread_override
        super().__init__(**kwargs)

    @property
    def _spread(self) -> float:
        if self._spread_override is not None:
            return float(self._spread_override)
        return MIN_SPREAD_M + (MAX_SPREAD_M - MIN_SPREAD_M) * self.difficulty

    @property
    def _count(self) -> int:
        return int(round(MIN_TARGETS + (MAX_TARGETS - MIN_TARGETS) * self.difficulty))

    def _task_reset(self) -> None:
        """Roll a fresh layout and scale wind with difficulty."""
        self.physics.wind_scale = MAX_WIND * self.difficulty
        self.next_index = 0
        self._all_cleared = False
        self.targets = self._layout()

    def _layout(self) -> list[Target]:
        """Generate this episode's targets."""
        spread = self._spread
        out: list[Target] = []
        for _ in range(self._count):
            roll = float(self.np_random.random())
            size = SIZE_MIN + float(self.np_random.random()) * SIZE_RANGE
            # No buildings are modelled yet, so a rooftop roll falls through to
            # the hill branch exactly as the browser sim does.
            hill = roll < 0.55
            best: tuple[float, float, float] | None = None
            for _ in range(HILL_SAMPLES if hill else 1):
                cx = (float(self.np_random.random()) - 0.5) * 2.0 * spread
                cy = (float(self.np_random.random()) - 0.5) * 2.0 * spread
                if np.hypot(cx, cy) < MIN_ORIGIN_DISTANCE:
                    continue
                h = self.terrain.height(cx, cy)
                if best is None or h > best[2]:
                    best = (cx, cy, h)
            if best is None:
                fx, fy = FALLBACK_XY
                best = (fx, fy, self.terrain.height(fx, fy))

            x, y, ground = best
            z = ground + size * 0.5 + 0.05
            perch = "GND"
            if self.terrain.water_z is not None and z < self.terrain.water_z + size * 0.5:
                z = self.terrain.water_z + size * 0.35
                perch = "BUOY"
            elif hill and ground > HILL_PERCH_MIN_HEIGHT + self.terrain.height(0.0, 0.0):
                perch = "HILL"
            out.append(
                Target(position=np.array([x, y, z]), size=size, perch=perch)
            )
        return out

    def _make_target(self) -> np.ndarray:
        """Return the active target's position."""
        if not self.targets:
            return np.zeros(3)
        idx = min(self.next_index, len(self.targets) - 1)
        return self.targets[idx].position.copy()

    def _build_reward(self) -> CompositeReward:
        """Dense progress shaping plus collection bonuses and crash penalties."""
        return CompositeReward(
            [
                ProgressReward(scale=1.0),
                TargetReachedBonus(bonus=10.0),
                AllTargetsBonus(bonus=25.0),
                CrashPenalty(penalty=10.0),
                BatteryDepletedPenalty(penalty=5.0),
                EnergyPenalty(weight=0.01),
            ]
        )

    def _task_step(self, state: DroneState) -> tuple[bool, bool]:
        """Collect the active target when within its radius."""
        if self._all_cleared or not self.targets:
            return False, False
        target = self.targets[self.next_index]
        if float(np.linalg.norm(target.position - state.position)) > target.collect_radius:
            return False, False
        self.next_index += 1
        if self.next_index >= len(self.targets):
            self._all_cleared = True
            return True, True
        return True, False

    def _task_terminated(self, state: DroneState) -> bool:
        """Terminate once every target is collected."""
        return self._all_cleared

    def _get_info(self, state: DroneState):
        """Add hunt progress to the base diagnostics."""
        info = super()._get_info(state)
        info["targets_collected"] = self.next_index
        info["targets_total"] = len(self.targets)
        info["all_cleared"] = self._all_cleared
        info["active_perch"] = (
            self.targets[self.next_index].perch
            if self.targets and self.next_index < len(self.targets)
            else None
        )
        return info
