"""Composable reward terms.

Reward shaping is expressed as **data** — a list of configured term objects
summed by :class:`CompositeReward` — rather than as a hard-coded function, so
reward stays independent of physics and observation.

:class:`ProgressReward` is stateful and **must** be reset at the start of every
episode. A dense progress signal is what makes goal-reaching learnable here;
absolute distance penalties are too sparse and stall at zero success.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, Sequence

import numpy as np

from propwash_gym.core.state import DroneState


@dataclass
class RewardContext:
    """Everything a reward term may read for one step.

    Attributes:
        state: Post-step drone state.
        throttle: Normalised throttle applied this step.
        target: World position of the active target.
        crashed: Whether this step ended in a crash.
        target_reached: Whether a target was collected this step.
        all_targets_cleared: Whether the final target was collected this step.
        battery_empty: Whether the pack ran out this step.
    """

    state: DroneState
    throttle: float
    target: np.ndarray
    crashed: bool
    target_reached: bool
    all_targets_cleared: bool
    battery_empty: bool


class RewardTerm(Protocol):
    """Interface for a single additive component of a reward function."""

    def __call__(self, ctx: RewardContext) -> float:
        """Return this term's contribution for one step."""
        ...


@dataclass
class ProgressReward:
    """Dense reward for closing distance to the active target.

    ``r = scale * (d_prev - d_curr)``: moving 1 m closer pays ``scale``,
    stalling pays 0, retreating pays negative. The previous distance is instance
    state and must be cleared via :meth:`reset` each episode.
    """

    scale: float = 1.0
    _prev_distance: float | None = field(default=None, repr=False)

    def reset(self, initial_position: np.ndarray, target: np.ndarray) -> None:
        """Seed the previous distance with the spawn distance (zero progress)."""
        self._prev_distance = float(np.linalg.norm(target - initial_position))

    def retarget(self, position: np.ndarray, target: np.ndarray) -> None:
        """Reseed after the active target changes, so the switch pays nothing."""
        self._prev_distance = float(np.linalg.norm(target - position))

    def __call__(self, ctx: RewardContext) -> float:
        """Return ``scale * (prev_distance - current_distance)``."""
        distance = float(np.linalg.norm(ctx.target - ctx.state.position))
        if self._prev_distance is None:
            self._prev_distance = distance
            return 0.0
        progress = self._prev_distance - distance
        self._prev_distance = distance
        return float(self.scale) * progress


@dataclass
class TargetReachedBonus:
    """One-off bonus for collecting a target."""

    bonus: float = 10.0

    def __call__(self, ctx: RewardContext) -> float:
        """Return ``bonus`` on the collection step, else 0."""
        return self.bonus if ctx.target_reached else 0.0


@dataclass
class AllTargetsBonus:
    """One-off bonus for clearing every target in the layout."""

    bonus: float = 25.0

    def __call__(self, ctx: RewardContext) -> float:
        """Return ``bonus`` on the clearing step, else 0."""
        return self.bonus if ctx.all_targets_cleared else 0.0


@dataclass
class CrashPenalty:
    """One-off penalty applied on a terminal crash."""

    penalty: float = 10.0

    def __call__(self, ctx: RewardContext) -> float:
        """Return ``-penalty`` when crashed, else 0."""
        return -self.penalty if ctx.crashed else 0.0


@dataclass
class BatteryDepletedPenalty:
    """One-off penalty for running the pack flat."""

    penalty: float = 5.0

    def __call__(self, ctx: RewardContext) -> float:
        """Return ``-penalty`` when the battery is empty, else 0."""
        return -self.penalty if ctx.battery_empty else 0.0


@dataclass
class EnergyPenalty:
    """Small penalty proportional to throttle², discouraging full-throttle flying."""

    weight: float = 0.01

    def __call__(self, ctx: RewardContext) -> float:
        """Return ``-weight * throttle²``."""
        t = float(ctx.throttle)
        return -self.weight * t * t


@dataclass
class CompositeReward:
    """Sums a sequence of reward terms and forwards resets to stateful ones."""

    terms: Sequence[RewardTerm]

    def reset(self, initial_position: np.ndarray, target: np.ndarray) -> None:
        """Reset every term that carries episode state."""
        for term in self.terms:
            reset = getattr(term, "reset", None)
            if callable(reset):
                reset(initial_position, target)

    def retarget(self, position: np.ndarray, target: np.ndarray) -> None:
        """Forward a target change to every term that tracks one."""
        for term in self.terms:
            retarget = getattr(term, "retarget", None)
            if callable(retarget):
                retarget(position, target)

    def __call__(self, ctx: RewardContext) -> float:
        """Return the sum of all term contributions."""
        return float(sum(term(ctx) for term in self.terms))
