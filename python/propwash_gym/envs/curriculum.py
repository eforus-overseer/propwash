"""Difficulty scheduling, as a wrapper.

Two schedules, matching the practice in reference drone-RL projects:

* ``"success"`` — performance-staged. Difficulty rises when the recent success
  rate clears ``success_threshold`` and falls below ``failure_threshold``.
* ``"step"`` — annealed linearly from ``start_difficulty`` to 1.0 across
  ``anneal_steps`` environment steps.

Curriculum state belongs here, never inside the env. The wrapped env only needs
to accept ``reset(options={"difficulty": float})``.

Starting at difficulty 0 matters: training directly at full difficulty stalls at
zero success, whereas mastering a flat single-target arena first converges.
"""

from __future__ import annotations

from collections import deque
from typing import Any

import gymnasium as gym
import numpy as np

MODES = ("success", "step")


class CurriculumWrapper(gym.Wrapper):
    """Anneal task difficulty across episodes.

    Args:
        env: Environment accepting a ``difficulty`` reset option.
        mode: ``"success"`` or ``"step"``.
        start_difficulty: Initial difficulty in ``[0, 1]``.
        difficulty_step: Increment applied per stage in success mode.
        window: Episodes considered when computing the success rate.
        success_threshold: Success rate at or above which difficulty rises.
        failure_threshold: Success rate at or below which difficulty falls.
        anneal_steps: Steps over which step mode ramps to 1.0.
    """

    def __init__(
        self,
        env: gym.Env,
        mode: str = "success",
        start_difficulty: float = 0.0,
        difficulty_step: float = 0.1,
        window: int = 20,
        success_threshold: float = 0.7,
        failure_threshold: float = 0.2,
        anneal_steps: int = 500_000,
    ) -> None:
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
        if int(window) < 1:
            # A zero-length deque never fills, so the staging check divides by
            # len(self._results) == 0 on the first reset.
            raise ValueError(f"window must be at least 1, got {window!r}")
        super().__init__(env)
        self.mode = mode
        self.difficulty = float(np.clip(start_difficulty, 0.0, 1.0))
        self.start_difficulty = self.difficulty
        self.difficulty_step = float(difficulty_step)
        self.success_threshold = float(success_threshold)
        self.failure_threshold = float(failure_threshold)
        self.anneal_steps = int(anneal_steps)
        self._results: deque[bool] = deque(maxlen=int(window))
        self._elapsed = 0

    def reset(self, **kwargs) -> tuple[Any, dict[str, Any]]:
        """Reset the inner env at the current difficulty."""
        if self.mode == "step":
            frac = min(1.0, self._elapsed / max(1, self.anneal_steps))
            self.difficulty = float(
                np.clip(
                    self.start_difficulty + (1.0 - self.start_difficulty) * frac,
                    0.0,
                    1.0,
                )
            )
        else:
            self._maybe_stage()
        options = dict(kwargs.pop("options", None) or {})
        options["difficulty"] = self.difficulty
        return self.env.reset(options=options, **kwargs)

    def _maybe_stage(self) -> None:
        """Raise or lower difficulty once the window has enough episodes."""
        if len(self._results) < self._results.maxlen:
            return
        rate = sum(self._results) / len(self._results)
        if rate >= self.success_threshold:
            self.difficulty = float(np.clip(self.difficulty + self.difficulty_step, 0.0, 1.0))
            self._results.clear()
        elif rate <= self.failure_threshold:
            self.difficulty = float(np.clip(self.difficulty - self.difficulty_step, 0.0, 1.0))
            self._results.clear()

    def step(self, action) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        """Step the inner env, recording episode outcomes for staging."""
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._elapsed += 1
        if terminated or truncated:
            self._results.append(bool(info.get("all_cleared", False)))
        info["difficulty"] = self.difficulty
        return obs, reward, terminated, truncated, info
