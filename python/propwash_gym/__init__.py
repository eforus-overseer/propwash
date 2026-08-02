"""propwash-gym — a Gymnasium RL environment on real satellite terrain.

Importing this package registers the environment IDs with Gymnasium's global
registry, so ``propwash_gym.make("PropwashHunt-v0")`` (or ``gymnasium.make``)
works after ``import propwash_gym``.

Many small registered variants over one task class follows MiniGrid; selecting
behaviour through registry kwargs follows gym-pybullet-drones.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
from gymnasium.envs.registration import register

__version__ = "0.1.0"

_HUNT = "propwash_gym.envs.hunt_env:HuntEnv"

# 60 s at dt=0.02 -> 3000 steps.
_EPISODE_STEPS = 3000

_VARIANTS: dict[str, tuple[str, int, dict[str, Any]]] = {
    "PropwashHunt-v0": (_HUNT, _EPISODE_STEPS, {}),
    # Windless, single-target, tightly-scattered arena — the curriculum's
    # starting point. Terrain stays real DEM at every difficulty.
    "PropwashHuntEasy-v0": (_HUNT, _EPISODE_STEPS, {"start_difficulty": 0.0}),
    "PropwashHuntRGB-v0": (_HUNT, _EPISODE_STEPS, {"perception": "rgb"}),
    "PropwashHuntDepth-v0": (_HUNT, _EPISODE_STEPS, {"perception": "depth"}),
}

for _env_id, (_entry, _max_steps, _kwargs) in _VARIANTS.items():
    if _env_id not in gym.registry:
        register(
            id=_env_id,
            entry_point=_entry,
            max_episode_steps=_max_steps,
            kwargs={k: v for k, v in _kwargs.items() if k != "start_difficulty"},
        )


def make(env_id: str, **kwargs: Any) -> gym.Env:
    """Create a registered propwash-gym environment.

    Variants carrying ``start_difficulty`` are returned already wrapped in a
    :class:`~propwash_gym.envs.curriculum.CurriculumWrapper`, so the difficulty
    they advertise is applied from the first reset.

    Args:
        env_id: A registered ID, e.g. ``"PropwashHunt-v0"``.
        **kwargs: Forwarded to the env constructor (``location``, ``offline``, …).
    """
    from propwash_gym.envs.curriculum import CurriculumWrapper

    env = gym.make(env_id, **kwargs)
    spec_kwargs = _VARIANTS.get(env_id, (None, 0, {}))[2]
    if "start_difficulty" in spec_kwargs:
        return CurriculumWrapper(
            env, mode="success", start_difficulty=spec_kwargs["start_difficulty"]
        )
    return env
