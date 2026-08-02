"""CurriculumWrapper: success-staged and step-annealed difficulty schedules."""

import gymnasium as gym
import numpy as np
import pytest

from propwash_gym.envs.curriculum import CurriculumWrapper
from propwash_gym.envs.hunt_env import HuntEnv


class _Stub(gym.Env):
    """Minimal env that records the difficulty it was reset with.

    Subclasses ``gym.Env`` because ``gym.Wrapper.__init__`` asserts
    ``isinstance(env, Env)`` (gymnasium >= 1.0); a plain object is rejected
    before any wrapper logic runs.
    """

    def __init__(self):
        self.seen: list[float] = []
        self.observation_space = None
        self.action_space = None
        self._success = False

    def reset(self, *, seed=None, options=None):
        self.seen.append(float((options or {}).get("difficulty", -1.0)))
        return np.zeros(1), {}

    def step(self, action):
        return np.zeros(1), 0.0, True, False, {"all_cleared": self._success}

    def close(self):
        pass


def test_rejects_an_unknown_mode():
    with pytest.raises(ValueError, match="mode"):
        CurriculumWrapper(_Stub(), mode="vibes")


def test_starts_at_the_configured_difficulty():
    env = _Stub()
    w = CurriculumWrapper(env, mode="success", start_difficulty=0.0)
    w.reset()
    assert env.seen[0] == 0.0


def test_success_mode_raises_difficulty_after_enough_wins():
    env = _Stub()
    w = CurriculumWrapper(
        env, mode="success", start_difficulty=0.0, difficulty_step=0.25,
        window=4, success_threshold=0.75,
    )
    w.reset()
    env._success = True
    for _ in range(4):
        w.step(None)
        w.reset()
    assert w.difficulty > 0.0


def test_success_mode_lowers_difficulty_after_enough_losses():
    env = _Stub()
    w = CurriculumWrapper(
        env, mode="success", start_difficulty=0.5, difficulty_step=0.25,
        window=4, failure_threshold=0.25,
    )
    w.reset()
    env._success = False
    for _ in range(4):
        w.step(None)
        w.reset()
    assert w.difficulty < 0.5


def test_difficulty_is_clamped_to_the_unit_interval():
    env = _Stub()
    w = CurriculumWrapper(
        env, mode="success", start_difficulty=0.9, difficulty_step=0.5,
        window=2, success_threshold=0.0,
    )
    w.reset()
    env._success = True
    for _ in range(6):
        w.step(None)
        w.reset()
    assert w.difficulty <= 1.0


def test_step_mode_anneals_linearly_with_elapsed_steps():
    env = _Stub()
    w = CurriculumWrapper(env, mode="step", start_difficulty=0.0, anneal_steps=100)
    w.reset()
    for _ in range(50):
        w.step(None)
    w.reset()
    assert 0.3 < w.difficulty < 0.7


def test_step_mode_reaches_full_difficulty_after_annealing():
    env = _Stub()
    w = CurriculumWrapper(env, mode="step", start_difficulty=0.0, anneal_steps=10)
    w.reset()
    for _ in range(20):
        w.step(None)
    w.reset()
    assert pytest.approx(1.0) == w.difficulty


def test_wraps_a_real_hunt_env_end_to_end():
    env = CurriculumWrapper(
        HuntEnv(offline=True), mode="success", start_difficulty=0.0
    )
    obs, info = env.reset(seed=0)
    assert info["difficulty"] == 0.0
    for _ in range(5):
        obs, r, term, trunc, info = env.step(env.action_space.sample())
        if term or trunc:
            break
    env.close()


@pytest.mark.parametrize("bad_window", [0, -1])
def test_window_below_one_is_rejected(bad_window):
    """A zero-length deque never fills, so staging divides by zero on reset."""
    with pytest.raises(ValueError, match="window must be at least 1"):
        CurriculumWrapper(_Stub(), mode="success", window=bad_window)


def test_wrapper_overrides_a_caller_supplied_difficulty():
    """The schedule wins over reset(options=...).

    Correct priority for training — the curriculum owns difficulty — but worth
    pinning so it is a decision rather than an accident.
    """
    env = _Stub()
    w = CurriculumWrapper(env, mode="success", start_difficulty=0.25)
    w.reset(options={"difficulty": 0.9})
    assert env.seen[0] == 0.25
