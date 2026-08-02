"""Gymnasium surface of PropwashEnv, exercised through a trivial subclass."""

import numpy as np
import pytest
from gymnasium import spaces

from propwash_gym.core.reward import CompositeReward, ProgressReward
from propwash_gym.envs.base_env import OBS_DIM, PropwashEnv
from propwash_gym.world.terrain import Heightfield


class _FixedTargetEnv(PropwashEnv):
    """Minimal task: one target 40 m north, for testing the base class."""

    def _build_terrain(self):
        return Heightfield(heights=np.full((16, 16), 100.0), extent_m=600.0)

    def _make_target(self):
        return np.array([0.0, 40.0, 120.0])

    def _build_reward(self):
        return CompositeReward([ProgressReward(1.0)])

    def _task_terminated(self, state):
        return False


def _env(**kw):
    kw.setdefault("offline", True)
    return _FixedTargetEnv(**kw)


def test_action_space_is_four_normalised_channels():
    env = _env()
    assert env.action_space.shape == (4,)
    assert np.all(env.action_space.low == -1.0)
    assert np.all(env.action_space.high == 1.0)


def test_state_observation_has_the_documented_width():
    env = _env(perception="state")
    assert env.observation_space.shape == (OBS_DIM,)
    assert OBS_DIM == 23


def test_reset_returns_obs_and_info():
    env = _env()
    obs, info = env.reset(seed=0)
    assert obs.shape == (OBS_DIM,)
    assert obs.dtype == np.float32
    assert "battery_soc" in info


def test_reset_with_the_same_seed_is_reproducible():
    a, _ = _env().reset(seed=42)
    b, _ = _env().reset(seed=42)
    np.testing.assert_array_equal(a, b)


def test_stepping_with_the_same_seed_and_actions_is_reproducible():
    def rollout():
        env = _env()
        env.reset(seed=7)
        out = []
        for _ in range(40):
            obs, r, term, trunc, _ = env.step(np.array([0.4, 0.0, 0.2, 0.0], np.float32))
            out.append((obs.copy(), r))
            if term or trunc:
                break
        return out

    a, b = rollout(), rollout()
    assert len(a) == len(b)
    for (oa, ra), (ob, rb) in zip(a, b):
        np.testing.assert_array_equal(oa, ob)
        assert ra == rb


def test_step_returns_the_five_tuple():
    env = _env()
    env.reset(seed=0)
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    assert obs.shape == (OBS_DIM,)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info, dict)


def test_observations_stay_inside_the_declared_space():
    env = _env()
    obs, _ = env.reset(seed=1)
    assert env.observation_space.contains(obs), obs
    for _ in range(60):
        obs, _, term, trunc, _ = env.step(np.array([0.6, 0.1, 0.3, 0.05], np.float32))
        assert env.observation_space.contains(obs), obs
        if term or trunc:
            break


def test_battery_drains_over_an_episode():
    env = _env()
    env.reset(seed=0)
    _, _, _, _, info = env.step(np.array([1.0, 0, 0, 0], np.float32))
    first = info["battery_soc"]
    for _ in range(50):
        _, _, term, trunc, info = env.step(np.array([1.0, 0, 0, 0], np.float32))
        if term or trunc:
            break
    assert info["battery_soc"] < first


def test_crashing_terminates_the_episode():
    env = _env(spawn_height=1.0)
    env.reset(seed=0)
    terminated = False
    for _ in range(400):
        _, _, terminated, truncated, info = env.step(
            np.array([-1.0, 0.0, 0.0, 0.0], np.float32)   # no thrust: fall
        )
        if terminated or truncated:
            break
    assert terminated
    assert info["crashed"] is True


def test_difficulty_option_is_recorded_and_clamped():
    env = _env()
    env.reset(seed=0, options={"difficulty": 0.42})
    assert pytest.approx(0.42) == env.difficulty
    env.reset(seed=0, options={"difficulty": 5.0})
    assert env.difficulty == 1.0


def test_unknown_perception_mode_is_rejected():
    with pytest.raises(ValueError, match="perception"):
        _env(perception="lidar")
