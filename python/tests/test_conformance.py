"""Registered variants, and Gymnasium's own env checker across all of them."""

import gymnasium as gym
import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

import propwash_gym

STATE_IDS = ["PropwashHunt-v0", "PropwashHuntEasy-v0"]
PIXEL_IDS = ["PropwashHuntRGB-v0", "PropwashHuntDepth-v0"]
ALL_IDS = STATE_IDS + PIXEL_IDS


def test_every_variant_is_registered():
    for env_id in ALL_IDS:
        assert env_id in gym.registry, env_id


@pytest.mark.parametrize("env_id", STATE_IDS)
def test_check_env_passes_for_state_variants(env_id):
    env = propwash_gym.make(env_id, offline=True)
    check_env(env.unwrapped, skip_render_check=True)
    env.close()


@pytest.mark.parametrize("env_id", PIXEL_IDS)
def test_pixel_variants_build_and_produce_correct_shapes(env_id):
    # Ray-marched pixel obs is slow; assert shape rather than running check_env.
    env = propwash_gym.make(env_id, offline=True)
    obs, _ = env.reset(seed=0)
    assert obs.shape == env.observation_space.shape
    assert env.observation_space.contains(obs)
    env.close()


@pytest.mark.parametrize("env_id", STATE_IDS)
def test_time_limit_truncates_rather_than_running_forever(env_id):
    env = propwash_gym.make(env_id, offline=True)
    env.reset(seed=0)
    for _ in range(10_000):
        _, _, terminated, truncated, _ = env.step(env.action_space.sample())
        if terminated or truncated:
            break
    else:
        pytest.fail("episode never ended")
    env.close()


def test_easy_variant_starts_at_zero_difficulty():
    env = propwash_gym.make("PropwashHuntEasy-v0", offline=True)
    _, info = env.reset(seed=0)
    assert info["difficulty"] == 0.0
    env.close()


def test_make_rejects_an_unknown_id():
    with pytest.raises(Exception):
        propwash_gym.make("PropwashNope-v0")


def test_location_can_be_overridden_at_construction():
    env = propwash_gym.make("PropwashHunt-v0", location="canyon", offline=True)
    assert env.unwrapped.location == "canyon"
    env.close()
