"""TARGET HUNT: layout generation, collection, and difficulty scaling."""

import numpy as np
import pytest

from propwash_gym.envs.hunt_env import (
    COLLECT_PAD,
    MAX_WIND,
    MAX_TARGETS,
    MIN_ORIGIN_DISTANCE,
    HuntEnv,
    Target,
)


def _env(**kw):
    kw.setdefault("offline", True)
    kw.setdefault("location", "negev")
    return HuntEnv(**kw)


def test_full_difficulty_places_twelve_targets():
    env = _env()
    env.reset(seed=0, options={"difficulty": 1.0})
    assert len(env.targets) == MAX_TARGETS


def test_zero_difficulty_places_a_single_target():
    env = _env()
    env.reset(seed=0, options={"difficulty": 0.0})
    assert len(env.targets) == 1


def test_layout_is_reproducible_for_a_seed():
    a, b = _env(), _env()
    a.reset(seed=99, options={"difficulty": 1.0})
    b.reset(seed=99, options={"difficulty": 1.0})
    assert [t.position.tolist() for t in a.targets] == [
        t.position.tolist() for t in b.targets
    ]


def test_layouts_differ_between_seeds():
    a, b = _env(), _env()
    a.reset(seed=1, options={"difficulty": 1.0})
    b.reset(seed=2, options={"difficulty": 1.0})
    assert [t.position.tolist() for t in a.targets] != [
        t.position.tolist() for t in b.targets
    ]


def test_targets_are_never_closer_than_the_minimum_to_the_pad():
    env = _env()
    env.reset(seed=3, options={"difficulty": 1.0})
    for t in env.targets:
        if t.perch == "GND" or t.perch == "HILL":
            assert float(np.linalg.norm(t.position[:2])) >= MIN_ORIGIN_DISTANCE - 1e-6


def test_target_sizes_are_within_the_documented_range():
    env = _env()
    env.reset(seed=4, options={"difficulty": 1.0})
    for t in env.targets:
        assert 1.3 <= t.size <= 3.3 + 1e-9


def test_collect_radius_follows_the_sim_formula():
    t = Target(position=np.zeros(3), size=2.0, perch="GND")
    assert pytest.approx(2.0 / 2 + COLLECT_PAD, abs=1e-12) == t.collect_radius


def test_spread_grows_with_difficulty():
    easy, hard = _env(), _env()
    easy.reset(seed=5, options={"difficulty": 0.0})
    hard.reset(seed=5, options={"difficulty": 1.0})
    easy_max = max(float(np.linalg.norm(t.position[:2])) for t in easy.targets)
    hard_max = max(float(np.linalg.norm(t.position[:2])) for t in hard.targets)
    assert hard_max > easy_max


def test_teleporting_onto_a_target_collects_it_and_pays_the_bonus():
    env = _env()
    env.reset(seed=6, options={"difficulty": 0.0})
    target = env.targets[0]
    env.state.position = target.position.copy()
    _, reward, _, _, info = env.step(np.array([0.0, 0.0, 0.0, 0.0], np.float32))
    assert info["targets_collected"] == 1
    assert reward > 10.0, "collection bonus plus clear bonus expected"


def test_clearing_the_only_target_terminates_the_episode():
    env = _env()
    env.reset(seed=7, options={"difficulty": 0.0})
    env.state.position = env.targets[0].position.copy()
    _, _, terminated, _, info = env.step(np.array([0.0, 0.0, 0.0, 0.0], np.float32))
    assert terminated
    assert info["all_cleared"] is True


def test_the_active_target_advances_after_a_collection():
    env = _env()
    env.reset(seed=8, options={"difficulty": 1.0})
    first = env.target.copy()
    env.state.position = env.targets[0].position.copy()
    env.step(np.array([0.0, 0.0, 0.0, 0.0], np.float32))
    assert not np.array_equal(first, env.target)
    assert env.next_index == 1


def test_hilltop_fallback_position_is_used_when_all_samples_are_rejected():
    # Force every hill sample inside the rejection radius by shrinking spread.
    env = _env(spread_override=1.0)
    env.reset(seed=9, options={"difficulty": 1.0})
    # The sim's fallback is (60, 60); at least one target should land there.
    assert any(
        abs(float(t.position[0]) - 60.0) < 1e-6 and abs(float(t.position[1]) - 60.0) < 1e-6
        for t in env.targets
    )


def test_info_reports_progress():
    env = _env()
    env.reset(seed=10, options={"difficulty": 1.0})
    _, _, _, _, info = env.step(np.array([0.2, 0, 0, 0], np.float32))
    assert info["targets_collected"] == 0
    assert info["targets_total"] == MAX_TARGETS


def test_explicit_wind_scale_survives_reset():
    """A constructor argument must not be silently discarded.

    Difficulty otherwise drives wind, but an env built with `wind_scale=0.5`
    that quietly flies at 0.7 from the first reset is the kind of thing that
    costs an afternoon to find.
    """
    env = _env(wind_scale=0.5)
    env.reset(seed=0, options={"difficulty": 1.0})
    assert pytest.approx(0.5) == env.physics.wind_scale


def test_wind_scales_with_difficulty_when_not_overridden():
    calm = _env()
    calm.reset(seed=0, options={"difficulty": 0.0})
    assert pytest.approx(0.0) == calm.physics.wind_scale

    windy = _env()
    windy.reset(seed=0, options={"difficulty": 1.0})
    assert pytest.approx(MAX_WIND) == windy.physics.wind_scale


def test_difficulty_persists_when_reset_omits_it():
    """Documented trap: a bare reset() inherits the previous difficulty.

    Gymnasium's vector-env autoreset calls `reset()` with no options, so a
    curriculum must pass difficulty on every reset rather than relying on the
    env to remember or to default.
    """
    env = _env()
    env.reset(seed=0, options={"difficulty": 0.0})
    assert len(env.targets) == 1
    env.reset(seed=1)                      # no options
    assert env.difficulty == 0.0
    assert len(env.targets) == 1, "inherited difficulty, not reset to full"
