"""Reward terms in isolation, and their composition."""

import numpy as np
import pytest

from propwash_gym.core.reward import (
    AllTargetsBonus,
    BatteryDepletedPenalty,
    CompositeReward,
    CrashPenalty,
    EnergyPenalty,
    ProgressReward,
    RewardContext,
    TargetReachedBonus,
)
from propwash_gym.core.state import DroneState


def _ctx(**kw):
    base = dict(
        state=DroneState([0, 0, 10], [0, 0, 0], [1, 0, 0, 0], [0, 0, 0], 0.0),
        throttle=0.0,
        target=np.array([10.0, 0.0, 10.0]),
        crashed=False,
        target_reached=False,
        all_targets_cleared=False,
        battery_empty=False,
    )
    base.update(kw)
    return RewardContext(**base)


def test_progress_reward_pays_for_closing_distance():
    r = ProgressReward(scale=1.0)
    r.reset(np.array([0.0, 0.0, 10.0]), np.array([10.0, 0.0, 10.0]))
    moved = _ctx(state=DroneState([3, 0, 10], [0, 0, 0], [1, 0, 0, 0], [0, 0, 0], 0.0))
    assert pytest.approx(3.0, abs=1e-9) == r(moved)


def test_progress_reward_is_negative_when_retreating():
    r = ProgressReward(scale=1.0)
    r.reset(np.array([0.0, 0.0, 10.0]), np.array([10.0, 0.0, 10.0]))
    back = _ctx(state=DroneState([-2, 0, 10], [0, 0, 0], [1, 0, 0, 0], [0, 0, 0], 0.0))
    assert pytest.approx(-2.0, abs=1e-9) == r(back)


def test_progress_reward_pays_zero_for_stalling():
    r = ProgressReward(scale=1.0)
    r.reset(np.array([0.0, 0.0, 10.0]), np.array([10.0, 0.0, 10.0]))
    same = _ctx(state=DroneState([0, 0, 10], [0, 0, 0], [1, 0, 0, 0], [0, 0, 0], 0.0))
    assert pytest.approx(0.0, abs=1e-9) == r(same)


def test_progress_reward_must_be_reset_between_episodes():
    r = ProgressReward(scale=1.0)
    r.reset(np.array([0.0, 0.0, 10.0]), np.array([10.0, 0.0, 10.0]))
    r(_ctx(state=DroneState([5, 0, 10], [0, 0, 0], [1, 0, 0, 0], [0, 0, 0], 0.0)))
    # New episode: spawn far away again. Without reset this would pay a huge
    # spurious negative.
    r.reset(np.array([0.0, 0.0, 10.0]), np.array([10.0, 0.0, 10.0]))
    first = r(_ctx(state=DroneState([1, 0, 10], [0, 0, 0], [1, 0, 0, 0], [0, 0, 0], 0.0)))
    assert pytest.approx(1.0, abs=1e-9) == first


def test_target_reached_bonus_only_fires_on_collection():
    b = TargetReachedBonus(bonus=10.0)
    assert b(_ctx(target_reached=True)) == 10.0
    assert b(_ctx(target_reached=False)) == 0.0


def test_all_targets_bonus_only_fires_once_cleared():
    b = AllTargetsBonus(bonus=25.0)
    assert b(_ctx(all_targets_cleared=True)) == 25.0
    assert b(_ctx(all_targets_cleared=False)) == 0.0


def test_crash_penalty_is_negative_on_crash_only():
    p = CrashPenalty(penalty=10.0)
    assert p(_ctx(crashed=True)) == -10.0
    assert p(_ctx(crashed=False)) == 0.0


def test_battery_penalty_fires_when_empty():
    p = BatteryDepletedPenalty(penalty=5.0)
    assert p(_ctx(battery_empty=True)) == -5.0
    assert p(_ctx(battery_empty=False)) == 0.0


def test_energy_penalty_scales_with_throttle_squared():
    p = EnergyPenalty(weight=1.0)
    assert pytest.approx(-0.25, abs=1e-9) == p(_ctx(throttle=0.5))
    assert pytest.approx(-1.0, abs=1e-9) == p(_ctx(throttle=1.0))


def test_composite_sums_its_terms():
    c = CompositeReward([TargetReachedBonus(10.0), CrashPenalty(10.0)])
    assert c(_ctx(target_reached=True, crashed=True)) == 0.0
    assert c(_ctx(target_reached=True)) == 10.0


def test_composite_resets_stateful_terms():
    progress = ProgressReward(scale=1.0)
    c = CompositeReward([progress, CrashPenalty(1.0)])
    c.reset(np.array([0.0, 0.0, 0.0]), np.array([5.0, 0.0, 0.0]))
    r = c(_ctx(state=DroneState([1, 0, 0], [0, 0, 0], [1, 0, 0, 0], [0, 0, 0], 0.0),
               target=np.array([5.0, 0.0, 0.0])))
    assert pytest.approx(1.0, abs=1e-9) == r
