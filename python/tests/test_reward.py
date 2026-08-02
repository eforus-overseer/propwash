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
    # Raw metres-closed signal.
    r = ProgressReward(scale=1.0, normalize=False)
    r.reset(np.array([0.0, 0.0, 10.0]), np.array([10.0, 0.0, 10.0]))
    moved = _ctx(state=DroneState([3, 0, 10], [0, 0, 0], [1, 0, 0, 0], [0, 0, 0], 0.0))
    assert pytest.approx(3.0, abs=1e-9) == r(moved)


def test_progress_reward_is_negative_when_retreating():
    r = ProgressReward(scale=1.0, normalize=False)
    r.reset(np.array([0.0, 0.0, 10.0]), np.array([10.0, 0.0, 10.0]))
    back = _ctx(state=DroneState([-2, 0, 10], [0, 0, 0], [1, 0, 0, 0], [0, 0, 0], 0.0))
    assert pytest.approx(-2.0, abs=1e-9) == r(back)


def test_progress_reward_pays_zero_for_stalling():
    r = ProgressReward(scale=1.0)
    r.reset(np.array([0.0, 0.0, 10.0]), np.array([10.0, 0.0, 10.0]))
    same = _ctx(state=DroneState([0, 0, 10], [0, 0, 0], [1, 0, 0, 0], [0, 0, 0], 0.0))
    assert pytest.approx(0.0, abs=1e-9) == r(same)


def test_progress_reward_must_be_reset_between_episodes():
    r = ProgressReward(scale=1.0, normalize=False)
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
    progress = ProgressReward(scale=1.0, normalize=False)
    c = CompositeReward([progress, CrashPenalty(1.0)])
    c.reset(np.array([0.0, 0.0, 0.0]), np.array([5.0, 0.0, 0.0]))
    r = c(_ctx(state=DroneState([1, 0, 0], [0, 0, 0], [1, 0, 0, 0], [0, 0, 0], 0.0),
               target=np.array([5.0, 0.0, 0.0])))
    assert pytest.approx(1.0, abs=1e-9) == r


def test_normalized_progress_pays_the_same_regardless_of_target_distance():
    """A full traverse pays `scale` whether the target is 30 m or 300 m away.

    Unnormalized, a 12-target hunt at ~100 m spacing pays ~+1200 in progress
    against a -10 crash penalty, so crashing costs under 1% of episode return.
    Normalizing puts every term on the same order and makes reward invariant to
    how far apart the targets are scattered.
    """
    for distance in (30.0, 100.0, 300.0):
        r = ProgressReward(scale=1.0)          # normalize=True by default
        start = np.array([0.0, 0.0, 10.0])
        target = np.array([distance, 0.0, 10.0])
        r.reset(start, target)
        arrived = _ctx(
            state=DroneState(target, [0, 0, 0], [1, 0, 0, 0], [0, 0, 0], 0.0),
            target=target,
        )
        assert pytest.approx(1.0, abs=1e-9) == r(arrived), (
            f"full traverse of {distance} m should pay exactly 1.0"
        )


def test_normalized_progress_keeps_crash_penalty_significant():
    """The whole point of normalizing: terminal penalties stay comparable."""
    r = ProgressReward(scale=1.0)
    start = np.array([0.0, 0.0, 10.0])
    target = np.array([100.0, 0.0, 10.0])
    r.reset(start, target)
    halfway = _ctx(
        state=DroneState([50.0, 0.0, 10.0], [0, 0, 0], [1, 0, 0, 0], [0, 0, 0], 0.0),
        target=target,
    )
    earned = r(halfway)
    assert pytest.approx(0.5, abs=1e-9) == earned
    # A crash costs 10; progress toward one target pays at most 1.
    assert CrashPenalty(10.0)(_ctx(crashed=True)) < -earned * 10


def test_retarget_makes_a_target_switch_pay_nothing():
    """Task 12 collects 12 targets in sequence.

    Without reseeding on the switch, the step's progress compares distance to
    the NEW target against distance to the OLD one, paying a large spurious
    jump. `retarget()` is the guard, and it was previously untested.
    """
    r = ProgressReward(scale=1.0, normalize=False)
    first = np.array([10.0, 0.0, 10.0])
    second = np.array([-90.0, 0.0, 10.0])     # far away, opposite direction
    at_first = np.array([10.0, 0.0, 10.0])

    r.reset(np.array([0.0, 0.0, 10.0]), first)
    r(_ctx(state=DroneState(at_first, [0, 0, 0], [1, 0, 0, 0], [0, 0, 0], 0.0), target=first))

    r.retarget(at_first, second)
    # Standing still on the new target's first step must pay exactly zero.
    stayed = r(
        _ctx(
            state=DroneState(at_first, [0, 0, 0], [1, 0, 0, 0], [0, 0, 0], 0.0),
            target=second,
        )
    )
    assert pytest.approx(0.0, abs=1e-9) == stayed


def test_composite_forwards_retarget_to_stateful_terms():
    progress = ProgressReward(scale=1.0, normalize=False)
    c = CompositeReward([progress, CrashPenalty(1.0)])
    here = np.array([5.0, 0.0, 0.0])
    c.reset(np.array([0.0, 0.0, 0.0]), np.array([5.0, 0.0, 0.0]))
    c.retarget(here, np.array([100.0, 0.0, 0.0]))
    stayed = c(
        _ctx(
            state=DroneState(here, [0, 0, 0], [1, 0, 0, 0], [0, 0, 0], 0.0),
            target=np.array([100.0, 0.0, 0.0]),
        )
    )
    assert pytest.approx(0.0, abs=1e-9) == stayed
