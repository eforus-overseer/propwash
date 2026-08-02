"""4S LiPo model ported from index.html `const Battery`."""

import pytest

from propwash_gym.core.battery import Battery


def test_fresh_pack_is_full_and_near_max_voltage():
    b = Battery(capacity_mah=1300.0)
    assert pytest.approx(1.0, abs=1e-12) == b.soc
    # A fresh pack draws no current, so there is no IR sag: the unloaded
    # (3.0 + 1.2) * 4 = 16.8 V sits exactly at the clamp ceiling.
    assert pytest.approx(16.8, abs=1e-9) == b.voltage


def test_idle_current_is_1_2_amps():
    b = Battery(capacity_mah=1300.0)
    b.update(throttle=0.0, dt=0.02)
    assert pytest.approx(1.2, abs=1e-12) == b.current


def test_full_throttle_current_is_79_2_amps():
    b = Battery(capacity_mah=1300.0)
    b.update(throttle=1.0, dt=0.02)
    assert pytest.approx(1.2 + 78.0, abs=1e-12) == b.current


def test_coulomb_counting_accumulates_mah():
    b = Battery(capacity_mah=1300.0)
    # 79.2 A for 1 second = 79.2/3600 Ah = 22.0 mAh
    for _ in range(100):
        b.update(throttle=1.0, dt=0.01)
    assert pytest.approx(79.2 / 3.6 * 1.0, rel=1e-9) == b.drawn_mah


def test_voltage_sags_under_load():
    idle = Battery(capacity_mah=1300.0)
    idle.update(throttle=0.0, dt=0.02)
    loaded = Battery(capacity_mah=1300.0)
    loaded.update(throttle=1.0, dt=0.02)
    assert loaded.voltage < idle.voltage


def test_voltage_is_clamped_to_pack_maximum():
    b = Battery(capacity_mah=1300.0)
    assert b.voltage <= 16.8


def test_depleted_pack_reports_zero_soc_and_is_empty():
    b = Battery(capacity_mah=10.0)
    for _ in range(2000):
        b.update(throttle=1.0, dt=0.02)
    assert pytest.approx(0.0, abs=1e-12) == b.soc
    assert b.is_empty


def test_thrust_scale_never_falls_below_floor():
    b = Battery(capacity_mah=10.0)
    for _ in range(2000):
        b.update(throttle=1.0, dt=0.02)
    # clamp((V-11.2)/5.6, .35, 1) * .35 + .65  ->  floor is .35*.35+.65
    assert b.thrust_scale >= 0.35 * 0.35 + 0.65 - 1e-12


def test_thrust_scale_is_one_at_full_charge_unloaded():
    b = Battery(capacity_mah=1300.0)
    assert pytest.approx(1.0, abs=1e-9) == b.thrust_scale


def test_reset_restores_full_pack():
    b = Battery(capacity_mah=1300.0)
    for _ in range(50):
        b.update(throttle=1.0, dt=0.02)
    b.reset(capacity_mah=1000.0)
    assert b.drawn_mah == 0.0
    assert b.capacity_mah == 1000.0
    assert pytest.approx(1.0, abs=1e-12) == b.soc
