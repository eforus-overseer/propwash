"""Heightfield sampling, slopes, water level, and the offline fallback path."""

import numpy as np
import pytest

from propwash_gym.world.terrain import Heightfield, build_heightfield
from propwash_gym.world.tiles import TileCache


def _ramp(size=16, lo=0.0, hi=150.0):
    """Heights increasing along +x only, so gradients are predictable."""
    row = np.linspace(lo, hi, size)
    return np.tile(row, (size, 1))


def test_height_at_centre_matches_the_grid():
    hf = Heightfield(heights=_ramp(), extent_m=100.0)
    # Centre of the field is the middle of the ramp.
    assert hf.height(0.0, 0.0) == pytest.approx(75.0, rel=0.05)


def test_height_is_bilinear_and_monotonic_along_x():
    hf = Heightfield(heights=_ramp(), extent_m=100.0)
    a = hf.height(-40.0, 0.0)
    b = hf.height(0.0, 0.0)
    c = hf.height(40.0, 0.0)
    assert a < b < c


def test_height_outside_extent_is_clamped_not_wrapped():
    hf = Heightfield(heights=_ramp(), extent_m=100.0)
    assert hf.height(10_000.0, 0.0) == pytest.approx(hf.height(49.99, 0.0), rel=0.05)


def test_slope_is_zero_on_flat_terrain():
    hf = Heightfield(heights=np.full((16, 16), 20.0), extent_m=100.0)
    assert hf.slope(0.0, 0.0) == pytest.approx(0.0, abs=1e-9)


def test_slope_is_positive_on_a_ramp():
    hf = Heightfield(heights=_ramp(), extent_m=100.0)
    assert hf.slope(0.0, 0.0) > 0.0


def test_probe_ahead_returns_one_height_per_distance():
    hf = Heightfield(heights=_ramp(), extent_m=100.0)
    probes = hf.probe_ahead(x=0.0, y=0.0, yaw=0.0, distances=(5.0, 10.0, 20.0, 40.0))
    assert probes.shape == (4,)
    assert np.all(np.isfinite(probes))


def test_water_level_is_sea_level_for_oceanic_terrain():
    # Terrarium is referenced to sea level, so water sits at z = 0 exactly.
    heights = np.where(np.arange(256).reshape(16, 16) < 128, -1200.0, 300.0)
    hf = Heightfield(heights=heights, extent_m=100.0, has_water=True)
    assert hf.water_z == 0.0
    # Sanity: that level actually splits this field into wet and dry.
    assert (heights < hf.water_z).any()
    assert (heights > hf.water_z).any()


def test_no_water_level_when_location_has_none():
    hf = Heightfield(heights=_ramp(), extent_m=100.0, has_water=False)
    assert hf.water_z is None


def test_build_heightfield_falls_back_to_procedural_when_offline():
    cache = TileCache(root="/nonexistent-propwash-test", fetcher=lambda u: b"", offline=True)
    hf = build_heightfield("negev", cache=cache, seed=5)
    assert hf.source == "procedural"
    assert np.all(np.isfinite(hf.heights))


def test_build_heightfield_is_deterministic_in_fallback_mode():
    cache = TileCache(root="/nonexistent-propwash-test", fetcher=lambda u: b"", offline=True)
    a = build_heightfield("negev", cache=cache, seed=11)
    b = build_heightfield("negev", cache=cache, seed=11)
    np.testing.assert_array_equal(a.heights, b.heights)
