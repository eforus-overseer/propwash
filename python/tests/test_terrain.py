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


def _tile_encoding_cache(value_fn):
    """A TileCache whose tiles encode an arbitrary function of (z, x, y)."""

    class _Fake(TileCache):
        def __init__(self):
            super().__init__(root="/nonexistent-propwash-crop", fetcher=lambda u: b"")

        def get_terrarium(self, z, x, y):
            return value_fn(z, x, y)

    return _Fake()


def test_dem_patch_is_centred_on_the_location_not_a_tile_corner():
    """World (0, 0) must be the place named in LOCATIONS.

    Truncating fractional tile coordinates puts the origin on a tile boundary
    instead — up to 1043 m away at Erg Chebbi, which would spawn the drone a
    kilometre from the dunes that define the location.

    Every tile here is uniformly 0 m except a single spike planted at the exact
    pixel the location falls on. A correctly centred crop samples that spike at
    world (0, 0); a corner-aligned one misses it entirely.
    """
    from propwash_gym.world.locations import get_location
    from propwash_gym.world.terrain import DEM_ZOOM, TILE_PX, build_heightfield
    from propwash_gym.world.tiles import lat_to_tile, lon_to_tile

    loc = get_location("sahara")
    fx = lon_to_tile(loc.lon, DEM_ZOOM)
    fy = lat_to_tile(loc.lat, DEM_ZOOM)
    home_tile = (int(fx), int(fy))
    spike_col = int((fx - int(fx)) * TILE_PX)
    spike_row = int((fy - int(fy)) * TILE_PX)

    def value_fn(z, x, y):
        # Flat 0 m everywhere: R=128, G=0, B=0 decodes to exactly 0.
        tile = np.zeros((TILE_PX, TILE_PX, 3), dtype=np.uint8)
        tile[..., 0] = 128
        if (x, y) == home_tile:
            tile[spike_row, spike_col, 1] = 200      # a 200 m spike
        return tile

    hf = build_heightfield("sahara", cache=_tile_encoding_cache(value_fn))
    assert hf.source == "dem"

    # The spike must land at the middle of the cropped grid. Assert on grid
    # position rather than on height(0, 0): bilinear interpolation averages a
    # one-pixel spike across four neighbours, so the sampled value is a quarter
    # of the peak even when the crop is exactly right.
    row, col = np.unravel_index(int(np.argmax(hf.heights)), hf.heights.shape)
    centre = hf.heights.shape[0] // 2
    assert abs(int(row) - centre) <= 1 and abs(int(col) - centre) <= 1, (
        f"spike landed at ({row}, {col}), expected ~({centre}, {centre}); "
        "the patch is not centred on the location"
    )
    # And it must be reachable from world origin at all.
    assert hf.height(0.0, 0.0) > 0.0


def test_partial_tile_loss_warns_instead_of_passing_off_holes_as_terrain():
    from propwash_gym.world.terrain import DEM_ZOOM, TILE_PX, build_heightfield

    calls = {"n": 0}

    def value_fn(z, x, y):
        calls["n"] += 1
        if calls["n"] == 3:          # drop exactly one tile
            return None
        return np.full((TILE_PX, TILE_PX, 3), 128, dtype=np.uint8)

    with pytest.warns(RuntimeWarning, match="elevation tiles missing"):
        hf = build_heightfield("negev", cache=_tile_encoding_cache(value_fn))
    assert hf.source == "dem", "one missing tile must not trigger the full fallback"
