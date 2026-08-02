"""Live-network checks against the real tile providers.

Marked ``network`` and skipped when unreachable, so the default suite stays
offline. Run explicitly with:  pytest -m network
"""

import pytest

from propwash_gym.world.locations import get_location
from propwash_gym.world.terrain import build_heightfield
from propwash_gym.world.tiles import (
    TileCache,
    decode_terrarium,
    lat_to_tile,
    lon_to_tile,
)

pytestmark = pytest.mark.network


@pytest.fixture(scope="module")
def cache(tmp_path_factory):
    return TileCache(root=tmp_path_factory.mktemp("tiles"))


def _fetch_dem(cache, key):
    loc = get_location(key)
    x = int(lon_to_tile(loc.lon, 15))
    y = int(lat_to_tile(loc.lat, 15))
    tile = cache.get_terrarium(15, x, y)
    if tile is None:
        pytest.skip("tile provider unreachable")
    return decode_terrarium(tile)


@pytest.mark.parametrize(
    "key,lo,hi",
    [
        # Ranges verified against live data on 2026-08-02.
        ("negev", 700.0, 1000.0),      # Makhtesh Ramon crater rim ~825-868 m
        ("alps", 800.0, 1300.0),       # Grindelwald valley ~953-1119 m
    ],
)
def test_known_locations_decode_to_expected_elevations(cache, key, lo, hi):
    h = _fetch_dem(cache, key)
    assert lo < float(h.min()) < hi, f"{key} min {h.min()}"
    assert lo < float(h.max()) < hi + 400.0, f"{key} max {h.max()}"


def test_bora_bora_contains_bathymetry_and_land(cache):
    h = _fetch_dem(cache, "borabora")
    assert float(h.min()) < -1000.0, "expected ocean floor below -1000 m"
    assert float(h.max()) > 200.0, "expected Mt Otemanu above 200 m"


def test_imagery_tile_is_fetchable(cache):
    loc = get_location("negev")
    x = int(lon_to_tile(loc.lon, 15))
    y = int(lat_to_tile(loc.lat, 15))
    img = cache.get_imagery(15, x, y)
    if img is None:
        pytest.skip("imagery provider unreachable")
    assert img.shape == (256, 256, 3)


def test_build_heightfield_uses_dem_when_online(cache):
    hf = build_heightfield("negev", cache=cache)
    if hf.source != "dem":
        pytest.skip("tiles unavailable")
    assert hf.heights.shape == (1024, 1024)
    assert hf.extent_m > 2000.0     # roughly a 3-4 km patch


def test_bora_bora_water_level_sits_between_floor_and_peak(cache):
    hf = build_heightfield("borabora", cache=cache)
    if hf.source != "dem":
        pytest.skip("tiles unavailable")
    assert hf.water_z is not None
    assert float(hf.heights.min()) < hf.water_z < float(hf.heights.max())
