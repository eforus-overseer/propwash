"""Tile maths, URL construction, DEM decode, and cache behaviour.

No network: fetching is exercised through an injected fake fetcher.
"""

import io

import numpy as np
import pytest
from PIL import Image

from propwash_gym.world.tiles import (
    TileCache,
    decode_terrarium,
    imagery_url,
    lat_to_tile,
    lon_to_tile,
    terrarium_url,
)


def test_lon_to_tile_matches_the_browser_sim():
    # SatTerrain.lon2t: (lon + 180) / 360 * 2**z
    assert pytest.approx(0.5 * 2**15, abs=1e-9) == lon_to_tile(0.0, 15)
    assert pytest.approx(0.0, abs=1e-9) == lon_to_tile(-180.0, 15)


def test_lat_to_tile_is_web_mercator():
    # At the equator the tile row is exactly half the grid.
    assert pytest.approx(0.5 * 2**15, abs=1e-6) == lat_to_tile(0.0, 15)


def test_known_locations_map_to_verified_tiles():
    # These tile indices were verified against live data while designing.
    assert (int(lon_to_tile(34.8020, 15)), int(lat_to_tile(30.6100, 15))) == (19551, 13454)
    assert (int(lon_to_tile(8.0414, 15)), int(lat_to_tile(46.6244, 15))) == (17115, 11575)
    assert (int(lon_to_tile(-151.7500, 15)), int(lat_to_tile(-16.5067, 15))) == (2571, 17907)


def test_imagery_url_uses_z_y_x_order():
    url = imagery_url(z=15, x=19551, y=13454)
    assert url.endswith("/15/13454/19551"), f"Esri is z/y/x, got {url}"


def test_terrarium_url_uses_z_x_y_order():
    url = terrarium_url(z=15, x=19551, y=13454)
    assert url.endswith("/15/19551/13454.png"), f"terrarium is z/x/y, got {url}"


def test_decode_terrarium_applies_the_documented_formula():
    # h = R*256 + G + B/256 - 32768
    px = np.zeros((2, 2, 3), dtype=np.uint8)
    px[0, 0] = (128, 0, 0)        # 128*256 - 32768 = 0
    px[0, 1] = (128, 100, 0)      # +100
    px[1, 0] = (128, 0, 128)      # +0.5
    px[1, 1] = (127, 0, 0)        # -256
    h = decode_terrarium(px)
    assert pytest.approx(0.0, abs=1e-9) == h[0, 0]
    assert pytest.approx(100.0, abs=1e-9) == h[0, 1]
    assert pytest.approx(0.5, abs=1e-9) == h[1, 0]
    assert pytest.approx(-256.0, abs=1e-9) == h[1, 1]


def _png_bytes(rgb):
    buf = io.BytesIO()
    Image.fromarray(np.asarray(rgb, dtype=np.uint8), "RGB").save(buf, "PNG")
    return buf.getvalue()


def test_cache_writes_then_reads_without_refetching(tmp_path):
    calls = []

    def fake_fetch(url):
        calls.append(url)
        return _png_bytes(np.full((4, 4, 3), 128, dtype=np.uint8))

    cache = TileCache(root=tmp_path, fetcher=fake_fetch)
    a = cache.get_terrarium(z=15, x=1, y=2)
    b = cache.get_terrarium(z=15, x=1, y=2)
    assert len(calls) == 1, "second read must come from disk"
    np.testing.assert_array_equal(a, b)


def test_cache_path_layout_separates_providers(tmp_path):
    cache = TileCache(root=tmp_path, fetcher=lambda url: _png_bytes(np.zeros((4, 4, 3))))
    cache.get_terrarium(z=15, x=1, y=2)
    cache.get_imagery(z=15, x=1, y=2)
    assert (tmp_path / "terrarium" / "15" / "1" / "2.png").exists()
    assert (tmp_path / "imagery" / "15" / "2" / "1.jpg").exists()


def test_fetch_failure_returns_none_rather_than_raising(tmp_path):
    def boom(url):
        raise OSError("network down")

    cache = TileCache(root=tmp_path, fetcher=boom)
    assert cache.get_terrarium(z=15, x=1, y=2) is None


def test_offline_cache_only_mode_never_calls_the_fetcher(tmp_path):
    called = []

    cache = TileCache(root=tmp_path, fetcher=lambda u: called.append(u), offline=True)
    assert cache.get_terrarium(z=15, x=9, y=9) is None
    assert called == []
