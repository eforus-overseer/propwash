"""The seven real locations, and the procedural fallback generator."""

import numpy as np
import pytest

from propwash_gym.world.locations import LOCATIONS, Location, get_location
from propwash_gym.world.procedural import procedural_heights


def test_all_seven_locations_are_present():
    assert set(LOCATIONS) == {
        "negev", "alps", "cascades", "canyon", "iceland", "borabora", "sahara",
    }


def test_coordinates_match_the_browser_sim():
    assert get_location("negev").lat == pytest.approx(30.6100)
    assert get_location("negev").lon == pytest.approx(34.8020)
    assert get_location("borabora").lat == pytest.approx(-16.5067)
    assert get_location("borabora").lon == pytest.approx(-151.7500)
    assert get_location("canyon").lat == pytest.approx(36.0997)
    assert get_location("canyon").lon == pytest.approx(-112.0964)


def test_only_bora_bora_has_water():
    watery = {k for k, v in LOCATIONS.items() if v.has_water}
    assert watery == {"borabora"}


def test_unknown_location_raises_with_a_helpful_message():
    with pytest.raises(KeyError, match="unknown location"):
        get_location("atlantis")


def test_location_is_immutable():
    loc = get_location("negev")
    with pytest.raises(Exception):
        loc.lat = 0.0  # frozen dataclass


def test_procedural_heights_is_deterministic_for_a_seed():
    a = procedural_heights("negev", size=32, seed=7)
    b = procedural_heights("negev", size=32, seed=7)
    np.testing.assert_array_equal(a, b)


def test_procedural_heights_differs_between_seeds():
    a = procedural_heights("negev", size=32, seed=1)
    b = procedural_heights("negev", size=32, seed=2)
    assert not np.array_equal(a, b)


def test_procedural_heights_has_expected_shape_and_finite_values():
    h = procedural_heights("alps", size=48, seed=0)
    assert h.shape == (48, 48)
    assert np.all(np.isfinite(h))


def test_alps_is_taller_than_sahara():
    alps = procedural_heights("alps", size=64, seed=3)
    sahara = procedural_heights("sahara", size=64, seed=3)
    assert alps.max() > sahara.max()


def test_water_locations_get_bathymetry_in_the_procedural_fallback():
    """Offline Bora Bora must still have a lagoon.

    Real terrarium tiles encode bathymetry, so the DEM spans deep negatives up
    to dry peaks. Plain noise is non-negative, so without a shift the offline
    island would be entirely dry — losing the splash-crash mechanic on the very
    path CI uses, and no existing test would fail.
    """
    h = procedural_heights("borabora", size=64, seed=3)
    assert h.min() < 0.0, "expected water below sea level"
    assert h.max() > 0.0, "expected dry land above sea level"
    submerged = float((h < 0.0).mean())
    assert 0.3 < submerged < 0.6, f"expected a real lagoon, got {submerged:.0%} wet"


def test_dry_locations_stay_entirely_above_sea_level():
    for key in ("negev", "alps", "cascades", "canyon", "iceland", "sahara"):
        h = procedural_heights(key, size=32, seed=3)
        assert h.min() >= 0.0, f"{key} should have no water"


def test_water_shift_preserves_determinism():
    a = procedural_heights("borabora", size=32, seed=12)
    b = procedural_heights("borabora", size=32, seed=12)
    np.testing.assert_array_equal(a, b)
