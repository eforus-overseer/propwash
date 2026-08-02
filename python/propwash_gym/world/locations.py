"""The seven real-world locations, with coordinates from the browser simulator."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Location:
    """A flyable real-world place.

    Attributes:
        key: Short identifier used in env kwargs.
        name: Display name, matching the browser sim.
        lat: Latitude in degrees.
        lon: Longitude in degrees.
        has_water: Whether a water plane exists (splash-crash on contact).
        relief: Vertical scale hint for the procedural fallback, in metres.
    """

    key: str
    name: str
    lat: float
    lon: float
    has_water: bool
    relief: float


LOCATIONS: dict[str, Location] = {
    "negev": Location("negev", "NEGEV DESERT", 30.6100, 34.8020, False, 60.0),
    "alps": Location("alps", "HIGH ALPS", 46.6244, 8.0414, False, 460.0),
    "cascades": Location("cascades", "CASCADE FOREST", 47.4880, -121.7220, False, 180.0),
    "canyon": Location("canyon", "GRAND CANYON", 36.0997, -112.0964, False, 320.0),
    "iceland": Location("iceland", "ICELAND VOLCANIC", 63.9830, -19.0650, False, 200.0),
    "borabora": Location("borabora", "BORA BORA", -16.5067, -151.7500, True, 220.0),
    "sahara": Location("sahara", "SAHARA ERG", 31.1450, -3.9880, False, 90.0),
}


def get_location(key: str) -> Location:
    """Return the location for ``key``, or raise ``KeyError``."""
    try:
        return LOCATIONS[key]
    except KeyError:
        raise KeyError(
            f"unknown location {key!r}; expected one of {sorted(LOCATIONS)}"
        ) from None
