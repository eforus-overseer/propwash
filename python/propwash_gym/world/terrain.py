"""Terrain as a metre-indexed heightfield.

One class, one job: given world ``(x, y)`` in metres, return ground height in
metres. Everything downstream — spawn, collision, target placement, GPS, and the
observation's terrain probes — reads terrain through this object, so swapping the
real DEM for procedural terrain changes nothing else.

Coordinates are Z-up: ``x`` east, ``y`` north, ``z`` altitude. The field is
centred on the location's lat/lon, spanning ``extent_m`` metres per side.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field

import numpy as np

from propwash_gym.world.locations import get_location
from propwash_gym.world.procedural import procedural_heights
from propwash_gym.world.tiles import TileCache, decode_terrarium, lat_to_tile, lon_to_tile

#: Zoom level for elevation tiles — terrarium's maximum is 15.
DEM_ZOOM = 15
#: Tiles per side stitched into one field (4x4 = 4.2 km at z15, as in the sim).
DEM_TILES = 4
#: Pixels per tile edge.
TILE_PX = 256
_WARNED_OFFLINE = False


@dataclass
class Heightfield:
    """A square terrain patch sampled in metres.

    Attributes:
        heights: ``(N, N)`` elevations in metres, row 0 at ``y = -extent/2``.
        extent_m: Side length of the patch in metres.
        has_water: Whether a water plane exists.
        source: ``"dem"`` or ``"procedural"``, for diagnostics.
    """

    heights: np.ndarray
    extent_m: float
    has_water: bool = False
    source: str = "procedural"
    water_z: float | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        """Validate the grid and derive the water level from the DEM."""
        self.heights = np.asarray(self.heights, dtype=np.float64)
        if self.heights.ndim != 2 or self.heights.shape[0] != self.heights.shape[1]:
            raise ValueError(f"heights must be square 2-D, got {self.heights.shape}")
        self.extent_m = float(self.extent_m)
        if self.has_water:
            # Terrarium elevations are referenced to sea level, so sea level IS
            # zero -- both for real tiles and for the procedural fallback, which
            # shifts water locations below zero to match. Deriving it from
            # min/max instead lands in the ocean trench (Bora Bora's DEM bottoms
            # out near -6100 m), leaving 0% of the map wet.
            self.water_z = 0.0

    def _to_grid(self, x: float, y: float) -> tuple[float, float]:
        """Map world metres to fractional grid indices, clamped to the patch."""
        n = self.heights.shape[0]
        half = self.extent_m / 2.0
        gx = (x + half) / self.extent_m * (n - 1)
        gy = (y + half) / self.extent_m * (n - 1)
        return (
            float(np.clip(gx, 0.0, n - 1 - 1e-9)),
            float(np.clip(gy, 0.0, n - 1 - 1e-9)),
        )

    def height(self, x: float, y: float) -> float:
        """Return bilinearly-interpolated ground height at world ``(x, y)``."""
        gx, gy = self._to_grid(x, y)
        x0, y0 = int(gx), int(gy)
        x1, y1 = min(x0 + 1, self.heights.shape[0] - 1), min(
            y0 + 1, self.heights.shape[0] - 1
        )
        tx, ty = gx - x0, gy - y0
        top = self.heights[y0, x0] * (1 - tx) + self.heights[y0, x1] * tx
        bot = self.heights[y1, x0] * (1 - tx) + self.heights[y1, x1] * tx
        return float(top * (1 - ty) + bot * ty)

    def slope(self, x: float, y: float, eps: float = 2.0) -> float:
        """Return the local gradient magnitude (rise over run) at ``(x, y)``."""
        dzdx = (self.height(x + eps, y) - self.height(x - eps, y)) / (2 * eps)
        dzdy = (self.height(x, y + eps) - self.height(x, y - eps)) / (2 * eps)
        return float(math.hypot(dzdx, dzdy))

    def probe_ahead(
        self,
        x: float,
        y: float,
        yaw: float,
        distances: tuple[float, ...] = (5.0, 10.0, 20.0, 40.0),
    ) -> np.ndarray:
        """Return ground heights at several distances along the heading.

        Without this lookahead an agent flying real terrain in ``state``
        perception cannot anticipate a ridge.
        """
        cx, sy = math.cos(yaw), math.sin(yaw)
        return np.array(
            [self.height(x + cx * d, y + sy * d) for d in distances],
            dtype=np.float64,
        )


def _dem_extent_metres(lat: float) -> float:
    """Ground width in metres of the stitched DEM patch at this latitude."""
    world_px = TILE_PX * (2**DEM_ZOOM)
    metres_per_px = 40075016.686 * math.cos(math.radians(lat)) / world_px
    return metres_per_px * TILE_PX * DEM_TILES


def build_heightfield(
    location_key: str,
    cache: TileCache | None = None,
    seed: int = 0,
) -> Heightfield:
    """Assemble a heightfield for a location, falling back to procedural terrain.

    Args:
        location_key: One of the keys in ``LOCATIONS``.
        cache: Tile source. A default disk cache is created when omitted.
        seed: Seed for the procedural fallback.
    """
    global _WARNED_OFFLINE
    loc = get_location(location_key)
    cache = cache if cache is not None else TileCache()

    x0 = int(lon_to_tile(loc.lon, DEM_ZOOM)) - DEM_TILES // 2 + 1
    y0 = int(lat_to_tile(loc.lat, DEM_ZOOM)) - DEM_TILES // 2 + 1

    grid = np.zeros((TILE_PX * DEM_TILES, TILE_PX * DEM_TILES), dtype=np.float64)
    missing = 0
    for j in range(DEM_TILES):
        for i in range(DEM_TILES):
            tile = cache.get_terrarium(DEM_ZOOM, x0 + i, y0 + j)
            if tile is None:
                missing += 1
                continue
            ys = slice(j * TILE_PX, (j + 1) * TILE_PX)
            xs = slice(i * TILE_PX, (i + 1) * TILE_PX)
            grid[ys, xs] = decode_terrarium(tile)

    if missing == DEM_TILES * DEM_TILES:
        if not _WARNED_OFFLINE:
            warnings.warn(
                "no elevation tiles available (offline or unreachable); "
                "using procedural terrain",
                RuntimeWarning,
                stacklevel=2,
            )
            _WARNED_OFFLINE = True
        return Heightfield(
            heights=procedural_heights(location_key, size=256, seed=seed),
            extent_m=_dem_extent_metres(loc.lat),
            has_water=loc.has_water,
            source="procedural",
        )

    # Terrarium rows run north-to-south; flip so +y is north.
    return Heightfield(
        heights=np.flipud(grid),
        extent_m=_dem_extent_metres(loc.lat),
        has_water=loc.has_water,
        source="dem",
    )
