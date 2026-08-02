"""Fetch and cache the same map tiles the browser simulator streams.

Two providers, with **opposite coordinate order** in their URLs:

* Esri World Imagery — ``/tile/{z}/{y}/{x}`` (row before column)
* AWS terrarium DEM  — ``/terrarium/{z}/{x}/{y}.png`` (column before row)

Swapping them yields valid-looking imagery of the wrong place, so the order is
pinned by tests.

**The cache directory layout deliberately mirrors each provider's URL order**,
so the same tile is stored at ``terrarium/15/19551/13454.png`` but
``imagery/15/13454/19551.jpg``. Seeing the two indices transposed between
providers on disk looks like a bug and is not — it keeps each path a direct
echo of the URL it came from.

Elevation is decoded with the documented terrarium formula::

    h = R * 256 + G + B / 256 - 32768

Terrarium encodes **bathymetry** below sea level, so oceanic tiles legitimately
return large negative values. Callers must not clamp them to zero.
"""

from __future__ import annotations

import io
import logging
import math
import pathlib
from typing import Callable

import numpy as np
from PIL import Image

_log = logging.getLogger(__name__)

IMAGERY_BASE = (
    "https://server.arcgisonline.com/ArcGIS/rest/services/"
    "World_Imagery/MapServer/tile"
)
TERRARIUM_BASE = "https://s3.amazonaws.com/elevation-tiles-prod/terrarium"
USER_AGENT = "propwash-gym/0.1 (+https://github.com/eforus-overseer)"
DEFAULT_CACHE_ROOT = pathlib.Path.home() / ".cache" / "propwash-gym" / "tiles"

#: Web Mercator is undefined beyond this latitude; inputs are clamped to it.
MAX_MERCATOR_LAT = 85.0511287798066

Fetcher = Callable[[str], bytes]


def lon_to_tile(lon: float, z: int) -> float:
    """Fractional tile column for a longitude at zoom ``z``."""
    return (lon + 180.0) / 360.0 * (2**z)


def lat_to_tile(lat: float, z: int) -> float:
    """Fractional tile row for a latitude at zoom ``z`` (Web Mercator).

    Latitude is clamped to Web Mercator's valid range (±85.0511°). Without the
    clamp the projection fails asymmetrically: ``-90`` raises from ``math.log``
    while ``+90`` silently returns a nonsense index that would reach a URL.
    """
    r = math.radians(max(-MAX_MERCATOR_LAT, min(MAX_MERCATOR_LAT, float(lat))))
    return (1.0 - math.log(math.tan(r) + 1.0 / math.cos(r)) / math.pi) / 2.0 * (2**z)


def imagery_url(z: int, x: int, y: int) -> str:
    """Esri World Imagery tile URL. Note the ``z/y/x`` ordering."""
    return f"{IMAGERY_BASE}/{z}/{y}/{x}"


def terrarium_url(z: int, x: int, y: int) -> str:
    """AWS terrarium DEM tile URL. Note the ``z/x/y`` ordering."""
    return f"{TERRARIUM_BASE}/{z}/{x}/{y}.png"


def decode_terrarium(rgb: np.ndarray) -> np.ndarray:
    """Decode a terrarium RGB tile to metres of elevation."""
    a = np.asarray(rgb, dtype=np.float64)
    return a[..., 0] * 256.0 + a[..., 1] + a[..., 2] / 256.0 - 32768.0


def _http_get(url: str) -> bytes:
    """Fetch ``url`` with a polite User-Agent. Imported lazily to keep tests offline."""
    import requests

    resp = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=30)
    resp.raise_for_status()
    return resp.content


class TileCache:
    """Disk-backed tile store.

    Tiles are content-addressed by ``(provider, z, x, y)``, so the cache never
    needs invalidating. A failed fetch returns ``None`` rather than raising —
    callers degrade to procedural terrain for that patch.

    Args:
        root: Cache directory. Defaults to ``~/.cache/propwash-gym/tiles``.
        fetcher: Injectable byte-fetcher, for tests.
        offline: When True, never touch the network; misses return ``None``.
    """

    def __init__(
        self,
        root: pathlib.Path | str | None = None,
        fetcher: Fetcher | None = None,
        offline: bool = False,
    ) -> None:
        self.root = pathlib.Path(root) if root is not None else DEFAULT_CACHE_ROOT
        self._fetch = fetcher if fetcher is not None else _http_get
        self.offline = bool(offline)

    def _load_or_fetch(
        self, path: pathlib.Path, url: str, fmt: str
    ) -> np.ndarray | None:
        if path.exists():
            try:
                return np.asarray(Image.open(path).convert("RGB"))
            except Exception as exc:   # truncated or malformed cache entry
                _log.debug("discarding unreadable cache entry %s: %r", path, exc)
                path.unlink(missing_ok=True)   # refetch below
        if self.offline:
            return None
        try:
            raw = self._fetch(url)
            if not raw:
                _log.debug("empty response for %s", url)
                return None
            img = Image.open(io.BytesIO(raw)).convert("RGB")
        except Exception as exc:
            # Degrade to procedural terrain rather than raising, but leave a
            # trace: a swallowed URL-builder or decode bug is otherwise
            # indistinguishable from "provider unreachable".
            _log.debug("tile fetch failed for %s: %r", url, exc)
            return None
        path.parent.mkdir(parents=True, exist_ok=True)
        img.save(path, fmt)
        return np.asarray(img)

    def get_terrarium(self, z: int, x: int, y: int) -> np.ndarray | None:
        """Return a terrarium tile as an RGB array, or ``None`` if unavailable."""
        path = self.root / "terrarium" / str(z) / str(x) / f"{y}.png"
        return self._load_or_fetch(path, terrarium_url(z, x, y), "PNG")

    def get_imagery(self, z: int, x: int, y: int) -> np.ndarray | None:
        """Return an imagery tile as an RGB array, or ``None`` if unavailable."""
        path = self.root / "imagery" / str(z) / str(y) / f"{x}.jpg"
        return self._load_or_fetch(path, imagery_url(z, x, y), "JPEG")
