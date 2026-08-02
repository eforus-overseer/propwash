"""Pre-fetch elevation tiles so later training runs are fully offline.

Usage:
    python examples/fetch_tiles.py                 # all seven locations
    python examples/fetch_tiles.py negev canyon    # just these
"""

from __future__ import annotations

import sys

from propwash_gym.world.locations import LOCATIONS
from propwash_gym.world.terrain import build_heightfield
from propwash_gym.world.tiles import TileCache


def main(keys: list[str]) -> int:
    """Build a heightfield per location, populating the cache as a side effect."""
    cache = TileCache()
    print(f"cache: {cache.root}")
    failures = 0
    for key in keys:
        hf = build_heightfield(key, cache=cache)
        status = "ok" if hf.source == "dem" else "FELL BACK to procedural"
        print(
            f"  {key:10s} {status:26s} "
            f"{hf.heights.min():8.1f} .. {hf.heights.max():8.1f} m"
        )
        failures += hf.source != "dem"
    if failures:
        print(f"\n{failures} location(s) could not fetch tiles.")
    return 1 if failures else 0


if __name__ == "__main__":
    requested = sys.argv[1:] or list(LOCATIONS)
    unknown = [k for k in requested if k not in LOCATIONS]
    if unknown:
        print(f"unknown location(s): {unknown}; expected {sorted(LOCATIONS)}")
        raise SystemExit(2)
    raise SystemExit(main(requested))
