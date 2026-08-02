"""Synthetic terrain used when satellite tiles are unavailable.

Value-noise octaves scaled by each location's ``relief``. This is a fallback so
that tests and CI never require network access — it is not trying to imitate the
real DEM.
"""

from __future__ import annotations

import numpy as np

from propwash_gym.world.locations import get_location


def _value_noise(rng: np.random.Generator, size: int, cells: int) -> np.ndarray:
    """Bilinearly-upsampled random lattice, in ``[0, 1]``."""
    lattice = rng.random((cells + 1, cells + 1))
    ys = np.linspace(0.0, cells, size)
    xs = np.linspace(0.0, cells, size)
    y0 = np.floor(ys).astype(int)
    x0 = np.floor(xs).astype(int)
    y1 = np.clip(y0 + 1, 0, cells)
    x1 = np.clip(x0 + 1, 0, cells)
    ty = (ys - y0)[:, None]
    tx = (xs - x0)[None, :]
    # Smoothstep for continuous slopes.
    ty = ty * ty * (3.0 - 2.0 * ty)
    tx = tx * tx * (3.0 - 2.0 * tx)
    top = lattice[np.ix_(y0, x0)] * (1 - tx) + lattice[np.ix_(y0, x1)] * tx
    bot = lattice[np.ix_(y1, x0)] * (1 - tx) + lattice[np.ix_(y1, x1)] * tx
    return top * (1 - ty) + bot * ty


def procedural_heights(location_key: str, size: int = 256, seed: int = 0) -> np.ndarray:
    """Return a ``(size, size)`` synthetic heightfield in metres.

    Args:
        location_key: Which location's relief scale to use.
        size: Grid resolution per side.
        seed: Seed for reproducibility. Same seed gives identical terrain.
    """
    loc = get_location(location_key)
    rng = np.random.default_rng(seed)
    h = np.zeros((size, size), dtype=np.float64)
    amplitude = 1.0
    total = 0.0
    for cells in (2, 4, 8, 16, 32):
        h += amplitude * _value_noise(rng, size, cells)
        total += amplitude
        amplitude *= 0.5
    h /= total
    return h * loc.relief
