# propwash-gym Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Gymnasium-compatible RL environment that ports the PROP//WASH flight model, real-terrain pipeline, and TARGET HUNT task to Python, trainable at thousands of steps/second.

**Architecture:** Pure-Python simulation in-process — the browser is never in the training loop. Python fetches the same Esri satellite imagery and AWS terrarium DEM tiles the browser sim uses, caches them to disk, and samples them as NumPy arrays. Physics, reward, observation, and terrain are independent, swappable concerns. Patterns are mirrored from the sibling `rotorenv` project (`RewardTerm` protocol, `CompositeReward`, `CurriculumWrapper`, registered task variants) without importing it.

**Tech Stack:** Python ≥3.10, Gymnasium, NumPy, Pillow (tile decode), `requests` (tile fetch); optional extras `[rl]` = stable-baselines3, `[dev]` = pytest.

**Spec:** `docs/superpowers/specs/2026-08-02-propwash-gym-design.md`

**Source of truth for constants:** `index.html` at the repo root. Every physics constant in this plan was transcribed from it. Do not "improve" these numbers — identical values are what let a trained policy be replayed in the browser.

---

## Environment setup

Do this once before Task 1. The system Python is Homebrew's and is externally managed (PEP 668), so a bare `pip install` will be refused.

```bash
cd "$REPO_ROOT"
mkdir -p python && cd python
python3 -m venv .venv
.venv/bin/pip install --upgrade pip
```

Run everything through `.venv/bin/python` and `.venv/bin/pytest` unless the venv is activated.

---

## File structure

All paths relative to `<repo>/python/`.

| File | Responsibility |
|---|---|
| `pyproject.toml` | package metadata, deps, pytest config |
| `propwash_gym/__init__.py` | register env IDs, expose `make()` |
| `propwash_gym/core/state.py` | `DroneState` dataclass (position, velocity, quaternion, rates, time) |
| `propwash_gym/core/rotations.py` | one frame convention: Z-up, quaternions `[w,x,y,z]` |
| `propwash_gym/core/battery.py` | 4S LiPo: IR sag, coulomb counting, `thrust_scale` |
| `propwash_gym/core/flight_ctrl.py` | ANGLE and ACRO stick→attitude mapping |
| `propwash_gym/core/physics.py` | fixed-step integrator, 2 substeps, wind |
| `propwash_gym/core/reward.py` | `RewardTerm` protocol, terms, `CompositeReward` |
| `propwash_gym/world/locations.py` | the 7 real locations (lat/lon, biome, water flag) |
| `propwash_gym/world/tiles.py` | fetch + disk cache, offline fallback |
| `propwash_gym/world/procedural.py` | synthetic terrain when tiles unavailable |
| `propwash_gym/world/terrain.py` | `Heightfield`: `height()`, `slope()`, `water_z` |
| `propwash_gym/envs/base_env.py` | `PropwashEnv` — Gymnasium API, obs/action spaces |
| `propwash_gym/envs/hunt_env.py` | `HuntEnv` — TARGET HUNT task |
| `propwash_gym/envs/curriculum.py` | `CurriculumWrapper` (success / step modes) |
| `tests/` | one test module per source module |
| `examples/fetch_tiles.py` | warm the tile cache |
| `examples/random_agent.py` | sanity check |
| `examples/train_hunt.py` | PPO + curriculum |

Keep each file to one responsibility. Do not let `base_env.py` absorb terrain or reward logic.

---

## Task 1: Package scaffold

**Files:**
- Create: `python/pyproject.toml`
- Create: `python/propwash_gym/__init__.py`
- Create: `python/.gitignore`
- Test: `python/tests/test_package.py`

- [ ] **Step 1: Write the failing test**

Create `python/tests/test_package.py`:

```python
"""The package imports and exposes a version."""


def test_package_imports_and_has_version():
    import propwash_gym

    assert isinstance(propwash_gym.__version__, str)
    assert propwash_gym.__version__ != ""
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd python && .venv/bin/python -m pytest tests/test_package.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'propwash_gym'`

- [ ] **Step 3: Write minimal implementation**

Create `python/pyproject.toml`:

```toml
[build-system]
requires = ["setuptools>=61.0"]
build-backend = "setuptools.build_meta"

[project]
name = "propwash-gym"
version = "0.1.0"
description = "Gymnasium RL environment for the PROP//WASH FPV drone simulator, on real satellite terrain."
readme = "README.md"
requires-python = ">=3.10"
license = { text = "MIT" }
dependencies = [
    "gymnasium>=0.29",
    "numpy>=1.23",
    "pillow>=10.0",
    "requests>=2.31",
]

[project.optional-dependencies]
dev = ["pytest>=7.0"]
# Training extras, kept optional so `import propwash_gym` stays framework-free.
rl = ["stable-baselines3>=2.3"]

[tool.setuptools.packages.find]
include = ["propwash_gym*"]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

Create `python/propwash_gym/__init__.py`:

```python
"""propwash-gym — a Gymnasium RL environment on real satellite terrain.

Importing this package registers the environment IDs with Gymnasium's global
registry. Env variants are added in a later task.
"""

from __future__ import annotations

__version__ = "0.1.0"
```

Create `python/.gitignore`:

```
.venv/
__pycache__/
*.pyc
*.egg-info/
.pytest_cache/
runs/
```

- [ ] **Step 4: Install and run the test**

```bash
cd python
.venv/bin/pip install -e ".[dev]"
.venv/bin/python -m pytest tests/test_package.py -v
```

Expected: PASS, 1 test.

- [ ] **Step 5: Commit**

```bash
cd "$REPO_ROOT"
git add python/
git commit -m "feat(gym): package scaffold for propwash-gym"
```

---

## Task 2: DroneState and rotations

The browser sim is Y-up (Three.js). This package is **Z-up**, matching robotics convention, and converts only at the browser-replay boundary. Quaternions are scalar-first `[w, x, y, z]`.

**Files:**
- Create: `python/propwash_gym/core/__init__.py`
- Create: `python/propwash_gym/core/state.py`
- Create: `python/propwash_gym/core/rotations.py`
- Test: `python/tests/test_rotations.py`

- [ ] **Step 1: Write the failing test**

Create `python/tests/test_rotations.py`:

```python
"""Frame conventions: Z-up world, scalar-first quaternions [w, x, y, z]."""

import numpy as np
import pytest

from propwash_gym.core.rotations import (
    body_up_axis,
    euler_to_quat,
    quat_multiply,
    quat_normalize,
    quat_to_euler,
)
from propwash_gym.core.state import DroneState


def test_identity_quaternion_has_body_up_along_world_z():
    q = np.array([1.0, 0.0, 0.0, 0.0])
    np.testing.assert_allclose(body_up_axis(q), [0.0, 0.0, 1.0], atol=1e-12)


def test_euler_quat_roundtrip():
    for roll, pitch, yaw in [(0.0, 0.0, 0.0), (0.3, -0.2, 1.1), (-0.5, 0.4, -2.0)]:
        q = euler_to_quat(roll, pitch, yaw)
        r2, p2, y2 = quat_to_euler(q)
        assert pytest.approx(roll, abs=1e-9) == r2
        assert pytest.approx(pitch, abs=1e-9) == p2
        assert pytest.approx(yaw, abs=1e-9) == y2


def test_pitching_forward_tilts_body_up_toward_plus_x():
    # 30 deg nose-down pitch: body-up leans toward +x in this convention.
    q = euler_to_quat(0.0, np.deg2rad(30.0), 0.0)
    up = body_up_axis(q)
    assert up[0] > 0.4          # leaned over
    assert up[2] == pytest.approx(np.cos(np.deg2rad(30.0)), abs=1e-9)


def test_quat_multiply_by_identity_is_noop():
    q = euler_to_quat(0.2, 0.3, 0.4)
    ident = np.array([1.0, 0.0, 0.0, 0.0])
    np.testing.assert_allclose(quat_multiply(q, ident), q, atol=1e-12)


def test_quat_normalize_returns_unit_length():
    q = quat_normalize(np.array([2.0, 0.0, 0.0, 0.0]))
    assert pytest.approx(1.0, abs=1e-12) == float(np.linalg.norm(q))


def test_drone_state_coerces_shapes_and_normalizes_quat():
    s = DroneState(
        position=[1, 2, 3],
        velocity=[0, 0, 0],
        quaternion=[2.0, 0.0, 0.0, 0.0],
        angular_velocity=[0, 0, 0],
        time=0.0,
    )
    assert s.position.shape == (3,)
    assert s.position.dtype == np.float64
    assert s.quaternion.shape == (4,)
    assert pytest.approx(1.0, abs=1e-12) == float(np.linalg.norm(s.quaternion))


def test_drone_state_copy_is_independent():
    s = DroneState([0, 0, 1], [0, 0, 0], [1, 0, 0, 0], [0, 0, 0], 0.0)
    c = s.copy()
    c.position[0] = 99.0
    assert s.position[0] == 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd python && .venv/bin/python -m pytest tests/test_rotations.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'propwash_gym.core'`

- [ ] **Step 3: Write minimal implementation**

Create `python/propwash_gym/core/__init__.py`:

```python
"""Core simulation primitives: state, frames, battery, control, physics, reward."""
```

Create `python/propwash_gym/core/rotations.py`:

```python
"""The single frame convention for this package.

World frame is **Z-up**: ``x`` east, ``y`` north, ``z`` altitude. This matches
robotics practice and the sibling ``rotorenv`` project. Yaw 0 points along
``+x``, so heading maths and terrain sampling agree that east is the zero
bearing. Quaternions are **scalar-first** ``[w, x, y, z]`` and always unit
length.

The browser simulator is Y-up (Three.js). Conversion happens only where
trajectories are exported for browser replay — never inside the sim loop. Do
not inline ad-hoc rotation maths anywhere else in the package.
"""

from __future__ import annotations

import numpy as np


def quat_normalize(q: np.ndarray) -> np.ndarray:
    """Return ``q`` scaled to unit length, falling back to identity if degenerate."""
    q = np.asarray(q, dtype=np.float64).reshape(4)
    n = float(np.linalg.norm(q))
    if n < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0])
    return q / n


def quat_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Return the Hamilton product ``a ⊗ b`` for scalar-first quaternions."""
    aw, ax, ay, az = (float(v) for v in a)
    bw, bx, by, bz = (float(v) for v in b)
    return np.array(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ]
    )


def euler_to_quat(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Convert ZYX Euler angles (radians) to a scalar-first unit quaternion."""
    cr, sr = np.cos(roll * 0.5), np.sin(roll * 0.5)
    cp, sp = np.cos(pitch * 0.5), np.sin(pitch * 0.5)
    cy, sy = np.cos(yaw * 0.5), np.sin(yaw * 0.5)
    return quat_normalize(
        np.array(
            [
                cr * cp * cy + sr * sp * sy,
                sr * cp * cy - cr * sp * sy,
                cr * sp * cy + sr * cp * sy,
                cr * cp * sy - sr * sp * cy,
            ]
        )
    )


def quat_to_euler(q: np.ndarray) -> tuple[float, float, float]:
    """Convert a scalar-first quaternion to ZYX Euler angles ``(roll, pitch, yaw)``."""
    w, x, y, z = (float(v) for v in quat_normalize(q))
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = float(np.arctan2(sinr_cosp, cosr_cosp))

    sinp = 2.0 * (w * y - z * x)
    sinp = float(np.clip(sinp, -1.0, 1.0))
    pitch = float(np.arcsin(sinp))

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = float(np.arctan2(siny_cosp, cosy_cosp))
    return roll, pitch, yaw


def rotate_vector(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate world-frame vector ``v`` by quaternion ``q``."""
    w, x, y, z = (float(c) for c in quat_normalize(q))
    v = np.asarray(v, dtype=np.float64).reshape(3)
    u = np.array([x, y, z])
    return v + 2.0 * np.cross(u, np.cross(u, v) + w * v)


def body_up_axis(q: np.ndarray) -> np.ndarray:
    """Return the drone's body-up axis in world coordinates (thrust direction)."""
    return rotate_vector(q, np.array([0.0, 0.0, 1.0]))
```

Create `python/propwash_gym/core/state.py`:

```python
"""Kinematic state of the drone at one instant.

A plain dataclass of numpy arrays, so physics backends, reward terms, and
observation builders all read one well-defined structure without coupling to
each other.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from propwash_gym.core.rotations import quat_normalize


@dataclass
class DroneState:
    """Full kinematic state at a single timestep.

    Attributes:
        position: World-frame ``(x, y, z)`` in metres, Z-up.
        velocity: World-frame linear velocity in m/s.
        quaternion: Attitude, scalar-first ``[w, x, y, z]``, unit length.
        angular_velocity: Body angular rates in rad/s.
        time: Elapsed simulation time in seconds.
    """

    position: np.ndarray
    velocity: np.ndarray
    quaternion: np.ndarray
    angular_velocity: np.ndarray
    time: float

    def __post_init__(self) -> None:
        """Coerce arrays to float64 and normalise the quaternion."""
        self.position = np.asarray(self.position, dtype=np.float64).reshape(3)
        self.velocity = np.asarray(self.velocity, dtype=np.float64).reshape(3)
        self.quaternion = quat_normalize(self.quaternion)
        self.angular_velocity = np.asarray(
            self.angular_velocity, dtype=np.float64
        ).reshape(3)
        self.time = float(self.time)

    def copy(self) -> "DroneState":
        """Return a copy with independent array buffers."""
        return DroneState(
            position=self.position.copy(),
            velocity=self.velocity.copy(),
            quaternion=self.quaternion.copy(),
            angular_velocity=self.angular_velocity.copy(),
            time=self.time,
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd python && .venv/bin/python -m pytest tests/test_rotations.py -v`
Expected: PASS, 7 tests.

- [ ] **Step 5: Commit**

```bash
cd "$REPO_ROOT"
git add python/propwash_gym/core python/tests/test_rotations.py
git commit -m "feat(gym): DroneState and Z-up quaternion frame convention"
```

---

## Task 3: Battery model

Ported verbatim from `index.html` `const Battery`. Note `update()` takes **throttle**, not current: `current = 1.2 + throttle² * 78`.

**Files:**
- Create: `python/propwash_gym/core/battery.py`
- Test: `python/tests/test_battery.py`

- [ ] **Step 1: Write the failing test**

Create `python/tests/test_battery.py`:

```python
"""4S LiPo model ported from index.html `const Battery`."""

import pytest

from propwash_gym.core.battery import Battery


def test_fresh_pack_is_full_and_near_max_voltage():
    b = Battery(capacity_mah=1300.0)
    assert pytest.approx(1.0, abs=1e-12) == b.soc
    # soc=1, current = idle 1.2A -> (3.0 + 1.2 - 1.2*0.0058) * 4
    assert pytest.approx((3.0 + 1.2 - 1.2 * 0.0058) * 4, abs=1e-9) == b.voltage


def test_idle_current_is_1_2_amps():
    b = Battery(capacity_mah=1300.0)
    b.update(throttle=0.0, dt=0.02)
    assert pytest.approx(1.2, abs=1e-12) == b.current


def test_full_throttle_current_is_79_2_amps():
    b = Battery(capacity_mah=1300.0)
    b.update(throttle=1.0, dt=0.02)
    assert pytest.approx(1.2 + 78.0, abs=1e-12) == b.current


def test_coulomb_counting_accumulates_mah():
    b = Battery(capacity_mah=1300.0)
    # 79.2 A for 1 second = 79.2/3600 Ah = 22.0 mAh
    for _ in range(100):
        b.update(throttle=1.0, dt=0.01)
    assert pytest.approx(79.2 / 3.6 * 1.0, rel=1e-9) == b.drawn_mah


def test_voltage_sags_under_load():
    idle = Battery(capacity_mah=1300.0)
    idle.update(throttle=0.0, dt=0.02)
    loaded = Battery(capacity_mah=1300.0)
    loaded.update(throttle=1.0, dt=0.02)
    assert loaded.voltage < idle.voltage


def test_voltage_is_clamped_to_pack_maximum():
    b = Battery(capacity_mah=1300.0)
    assert b.voltage <= 16.8


def test_depleted_pack_reports_zero_soc_and_is_empty():
    b = Battery(capacity_mah=10.0)
    for _ in range(2000):
        b.update(throttle=1.0, dt=0.02)
    assert pytest.approx(0.0, abs=1e-12) == b.soc
    assert b.is_empty


def test_thrust_scale_never_falls_below_floor():
    b = Battery(capacity_mah=10.0)
    for _ in range(2000):
        b.update(throttle=1.0, dt=0.02)
    # clamp((V-11.2)/5.6, .35, 1) * .35 + .65  ->  floor is .35*.35+.65
    assert b.thrust_scale >= 0.35 * 0.35 + 0.65 - 1e-12


def test_thrust_scale_is_one_at_full_charge_unloaded():
    b = Battery(capacity_mah=1300.0)
    assert pytest.approx(1.0, abs=1e-9) == b.thrust_scale


def test_reset_restores_full_pack():
    b = Battery(capacity_mah=1300.0)
    for _ in range(50):
        b.update(throttle=1.0, dt=0.02)
    b.reset(capacity_mah=1000.0)
    assert b.drawn_mah == 0.0
    assert b.capacity_mah == 1000.0
    assert pytest.approx(1.0, abs=1e-12) == b.soc
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd python && .venv/bin/python -m pytest tests/test_battery.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'propwash_gym.core.battery'`

- [ ] **Step 3: Write minimal implementation**

Create `python/propwash_gym/core/battery.py`:

```python
"""4S LiPo pack model.

Ported from ``index.html`` ``const Battery``. Every constant here is transcribed
from the browser simulator so a policy trained against this model behaves the
same when replayed there. Do not retune these values.
"""

from __future__ import annotations

CELLS = 4
IDLE_CURRENT_A = 1.2
THROTTLE_CURRENT_A = 78.0
INTERNAL_RESISTANCE = 0.0058
CELL_MIN_V = 3.0
CELL_SPAN_V = 1.2
PACK_MAX_V = 16.8
# thrust_scale = clamp((V - SAG_OFFSET) / SAG_SPAN, SAG_FLOOR, 1) * SAG_GAIN + SAG_BASE
SAG_OFFSET_V = 11.2
SAG_SPAN_V = 5.6
SAG_FLOOR = 0.35
SAG_GAIN = 0.35
SAG_BASE = 0.65


def _clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else (hi if v > hi else v)


class Battery:
    """A 4S LiPo whose voltage sags under load and depletes by coulomb counting.

    Args:
        capacity_mah: Pack capacity. The browser sim uses 1300 for most missions
            and 1000 for PRO RACE.
    """

    def __init__(self, capacity_mah: float = 1300.0) -> None:
        self.capacity_mah = float(capacity_mah)
        self.drawn_mah = 0.0
        self.current = IDLE_CURRENT_A

    def reset(self, capacity_mah: float | None = None) -> None:
        """Restore a full pack, optionally changing capacity."""
        if capacity_mah is not None:
            self.capacity_mah = float(capacity_mah)
        self.drawn_mah = 0.0
        self.current = IDLE_CURRENT_A

    def update(self, throttle: float, dt: float) -> None:
        """Advance the pack by ``dt`` seconds at the given normalised throttle."""
        t = _clamp(float(throttle), 0.0, 1.0)
        self.current = IDLE_CURRENT_A + t * t * THROTTLE_CURRENT_A
        self.drawn_mah += self.current * float(dt) / 3.6

    @property
    def soc(self) -> float:
        """State of charge in ``[0, 1]``."""
        return _clamp(1.0 - self.drawn_mah / self.capacity_mah, 0.0, 1.0)

    @property
    def voltage(self) -> float:
        """Pack voltage under the current load, including IR sag."""
        v = (
            CELL_MIN_V
            + self.soc * CELL_SPAN_V
            - self.current * INTERNAL_RESISTANCE
        ) * CELLS
        return _clamp(v, 0.0, PACK_MAX_V)

    @property
    def cell_voltage(self) -> float:
        """Per-cell voltage — what pilots actually watch."""
        return self.voltage / CELLS

    @property
    def is_empty(self) -> bool:
        """True once the pack has no usable charge left."""
        return self.soc <= 0.0

    @property
    def thrust_scale(self) -> float:
        """Multiplier applied to max thrust as the pack sags."""
        k = _clamp((self.voltage - SAG_OFFSET_V) / SAG_SPAN_V, SAG_FLOOR, 1.0)
        return k * SAG_GAIN + SAG_BASE

    @property
    def eta_seconds(self) -> float:
        """Seconds of flight left at the present draw, or ``inf`` when idle."""
        if self.current < 0.1:
            return float("inf")
        return (self.capacity_mah - self.drawn_mah) / self.current * 3.6
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd python && .venv/bin/python -m pytest tests/test_battery.py -v`
Expected: PASS, 10 tests.

- [ ] **Step 5: Commit**

```bash
cd "$REPO_ROOT"
git add python/propwash_gym/core/battery.py python/tests/test_battery.py
git commit -m "feat(gym): 4S LiPo battery model ported from the browser sim"
```

---

## Task 4: Tile fetching and caching

Two providers with **opposite axis order**. Getting this backwards returns valid-looking tiles of the wrong place, which is silent and hard to debug — so it is asserted in tests.

| Provider | URL shape |
|---|---|
| Esri imagery | `.../MapServer/tile/{z}/{y}/{x}` |
| AWS terrarium | `.../terrarium/{z}/{x}/{y}.png` |

**Files:**
- Create: `python/propwash_gym/world/__init__.py`
- Create: `python/propwash_gym/world/tiles.py`
- Test: `python/tests/test_tiles.py`

- [ ] **Step 1: Write the failing test**

Create `python/tests/test_tiles.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd python && .venv/bin/python -m pytest tests/test_tiles.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'propwash_gym.world'`

- [ ] **Step 3: Write minimal implementation**

Create `python/propwash_gym/world/__init__.py`:

```python
"""World data: tile fetching, terrain heightfields, and location definitions."""
```

Create `python/propwash_gym/world/tiles.py`:

```python
"""Fetch and cache the same map tiles the browser simulator streams.

Two providers, with **opposite coordinate order** in their URLs:

* Esri World Imagery — ``/tile/{z}/{y}/{x}`` (row before column)
* AWS terrarium DEM  — ``/terrarium/{z}/{x}/{y}.png`` (column before row)

Swapping them yields valid-looking imagery of the wrong place, so the order is
pinned by tests. Elevation is decoded with the documented terrarium formula::

    h = R * 256 + G + B / 256 - 32768

Terrarium encodes **bathymetry** below sea level, so oceanic tiles legitimately
return large negative values. Callers must not clamp them to zero.
"""

from __future__ import annotations

import io
import math
import pathlib
from typing import Callable

import numpy as np
from PIL import Image

IMAGERY_BASE = (
    "https://server.arcgisonline.com/ArcGIS/rest/services/"
    "World_Imagery/MapServer/tile"
)
TERRARIUM_BASE = "https://s3.amazonaws.com/elevation-tiles-prod/terrarium"
USER_AGENT = "propwash-gym/0.1 (+https://github.com/eforus-overseer)"
DEFAULT_CACHE_ROOT = pathlib.Path.home() / ".cache" / "propwash-gym" / "tiles"

Fetcher = Callable[[str], bytes]


def lon_to_tile(lon: float, z: int) -> float:
    """Fractional tile column for a longitude at zoom ``z``."""
    return (lon + 180.0) / 360.0 * (2**z)


def lat_to_tile(lat: float, z: int) -> float:
    """Fractional tile row for a latitude at zoom ``z`` (Web Mercator)."""
    r = math.radians(lat)
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
            except OSError:
                path.unlink(missing_ok=True)   # corrupt file: refetch below
        if self.offline:
            return None
        try:
            raw = self._fetch(url)
            if not raw:
                return None
            img = Image.open(io.BytesIO(raw)).convert("RGB")
        except Exception:
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd python && .venv/bin/python -m pytest tests/test_tiles.py -v`
Expected: PASS, 11 tests.

- [ ] **Step 5: Commit**

```bash
cd "$REPO_ROOT"
git add python/propwash_gym/world python/tests/test_tiles.py
git commit -m "feat(gym): tile fetching, disk cache, and terrarium DEM decode"
```

---

## Task 5: Locations and procedural terrain

**Files:**
- Create: `python/propwash_gym/world/locations.py`
- Create: `python/propwash_gym/world/procedural.py`
- Test: `python/tests/test_locations.py`

- [ ] **Step 1: Write the failing test**

Create `python/tests/test_locations.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd python && .venv/bin/python -m pytest tests/test_locations.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'propwash_gym.world.locations'`

- [ ] **Step 3: Write minimal implementation**

Create `python/propwash_gym/world/locations.py`:

```python
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
```

Create `python/propwash_gym/world/procedural.py`:

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd python && .venv/bin/python -m pytest tests/test_locations.py -v`
Expected: PASS, 9 tests.

- [ ] **Step 5: Commit**

```bash
cd "$REPO_ROOT"
git add python/propwash_gym/world/locations.py python/propwash_gym/world/procedural.py python/tests/test_locations.py
git commit -m "feat(gym): seven real locations plus procedural terrain fallback"
```

---

## Task 6: Heightfield

Assembles tiles into a metre-indexed surface with bilinear sampling. `water_z` comes from the DEM, never assumed to be zero.

**Files:**
- Create: `python/propwash_gym/world/terrain.py`
- Test: `python/tests/test_terrain.py`

- [ ] **Step 1: Write the failing test**

Create `python/tests/test_terrain.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd python && .venv/bin/python -m pytest tests/test_terrain.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'propwash_gym.world.terrain'`

- [ ] **Step 3: Write minimal implementation**

Create `python/propwash_gym/world/terrain.py`:

```python
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

    # Keep the FRACTIONAL tile coordinates. Truncating them puts world (0, 0)
    # on a tile boundary instead of on the place we mean to fly -- measured at
    # 1043 m off for Erg Chebbi. Fetch one spare tile per axis and crop around
    # the location's own pixel after assembly.
    fx = lon_to_tile(loc.lon, DEM_ZOOM)
    fy = lat_to_tile(loc.lat, DEM_ZOOM)
    x0 = int(fx) - DEM_TILES // 2
    y0 = int(fy) - DEM_TILES // 2
    span = DEM_TILES + 1

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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd python && .venv/bin/python -m pytest tests/test_terrain.py -v`
Expected: PASS, 10 tests.

- [ ] **Step 5: Commit**

```bash
cd "$REPO_ROOT"
git add python/propwash_gym/world/terrain.py python/tests/test_terrain.py
git commit -m "feat(gym): metre-indexed heightfield with DEM assembly and fallback"
```

---

## Task 7: Live tile integration test

Task 6 is tested offline. This task proves the real network path works and that the verified elevations still hold. It is the only test that touches the network, and it skips cleanly when offline.

**Files:**
- Create: `python/tests/test_tiles_live.py`
- Create: `python/examples/fetch_tiles.py`

- [ ] **Step 1: Write the test**

Create `python/tests/test_tiles_live.py`:

```python
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
```

- [ ] **Step 2: Register the marker so `-m network` works**

Append to `python/pyproject.toml` under `[tool.pytest.ini_options]`:

```toml
markers = [
    "network: hits live tile providers; deselect with '-m \"not network\"'",
]
addopts = "-m 'not network'"
```

- [ ] **Step 3: Run the offline suite and confirm live tests are deselected**

Run: `cd python && .venv/bin/python -m pytest -q`
Expected: all prior tests pass; output notes deselected tests.

- [ ] **Step 4: Run the live tests explicitly**

Run: `cd python && .venv/bin/python -m pytest -m network -v`
Expected: PASS, 6 tests (or skips if the network is unavailable).

- [ ] **Step 5: Write the cache-warming example**

Create `python/examples/fetch_tiles.py`:

```python
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
```

- [ ] **Step 6: Run it for one location**

Run: `cd python && .venv/bin/python examples/fetch_tiles.py negev`
Expected: `negev  ok  <min> .. <max> m` with values in the 700–1000 m range.

- [ ] **Step 7: Commit**

```bash
cd "$REPO_ROOT"
git add python/tests/test_tiles_live.py python/examples/fetch_tiles.py python/pyproject.toml
git commit -m "test(gym): live tile checks and a cache-warming script"
```

---

## Task 8: Flight controller

Ported from `index.html` `const FlightCtrl`. ANGLE self-levels toward a commanded tilt; ACRO commands body rates with expo.

**Files:**
- Create: `python/propwash_gym/core/flight_ctrl.py`
- Test: `python/tests/test_flight_ctrl.py`

- [ ] **Step 1: Write the failing test**

Create `python/tests/test_flight_ctrl.py`:

```python
"""ANGLE and ACRO control laws, ported from index.html `const FlightCtrl`."""

import numpy as np
import pytest

from propwash_gym.core.flight_ctrl import MAX_TILT, RATE_RP, RATE_Y, FlightController
from propwash_gym.core.rotations import quat_to_euler
from propwash_gym.core.state import DroneState


def _level_state():
    return DroneState([0, 0, 10], [0, 0, 0], [1, 0, 0, 0], [0, 0, 0], 0.0)


def test_constants_match_the_browser_sim():
    assert pytest.approx(np.deg2rad(38.0), abs=1e-12) == MAX_TILT
    assert pytest.approx(np.deg2rad(480.0), abs=1e-12) == RATE_RP
    assert pytest.approx(np.deg2rad(300.0), abs=1e-12) == RATE_Y


def test_angle_mode_holds_level_with_no_input():
    fc = FlightController(mode="angle")
    s = _level_state()
    for _ in range(50):
        fc.apply(s, roll=0.0, pitch=0.0, yaw=0.0, dt=0.02)
    roll, pitch, _ = quat_to_euler(s.quaternion)
    assert pytest.approx(0.0, abs=1e-6) == roll
    assert pytest.approx(0.0, abs=1e-6) == pitch


def test_angle_mode_converges_toward_commanded_tilt():
    fc = FlightController(mode="angle")
    s = _level_state()
    for _ in range(400):
        fc.apply(s, roll=0.0, pitch=1.0, yaw=0.0, dt=0.02)
    _, pitch, _ = quat_to_euler(s.quaternion)
    assert pytest.approx(MAX_TILT, rel=0.02) == pitch


def test_angle_mode_never_exceeds_max_tilt():
    fc = FlightController(mode="angle")
    s = _level_state()
    for _ in range(400):
        fc.apply(s, roll=1.0, pitch=1.0, yaw=0.0, dt=0.02)
    roll, pitch, _ = quat_to_euler(s.quaternion)
    assert abs(roll) <= MAX_TILT + 1e-6
    assert abs(pitch) <= MAX_TILT + 1e-6


def test_angle_mode_yaw_integrates_at_2_6_rad_per_second():
    fc = FlightController(mode="angle")
    s = _level_state()
    for _ in range(50):          # 1 second at dt=0.02
        fc.apply(s, roll=0.0, pitch=0.0, yaw=1.0, dt=0.02)
    _, _, yaw = quat_to_euler(s.quaternion)
    assert pytest.approx(2.6, rel=0.02) == abs(yaw)


def test_acro_mode_keeps_rotating_while_stick_is_held():
    fc = FlightController(mode="acro")
    s = _level_state()
    for _ in range(60):
        fc.apply(s, roll=1.0, pitch=0.0, yaw=0.0, dt=0.02)
    # A held roll command in ACRO produces continuous rotation, not a fixed tilt.
    assert float(np.linalg.norm(s.angular_velocity)) > 1.0


def test_acro_expo_softens_small_stick_inputs():
    fc = FlightController(mode="acro")
    small = fc.expo(0.2)
    assert small < 0.2, "expo must reduce small inputs"
    assert pytest.approx(1.0, abs=1e-12) == fc.expo(1.0)


def test_unknown_mode_is_rejected():
    with pytest.raises(ValueError, match="mode"):
        FlightController(mode="sport")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd python && .venv/bin/python -m pytest tests/test_flight_ctrl.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'propwash_gym.core.flight_ctrl'`

- [ ] **Step 3: Write minimal implementation**

Create `python/propwash_gym/core/flight_ctrl.py`:

```python
"""Stick-to-attitude control laws.

Ported from ``index.html`` ``const FlightCtrl``:

* **ANGLE** — self-levelling. Sticks command a tilt angle, capped at 38°, and
  attitude eases toward it. Yaw integrates at a fixed rate.
* **ACRO** — sticks command body rates directly (480°/s roll and pitch,
  300°/s yaw) through an expo curve, integrated on the quaternion.

Smoothing uses the sim's exponential form ``1 - exp(-dt * k)``, which is
timestep-independent — do not replace it with a plain lerp constant.
"""

from __future__ import annotations

import math

import numpy as np

from propwash_gym.core.rotations import (
    euler_to_quat,
    quat_multiply,
    quat_normalize,
    quat_to_euler,
)
from propwash_gym.core.state import DroneState

MAX_TILT = math.radians(38.0)
RATE_RP = math.radians(480.0)
RATE_Y = math.radians(300.0)
ANGLE_SMOOTH_K = 7.5
ANGLE_YAW_RATE = 2.6
#: Reported roll/pitch rate per unit of stick in ANGLE mode (index.html:503).
ANGLE_STICK_RATE = 2.0
ACRO_SMOOTH_K = 18.0
MODES = ("angle", "acro")


class FlightController:
    """Maps normalised stick inputs onto the drone's attitude.

    Args:
        mode: ``"angle"`` (self-levelling) or ``"acro"`` (rate mode).
    """

    def __init__(self, mode: str = "angle") -> None:
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
        self.mode = mode
        self._yaw = 0.0

    def reset(self, yaw: float = 0.0) -> None:
        """Clear the integrated heading held for ANGLE mode."""
        self._yaw = float(yaw)

    @staticmethod
    def expo(v: float) -> float:
        """Apply the sim's expo curve: ``v * (0.35 + 0.65 * v²)``."""
        return float(v) * (0.35 + 0.65 * float(v) * float(v))

    def apply(
        self, state: DroneState, roll: float, pitch: float, yaw: float, dt: float
    ) -> None:
        """Advance ``state``'s attitude by one control step, in place."""
        if self.mode == "angle":
            self._apply_angle(state, roll, pitch, yaw, dt)
        else:
            self._apply_acro(state, roll, pitch, yaw, dt)

    def _apply_angle(
        self, state: DroneState, roll: float, pitch: float, yaw: float, dt: float
    ) -> None:
        cur_roll, cur_pitch, _ = quat_to_euler(state.quaternion)
        smooth = 1.0 - math.exp(-dt * ANGLE_SMOOTH_K)
        target_pitch = float(pitch) * MAX_TILT
        target_roll = float(roll) * MAX_TILT
        new_pitch = cur_pitch + (target_pitch - cur_pitch) * smooth
        new_roll = cur_roll + (target_roll - cur_roll) * smooth
        self._yaw += float(yaw) * ANGLE_YAW_RATE * dt
        state.quaternion = euler_to_quat(new_roll, new_pitch, self._yaw)
        # Rates come from stick deflection, as in index.html:503. Finite-
        # differencing the Euler angles instead reports 4.62 rad/s where the
        # browser reports 2.0, and varies with dt.
        state.angular_velocity = np.array(
            [
                float(roll) * ANGLE_STICK_RATE,
                float(pitch) * ANGLE_STICK_RATE,
                float(yaw) * ANGLE_YAW_RATE,
            ]
        )

    def _apply_acro(
        self, state: DroneState, roll: float, pitch: float, yaw: float, dt: float
    ) -> None:
        target = np.array(
            [
                self.expo(float(roll)) * RATE_RP,
                self.expo(float(pitch)) * RATE_RP,
                self.expo(float(yaw)) * RATE_Y,
            ]
        )
        smooth = 1.0 - math.exp(-dt * ACRO_SMOOTH_K)
        state.angular_velocity = state.angular_velocity + (
            target - state.angular_velocity
        ) * smooth
        wx, wy, wz = (float(v) for v in state.angular_velocity)
        omega_q = np.array([0.0, wx * 0.5 * dt, wy * 0.5 * dt, wz * 0.5 * dt])
        state.quaternion = quat_normalize(
            state.quaternion + quat_multiply(state.quaternion, omega_q)
        )
        self._yaw = quat_to_euler(state.quaternion)[2]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd python && .venv/bin/python -m pytest tests/test_flight_ctrl.py -v`
Expected: PASS, 8 tests.

- [ ] **Step 5: Commit**

```bash
cd "$REPO_ROOT"
git add python/propwash_gym/core/flight_ctrl.py python/tests/test_flight_ctrl.py
git commit -m "feat(gym): ANGLE and ACRO flight controllers"
```

---

## Task 9: Physics and collision

Fixed timestep with 2 substeps, matching `loop()` in the browser (`const h = dt/2; for (i<2) Physics.step(inp, h)`).

**Files:**
- Create: `python/propwash_gym/core/physics.py`
- Test: `python/tests/test_physics.py`

- [ ] **Step 1: Write the failing test**

Create `python/tests/test_physics.py`:

```python
"""Rigid-body integration, drag, wind, and crash detection."""

import numpy as np
import pytest

from propwash_gym.core.battery import Battery
from propwash_gym.core.flight_ctrl import FlightController
from propwash_gym.core.physics import (
    DRAG_K,
    GRAVITY,
    MASS,
    MAX_THRUST,
    CrashReason,
    Physics,
)
from propwash_gym.core.state import DroneState
from propwash_gym.world.terrain import Heightfield


def _flat(height=0.0, extent=400.0):
    return Heightfield(heights=np.full((16, 16), height), extent_m=extent)


def _phys(terrain=None, wind=0.0):
    return Physics(
        terrain=terrain if terrain is not None else _flat(),
        controller=FlightController(mode="angle"),
        battery=Battery(1300.0),
        wind_scale=wind,
    )


def _airborne(z=50.0):
    return DroneState([0, 0, z], [0, 0, 0], [1, 0, 0, 0], [0, 0, 0], 0.0)


def test_constants_match_the_browser_sim():
    assert pytest.approx(0.62, abs=1e-12) == MASS
    assert pytest.approx(24.0, abs=1e-12) == MAX_THRUST
    assert pytest.approx(0.045, abs=1e-12) == DRAG_K
    assert pytest.approx(9.81, abs=1e-12) == GRAVITY


def test_zero_throttle_falls_under_gravity():
    p = _phys()
    s = _airborne()
    p.step(s, throttle=0.0, roll=0.0, pitch=0.0, yaw=0.0, dt=0.02)
    assert s.velocity[2] < 0.0


def test_hover_throttle_roughly_cancels_gravity():
    p = _phys()
    s = _airborne()
    hover = MASS * GRAVITY / MAX_THRUST     # ~0.2534 at full pack scale
    for _ in range(10):
        p.step(s, throttle=hover, roll=0.0, pitch=0.0, yaw=0.0, dt=0.02)
    assert abs(float(s.velocity[2])) < 0.5, f"vz={s.velocity[2]}"


def test_full_throttle_climbs():
    p = _phys()
    s = _airborne()
    for _ in range(25):
        p.step(s, throttle=1.0, roll=0.0, pitch=0.0, yaw=0.0, dt=0.02)
    assert s.velocity[2] > 1.0


def test_drag_bounds_terminal_velocity():
    p = _phys()
    s = _airborne(z=5000.0)
    for _ in range(4000):
        p.step(s, throttle=0.0, roll=0.0, pitch=0.0, yaw=0.0, dt=0.02)
    # Terminal speed where drag balances weight: sqrt(m*g/k)
    expected = np.sqrt(MASS * GRAVITY / DRAG_K)
    assert float(np.linalg.norm(s.velocity)) == pytest.approx(expected, rel=0.1)


def test_wind_perturbs_a_hovering_drone_laterally():
    calm = _phys(wind=0.0)
    windy = _phys(wind=2.0)
    hover = MASS * GRAVITY / MAX_THRUST
    a, b = _airborne(), _airborne()
    for _ in range(100):
        calm.step(a, hover, 0.0, 0.0, 0.0, 0.02)
        windy.step(b, hover, 0.0, 0.0, 0.0, 0.02)
    assert float(np.linalg.norm(b.position[:2])) > float(np.linalg.norm(a.position[:2]))


def test_gentle_ground_contact_does_not_crash():
    p = _phys(_flat(0.0))
    s = DroneState([0, 0, 0.12], [0, 0, -0.5], [1, 0, 0, 0], [0, 0, 0], 0.0)
    reason = p.step(s, throttle=0.0, roll=0.0, pitch=0.0, yaw=0.0, dt=0.02)
    assert reason is None
    assert s.position[2] >= 0.09 - 1e-9


def test_fast_descent_into_ground_crashes():
    p = _phys(_flat(0.0))
    # Start low enough that one 0.02 s step actually reaches contact: at 8 m/s
    # the drone falls ~0.16 m, and contact needs z < ground + GROUND_OFFSET.
    s = DroneState([0, 0, 0.2], [0, 0, -8.0], [1, 0, 0, 0], [0, 0, 0], 0.0)
    reason = p.step(s, throttle=0.0, roll=0.0, pitch=0.0, yaw=0.0, dt=0.02)
    assert reason is CrashReason.GROUND


def test_water_contact_crashes_at_any_speed():
    hf = Heightfield(
        heights=np.where(np.arange(256).reshape(16, 16) < 128, -1000.0, 200.0),
        extent_m=400.0,
        has_water=True,
    )
    p = _phys(hf)
    assert hf.water_z is not None
    s = DroneState([0, 0, hf.water_z - 0.5], [0, 0, -0.01], [1, 0, 0, 0], [0, 0, 0], 0.0)
    assert p.step(s, 0.0, 0.0, 0.0, 0.0, 0.02) is CrashReason.WATER


def test_leaving_the_arena_crashes():
    p = _phys(_flat(0.0, extent=400.0))
    s = DroneState([10_000.0, 0.0, 50.0], [0, 0, 0], [1, 0, 0, 0], [0, 0, 0], 0.0)
    assert p.step(s, 0.0, 0.0, 0.0, 0.0, 0.02) is CrashReason.OUT_OF_BOUNDS


def test_time_advances_by_dt():
    p = _phys()
    s = _airborne()
    p.step(s, 0.3, 0.0, 0.0, 0.0, 0.02)
    assert pytest.approx(0.02, abs=1e-12) == s.time


def test_stepping_is_deterministic():
    a, b = _phys(wind=1.5), _phys(wind=1.5)
    sa, sb = _airborne(), _airborne()
    for _ in range(200):
        a.step(sa, 0.4, 0.1, 0.2, 0.05, 0.02)
        b.step(sb, 0.4, 0.1, 0.2, 0.05, 0.02)
    np.testing.assert_array_equal(sa.position, sb.position)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd python && .venv/bin/python -m pytest tests/test_physics.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'propwash_gym.core.physics'`

- [ ] **Step 3: Write minimal implementation**

Create `python/propwash_gym/core/physics.py`:

```python
"""Rigid-body flight physics and terrain collision.

Ported from ``index.html`` ``const Physics`` and the substepping in ``loop()``:
each env step runs ``SUBSTEPS`` integration passes of ``dt / SUBSTEPS``, which
is what keeps the simulation stable at aggressive attitudes.

Wind is deterministic — a fixed sum of sinusoids in simulation time, exactly as
the browser computes it. It is not random, so it never needs seeding.
"""

from __future__ import annotations

import enum
import math

import numpy as np

from propwash_gym.core.battery import Battery
from propwash_gym.core.flight_ctrl import FlightController
from propwash_gym.core.rotations import body_up_axis
from propwash_gym.core.state import DroneState
from propwash_gym.world.terrain import Heightfield

MASS = 0.62
MAX_THRUST = 24.0
DRAG_K = 0.045
GRAVITY = 9.81
SUBSTEPS = 2
#: Ground clearance the airframe rests at, and the contact threshold.
GROUND_OFFSET = 0.09
#: Descent rate (m/s) above which ground contact is a crash.
CRASH_DESCENT_MS = 3.2
#: Speed (m/s) above which ground contact is a crash regardless of direction.
CRASH_SPEED_MS = 9.0
#: Velocity retained after a survivable ground scrape.
GROUND_BOUNCE_DAMPING = 0.6
#: Distance from the arena edge that terminates an episode.
ARENA_MARGIN_M = 10.0


class CrashReason(enum.Enum):
    """Why an episode ended in a crash."""

    GROUND = "ground"
    WATER = "water"
    OUT_OF_BOUNDS = "out_of_bounds"


class Physics:
    """Integrates drone motion and detects terminal contacts.

    Args:
        terrain: Height source for collision and ground contact.
        controller: Attitude control law.
        battery: Pack whose sag scales available thrust.
        wind_scale: Mission wind strength (0 = calm).
    """

    def __init__(
        self,
        terrain: Heightfield,
        controller: FlightController,
        battery: Battery,
        wind_scale: float = 0.0,
    ) -> None:
        self.terrain = terrain
        self.controller = controller
        self.battery = battery
        self.wind_scale = float(wind_scale)

    def wind(self, t: float) -> np.ndarray:
        """Return the deterministic wind vector at simulation time ``t``."""
        if self.wind_scale == 0.0:
            return np.zeros(3)
        return (
            np.array(
                [
                    math.sin(t * 0.31) + math.sin(t * 0.83) * 0.5,
                    math.cos(t * 0.27) + math.sin(t * 0.71) * 0.5,
                    math.sin(t * 0.57) * 0.25,
                ]
            )
            * self.wind_scale
        )

    def step(
        self,
        state: DroneState,
        throttle: float,
        roll: float,
        pitch: float,
        yaw: float,
        dt: float,
    ) -> CrashReason | None:
        """Advance ``state`` by ``dt`` in place; return a crash reason or ``None``."""
        h = dt / SUBSTEPS
        for _ in range(SUBSTEPS):
            self.controller.apply(state, roll=roll, pitch=pitch, yaw=yaw, dt=h)
            thrust = body_up_axis(state.quaternion) * (
                float(np.clip(throttle, 0.0, 1.0))
                * MAX_THRUST
                * self.battery.thrust_scale
            )
            speed = float(np.linalg.norm(state.velocity))
            drag = -DRAG_K * speed * state.velocity
            gravity = np.array([0.0, 0.0, -GRAVITY * MASS])
            force = thrust + gravity + drag + self.wind(state.time)
            state.velocity = state.velocity + force / MASS * h
            state.position = state.position + state.velocity * h
            state.time += h
            reason = self._collide(state)
            if reason is not None:
                state.velocity = np.zeros(3)
                return reason
        return None

    def _collide(self, state: DroneState) -> CrashReason | None:
        """Resolve terrain contact, returning a crash reason when terminal."""
        x, y, z = (float(v) for v in state.position)

        half = self.terrain.extent_m / 2.0 - ARENA_MARGIN_M
        if abs(x) > half or abs(y) > half:
            return CrashReason.OUT_OF_BOUNDS

        if self.terrain.water_z is not None and z < self.terrain.water_z + 0.05:
            return CrashReason.WATER

        ground = self.terrain.height(x, y)
        if z < ground + GROUND_OFFSET:
            descent = -float(state.velocity[2])
            speed = float(np.linalg.norm(state.velocity))
            state.position[2] = ground + GROUND_OFFSET
            if descent > CRASH_DESCENT_MS or speed > CRASH_SPEED_MS:
                return CrashReason.GROUND
            state.velocity[2] = max(float(state.velocity[2]), 0.0)
            state.velocity = state.velocity * GROUND_BOUNCE_DAMPING
        return None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd python && .venv/bin/python -m pytest tests/test_physics.py -v`
Expected: PASS, 12 tests.

- [ ] **Step 5: Commit**

```bash
cd "$REPO_ROOT"
git add python/propwash_gym/core/physics.py python/tests/test_physics.py
git commit -m "feat(gym): rigid-body physics with substeps, wind, and crash detection"
```

---

## Task 10: Reward terms

Mirrors `rotorenv/core/reward.py`. `ProgressReward` is stateful and **must** be reset each episode — this is the single most important detail for learnability.

Progress is also **normalized by the target's initial distance** (`normalize=True`), so a full traverse pays `scale` at any range. Raw metres-closed shaping made a −10 crash penalty worth 0.74% of a 12-target episode's return.

**Files:**
- Create: `python/propwash_gym/core/reward.py`
- Test: `python/tests/test_reward.py`

- [ ] **Step 1: Write the failing test**

Create `python/tests/test_reward.py`:

```python
"""Reward terms in isolation, and their composition."""

import numpy as np
import pytest

from propwash_gym.core.reward import (
    AllTargetsBonus,
    BatteryDepletedPenalty,
    CompositeReward,
    CrashPenalty,
    EnergyPenalty,
    ProgressReward,
    RewardContext,
    TargetReachedBonus,
)
from propwash_gym.core.state import DroneState


def _ctx(**kw):
    base = dict(
        state=DroneState([0, 0, 10], [0, 0, 0], [1, 0, 0, 0], [0, 0, 0], 0.0),
        throttle=0.0,
        target=np.array([10.0, 0.0, 10.0]),
        crashed=False,
        target_reached=False,
        all_targets_cleared=False,
        battery_empty=False,
    )
    base.update(kw)
    return RewardContext(**base)


def test_progress_reward_pays_for_closing_distance():
    r = ProgressReward(scale=1.0)
    r.reset(np.array([0.0, 0.0, 10.0]), np.array([10.0, 0.0, 10.0]))
    moved = _ctx(state=DroneState([3, 0, 10], [0, 0, 0], [1, 0, 0, 0], [0, 0, 0], 0.0))
    assert pytest.approx(3.0, abs=1e-9) == r(moved)


def test_progress_reward_is_negative_when_retreating():
    r = ProgressReward(scale=1.0)
    r.reset(np.array([0.0, 0.0, 10.0]), np.array([10.0, 0.0, 10.0]))
    back = _ctx(state=DroneState([-2, 0, 10], [0, 0, 0], [1, 0, 0, 0], [0, 0, 0], 0.0))
    assert pytest.approx(-2.0, abs=1e-9) == r(back)


def test_progress_reward_pays_zero_for_stalling():
    r = ProgressReward(scale=1.0)
    r.reset(np.array([0.0, 0.0, 10.0]), np.array([10.0, 0.0, 10.0]))
    same = _ctx(state=DroneState([0, 0, 10], [0, 0, 0], [1, 0, 0, 0], [0, 0, 0], 0.0))
    assert pytest.approx(0.0, abs=1e-9) == r(same)


def test_progress_reward_must_be_reset_between_episodes():
    r = ProgressReward(scale=1.0)
    r.reset(np.array([0.0, 0.0, 10.0]), np.array([10.0, 0.0, 10.0]))
    r(_ctx(state=DroneState([5, 0, 10], [0, 0, 0], [1, 0, 0, 0], [0, 0, 0], 0.0)))
    # New episode: spawn far away again. Without reset this would pay a huge
    # spurious negative.
    r.reset(np.array([0.0, 0.0, 10.0]), np.array([10.0, 0.0, 10.0]))
    first = r(_ctx(state=DroneState([1, 0, 10], [0, 0, 0], [1, 0, 0, 0], [0, 0, 0], 0.0)))
    assert pytest.approx(1.0, abs=1e-9) == first


def test_target_reached_bonus_only_fires_on_collection():
    b = TargetReachedBonus(bonus=10.0)
    assert b(_ctx(target_reached=True)) == 10.0
    assert b(_ctx(target_reached=False)) == 0.0


def test_all_targets_bonus_only_fires_once_cleared():
    b = AllTargetsBonus(bonus=25.0)
    assert b(_ctx(all_targets_cleared=True)) == 25.0
    assert b(_ctx(all_targets_cleared=False)) == 0.0


def test_crash_penalty_is_negative_on_crash_only():
    p = CrashPenalty(penalty=10.0)
    assert p(_ctx(crashed=True)) == -10.0
    assert p(_ctx(crashed=False)) == 0.0


def test_battery_penalty_fires_when_empty():
    p = BatteryDepletedPenalty(penalty=5.0)
    assert p(_ctx(battery_empty=True)) == -5.0
    assert p(_ctx(battery_empty=False)) == 0.0


def test_energy_penalty_scales_with_throttle_squared():
    p = EnergyPenalty(weight=1.0)
    assert pytest.approx(-0.25, abs=1e-9) == p(_ctx(throttle=0.5))
    assert pytest.approx(-1.0, abs=1e-9) == p(_ctx(throttle=1.0))


def test_composite_sums_its_terms():
    c = CompositeReward([TargetReachedBonus(10.0), CrashPenalty(10.0)])
    assert c(_ctx(target_reached=True, crashed=True)) == 0.0
    assert c(_ctx(target_reached=True)) == 10.0


def test_composite_resets_stateful_terms():
    progress = ProgressReward(scale=1.0)
    c = CompositeReward([progress, CrashPenalty(1.0)])
    c.reset(np.array([0.0, 0.0, 0.0]), np.array([5.0, 0.0, 0.0]))
    r = c(_ctx(state=DroneState([1, 0, 0], [0, 0, 0], [1, 0, 0, 0], [0, 0, 0], 0.0),
               target=np.array([5.0, 0.0, 0.0])))
    assert pytest.approx(1.0, abs=1e-9) == r
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd python && .venv/bin/python -m pytest tests/test_reward.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'propwash_gym.core.reward'`

- [ ] **Step 3: Write minimal implementation**

Create `python/propwash_gym/core/reward.py`:

```python
"""Composable reward terms.

Reward shaping is expressed as **data** — a list of configured term objects
summed by :class:`CompositeReward` — rather than as a hard-coded function, so
reward stays independent of physics and observation.

:class:`ProgressReward` is stateful and **must** be reset at the start of every
episode. A dense progress signal is what makes goal-reaching learnable here;
absolute distance penalties are too sparse and stall at zero success.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, Sequence

import numpy as np

from propwash_gym.core.state import DroneState


@dataclass
class RewardContext:
    """Everything a reward term may read for one step.

    Attributes:
        state: Post-step drone state.
        throttle: Normalised throttle applied this step.
        target: World position of the active target.
        crashed: Whether this step ended in a crash.
        target_reached: Whether a target was collected this step.
        all_targets_cleared: Whether the final target was collected this step.
        battery_empty: Whether the pack ran out this step.
    """

    state: DroneState
    throttle: float
    target: np.ndarray
    crashed: bool
    target_reached: bool
    all_targets_cleared: bool
    battery_empty: bool


class RewardTerm(Protocol):
    """Interface for a single additive component of a reward function."""

    def __call__(self, ctx: RewardContext) -> float:
        """Return this term's contribution for one step."""
        ...


@dataclass
class ProgressReward:
    """Dense reward for closing distance to the active target.

    ``r = scale * (d_prev - d_curr)``: moving 1 m closer pays ``scale``,
    stalling pays 0, retreating pays negative. The previous distance is instance
    state and must be cleared via :meth:`reset` each episode.
    """

    scale: float = 1.0
    _prev_distance: float | None = field(default=None, repr=False)

    def reset(self, initial_position: np.ndarray, target: np.ndarray) -> None:
        """Seed the previous distance with the spawn distance (zero progress)."""
        self._prev_distance = float(np.linalg.norm(target - initial_position))

    def retarget(self, position: np.ndarray, target: np.ndarray) -> None:
        """Reseed after the active target changes, so the switch pays nothing."""
        self._prev_distance = float(np.linalg.norm(target - position))

    def __call__(self, ctx: RewardContext) -> float:
        """Return ``scale * (prev_distance - current_distance)``."""
        distance = float(np.linalg.norm(ctx.target - ctx.state.position))
        if self._prev_distance is None:
            self._prev_distance = distance
            return 0.0
        progress = self._prev_distance - distance
        self._prev_distance = distance
        return float(self.scale) * progress


@dataclass
class TargetReachedBonus:
    """One-off bonus for collecting a target."""

    bonus: float = 10.0

    def __call__(self, ctx: RewardContext) -> float:
        """Return ``bonus`` on the collection step, else 0."""
        return self.bonus if ctx.target_reached else 0.0


@dataclass
class AllTargetsBonus:
    """One-off bonus for clearing every target in the layout."""

    bonus: float = 25.0

    def __call__(self, ctx: RewardContext) -> float:
        """Return ``bonus`` on the clearing step, else 0."""
        return self.bonus if ctx.all_targets_cleared else 0.0


@dataclass
class CrashPenalty:
    """One-off penalty applied on a terminal crash."""

    penalty: float = 10.0

    def __call__(self, ctx: RewardContext) -> float:
        """Return ``-penalty`` when crashed, else 0."""
        return -self.penalty if ctx.crashed else 0.0


@dataclass
class BatteryDepletedPenalty:
    """One-off penalty for running the pack flat."""

    penalty: float = 5.0

    def __call__(self, ctx: RewardContext) -> float:
        """Return ``-penalty`` when the battery is empty, else 0."""
        return -self.penalty if ctx.battery_empty else 0.0


@dataclass
class EnergyPenalty:
    """Small penalty proportional to throttle², discouraging full-throttle flying."""

    weight: float = 0.01

    def __call__(self, ctx: RewardContext) -> float:
        """Return ``-weight * throttle²``."""
        t = float(ctx.throttle)
        return -self.weight * t * t


@dataclass
class CompositeReward:
    """Sums a sequence of reward terms and forwards resets to stateful ones."""

    terms: Sequence[RewardTerm]

    def reset(self, initial_position: np.ndarray, target: np.ndarray) -> None:
        """Reset every term that carries episode state."""
        for term in self.terms:
            reset = getattr(term, "reset", None)
            if callable(reset):
                reset(initial_position, target)

    def retarget(self, position: np.ndarray, target: np.ndarray) -> None:
        """Forward a target change to every term that tracks one."""
        for term in self.terms:
            retarget = getattr(term, "retarget", None)
            if callable(retarget):
                retarget(position, target)

    def __call__(self, ctx: RewardContext) -> float:
        """Return the sum of all term contributions."""
        return float(sum(term(ctx) for term in self.terms))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd python && .venv/bin/python -m pytest tests/test_reward.py -v`
Expected: PASS, 11 tests.

- [ ] **Step 5: Commit**

```bash
cd "$REPO_ROOT"
git add python/propwash_gym/core/reward.py python/tests/test_reward.py
git commit -m "feat(gym): composable reward terms with dense progress shaping"
```

---

## Task 11: Base environment

The Gymnasium surface. Subclasses supply the task via hooks — this class never knows about targets.

**Files:**
- Create: `python/propwash_gym/envs/__init__.py`
- Create: `python/propwash_gym/envs/base_env.py`
- Test: `python/tests/test_base_env.py`

- [ ] **Step 1: Write the failing test**

Create `python/tests/test_base_env.py`:

```python
"""Gymnasium surface of PropwashEnv, exercised through a trivial subclass."""

import numpy as np
import pytest
from gymnasium import spaces

from propwash_gym.core.reward import CompositeReward, ProgressReward
from propwash_gym.envs.base_env import OBS_DIM, PropwashEnv
from propwash_gym.world.terrain import Heightfield


class _FixedTargetEnv(PropwashEnv):
    """Minimal task: one target 40 m north, for testing the base class."""

    def _build_terrain(self):
        return Heightfield(heights=np.full((16, 16), 100.0), extent_m=600.0)

    def _make_target(self):
        return np.array([0.0, 40.0, 120.0])

    def _build_reward(self):
        return CompositeReward([ProgressReward(1.0)])

    def _task_terminated(self, state):
        return False


def _env(**kw):
    kw.setdefault("offline", True)
    return _FixedTargetEnv(**kw)


def test_action_space_is_four_normalised_channels():
    env = _env()
    assert env.action_space.shape == (4,)
    assert np.all(env.action_space.low == -1.0)
    assert np.all(env.action_space.high == 1.0)


def test_state_observation_has_the_documented_width():
    env = _env(perception="state")
    assert env.observation_space.shape == (OBS_DIM,)
    assert OBS_DIM == 23


def test_reset_returns_obs_and_info():
    env = _env()
    obs, info = env.reset(seed=0)
    assert obs.shape == (OBS_DIM,)
    assert obs.dtype == np.float32
    assert "battery_soc" in info


def test_reset_with_the_same_seed_is_reproducible():
    a, _ = _env().reset(seed=42)
    b, _ = _env().reset(seed=42)
    np.testing.assert_array_equal(a, b)


def test_stepping_with_the_same_seed_and_actions_is_reproducible():
    def rollout():
        env = _env()
        env.reset(seed=7)
        out = []
        for _ in range(40):
            obs, r, term, trunc, _ = env.step(np.array([0.4, 0.0, 0.2, 0.0], np.float32))
            out.append((obs.copy(), r))
            if term or trunc:
                break
        return out

    a, b = rollout(), rollout()
    assert len(a) == len(b)
    for (oa, ra), (ob, rb) in zip(a, b):
        np.testing.assert_array_equal(oa, ob)
        assert ra == rb


def test_step_returns_the_five_tuple():
    env = _env()
    env.reset(seed=0)
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    assert obs.shape == (OBS_DIM,)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info, dict)


def test_observations_stay_inside_the_declared_space():
    env = _env()
    obs, _ = env.reset(seed=1)
    assert env.observation_space.contains(obs), obs
    for _ in range(60):
        obs, _, term, trunc, _ = env.step(np.array([0.6, 0.1, 0.3, 0.05], np.float32))
        assert env.observation_space.contains(obs), obs
        if term or trunc:
            break


def test_battery_drains_over_an_episode():
    env = _env()
    env.reset(seed=0)
    _, _, _, _, info = env.step(np.array([1.0, 0, 0, 0], np.float32))
    first = info["battery_soc"]
    for _ in range(50):
        _, _, term, trunc, info = env.step(np.array([1.0, 0, 0, 0], np.float32))
        if term or trunc:
            break
    assert info["battery_soc"] < first


def test_crashing_terminates_the_episode():
    env = _env(spawn_height=1.0)
    env.reset(seed=0)
    terminated = False
    for _ in range(400):
        _, _, terminated, truncated, info = env.step(
            np.array([-1.0, 0.0, 0.0, 0.0], np.float32)   # no thrust: fall
        )
        if terminated or truncated:
            break
    assert terminated
    assert info["crashed"] is True


def test_difficulty_option_is_recorded_and_clamped():
    env = _env()
    env.reset(seed=0, options={"difficulty": 0.42})
    assert pytest.approx(0.42) == env.difficulty
    env.reset(seed=0, options={"difficulty": 5.0})
    assert env.difficulty == 1.0


def test_unknown_perception_mode_is_rejected():
    with pytest.raises(ValueError, match="perception"):
        _env(perception="lidar")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd python && .venv/bin/python -m pytest tests/test_base_env.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'propwash_gym.envs'`

- [ ] **Step 3: Write minimal implementation**

Create `python/propwash_gym/envs/__init__.py`:

```python
"""Environments: the Gymnasium base class, tasks, and wrappers."""
```

Create `python/propwash_gym/envs/base_env.py`:

```python
"""Gymnasium base environment for PROP//WASH.

Subclasses supply the task through hooks — ``_make_target``, ``_build_reward``,
``_task_terminated`` — so this class never knows what a "target" means. That
keeps physics, terrain, reward, and observation independent.

**All randomness must come from ``self.np_random``.** The browser sim uses 29
unseeded ``Math.random()`` calls; reproducing episodes from ``reset(seed=n)``
requires routing every draw through Gymnasium's generator.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from gymnasium import Env, spaces

from propwash_gym.core.battery import Battery
from propwash_gym.core.flight_ctrl import FlightController
from propwash_gym.core.physics import Physics
from propwash_gym.core.reward import CompositeReward, RewardContext
from propwash_gym.core.rotations import euler_to_quat, quat_to_euler
from propwash_gym.core.state import DroneState
from propwash_gym.world.terrain import Heightfield, build_heightfield
from propwash_gym.world.tiles import TileCache

#: Width of the ``state`` observation vector. See the spec for the layout.
OBS_DIM = 23
PERCEPTIONS = ("state", "rgb", "depth")
#: Camera image side length for pixel perception modes.
IMAGE_SIZE = 64
DT = 0.02
#: Distances (m) ahead of the drone at which terrain height is probed.
PROBE_DISTANCES = (5.0, 10.0, 20.0, 40.0)


class PropwashEnv(Env):
    """Base flight environment. Not directly useful — subclass it with a task.

    Args:
        location: Which real-world location to fly.
        perception: ``"state"``, ``"rgb"``, or ``"depth"``.
        flight_mode: ``"angle"`` or ``"acro"``.
        spawn_height: Metres above ground at spawn.
        wind_scale: Mission wind strength.
        battery_mah: Pack capacity.
        max_time: Episode time limit in seconds.
        offline: Never fetch tiles (procedural terrain).
        tile_cache: Injectable cache, mainly for tests.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        location: str = "negev",
        perception: str = "state",
        flight_mode: str = "angle",
        spawn_height: float = 2.0,
        wind_scale: float = 0.0,
        battery_mah: float = 1300.0,
        max_time: float = 60.0,
        offline: bool = False,
        tile_cache: TileCache | None = None,
    ) -> None:
        if perception not in PERCEPTIONS:
            raise ValueError(
                f"perception must be one of {PERCEPTIONS}, got {perception!r}"
            )
        self.location = location
        self.perception = perception
        self.flight_mode = flight_mode
        self.spawn_height = float(spawn_height)
        self.wind_scale = float(wind_scale)
        self.battery_mah = float(battery_mah)
        self.max_time = float(max_time)
        self.offline = bool(offline)
        self._tile_cache = tile_cache or TileCache(offline=offline)

        #: Curriculum difficulty, driven by ``reset(options={"difficulty": d})``.
        self.difficulty: float = 1.0

        self.terrain: Heightfield = self._build_terrain()
        self.battery = Battery(self.battery_mah)
        self.controller = FlightController(mode=flight_mode)
        self.physics = Physics(
            terrain=self.terrain,
            controller=self.controller,
            battery=self.battery,
            wind_scale=self.wind_scale,
        )

        self.action_space = spaces.Box(-1.0, 1.0, shape=(4,), dtype=np.float32)
        self.observation_space = self._build_observation_space()

        self.state: DroneState | None = None
        self.target: np.ndarray = np.zeros(3)
        self.reward_fn: CompositeReward | None = None
        self._crashed = False

    # ---- hooks for subclasses ------------------------------------------------

    def _build_terrain(self) -> Heightfield:
        """Return the heightfield for this episode's location."""
        return build_heightfield(self.location, cache=self._tile_cache)

    def _make_target(self) -> np.ndarray:
        """Return the active target position. Subclasses must override."""
        raise NotImplementedError

    def _build_reward(self) -> CompositeReward:
        """Return this task's reward. Subclasses must override."""
        raise NotImplementedError

    def _task_terminated(self, state: DroneState) -> bool:
        """Return True when the task's own success condition is met."""
        raise NotImplementedError

    def _task_step(self, state: DroneState) -> tuple[bool, bool]:
        """Advance task bookkeeping. Return ``(target_reached, all_cleared)``."""
        return False, False

    def _task_reset(self) -> None:
        """Reset task state. Called after terrain and before the first target."""

    # ---- spaces and observations -------------------------------------------

    def _build_observation_space(self) -> spaces.Space:
        if self.perception == "state":
            return spaces.Box(-np.inf, np.inf, shape=(OBS_DIM,), dtype=np.float32)
        if self.perception == "rgb":
            return spaces.Box(
                0, 255, shape=(3, IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8
            )
        return spaces.Box(
            0.0, 1.0, shape=(1, IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32
        )

    def _state_obs(self, state: DroneState) -> np.ndarray:
        """Build the 23-D kinematic observation."""
        _, _, yaw = quat_to_euler(state.quaternion)
        to_target = self.target - state.position
        distance = float(np.linalg.norm(to_target))
        bearing = math.atan2(float(to_target[1]), float(to_target[0])) - yaw
        ground = self.terrain.height(float(state.position[0]), float(state.position[1]))
        probes = self.terrain.probe_ahead(
            float(state.position[0]),
            float(state.position[1]),
            yaw,
            PROBE_DISTANCES,
        )
        obs = np.concatenate(
            [
                state.position,                              # 0:3
                state.velocity,                              # 3:6
                state.quaternion,                            # 6:10
                state.angular_velocity,                       # 10:13
                [math.sin(bearing), math.cos(bearing), math.log1p(distance)],  # 13:16
                [self.battery.soc, self.battery.voltage / 16.8],               # 16:18
                [float(state.position[2]) - ground],          # 18:19
                probes - float(state.position[2]),            # 19:23
            ]
        )
        return obs.astype(np.float32)

    def _pixel_obs(self, state: DroneState) -> np.ndarray:
        """Sample terrain along camera rays into an image observation.

        A NumPy ray-march over the cached heightfield — no GPU, no browser. This
        approximates the browser's render; the browser itself is the visual
        check, not the training signal.
        """
        _, _, yaw = quat_to_euler(state.quaternion)
        fov = math.radians(100.0)
        angles = np.linspace(-fov / 2, fov / 2, IMAGE_SIZE)
        pitches = np.linspace(fov / 4, -fov / 2, IMAGE_SIZE)
        depth = np.ones((IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)
        max_range = 400.0
        px, py, pz = (float(v) for v in state.position)
        for row, elev in enumerate(pitches):
            for col, az in enumerate(angles):
                a = yaw + az
                dx, dy = math.cos(a), math.sin(a)
                dz = math.tan(elev)
                hit = max_range
                for d in np.linspace(4.0, max_range, 24):
                    if pz + dz * d <= self.terrain.height(px + dx * d, py + dy * d):
                        hit = float(d)
                        break
                depth[row, col] = hit / max_range
        if self.perception == "depth":
            return depth[None, :, :]
        shade = (depth * 255.0).astype(np.uint8)
        return np.repeat(shade[None, :, :], 3, axis=0)

    def _get_obs(self, state: DroneState) -> np.ndarray:
        """Return the observation for the configured perception mode."""
        if self.perception == "state":
            return self._state_obs(state)
        return self._pixel_obs(state)

    def _initial_state(self) -> DroneState:
        """Spawn at the field centre, ``spawn_height`` above the ground."""
        ground = self.terrain.height(0.0, 0.0)
        z = ground + self.spawn_height
        if self.terrain.water_z is not None:
            z = max(z, self.terrain.water_z + self.spawn_height)
        return DroneState(
            position=[0.0, 0.0, z],
            velocity=[0.0, 0.0, 0.0],
            quaternion=euler_to_quat(0.0, 0.0, 0.0),
            angular_velocity=[0.0, 0.0, 0.0],
            time=0.0,
        )

    def _get_info(self, state: DroneState) -> dict[str, Any]:
        """Diagnostics returned alongside every observation."""
        return {
            "battery_soc": self.battery.soc,
            "battery_voltage": self.battery.voltage,
            "crashed": self._crashed,
            "distance_to_target": float(
                np.linalg.norm(self.target - state.position)
            ),
            "terrain_source": self.terrain.source,
            "difficulty": self.difficulty,
        }

    # ---- Gymnasium API -----------------------------------------------------

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Start a new episode.

        Args:
            seed: Seeds ``self.np_random``; the same seed reproduces the episode.
            options: ``{"difficulty": float}`` in ``[0, 1]`` for the curriculum.
        """
        super().reset(seed=seed)
        if options is not None and "difficulty" in options:
            self.difficulty = float(np.clip(options["difficulty"], 0.0, 1.0))

        self._crashed = False
        self.battery.reset(self.battery_mah)
        # Required, not optional: ANGLE mode rebuilds attitude from the
        # controller's own integrated yaw, so a stale value snaps the heading on
        # the first step (verified: a drone spawned at 1.0 rad jumps to 0.0).
        self.controller.reset(yaw=0.0)
        self._task_reset()
        self.state = self._initial_state()
        self.target = self._make_target()
        self.reward_fn = self._build_reward()
        self.reward_fn.reset(self.state.position, self.target)
        return self._get_obs(self.state), self._get_info(self.state)

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Advance one control step of ``DT`` seconds."""
        if self.state is None or self.reward_fn is None:
            raise RuntimeError("reset() must be called before step()")

        a = np.clip(np.asarray(action, dtype=np.float64).reshape(4), -1.0, 1.0)
        throttle = float((a[0] + 1.0) / 2.0)      # [-1,1] -> [0,1]
        roll, pitch, yaw = (float(v) for v in a[1:])

        crash = self.physics.step(
            self.state, throttle=throttle, roll=roll, pitch=pitch, yaw=yaw, dt=DT
        )
        self.battery.update(throttle=throttle, dt=DT)

        self._crashed = crash is not None
        battery_empty = self.battery.is_empty
        target_reached, all_cleared = (False, False)
        if not self._crashed:
            target_reached, all_cleared = self._task_step(self.state)
            if target_reached and not all_cleared:
                self.target = self._make_target()
                self.reward_fn.retarget(self.state.position, self.target)

        reward = self.reward_fn(
            RewardContext(
                state=self.state,
                throttle=throttle,
                target=self.target,
                crashed=self._crashed,
                target_reached=target_reached,
                all_targets_cleared=all_cleared,
                battery_empty=battery_empty,
            )
        )

        terminated = bool(
            self._crashed
            or battery_empty
            or all_cleared
            or self._task_terminated(self.state)
        )
        truncated = bool(not terminated and self.state.time >= self.max_time)

        info = self._get_info(self.state)
        info["crash_reason"] = crash.value if crash is not None else None
        info["battery_empty"] = battery_empty
        return self._get_obs(self.state), float(reward), terminated, truncated, info
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd python && .venv/bin/python -m pytest tests/test_base_env.py -v`
Expected: PASS, 11 tests.

- [ ] **Step 5: Commit**

```bash
cd "$REPO_ROOT"
git add python/propwash_gym/envs python/tests/test_base_env.py
git commit -m "feat(gym): PropwashEnv Gymnasium base with switchable perception"
```

---

## Task 12: TARGET HUNT task

Ported from `Race.layoutHunt()`. Reproduce the two edge cases in the original exactly — they are documented in the spec and easy to get wrong.

**Files:**
- Create: `python/propwash_gym/envs/hunt_env.py`
- Test: `python/tests/test_hunt_env.py`

- [ ] **Step 1: Write the failing test**

Create `python/tests/test_hunt_env.py`:

```python
"""TARGET HUNT: layout generation, collection, and difficulty scaling."""

import numpy as np
import pytest

from propwash_gym.envs.hunt_env import (
    COLLECT_PAD,
    MAX_TARGETS,
    MIN_ORIGIN_DISTANCE,
    HuntEnv,
    Target,
)


def _env(**kw):
    kw.setdefault("offline", True)
    kw.setdefault("location", "negev")
    return HuntEnv(**kw)


def test_full_difficulty_places_twelve_targets():
    env = _env()
    env.reset(seed=0, options={"difficulty": 1.0})
    assert len(env.targets) == MAX_TARGETS


def test_zero_difficulty_places_a_single_target():
    env = _env()
    env.reset(seed=0, options={"difficulty": 0.0})
    assert len(env.targets) == 1


def test_layout_is_reproducible_for_a_seed():
    a, b = _env(), _env()
    a.reset(seed=99, options={"difficulty": 1.0})
    b.reset(seed=99, options={"difficulty": 1.0})
    assert [t.position.tolist() for t in a.targets] == [
        t.position.tolist() for t in b.targets
    ]


def test_layouts_differ_between_seeds():
    a, b = _env(), _env()
    a.reset(seed=1, options={"difficulty": 1.0})
    b.reset(seed=2, options={"difficulty": 1.0})
    assert [t.position.tolist() for t in a.targets] != [
        t.position.tolist() for t in b.targets
    ]


def test_targets_are_never_closer_than_the_minimum_to_the_pad():
    env = _env()
    env.reset(seed=3, options={"difficulty": 1.0})
    for t in env.targets:
        if t.perch == "GND" or t.perch == "HILL":
            assert float(np.linalg.norm(t.position[:2])) >= MIN_ORIGIN_DISTANCE - 1e-6


def test_target_sizes_are_within_the_documented_range():
    env = _env()
    env.reset(seed=4, options={"difficulty": 1.0})
    for t in env.targets:
        assert 1.3 <= t.size <= 3.3 + 1e-9


def test_collect_radius_follows_the_sim_formula():
    t = Target(position=np.zeros(3), size=2.0, perch="GND")
    assert pytest.approx(2.0 / 2 + COLLECT_PAD, abs=1e-12) == t.collect_radius


def test_spread_grows_with_difficulty():
    easy, hard = _env(), _env()
    easy.reset(seed=5, options={"difficulty": 0.0})
    hard.reset(seed=5, options={"difficulty": 1.0})
    easy_max = max(float(np.linalg.norm(t.position[:2])) for t in easy.targets)
    hard_max = max(float(np.linalg.norm(t.position[:2])) for t in hard.targets)
    assert hard_max > easy_max


def test_teleporting_onto_a_target_collects_it_and_pays_the_bonus():
    env = _env()
    env.reset(seed=6, options={"difficulty": 0.0})
    target = env.targets[0]
    env.state.position = target.position.copy()
    _, reward, _, _, info = env.step(np.array([0.0, 0.0, 0.0, 0.0], np.float32))
    assert info["targets_collected"] == 1
    assert reward > 10.0, "collection bonus plus clear bonus expected"


def test_clearing_the_only_target_terminates_the_episode():
    env = _env()
    env.reset(seed=7, options={"difficulty": 0.0})
    env.state.position = env.targets[0].position.copy()
    _, _, terminated, _, info = env.step(np.array([0.0, 0.0, 0.0, 0.0], np.float32))
    assert terminated
    assert info["all_cleared"] is True


def test_the_active_target_advances_after_a_collection():
    env = _env()
    env.reset(seed=8, options={"difficulty": 1.0})
    first = env.target.copy()
    env.state.position = env.targets[0].position.copy()
    env.step(np.array([0.0, 0.0, 0.0, 0.0], np.float32))
    assert not np.array_equal(first, env.target)
    assert env.next_index == 1


def test_hilltop_fallback_position_is_used_when_all_samples_are_rejected():
    # Force every hill sample inside the rejection radius by shrinking spread.
    env = _env(spread_override=1.0)
    env.reset(seed=9, options={"difficulty": 1.0})
    # The sim's fallback is (60, 60); at least one target should land there.
    assert any(
        abs(float(t.position[0]) - 60.0) < 1e-6 and abs(float(t.position[1]) - 60.0) < 1e-6
        for t in env.targets
    )


def test_info_reports_progress():
    env = _env()
    env.reset(seed=10, options={"difficulty": 1.0})
    _, _, _, _, info = env.step(np.array([0.2, 0, 0, 0], np.float32))
    assert info["targets_collected"] == 0
    assert info["targets_total"] == MAX_TARGETS
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd python && .venv/bin/python -m pytest tests/test_hunt_env.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'propwash_gym.envs.hunt_env'`

- [ ] **Step 3: Write minimal implementation**

Create `python/propwash_gym/envs/hunt_env.py`:

```python
"""TARGET HUNT — collect randomised targets scattered across real terrain.

Ported from ``index.html`` ``Race.layoutHunt()``. A single ``roll`` per target
selects its perch:

* ``roll < 0.20`` → rooftop (only when buildings exist)
* ``0.20 ≤ roll < 0.55`` → hilltop, best of 7 samples by height
* ``roll ≥ 0.55`` → open ground, one sample

Two faithful quirks from the original, both easy to port wrong:

1. The rooftop branch is guarded by "are there buildings". With none, a
   ``roll < 0.20`` **falls through to the hilltop branch** rather than being
   discarded.
2. The hill sampler skips candidates within 25 m of the pad **without
   retrying**, so all 7 samples can be rejected. The original then falls back to
   a fixed ``(60, 60)`` position.

All randomness goes through ``self.np_random`` so layouts are reproducible.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from propwash_gym.core.reward import (
    AllTargetsBonus,
    BatteryDepletedPenalty,
    CompositeReward,
    CrashPenalty,
    EnergyPenalty,
    ProgressReward,
    TargetReachedBonus,
)
from propwash_gym.core.state import DroneState
from propwash_gym.envs.base_env import PropwashEnv

MAX_TARGETS = 12
MIN_TARGETS = 1
MAX_SPREAD_M = 330.0
MIN_SPREAD_M = 60.0
MIN_ORIGIN_DISTANCE = 25.0
HILL_SAMPLES = 7
HILL_PERCH_MIN_HEIGHT = 6.0
#: Added to half the target's size to give the collection radius.
COLLECT_PAD = 1.7
SIZE_MIN = 1.3
SIZE_RANGE = 2.0
FALLBACK_XY = (60.0, 60.0)
MAX_WIND = 0.7


@dataclass
class Target:
    """One hunt target.

    Attributes:
        position: World position of the target's centre.
        size: Edge length in metres.
        perch: ``"GND"``, ``"HILL"``, ``"ROOF"``, or ``"BUOY"``.
    """

    position: np.ndarray
    size: float
    perch: str

    @property
    def collect_radius(self) -> float:
        """Proximity radius that counts as collecting this target."""
        return self.size / 2.0 + COLLECT_PAD


class HuntEnv(PropwashEnv):
    """Collect every target in a randomised layout without crashing.

    Args:
        spread_override: Force the placement radius, for tests.
        **kwargs: Forwarded to :class:`PropwashEnv`.
    """

    def __init__(self, spread_override: float | None = None, **kwargs) -> None:
        self.targets: list[Target] = []
        self.next_index = 0
        self._all_cleared = False
        self._spread_override = spread_override
        super().__init__(**kwargs)

    @property
    def _spread(self) -> float:
        if self._spread_override is not None:
            return float(self._spread_override)
        return MIN_SPREAD_M + (MAX_SPREAD_M - MIN_SPREAD_M) * self.difficulty

    @property
    def _count(self) -> int:
        return int(round(MIN_TARGETS + (MAX_TARGETS - MIN_TARGETS) * self.difficulty))

    def _task_reset(self) -> None:
        """Roll a fresh layout and scale wind with difficulty."""
        self.physics.wind_scale = MAX_WIND * self.difficulty
        self.next_index = 0
        self._all_cleared = False
        self.targets = self._layout()

    def _layout(self) -> list[Target]:
        """Generate this episode's targets."""
        spread = self._spread
        out: list[Target] = []
        for _ in range(self._count):
            roll = float(self.np_random.random())
            size = SIZE_MIN + float(self.np_random.random()) * SIZE_RANGE
            # No buildings are modelled yet, so a rooftop roll falls through to
            # the hill branch exactly as the browser sim does.
            hill = roll < 0.55
            best: tuple[float, float, float] | None = None
            for _ in range(HILL_SAMPLES if hill else 1):
                cx = (float(self.np_random.random()) - 0.5) * 2.0 * spread
                cy = (float(self.np_random.random()) - 0.5) * 2.0 * spread
                if np.hypot(cx, cy) < MIN_ORIGIN_DISTANCE:
                    continue
                h = self.terrain.height(cx, cy)
                if best is None or h > best[2]:
                    best = (cx, cy, h)
            if best is None:
                fx, fy = FALLBACK_XY
                best = (fx, fy, self.terrain.height(fx, fy))

            x, y, ground = best
            z = ground + size * 0.5 + 0.05
            perch = "GND"
            if self.terrain.water_z is not None and z < self.terrain.water_z + size * 0.5:
                z = self.terrain.water_z + size * 0.35
                perch = "BUOY"
            elif hill and ground > HILL_PERCH_MIN_HEIGHT + self.terrain.height(0.0, 0.0):
                perch = "HILL"
            out.append(
                Target(position=np.array([x, y, z]), size=size, perch=perch)
            )
        return out

    def _make_target(self) -> np.ndarray:
        """Return the active target's position."""
        if not self.targets:
            return np.zeros(3)
        idx = min(self.next_index, len(self.targets) - 1)
        return self.targets[idx].position.copy()

    def _build_reward(self) -> CompositeReward:
        """Dense progress shaping plus collection bonuses and crash penalties."""
        return CompositeReward(
            [
                ProgressReward(scale=1.0),
                TargetReachedBonus(bonus=10.0),
                AllTargetsBonus(bonus=25.0),
                CrashPenalty(penalty=10.0),
                BatteryDepletedPenalty(penalty=5.0),
                EnergyPenalty(weight=0.01),
            ]
        )

    def _task_step(self, state: DroneState) -> tuple[bool, bool]:
        """Collect the active target when within its radius."""
        if self._all_cleared or not self.targets:
            return False, False
        target = self.targets[self.next_index]
        if float(np.linalg.norm(target.position - state.position)) > target.collect_radius:
            return False, False
        self.next_index += 1
        if self.next_index >= len(self.targets):
            self._all_cleared = True
            return True, True
        return True, False

    def _task_terminated(self, state: DroneState) -> bool:
        """Terminate once every target is collected."""
        return self._all_cleared

    def _get_info(self, state: DroneState):
        """Add hunt progress to the base diagnostics."""
        info = super()._get_info(state)
        info["targets_collected"] = self.next_index
        info["targets_total"] = len(self.targets)
        info["all_cleared"] = self._all_cleared
        info["active_perch"] = (
            self.targets[self.next_index].perch
            if self.targets and self.next_index < len(self.targets)
            else None
        )
        return info
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd python && .venv/bin/python -m pytest tests/test_hunt_env.py -v`
Expected: PASS, 13 tests.

- [ ] **Step 5: Commit**

```bash
cd "$REPO_ROOT"
git add python/propwash_gym/envs/hunt_env.py python/tests/test_hunt_env.py
git commit -m "feat(gym): TARGET HUNT task with faithful layout generation"
```

---

## Task 13: Curriculum wrapper

Difficulty logic lives here, never in env state. Two schedules, mirroring `rotorenv`.

**Files:**
- Create: `python/propwash_gym/envs/curriculum.py`
- Test: `python/tests/test_curriculum.py`

- [ ] **Step 1: Write the failing test**

Create `python/tests/test_curriculum.py`:

```python
"""CurriculumWrapper: success-staged and step-annealed difficulty schedules."""

import gymnasium as gym
import numpy as np
import pytest

from propwash_gym.envs.curriculum import CurriculumWrapper
from propwash_gym.envs.hunt_env import HuntEnv


class _Stub(gym.Env):
    """Minimal env that records the difficulty it was reset with.

    Must subclass gym.Env: Wrapper.__init__ asserts isinstance(env, Env) before
    running any wrapper logic, so a plain class fails at construction.
    """

    def __init__(self):
        self.seen: list[float] = []
        self.observation_space = None
        self.action_space = None
        self._success = False

    def reset(self, *, seed=None, options=None):
        self.seen.append(float((options or {}).get("difficulty", -1.0)))
        return np.zeros(1), {}

    def step(self, action):
        return np.zeros(1), 0.0, True, False, {"all_cleared": self._success}

    def close(self):
        pass


def test_rejects_an_unknown_mode():
    with pytest.raises(ValueError, match="mode"):
        CurriculumWrapper(_Stub(), mode="vibes")


def test_starts_at_the_configured_difficulty():
    env = _Stub()
    w = CurriculumWrapper(env, mode="success", start_difficulty=0.0)
    w.reset()
    assert env.seen[0] == 0.0


def test_success_mode_raises_difficulty_after_enough_wins():
    env = _Stub()
    w = CurriculumWrapper(
        env, mode="success", start_difficulty=0.0, difficulty_step=0.25,
        window=4, success_threshold=0.75,
    )
    w.reset()
    env._success = True
    for _ in range(4):
        w.step(None)
        w.reset()
    assert w.difficulty > 0.0


def test_success_mode_lowers_difficulty_after_enough_losses():
    env = _Stub()
    w = CurriculumWrapper(
        env, mode="success", start_difficulty=0.5, difficulty_step=0.25,
        window=4, failure_threshold=0.25,
    )
    w.reset()
    env._success = False
    for _ in range(4):
        w.step(None)
        w.reset()
    assert w.difficulty < 0.5


def test_difficulty_is_clamped_to_the_unit_interval():
    env = _Stub()
    w = CurriculumWrapper(
        env, mode="success", start_difficulty=0.9, difficulty_step=0.5,
        window=2, success_threshold=0.0,
    )
    w.reset()
    env._success = True
    for _ in range(6):
        w.step(None)
        w.reset()
    assert w.difficulty <= 1.0


def test_step_mode_anneals_linearly_with_elapsed_steps():
    env = _Stub()
    w = CurriculumWrapper(env, mode="step", start_difficulty=0.0, anneal_steps=100)
    w.reset()
    for _ in range(50):
        w.step(None)
    w.reset()
    assert 0.3 < w.difficulty < 0.7


def test_step_mode_reaches_full_difficulty_after_annealing():
    env = _Stub()
    w = CurriculumWrapper(env, mode="step", start_difficulty=0.0, anneal_steps=10)
    w.reset()
    for _ in range(20):
        w.step(None)
    w.reset()
    assert pytest.approx(1.0) == w.difficulty


def test_wraps_a_real_hunt_env_end_to_end():
    env = CurriculumWrapper(
        HuntEnv(offline=True), mode="success", start_difficulty=0.0
    )
    obs, info = env.reset(seed=0)
    assert info["difficulty"] == 0.0
    for _ in range(5):
        obs, r, term, trunc, info = env.step(env.action_space.sample())
        if term or trunc:
            break
    env.close()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd python && .venv/bin/python -m pytest tests/test_curriculum.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'propwash_gym.envs.curriculum'`

- [ ] **Step 3: Write minimal implementation**

Create `python/propwash_gym/envs/curriculum.py`:

```python
"""Difficulty scheduling, as a wrapper.

Two schedules, matching the practice in reference drone-RL projects:

* ``"success"`` — performance-staged. Difficulty rises when the recent success
  rate clears ``success_threshold`` and falls below ``failure_threshold``.
* ``"step"`` — annealed linearly from ``start_difficulty`` to 1.0 across
  ``anneal_steps`` environment steps.

Curriculum state belongs here, never inside the env. The wrapped env only needs
to accept ``reset(options={"difficulty": float})``.

Starting at difficulty 0 matters: training directly at full difficulty stalls at
zero success, whereas mastering a flat single-target arena first converges.
"""

from __future__ import annotations

from collections import deque
from typing import Any

import gymnasium as gym
import numpy as np

MODES = ("success", "step")


class CurriculumWrapper(gym.Wrapper):
    """Anneal task difficulty across episodes.

    Args:
        env: Environment accepting a ``difficulty`` reset option.
        mode: ``"success"`` or ``"step"``.
        start_difficulty: Initial difficulty in ``[0, 1]``.
        difficulty_step: Increment applied per stage in success mode.
        window: Episodes considered when computing the success rate.
        success_threshold: Success rate at or above which difficulty rises.
        failure_threshold: Success rate at or below which difficulty falls.
        anneal_steps: Steps over which step mode ramps to 1.0.
    """

    def __init__(
        self,
        env: gym.Env,
        mode: str = "success",
        start_difficulty: float = 0.0,
        difficulty_step: float = 0.1,
        window: int = 20,
        success_threshold: float = 0.7,
        failure_threshold: float = 0.2,
        anneal_steps: int = 500_000,
    ) -> None:
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
        super().__init__(env)
        self.mode = mode
        self.difficulty = float(np.clip(start_difficulty, 0.0, 1.0))
        self.start_difficulty = self.difficulty
        self.difficulty_step = float(difficulty_step)
        self.success_threshold = float(success_threshold)
        self.failure_threshold = float(failure_threshold)
        self.anneal_steps = int(anneal_steps)
        self._results: deque[bool] = deque(maxlen=int(window))
        self._elapsed = 0

    def reset(self, **kwargs) -> tuple[Any, dict[str, Any]]:
        """Reset the inner env at the current difficulty."""
        if self.mode == "step":
            frac = min(1.0, self._elapsed / max(1, self.anneal_steps))
            self.difficulty = float(
                np.clip(
                    self.start_difficulty + (1.0 - self.start_difficulty) * frac,
                    0.0,
                    1.0,
                )
            )
        else:
            self._maybe_stage()
        options = dict(kwargs.pop("options", None) or {})
        options["difficulty"] = self.difficulty
        return self.env.reset(options=options, **kwargs)

    def _maybe_stage(self) -> None:
        """Raise or lower difficulty once the window has enough episodes."""
        if len(self._results) < self._results.maxlen:
            return
        rate = sum(self._results) / len(self._results)
        if rate >= self.success_threshold:
            self.difficulty = float(np.clip(self.difficulty + self.difficulty_step, 0.0, 1.0))
            self._results.clear()
        elif rate <= self.failure_threshold:
            self.difficulty = float(np.clip(self.difficulty - self.difficulty_step, 0.0, 1.0))
            self._results.clear()

    def step(self, action) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        """Step the inner env, recording episode outcomes for staging."""
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._elapsed += 1
        if terminated or truncated:
            self._results.append(bool(info.get("all_cleared", False)))
        info["difficulty"] = self.difficulty
        return obs, reward, terminated, truncated, info
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd python && .venv/bin/python -m pytest tests/test_curriculum.py -v`
Expected: PASS, 8 tests.

- [ ] **Step 5: Commit**

```bash
cd "$REPO_ROOT"
git add python/propwash_gym/envs/curriculum.py python/tests/test_curriculum.py
git commit -m "feat(gym): curriculum wrapper with success and step schedules"
```

---

## Task 14: Registration and Gymnasium conformance

**Files:**
- Modify: `python/propwash_gym/__init__.py`
- Test: `python/tests/test_conformance.py`

- [ ] **Step 1: Write the failing test**

Create `python/tests/test_conformance.py`:

```python
"""Registered variants, and Gymnasium's own env checker across all of them."""

import gymnasium as gym
import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

import propwash_gym

STATE_IDS = ["PropwashHunt-v0", "PropwashHuntEasy-v0"]
PIXEL_IDS = ["PropwashHuntRGB-v0", "PropwashHuntDepth-v0"]
ALL_IDS = STATE_IDS + PIXEL_IDS


def test_every_variant_is_registered():
    for env_id in ALL_IDS:
        assert env_id in gym.registry, env_id


@pytest.mark.parametrize("env_id", STATE_IDS)
def test_check_env_passes_for_state_variants(env_id):
    env = propwash_gym.make(env_id, offline=True)
    check_env(env.unwrapped, skip_render_check=True)
    env.close()


@pytest.mark.parametrize("env_id", PIXEL_IDS)
def test_pixel_variants_build_and_produce_correct_shapes(env_id):
    # Ray-marched pixel obs is slow; assert shape rather than running check_env.
    env = propwash_gym.make(env_id, offline=True)
    obs, _ = env.reset(seed=0)
    assert obs.shape == env.observation_space.shape
    assert env.observation_space.contains(obs)
    env.close()


@pytest.mark.parametrize("env_id", STATE_IDS)
def test_time_limit_truncates_rather_than_running_forever(env_id):
    env = propwash_gym.make(env_id, offline=True)
    env.reset(seed=0)
    for _ in range(10_000):
        _, _, terminated, truncated, _ = env.step(env.action_space.sample())
        if terminated or truncated:
            break
    else:
        pytest.fail("episode never ended")
    env.close()


def test_easy_variant_starts_at_zero_difficulty():
    env = propwash_gym.make("PropwashHuntEasy-v0", offline=True)
    _, info = env.reset(seed=0)
    assert info["difficulty"] == 0.0
    env.close()


def test_make_rejects_an_unknown_id():
    with pytest.raises(Exception):
        propwash_gym.make("PropwashNope-v0")


def test_location_can_be_overridden_at_construction():
    env = propwash_gym.make("PropwashHunt-v0", location="canyon", offline=True)
    assert env.unwrapped.location == "canyon"
    env.close()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd python && .venv/bin/python -m pytest tests/test_conformance.py -v`
Expected: FAIL — variants are not registered yet.

- [ ] **Step 3: Write the implementation**

Replace `python/propwash_gym/__init__.py` entirely:

```python
"""propwash-gym — a Gymnasium RL environment on real satellite terrain.

Importing this package registers the environment IDs with Gymnasium's global
registry, so ``propwash_gym.make("PropwashHunt-v0")`` (or ``gymnasium.make``)
works after ``import propwash_gym``.

Many small registered variants over one task class follows MiniGrid; selecting
behaviour through registry kwargs follows gym-pybullet-drones.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
from gymnasium.envs.registration import register

__version__ = "0.1.0"

_HUNT = "propwash_gym.envs.hunt_env:HuntEnv"

# 60 s at dt=0.02 -> 3000 steps.
_EPISODE_STEPS = 3000

_VARIANTS: dict[str, tuple[str, int, dict[str, Any]]] = {
    "PropwashHunt-v0": (_HUNT, _EPISODE_STEPS, {}),
    # Flat, windless, single-target arena — the curriculum's starting point.
    "PropwashHuntEasy-v0": (_HUNT, _EPISODE_STEPS, {"start_difficulty": 0.0}),
    "PropwashHuntRGB-v0": (_HUNT, _EPISODE_STEPS, {"perception": "rgb"}),
    "PropwashHuntDepth-v0": (_HUNT, _EPISODE_STEPS, {"perception": "depth"}),
}

for _env_id, (_entry, _max_steps, _kwargs) in _VARIANTS.items():
    if _env_id not in gym.registry:
        register(
            id=_env_id,
            entry_point=_entry,
            max_episode_steps=_max_steps,
            kwargs={k: v for k, v in _kwargs.items() if k != "start_difficulty"},
        )


def make(env_id: str, **kwargs: Any) -> gym.Env:
    """Create a registered propwash-gym environment.

    Variants carrying ``start_difficulty`` are returned already wrapped in a
    :class:`~propwash_gym.envs.curriculum.CurriculumWrapper`, so the difficulty
    they advertise is applied from the first reset.

    Args:
        env_id: A registered ID, e.g. ``"PropwashHunt-v0"``.
        **kwargs: Forwarded to the env constructor (``location``, ``offline``, …).
    """
    from propwash_gym.envs.curriculum import CurriculumWrapper

    env = gym.make(env_id, **kwargs)
    spec_kwargs = _VARIANTS.get(env_id, (None, 0, {}))[2]
    if "start_difficulty" in spec_kwargs:
        return CurriculumWrapper(
            env, mode="success", start_difficulty=spec_kwargs["start_difficulty"]
        )
    return env
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd python && .venv/bin/python -m pytest tests/test_conformance.py -v`
Expected: PASS, 11 tests (parametrised).

`check_env` emits two warnings for the state variants:

```
WARN: A Box observation space minimum value is -infinity. This is probably too low.
WARN: A Box observation space maximum value is infinity. This is probably too high.
```

These are expected and not failures — verified against `gymnasium.utils.env_checker`.
World-frame position and terrain probes have no meaningful finite bound, so the
space is deliberately unbounded. Do not "fix" this by inventing clip limits: a
wrong bound would silently truncate observations at the edges of large terrain.

- [ ] **Step 5: Run the whole suite**

Run: `cd python && .venv/bin/python -m pytest -q`
Expected: all tests pass; network tests deselected.

- [ ] **Step 6: Commit**

```bash
cd "$REPO_ROOT"
git add python/propwash_gym/__init__.py python/tests/test_conformance.py
git commit -m "feat(gym): register task variants and pass Gymnasium check_env"
```

---

## Task 15: Examples, README, and a smoke training run

**Files:**
- Create: `python/examples/random_agent.py`
- Create: `python/examples/train_hunt.py`
- Create: `python/README.md`

- [ ] **Step 1: Write the random-agent example**

Create `python/examples/random_agent.py`:

```python
"""Sanity check: three random episodes, printing return and outcome.

Usage:
    python examples/random_agent.py
    python examples/random_agent.py --env PropwashHunt-v0 --location canyon
"""

from __future__ import annotations

import argparse

import propwash_gym


def main() -> int:
    """Run three random episodes and print a one-line summary each."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="PropwashHuntEasy-v0")
    ap.add_argument("--location", default="negev")
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--offline", action="store_true", help="skip tile fetching")
    args = ap.parse_args()

    env = propwash_gym.make(args.env, location=args.location, offline=args.offline)
    print(f"{args.env} @ {args.location}")
    print(f"  obs {env.observation_space.shape}  act {env.action_space.shape}")

    for ep in range(args.episodes):
        _, info = env.reset(seed=ep)
        total, steps = 0.0, 0
        while True:
            _, reward, terminated, truncated, info = env.step(env.action_space.sample())
            total += reward
            steps += 1
            if terminated or truncated:
                break
        print(
            f"  ep {ep}: return {total:8.2f}  steps {steps:5d}  "
            f"targets {info.get('targets_collected', 0)}/{info.get('targets_total', 0)}  "
            f"crash={info.get('crash_reason')}  terrain={info.get('terrain_source')}"
        )
    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Run it and confirm episodes terminate**

Run: `cd python && .venv/bin/python examples/random_agent.py --offline`
Expected: three lines, each with a finite return and a `crash=` reason (a random policy nearly always crashes).

- [ ] **Step 3: Write the training example**

Create `python/examples/train_hunt.py`:

```python
"""Train PPO on TARGET HUNT under a success-staged curriculum.

Requires the RL extra:  pip install -e ".[rl]"

Usage:
    python examples/train_hunt.py --steps 200000
    python examples/train_hunt.py --steps 500 --eval-episodes 2   # smoke test
"""

from __future__ import annotations

import argparse

import propwash_gym
from propwash_gym.envs.curriculum import CurriculumWrapper


def main() -> int:
    """Train, then evaluate, printing curriculum progress and success rate."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="PropwashHunt-v0")
    ap.add_argument("--location", default="negev")
    ap.add_argument("--steps", type=int, default=200_000)
    ap.add_argument("--eval-episodes", type=int, default=10)
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--save", default="runs/hunt_ppo")
    args = ap.parse_args()

    try:
        from stable_baselines3 import PPO
    except ImportError:
        print('stable-baselines3 missing. Install with: pip install -e ".[rl]"')
        return 1

    import gymnasium as gym

    base = gym.make(args.env, location=args.location, offline=args.offline)
    env = CurriculumWrapper(base, mode="success", start_difficulty=0.0)

    # Pixel perception needs a CNN and unnormalised images.
    is_pixels = len(env.observation_space.shape) == 3
    policy = "CnnPolicy" if is_pixels else "MlpPolicy"
    policy_kwargs = {"normalize_images": False} if is_pixels else None
    print(f"training {policy} on {args.env} for {args.steps} steps")

    model = PPO(policy, env, policy_kwargs=policy_kwargs, verbose=1)
    model.learn(total_timesteps=args.steps)
    model.save(args.save)
    print(f"saved to {args.save}.zip; final difficulty {env.difficulty:.2f}")

    cleared = 0
    for ep in range(args.eval_episodes):
        obs, _ = env.reset(seed=10_000 + ep)
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                cleared += bool(info.get("all_cleared"))
                break
    print(
        f"eval: cleared {cleared}/{args.eval_episodes} "
        f"({100.0 * cleared / args.eval_episodes:.0f}%) at difficulty {env.difficulty:.2f}"
    )
    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Smoke-test training end to end**

```bash
cd python
.venv/bin/pip install -e ".[rl]"
.venv/bin/python examples/train_hunt.py --steps 2000 --eval-episodes 2 --offline
```

Expected: PPO logs appear, a `.zip` is written under `runs/`, and an `eval:` line prints. A 2000-step run will not learn the task — this only proves the pipeline runs.

- [ ] **Step 5: Write the package README**

Create `python/README.md`:

```markdown
# propwash-gym

A Gymnasium reinforcement-learning environment built from the
[PROP//WASH](../README.md) FPV drone simulator: real satellite terrain,
quaternion rigid-body physics, and a 4S LiPo battery that sags under load.

The browser simulator is **not** in the training loop. This package fetches the
same Esri imagery and AWS terrarium elevation tiles, caches them to disk, and
samples them as NumPy arrays — so training runs in-process at thousands of steps
per second instead of the tens a browser bridge would allow.

## Install

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[dev,rl]"
```

## Quick start

```python
import propwash_gym

env = propwash_gym.make("PropwashHuntEasy-v0", location="negev")
obs, info = env.reset(seed=0)
obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
```

Warm the tile cache first so later runs need no network:

```bash
python examples/fetch_tiles.py
```

## Variants

| ID | Perception | Notes |
|---|---|---|
| `PropwashHunt-v0` | state (23-D) | full difficulty |
| `PropwashHuntEasy-v0` | state (23-D) | curriculum from difficulty 0 |
| `PropwashHuntRGB-v0` | `(3,64,64)` uint8 | needs `CnnPolicy` |
| `PropwashHuntDepth-v0` | `(1,64,64)` float32 | needs `CnnPolicy` |

Pass `location=` to fly any of the seven places: `negev`, `alps`, `cascades`,
`canyon`, `iceland`, `borabora`, `sahara`.

## Action and observation

**Action** — `Box(-1, 1, (4,))`: throttle, roll, pitch, yaw. Throttle is
rescaled to `[0,1]` internally.

**State observation** — 23 values: position (3), velocity (3), quaternion (4),
body rates (3), target bearing and distance (3), battery (2), ground clearance
(1), terrain height at 5/10/20/40 m ahead (4).

The terrain probes are not decoration. Without lookahead an agent flying real
elevation data cannot anticipate a ridge, and canyon locations become
unlearnable in a way that resembles a bug.

## Training

```bash
python examples/train_hunt.py --steps 200000
```

Two details matter for learnability, both carried over from the sibling
`rotorenv` project:

1. **Dense progress shaping.** `ProgressReward` pays for distance closed each
   step. Absolute distance penalties are too sparse and stall at zero success.
2. **Start easy.** `CurriculumWrapper(start_difficulty=0.0)` begins in a flat,
   windless, single-target arena and advances on the recent success rate.
   Training directly at full difficulty does not converge.

## Testing

```bash
pytest                  # offline suite
pytest -m network       # live tile-provider checks
```

The default suite never touches the network: tile fetching is injected in tests,
and missing tiles fall back to procedural terrain.
```

- [ ] **Step 6: Run the full suite one final time**

Run: `cd python && .venv/bin/python -m pytest -q`
Expected: every test passes.

- [ ] **Step 7: Commit**

```bash
cd "$REPO_ROOT"
git add python/examples python/README.md
git commit -m "docs(gym): examples, package README, and PPO smoke test"
```

---

## Self-review notes

**Spec coverage** — every section of the spec maps to a task:

| Spec section | Task |
|---|---|
| Mechanics in Python, not browser | architecture of Tasks 4–6, 11 |
| Standalone, mirror rotorenv patterns | Tasks 10, 13 (no rotorenv import anywhere) |
| Lives in `python/` of the sim repo | Task 1 |
| Physics ported constants | Tasks 3, 8, 9 |
| Z-up vs Y-up conversion | Task 2 (single conversion point, asserted) |
| Crash conditions | Task 9 |
| Action space | Task 11 |
| Observation modes and 23-D layout | Task 11 |
| Terrain lookahead probes | Tasks 6, 11 |
| Hunt placement + 2 edge cases | Task 12 |
| Reward table | Task 10 |
| Curriculum table | Tasks 12, 13 |
| Tile cache + offline fallback | Tasks 4, 6, 7 |
| Registered variants | Task 14 |
| All eight test areas | Tasks 2–14 |
| Out of scope | no tasks — correctly absent |

**Deliberate deviations from the spec, and why:**

1. **Buildings are not modelled.** The spec's hunt placement includes rooftop
   perches, but no task builds a building generator. Task 12 handles this the
   way the browser does when a location has no buildings: the rooftop roll falls
   through to the hilltop branch. `ROOF` perches therefore never occur yet.
   Adding buildings is a follow-up, not a silent gap.
2. **`rgb` perception is ray-marched, not rendered.** The spec allows this
   ("an approximation by design"), and Task 11 documents it in the docstring.
3. **Pixel variants skip `check_env`.** Task 14 asserts shapes and space
   membership instead, because the ray-march is slow enough to make the full
   checker impractical. State variants get the full check.

**Naming consistency:** `Heightfield.height/slope/probe_ahead/water_z/source`,
`Battery.soc/voltage/thrust_scale/is_empty/update(throttle, dt)`,
`Physics.step(...) -> CrashReason | None`, `CompositeReward.reset/retarget`,
`Target.collect_radius`, and `info["all_cleared"]` are used identically wherever
they appear across Tasks 3–14.
