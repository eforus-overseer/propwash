# PROP//WASH Gym — design

**Date:** 2026-08-02
**Status:** approved, pending implementation plan

A Gymnasium-compatible reinforcement learning environment that ports the PROP//WASH
flight model, terrain pipeline, and TARGET HUNT task to Python, so agents can be
trained on real satellite terrain at usable step rates.

## Goal

Train RL agents to fly the PROP//WASH drone over real-world terrain. The browser
sim stays the human-facing artifact; the Python package is the training surface.

Success means: `check_env` passes on every registered variant, a PPO agent learns
TARGET HUNT to a measurable success rate under a curriculum, and the same physics
constants drive both the Python env and the browser sim so a trained policy can be
replayed visually.

## Architecture decisions

These were settled during brainstorming and are load-bearing.

### Mechanics run in Python, not the browser

The browser is **not** in the training loop. No mature RL environment drives a
browser per step: MiniGrid mutates a NumPy array in-process, gym-pybullet-drones
calls PyBullet in-process, and rotorenv renders off-screen with PyVista. A
`page.evaluate()` round trip costs ~1–5 ms before any physics or pixel readback,
which caps training at roughly 30–100 steps/s with pixels — 3–9 hours for 1M steps.

Instead, Python fetches the **same tile data** the sim uses and samples it as NumPy
arrays. Verified reachable outside the browser:

| Dataset | Endpoint | Verified |
|---|---|---|
| Imagery | `server.arcgisonline.com/.../World_Imagery/MapServer/tile/{z}/{y}/{x}` | HTTP 200, 256×256 JPEG |
| Elevation | `s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png` | HTTP 200, 256×256 PNG |

Note the **axis order differs between providers** — Esri is `{z}/{y}/{x}`, terrarium
is `{z}/{x}/{y}`. Getting this backwards yields valid-looking tiles of the wrong
place, which is silent and hard to debug.

Decode, confirmed against three locations:

```
h = R*256 + G + B/256 - 32768
```

| Location | Tile (z15) | Decoded range | Sanity |
|---|---|---|---|
| Makhtesh Ramon | 19551/13454 | 824.8 – 868.3 m | crater rim |
| Grindelwald | 17115/11575 | 952.6 – 1119.4 m | alpine valley floor |
| Bora Bora | 2571/17907 | −2464.4 – 407.2 m | bathymetry + Mt Otemanu |

Bora Bora's negative values matter: terrarium encodes **bathymetry** below sea
level. `waterY` must be derived from the DEM offset, or the drone spawns underwater.

### Standalone package, mirroring rotorenv's patterns

`propwash-gym` does not import `rotorenv`. It copies the shapes that are already
proven there — `RewardTerm` protocol, `CompositeReward`, `CurriculumWrapper`,
enum-configurable spaces, registered task variants — so each repo installs and
releases independently. The cost is a small amount of duplicated scaffolding,
accepted deliberately.

### Lives inside the propwash-fpv-sim repo

A `python/` subdirectory of the existing (private) `propwash-fpv-sim` repo, so the
game and its RL env stay together.

## Layout

```
propwash-fpv-sim/
├── index.html                    the browser sim (unchanged)
└── python/
    ├── pyproject.toml            propwash-gym, extras: [rl] [render] [dev]
    ├── propwash_gym/
    │   ├── __init__.py           registers env IDs, exposes make()
    │   ├── world/
    │   │   ├── tiles.py          fetch + disk cache, offline fallback
    │   │   ├── terrain.py        Heightfield: height(), slope(), waterY
    │   │   ├── procedural.py     synthetic terrain when tiles unavailable
    │   │   └── locations.py      the 7 real locations
    │   ├── core/
    │   │   ├── drone.py          DroneState dataclass
    │   │   ├── physics.py        fixed-step integrator, 2 substeps
    │   │   ├── battery.py        4S LiPo: IR sag, coulomb counting
    │   │   ├── flight_ctrl.py    ANGLE / ACRO
    │   │   ├── rotations.py      one frame convention, quaternions [w,x,y,z]
    │   │   └── reward.py         RewardTerm protocol + terms + CompositeReward
    │   ├── envs/
    │   │   ├── base_env.py       PropwashEnv (Gymnasium API)
    │   │   ├── hunt_env.py       TARGET HUNT
    │   │   ├── curriculum.py     CurriculumWrapper (success / step modes)
    │   │   └── wrappers.py       obs/reward transforms
    │   └── render/
    │       └── viewer.py         optional trajectory export → browser replay
    ├── examples/
    │   ├── random_agent.py
    │   ├── fetch_tiles.py        warm the cache for chosen locations
    │   └── train_hunt.py         PPO + curriculum
    └── tests/
```

Keep physics, reward, observation, and terrain independent and swappable. Do not
collapse them into the env loop.

## Physics — ported constants

Taken from `index.html`, not reinvented. Identical values are what make browser
replay of a trained policy meaningful.

| Quantity | Value |
|---|---|
| Mass | 0.62 kg |
| Max thrust | 24 N (≈4:1 TWR) |
| Drag coefficient | 0.045, quadratic (`-dragK * v * |v|`) |
| Gravity | 9.81 m/s² |
| Integration | fixed `dt`, 2 substeps per step (`h = dt/2`) |
| Battery | 4 cells, 1300 mAh (1000 for PRO RACE) |
| Current draw | `1.2 + throttle² * 78` A |
| Coulomb counting | `drawn += current * dt / 3.6` (mAh) |
| Voltage | `clamp((3.0 + soc*1.2 - current*0.0058) * cells, 0, 16.8)` |
| Thrust scale | `clamp((V - 11.2)/5.6, .35, 1) * .35 + .65` |
| ANGLE max tilt | 38° |
| ANGLE smoothing | `1 - exp(-dt * 7.5)` |
| ANGLE yaw rate | 2.6 rad/s |
| ACRO roll/pitch rate | 480°/s |
| ACRO yaw rate | 300°/s |
| ACRO expo | `v * (0.35 + 0.65 * v²)` |
| ACRO rate smoothing | `1 - exp(-dt * 18)` |
| Wind | `sin(t*.31)+sin(t*.83)*.5`, `sin(t*.57)*.25`, `cos(t*.27)+sin(t*.71)*.5`, × mission wind |

**Coordinate frames differ and this must be handled explicitly.** Three.js is
Y-up; rotorenv and most robotics code are Z-up. The Python port uses **Z-up**
internally (matching rotorenv convention, so `rotations.py` is directly
comparable) and converts only at the browser-replay boundary. All ported formulas
above are frame-agnostic except the thrust and gravity axis, which become
`+Z` and `−Z`.

Crash conditions, ported from `Physics.collide()`:

- Ground contact with descent rate > 3.2 m/s or speed > 9 m/s
- Below `waterY + 0.05` on water locations (instant, any speed)
- Inside a tree cylinder or building AABB
- Beyond `worldSize/2 - 10` horizontally

Sub-crash ground contact clamps to `ground + 0.09`, zeroes downward velocity, and
scales the rest by 0.6.

## Spaces

**Action** — `Box(-1, 1, shape=(4,), float32)`: `[throttle, roll, pitch, yaw]`.
Throttle maps to `[0,1]` internally; the other three are bipolar. Mirrors the sim's
Mode-2 layout and rotorenv's normalized convention.

**Observation** — selected by `perception`:

| Mode | Space | Notes |
|---|---|---|
| `state` | `Box(23,, float32)` | no rendering, fastest |
| `rgb` | `Box(0,255,(3,64,64), uint8)` | sampled satellite imagery |
| `depth` | `Box(0,1,(1,64,64), float32)` | terrain raycast depth |

The 23-D state vector:

| Slice | Contents |
|---|---|
| 0:3 | position (world, metres) |
| 3:6 | velocity (world) |
| 6:10 | attitude quaternion `[w,x,y,z]` |
| 10:13 | body angular rates |
| 13:16 | target bearing (sin, cos), distance (log-scaled) |
| 16:18 | battery SoC, voltage (normalized) |
| 18:19 | terrain clearance directly below (AGL) |
| 19:23 | terrain height at 4 forward probes (5/10/20/40 m ahead) |

The forward probes are not optional. Without local terrain lookahead, an agent in
`state` mode flying real DEM cannot anticipate a ridge, and canyon locations become
unlearnable in a way that resembles a bug but is an observability gap.

`rgb` and `depth` observations are generated by sampling the cached terrain arrays
along camera rays in NumPy — no GPU, no browser. This gives real imagery content at
a fraction of a render's cost, and is the same tradeoff rotorenv's `depth` mode makes.

## TARGET HUNT task

Ported from `Race.layoutHunt()`. Each reset rolls a fresh scenario.

Target placement per target (12 at full difficulty), driven by one `roll ∈ [0,1)`:
- **`roll < 0.20` → rooftop** — on a building, `y = b.h + size/2 + 0.05`, perch `ROOF`
- **`0.20 ≤ roll < 0.55` → hilltop** — best of 7 sampled positions by height,
  perch `HILL` if `h > 6`
- **`roll ≥ 0.55` → open ground** — single sample, perch `GND`
- Any placement below `waterY + size/2` becomes `BUOY` at `waterY + size*0.35`
- Samples closer than 25 m to origin are rejected
- Size `1.3 + rand*2.0`; collection radius `size/2 + 1.7`

Two edge cases in the original that the port must preserve, both easy to get wrong:

1. The rooftop branch is guarded by `World.buildings.length`. On a location with no
   buildings, a `roll < 0.20` **falls through to the hilltop branch** (since
   `roll < 0.55` still holds) rather than being discarded. Effective split there is
   55% hill / 45% ground.
2. The hilltop sampler `continue`s on samples within 25 m of origin without
   retrying, so a hill target can end up with `best = null` after 7 rejections. The
   original falls back to a fixed `(60, 60)` position. Reproduce that fallback.

Collection is proximity-based (sphere test), matching the sim. Targets are
sequenced in generation order.

### Reward

Dense progress shaping, per the training lesson already established in rotorenv:
absolute distance penalties are too sparse and stall at 0% success.

| Term | Value |
|---|---|
| `ProgressReward` | `+1.0 ×` distance closed toward the current target |
| `TargetReached` | `+10` per target |
| `AllTargetsBonus` | `+25` on clearing the layout |
| `CrashPenalty` | `−10`, terminates |
| `BatteryDepleted` | `−5`, terminates |
| `EnergyPenalty` | small, `∝ throttle²` |

`ProgressReward` is stateful and must be reset with the spawn and first target on
every `reset()`.

**Terminate** on: all targets collected, crash, or battery empty.
**Truncate** at the mission time limit (`max_episode_steps`).

### Curriculum

`difficulty ∈ [0,1]` passed via `reset(options={"difficulty": d})`, consumed as:

| Parameter | d=0 | d=1 |
|---|---|---|
| Target count | 1 | 12 |
| Spread | 60 m | 330 m |
| Wind scale | 0.0 | 0.7 |
| Terrain | flat synthetic | real DEM |
| Obstacles | none | trees + buildings |

Difficulty 0 is a flat, windless, single-target arena on purpose. Curriculum logic
lives in the wrapper, never in env state.

## Tile cache

```
~/.cache/propwash-gym/tiles/
├── imagery/{z}/{y}/{x}.jpg
└── terrarium/{z}/{x}/{y}.png
```

- First construction for a location fetches and caches; later runs are offline.
- `examples/fetch_tiles.py` warms the cache deliberately.
- If tiles are missing **and** unfetchable, fall back to procedural terrain and warn
  **once** per process. Tests and CI never require network.
- Cache is content-addressed by tile coordinates, so it is safe to share and never
  needs invalidation (imagery updates are not a correctness concern here).
- Requests send a `User-Agent` and are rate-limited politely; failures on individual
  tiles degrade to the procedural fallback for that patch rather than raising.

## Registered variants

| ID | Task | Perception | Physics |
|---|---|---|---|
| `PropwashHunt-v0` | hunt | state | 6-DoF |
| `PropwashHuntEasy-v0` | hunt, d=0 fixed | state | 6-DoF |
| `PropwashHuntRGB-v0` | hunt | rgb | 6-DoF |
| `PropwashHuntDepth-v0` | hunt | depth | 6-DoF |

Location is a constructor kwarg (`location="negev"`), defaulting to `negev`.
Many small registered variants over one task class follows MiniGrid; selecting
behaviour through registry kwargs follows gym-pybullet-drones.

## Testing

| Area | What it checks |
|---|---|
| Conformance | `gymnasium.utils.env_checker.check_env` on every variant |
| Determinism | same seed → bit-identical trajectory; **all randomness through `self.np_random`**, never `random` or bare `np.random` |
| Physics parity | hover equilibrium throttle, terminal velocity, and battery curve match the JS constants within tolerance |
| Terrain | known lat/lon → known elevation, using the three verified locations above; tile axis order asserted explicitly |
| Water | Bora Bora `waterY` derived correctly from negative DEM values; spawn is above water |
| Offline | env constructs with no network and an empty cache (procedural path) |
| Reward | each term in isolation; `ProgressReward` resets between episodes |
| Curriculum | difficulty 0 → 1 changes target count, spread, wind as specified |

The sim's 29 `Math.random()` call sites become seeded draws from `self.np_random`.
This is a Gymnasium requirement, not a nicety — without it `reset(seed=n)` cannot
reproduce an episode.

## Out of scope

Deliberately excluded to keep this to one implementable plan:

- Gate racing (TRAINING/CIRCUIT/PRO RACE) — a second task, after hunt trains
- Browser-in-the-loop training — rejected on performance grounds above
- Audio, OSD, HUD, explosion particles — presentation, not simulation
- Multi-agent / swarm
- Sim-to-real transfer

## Risks

| Risk | Mitigation |
|---|---|
| Ported physics silently diverges from the JS | parity tests on hover throttle, terminal velocity, battery curve |
| Tile providers change URLs or rate-limit | procedural fallback keeps tests green; cache means one fetch per location |
| `rgb` sampling looks nothing like the Three.js render | it is an approximation by design; browser replay is the visual check, not the training signal |
| Y-up vs Z-up confusion | one conversion point, asserted in tests |
| Hunt proves unlearnable at high difficulty | the curriculum + `ProgressReward` combination is already proven in rotorenv for exactly this failure |
