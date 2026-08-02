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
