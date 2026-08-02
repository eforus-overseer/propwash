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
