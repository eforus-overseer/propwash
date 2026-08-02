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
