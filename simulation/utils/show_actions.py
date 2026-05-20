"""
Print example action chunks from the scripted policy.

Usage:
  python show_actions.py
  python show_actions.py --seed 8 --chunks 3
"""

import sys
import os
import argparse
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from env.pusht_env import PushTEnv


class ScriptedPolicy:
    def __init__(self, speed=6.0, noise_std=0.2):
        self.speed = speed
        self.noise_std = noise_std

    def get_actions(self, env, chunk_size=32):
        actions = []
        ee = np.array([env.ee_x, env.ee_y])
        block = np.array([env.block_x, env.block_y])
        target = np.array([env.target_x, env.target_y])
        for _ in range(chunk_size):
            b2t = target - block
            dist = np.linalg.norm(b2t)
            if dist < 1.0:
                action = np.random.randn(2) * 0.5
            else:
                push_dir = b2t / dist
                approach = block - push_dir * 40
                if np.linalg.norm(approach - ee) > 20:
                    d = approach - ee
                    action = d / max(np.linalg.norm(d), 1e-6) * self.speed
                else:
                    action = push_dir * self.speed
            action += np.random.randn(2) * self.noise_std
            actions.append(action)
            ee = np.clip(ee + action, 15, 497)
            if np.linalg.norm(ee - block) < 45:
                block += action * 0.15
        return np.array(actions, dtype=np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=8)
    parser.add_argument("--chunks", type=int, default=3, help="Number of inference chunks to show")
    args = parser.parse_args()

    np.random.seed(args.seed)
    env = PushTEnv(seed=args.seed)
    env.reset()
    policy = ScriptedPolicy()

    print(f"EE start:    ({env.ee_x:.0f}, {env.ee_y:.0f})")
    print(f"Block start: ({env.block_x:.0f}, {env.block_y:.0f})")
    print(f"Target:      ({env.target_x:.0f}, {env.target_y:.0f})")

    for chunk in range(args.chunks):
        actions = policy.get_actions(env, 32)
        print(f"\n--- Inference #{chunk+1} ---")
        print(f"Actions [32 x (dx, dy)]:")
        for i, a in enumerate(actions):
            print(f"  step {i+1:2d}: ({a[0]:+6.2f}, {a[1]:+6.2f})")
            env.step(a)
        print(f"EE now:  ({env.ee_x:.0f}, {env.ee_y:.0f})")
        print(f"Block:   ({env.block_x:.0f}, {env.block_y:.0f})")


if __name__ == "__main__":
    main()
