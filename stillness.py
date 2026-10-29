"""Stand *still*: an extra reward that penalises wandering, swaying and twitching once standing.

The stock reward only pays for torso height (~410/step when upright) and charges ~0.1/step for
torques, so a standing policy is free to shuffle around the floor and jitter its joints. This adds,
only while the torso is above ~0.9-1.0 m (so getting up is not penalised):

    penalty = gate * (v_coef * |torso xy speed| + qvel_coef * mean(joint speed^2)
                      + da_coef * mean((a_t - a_{t-1})^2))

With the defaults, the motion of crossq_posture_s1 (0.2/step action change, 1.3 rad/s joint speed)
costs roughly 40-50/step, about a tenth of the height reward: standing still pays, but standing still
never beats standing up.

  # continue the posture run with the stillness penalty (trainer args are passed to offpolicy.py)
  python stillness.py --algo crossq --total-steps 3500000 --init-from runs/crossq_posture_s1/last.pt \
      --posture-coef 150 --run-name crossq_still_s1

  # how much does a checkpoint move while standing?
  python stillness.py measure runs/crossq_still_s1/best.pt runs/crossq_posture_s1/best.pt
"""
import argparse
import ctypes
import json
import os
import sys

import gymnasium as gym
import numpy as np

from common import STAND_Z, make_env


class StillnessReward(gym.Wrapper):
    """Subtracts the stillness penalty above; wraps an env whose actions are already in [-1, 1].
    The stock reward in info["stock_reward"] (set by common.PostureReward) is left untouched."""

    def __init__(self, env, v_coef=50.0, qvel_coef=10.0, da_coef=200.0):
        super().__init__(env)
        self.v_coef, self.qvel_coef, self.da_coef = v_coef, qvel_coef, da_coef
        self.prev_action = None

    def reset(self, **kwargs):
        self.prev_action = None
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, term, trunc, info = self.env.step(action)
        d = self.env.unwrapped.data
        gate = np.clip((d.qpos[2] - (STAND_Z - 0.1)) / 0.1, 0, 1)  # 0 below 0.9 m, 1 above 1.0 m
        speed = float(np.linalg.norm(d.qvel[:2]))  # root (torso) horizontal speed, m/s
        joint_sq = float(np.mean(d.qvel[6:] ** 2))  # joint speeds, excluding the free root joint
        da_sq = 0.0 if self.prev_action is None else float(np.mean((action - self.prev_action) ** 2))
        self.prev_action = np.array(action, copy=True)
        info.update(xy_speed=speed, joint_speed_sq=joint_sq, action_change_sq=da_sq)
        penalty = gate * (self.v_coef * speed + self.qvel_coef * joint_sq + self.da_coef * da_sq)
        return obs, reward - penalty, term, trunc, info


def make_still_env(v_coef, qvel_coef, da_coef):
    def factory(render_mode=None, posture_coef=0.0, **kwargs):
        return StillnessReward(make_env(render_mode, posture_coef, **kwargs), v_coef, qvel_coef, da_coef)

    return factory


def train():
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--v-coef", type=float, default=50.0, help="per m/s of torso horizontal speed")
    p.add_argument("--qvel-coef", type=float, default=10.0, help="per (rad/s)^2 of mean joint speed")
    p.add_argument("--da-coef", type=float, default=200.0, help="per unit of mean squared action change")
    p.add_argument("--run-name", default=None)
    still, rest = p.parse_known_args()

    import offpolicy

    # offpolicy.main() builds its envs through the module-level name make_env; point it at ours.
    offpolicy.make_env = make_still_env(still.v_coef, still.qvel_coef, still.da_coef)
    sys.argv = [sys.argv[0]] + rest + (["--run-name", still.run_name] if still.run_name else [])
    if still.run_name:
        os.makedirs(os.path.join("runs", still.run_name), exist_ok=True)
        with open(os.path.join("runs", still.run_name, "stillness.json"), "w") as f:
            json.dump({k: v for k, v in vars(still).items() if k != "run_name"}, f, indent=2)
    if os.name == "nt":
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)  # no idle sleep
    offpolicy.main()


def measure(paths, episodes=3):
    """Motion while standing (torso > 1 m, after step 300): drift, joint speed, action jitter."""
    from evaluate import load_policy

    env = make_env()
    data = env.unwrapped.data
    for path in paths:
        act = load_policy(path, env)
        rows = []
        for ep in range(episodes):
            obs, _ = env.reset(seed=10_000 + ep)
            prev = None
            for t in range(1000):
                a = act(obs)
                obs, *_ = env.step(a)
                if t >= 300 and obs[0] > STAND_Z and prev is not None:
                    rows.append([np.linalg.norm(data.qvel[:2]), np.sqrt(np.mean(data.qvel[6:] ** 2)),
                                 np.abs(a - prev).mean(), (np.abs(a) > 0.95).mean(), obs[0]])
                prev = a
        if not rows:
            print(f"{path}: never standing after step 300")
            continue
        r = np.array(rows)
        print(f"{path}: standing {len(r) / (episodes * 699):.0%} of steps 300-1000")
        print(f"  torso xy speed      {r[:, 0].mean():.2f} m/s")
        print(f"  joint speed RMS     {r[:, 1].mean():.2f} rad/s")
        print(f"  action change/step  {r[:, 2].mean():.2f}")
        print(f"  saturated actions   {r[:, 3].mean():.0%}")
        print(f"  torso z std         {r[:, 4].std() * 100:.1f} cm")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "measure":
        measure(sys.argv[2:])
    else:
        train()
