"""Shared pieces for every trainer: env factory, running normalisation, posture metrics, logging."""
import csv
import json
import os
import random

import gymnasium as gym
import numpy as np
import torch

ENV_ID = "HumanoidStandup-v5"

# Posture thresholds on torso height (qpos[2], metres). Lying flat is ~0.1 m, sitting upright on the
# floor ~0.35-0.45 m (checked visually), fully upright ~1.4 m. Gymnasium PR #1767 used "torso above
# 1 m" as its standing criterion.
STAND_Z = 1.0
SIT_Z = 0.3


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class PostureReward(gym.Wrapper):
    """Reports spine tilt and head height in info, and optionally rewards an upright spine.

    The stock reward only counts torso height, so a stance bent ~30 deg at the hips scores almost the
    same as a straight one. The bonus is coef * gate * (1 - tilt / 90 deg), where tilt is the angle of
    the pelvis->head vector from vertical (forward and sideways lean alike) and gate ramps from 0 at
    sitting height to 1 at standing height, so sitting up straight earns little. With coef=150, going
    from a 30 deg lean to vertical is worth about as much as 15 cm of extra torso height (333/m).
    """

    def __init__(self, env, coef=0.0):
        import mujoco

        super().__init__(env)
        self.coef = coef
        m = env.unwrapped.model
        self.head = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "head")
        self.pelvis = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "pelvis")

    def step(self, action):
        obs, reward, term, trunc, info = self.env.step(action)
        d = self.env.unwrapped.data
        head = d.geom_xpos[self.head]
        spine = head - d.xpos[self.pelvis]
        tilt = float(np.degrees(np.arccos(np.clip(spine[2] / np.linalg.norm(spine), -1, 1))))
        gate = np.clip((d.qpos[2] - SIT_Z) / (STAND_Z - SIT_Z), 0, 1)
        info.update(stock_reward=reward, spine_tilt=tilt, head_height=float(head[2]))
        return obs, reward + self.coef * gate * max(0.0, 1 - tilt / 90), term, trunc, info


def make_env(render_mode=None, posture_coef=0.0, **kwargs):
    """HumanoidStandup-v5 with actions rescaled to [-1, 1], plus the PostureReward wrapper (a pure
    readout when posture_coef is 0).

    The env's native action box is [-0.4, 0.4]^17 (joint torques). Every policy here outputs in
    [-1, 1] (tanh-squashed or clipped Gaussian), so we rescale once here instead of in each algorithm.
    """
    env = gym.make(ENV_ID, render_mode=render_mode, **kwargs)
    env = PostureReward(env, posture_coef)
    env = gym.wrappers.RescaleAction(env, -1.0, 1.0)
    env = gym.wrappers.ClipAction(env)
    return env


def torso_height(obs):
    """v5 observations start at qpos[2] (x/y are excluded), so obs[..., 0] is torso height."""
    return obs[..., 0]


class RunningMeanStd:
    """Parallel-variance running mean/std (as in OpenAI baselines)."""

    def __init__(self, shape=(), epsilon=1e-4):
        self.mean = np.zeros(shape, np.float64)
        self.var = np.ones(shape, np.float64)
        self.count = epsilon

    def update(self, x):
        x = np.asarray(x, np.float64)
        batch_mean, batch_var, batch_count = x.mean(0), x.var(0), x.shape[0]
        delta = batch_mean - self.mean
        total = self.count + batch_count
        self.mean = self.mean + delta * batch_count / total
        m2 = self.var * self.count + batch_var * batch_count + delta**2 * self.count * batch_count / total
        self.var = m2 / total
        self.count = total

    def normalize(self, x, clip=10.0):
        return np.clip((x - self.mean) / np.sqrt(self.var + 1e-8), -clip, clip).astype(np.float32)

    def state_dict(self):
        return {"mean": self.mean, "var": self.var, "count": self.count}

    def load_state_dict(self, d):
        self.mean, self.var, self.count = d["mean"], d["var"], d["count"]


def posture_stats(heights, tilts=None):
    """Summarise one episode's torso-height (and optionally spine-tilt) trace into standing metrics."""
    h = np.asarray(heights)
    standing = h > STAND_Z
    stats = {
        "mean_height": float(h.mean()),
        "final_height": float(h[-100:].mean()),
        "max_height": float(h.max()),
        "frac_standing": float(standing.mean()),
        "frac_sitting": float(((h > SIT_Z) & ~standing).mean()),
        "first_stand_step": int(np.argmax(standing)) if standing.any() else -1,
    }
    if tilts is not None:
        stats["final_tilt"] = float(np.mean(tilts[-100:]))  # spine lean from vertical (deg), end of episode
    return stats


def posture_label(final_height):
    if final_height > STAND_Z:
        return "standing"
    if final_height > SIT_Z:
        return "sitting/kneeling"
    return "lying"


def run_episode(env, act_fn, seed=None, frames=None):
    """Roll out one episode; act_fn maps a raw observation to an action in [-1, 1].
    Returns (return, stock return without the posture bonus, torso heights, spine tilts)."""
    obs, _ = env.reset(seed=seed)
    ret, stock_ret, heights, tilts, done = 0.0, 0.0, [], [], False
    while not done:
        obs, r, term, trunc, info = env.step(act_fn(obs))
        ret += r
        stock_ret += info["stock_reward"]
        heights.append(float(torso_height(obs)))
        tilts.append(info["spine_tilt"])
        if frames is not None:
            frames.append(env.render())
        done = term or trunc
    return ret, stock_ret, np.array(heights), np.array(tilts)


def evaluate(env, act_fn, n_episodes, seed=10_000):
    """Average return and posture metrics over n_episodes with fixed reset seeds."""
    rows = []
    for i in range(n_episodes):
        ret, stock_ret, heights, tilts = run_episode(env, act_fn, seed=seed + i)
        rows.append({"return": ret, "stock_return": stock_ret, **posture_stats(heights, tilts)})
    return {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}


def mirror_action_pairs(env):
    """(right_idx, left_idx, sign) per left/right actuator pair, such that a_left = sign * a_right
    is the mirror-image torque. Signs come from the model's joint axes reflected through the sagittal
    plane, so they stay correct if the XML changes."""
    import mujoco

    model, data = env.unwrapped.model, env.unwrapped.data
    mujoco.mj_forward(model, data)
    body = lambda n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n)
    lateral = data.xpos[body("left_thigh")] - data.xpos[body("right_thigh")]
    n = lateral / np.linalg.norm(lateral)
    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(model.nu)]
    pairs = []
    for i, name in enumerate(names):
        if not name.startswith("right_"):
            continue
        j = names.index("left_" + name[len("right_"):])
        axis_r = data.xaxis[model.actuator_trnid[i, 0]]
        axis_l = data.xaxis[model.actuator_trnid[j, 0]]
        # A rotation axis is a pseudovector: reflecting through the plane with normal n gives -(v - 2(v.n)n).
        mirrored = -(axis_r - 2 * axis_r.dot(n) * n)
        cos = mirrored.dot(axis_l) / (np.linalg.norm(mirrored) * np.linalg.norm(axis_l))
        assert abs(cos) > 0.9, f"{name}: left/right axes are not mirror images (cos={cos:.2f})"
        pairs.append((i, j, float(np.sign(cos))))
    return pairs


class Logger:
    """Appends scalars to <run_dir>/<stream>.csv and TensorBoard. With resume_step set, existing CSVs
    are cut back to that step and continued instead of overwritten."""

    def __init__(self, run_dir, config, resume_step=None):
        from torch.utils.tensorboard import SummaryWriter

        os.makedirs(run_dir, exist_ok=True)
        self.run_dir, self.resume = run_dir, resume_step is not None
        if self.resume:
            for name in os.listdir(run_dir):
                if name.endswith(".csv"):
                    path = os.path.join(run_dir, name)
                    with open(path, newline="") as f:
                        rows = list(csv.reader(f))
                    with open(path, "w", newline="") as f:
                        csv.writer(f).writerows([rows[0]] + [r for r in rows[1:] if int(r[0]) <= resume_step])
        else:
            with open(os.path.join(run_dir, "config.json"), "w") as f:
                json.dump(config, f, indent=2)
        self.tb = SummaryWriter(run_dir)
        self.writers = {}

    def log(self, stream, step, **scalars):
        row = {"step": step, **{k: float(v) for k, v in scalars.items()}}
        if stream not in self.writers:
            path = os.path.join(self.run_dir, f"{stream}.csv")
            if self.resume and os.path.exists(path):
                with open(path, newline="") as f:
                    fieldnames = next(csv.reader(f))
                f = open(path, "a", newline="")
                # Metrics added since the run started (e.g. final_tilt) are dropped, keeping the CSV rectangular.
                w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            else:
                f = open(path, "w", newline="")
                w = csv.DictWriter(f, fieldnames=list(row))
                w.writeheader()
            self.writers[stream] = (f, w)
        f, w = self.writers[stream]
        w.writerow(row)
        f.flush()
        for k, v in scalars.items():
            self.tb.add_scalar(f"{stream}/{k}", v, step)

    def close(self):
        for f, _ in self.writers.values():
            f.close()
        self.tb.close()
