"""Measure how well a trained policy stands, and optionally record a video.

  python evaluate.py runs/bac_s1/best.pt --episodes 10 --video
  python evaluate.py random            # baselines: "random" or "zero" actions

Writes <ckpt dir>/eval_<name>/: per-episode metrics (episodes.csv), torso-height traces (heights.png)
and video.mp4 of the first episode.
"""
import argparse
import csv
import os

import numpy as np
import torch

from common import SIT_Z, STAND_Z, make_env, posture_label, posture_stats, run_episode


def load_policy(path, env, device="cpu"):
    """Return act_fn(raw_obs) -> action for a checkpoint, or a baseline policy."""
    if path == "random":
        return lambda obs: env.action_space.sample()
    if path == "zero":
        return lambda obs: np.zeros(env.action_space.shape, np.float32)
    ckpt = torch.load(path, map_location=device, weights_only=False)
    obs_dim, act_dim = env.observation_space.shape[0], env.action_space.shape[0]
    args = argparse.Namespace(**ckpt["args"])
    if ckpt["algo"] == "ppo":
        from ppo import Agent

        net = Agent(obs_dim, act_dim, args.hidden)
        net.load_state_dict(ckpt["agent"])
        policy = net.actor_mean
    else:
        from offpolicy import build_actor

        net = build_actor(args, obs_dim, act_dim)
        net.load_state_dict(ckpt["actor"])
        policy = lambda x: net(x, deterministic=True)[0]
    net.eval()
    rms = ckpt.get("obs_rms")
    mean, std = (rms["mean"], np.sqrt(rms["var"] + 1e-8)) if rms else (0.0, 1.0)

    @torch.no_grad()
    def act(obs):
        x = np.clip((obs - mean) / std, -10, 10) if rms else obs
        return policy(torch.as_tensor(x, dtype=torch.float32)[None])[0].numpy()

    print(f"loaded {ckpt['algo']} checkpoint from step {ckpt['step']:,}")
    return act


def main():
    p = argparse.ArgumentParser()
    p.add_argument("checkpoint", help="path to a .pt file, or 'random' / 'zero'")
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--video", action="store_true")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    if args.checkpoint in ("random", "zero"):
        out = args.out or os.path.join("runs", f"baseline_{args.checkpoint}")
    else:
        name = os.path.splitext(os.path.basename(args.checkpoint))[0]
        out = args.out or os.path.join(os.path.dirname(args.checkpoint), f"eval_{name}")
    os.makedirs(out, exist_ok=True)

    env = make_env(render_mode="rgb_array" if args.video else None, width=640, height=480)
    act = load_policy(args.checkpoint, env)

    rows, traces = [], []
    for ep in range(args.episodes):
        frames = [] if (args.video and ep == 0) else None
        ret, _, heights, tilts = run_episode(env, act, seed=args.seed + ep, frames=frames)
        s = posture_stats(heights, tilts)
        rows.append({"episode": ep, "return": ret, **s, "posture": posture_label(s["final_height"])})
        traces.append(heights)
        print(f"ep {ep}: return {ret:>9,.0f}  mean z {s['mean_height']:.2f}  final z {s['final_height']:.2f}  "
              f"max z {s['max_height']:.2f}  standing {s['frac_standing']:>4.0%}  tilt {s['final_tilt']:>3.0f}°  "
              f"first stand @ {s['first_stand_step']:>4}  -> {rows[-1]['posture']}")
        if frames:
            import imageio

            imageio.mimsave(os.path.join(out, "video.mp4"), frames, fps=int(round(1 / env.unwrapped.dt)))

    with open(os.path.join(out, "episodes.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    keys = ["return", "mean_height", "final_height", "max_height", "frac_standing", "final_tilt"]
    print("\nmean over episodes: " + "  ".join(f"{k} {np.mean([r[k] for r in rows]):,.3f}" for k in keys))
    labels = [r["posture"] for r in rows]
    print("final posture: " + ", ".join(f"{l} {labels.count(l)}/{len(labels)}" for l in sorted(set(labels))))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 4))
    t = np.arange(len(traces[0])) * env.unwrapped.dt
    for h in traces:
        ax.plot(t, h, lw=1, alpha=0.7)
    ax.axhline(STAND_Z, ls="--", c="k", lw=0.8)
    ax.axhline(SIT_Z, ls=":", c="k", lw=0.8)
    ax.set(xlabel="time (s)", ylabel="torso height (m)", title=f"{args.checkpoint}: torso height per episode")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "heights.png"), dpi=120)
    print(f"saved results to {out}")


if __name__ == "__main__":
    main()
