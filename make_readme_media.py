"""Render the README media: per part, a start / best / final GIF strip and a learning-curve plot.

  python make_readme_media.py            # -> media/<part>.gif, media/<part>_curve.png, media/summary.json

Only best.pt and last.pt are saved per run, so the three panels are: start (uniform-random actions,
which is what every from-scratch run does for its first 10k steps, or the checkpoint a continuation
run started from), the best-eval checkpoint, and the final checkpoint.
"""
import json
import os

import matplotlib
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw, ImageFont

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import STAND_Z, make_env, posture_stats, run_episode
from evaluate import load_policy

W, H, BAR = 240, 200, 22  # panel size and caption bar height (px)
EVERY = 8  # keep every 8th sim step: 0.12 s per GIF frame, played back in real time
CAMERA = {"trackbodyid": 1, "distance": 2.6, "elevation": -10.0, "azimuth": 120.0, "lookat": np.array([0, 0, 0.7])}
SEED = 0

# (part, title, panels, curve runs). A panel is (label, checkpoint path or "random").
# Curve runs are plotted left to right; earlier ones (the run this part continued from) in grey.
PARTS = [
    ("ppo", "PPO", [("start", "random"), ("best", "runs/ppo_s1/best.pt"), ("final", "runs/ppo_s1/last.pt")],
     ["ppo_s1"]),
    ("rpo_sym", "RPO + symmetry loss",
     [("start", "random"), ("best", "runs/rpo_sym_s1/best.pt"), ("final", "runs/rpo_sym_s1/last.pt")],
     ["rpo_sym_s1"]),
    ("sac", "SAC", [("start", "random"), ("best", "runs/sac_s1/best.pt"), ("final", "runs/sac_s1/last.pt")],
     ["sac_s1"]),
    ("td3", "TD3", [("start", "random"), ("best", "runs/td3_s1/best.pt"), ("final", "runs/td3_s1/last.pt")],
     ["td3_s1"]),
    ("bac", "BAC", [("start", "random"), ("best", "runs/bac_s1/best.pt"), ("final", "runs/bac_s1/last.pt")],
     ["bac_s1"]),
    ("crossq", "CrossQ (1M steps)", [("start", "random"), ("final", "runs/crossq_s1/last.pt")], ["crossq_s1"]),
    ("crossq_long", "CrossQ continued to 2.5M (no posture reward)",
     [("start", "runs/crossq_s1/last.pt"), ("best", "runs/crossq_long_s1/best.pt"),
      ("final", "runs/crossq_long_s1/last.pt")], ["crossq_s1", "crossq_long_s1"]),
    ("crossq_posture", "CrossQ + posture reward",
     [("start", "runs/crossq_s1/last.pt"), ("best", "runs/crossq_posture_s1/best.pt"),
      ("final", "runs/crossq_posture_s1/last.pt")], ["crossq_s1", "crossq_posture_s1"]),
    ("crossq_still", "CrossQ + posture + stillness",
     [("start", "runs/crossq_posture_s1/last.pt"), ("best", "runs/crossq_still_s1/best.pt"),
      ("final", "runs/crossq_still_s1/last.pt")], ["crossq_s1", "crossq_posture_s1", "crossq_still_s1"]),
]


def font(size):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # Pillow < 10.1
        return ImageFont.load_default()


def step_of(path):
    return 0 if path == "random" else torch.load(path, map_location="cpu", weights_only=False)["step"]


def render_panel(env, label, path):
    if path == "random":
        rng = np.random.default_rng(SEED)
        act = lambda obs: rng.uniform(-1, 1, env.action_space.shape).astype(np.float32)
    else:
        act = load_policy(path, env)
    frames = []
    ret, stock_ret, heights, tilts = run_episode(env, act, seed=SEED, frames=frames)
    stats = {"stock_return": stock_ret, **posture_stats(heights, tilts), "step": step_of(path)}
    caption = f"{label} @ {stats['step'] / 1e6:.2f}M  |  standing {stats['frac_standing']:.0%}"
    out = []
    for f in frames[::EVERY]:
        im = Image.new("RGB", (W, H + BAR), "white")
        im.paste(Image.fromarray(f), (0, BAR))
        ImageDraw.Draw(im).text((6, 4), caption, fill="black", font=font(13))
        out.append(im)
    return out, stats


def save_gif(panels, path):
    n = min(len(p) for p in panels)
    frames = []
    for i in range(n):
        strip = Image.new("RGB", (W * len(panels) + 4 * (len(panels) - 1), H + BAR), "white")
        for j, p in enumerate(panels):
            strip.paste(p[i], (j * (W + 4), 0))
        frames.append(strip.quantize(colors=64, method=Image.Quantize.MEDIANCUT))
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=int(EVERY * 15), loop=0, optimize=True)


def save_curve(runs, marks, title, path):
    """Eval curves of the run (blue) and the runs it continued from (grey); the GIF's checkpoints are
    marked on the curve."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.2))
    dfs = [pd.read_csv(f"runs/{run}/eval.csv") for run in runs]
    for k, (run, df) in enumerate(zip(runs, dfs)):
        style = dict(lw=1.6, color="C0") if k == len(runs) - 1 else dict(lw=1.2, color="0.6")
        axes[0].plot(df["step"] / 1e6, df["mean_height"], **style, label=run)
        axes[1].plot(df["step"] / 1e6, df["frac_standing"], **style, label=run)
    evals = pd.concat(dfs).drop_duplicates("step", keep="last").set_index("step")
    for (label, step), marker in zip(marks, ["o", "D", "s"]):
        if step in evals.index:
            for ax, col in zip(axes, ["mean_height", "frac_standing"]):
                ax.plot(step / 1e6, evals.loc[step, col], marker, ms=8, mfc="none", mew=1.8, color="C3",
                        label=f"{label} ({step / 1e6:.2f}M)")
    for ax in axes:
        ax.set_xlabel("env steps (M)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7, loc="lower right")
    axes[0].axhline(STAND_Z, ls="--", c="k", lw=0.8)
    axes[0].set_ylabel("mean torso height (m)")
    axes[1].set_ylabel(f"fraction of steps standing")
    axes[1].set_ylim(-0.03, 1.03)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main():
    os.makedirs("media", exist_ok=True)
    env = make_env(render_mode="rgb_array", width=W, height=H, default_camera_config=CAMERA)
    summary = {}
    for part, title, panels, runs in PARTS:
        rendered, stats = [], []
        for label, path in panels:
            frames, s = render_panel(env, label, path)
            rendered.append(frames)
            stats.append({"label": label, "checkpoint": path, **s})
        save_gif(rendered, f"media/{part}.gif")
        save_curve(runs, [(s["label"], s["step"]) for s in stats if s["step"] > 0], title, f"media/{part}_curve.png")
        summary[part] = {"title": title, "panels": stats}
        print(f"{part}: " + "  ".join(f"{s['label']}@{s['step']:,} stand={s['frac_standing']:.0%}" for s in stats),
              f"({os.path.getsize(f'media/{part}.gif') / 1e6:.1f} MB)", flush=True)
    with open("media/summary.json", "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
