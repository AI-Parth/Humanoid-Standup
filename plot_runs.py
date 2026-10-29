"""Compare learning curves of every run in runs/: eval return, mean torso height, fraction standing.

  python plot_runs.py            # -> runs/comparison.png and a summary table
"""
import glob
import os

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import STAND_Z


def main():
    runs = {os.path.basename(os.path.dirname(p)): pd.read_csv(p) for p in sorted(glob.glob("runs/*/eval.csv"))}
    runs = {k: v for k, v in runs.items() if len(v)}
    for df in runs.values():
        if "stock_return" in df:  # compare on the benchmark reward, without any posture bonus
            df["return"] = df["stock_return"]
    if not runs:
        print("no runs/*/eval.csv yet")
        return
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for name, df in runs.items():
        for ax, col in zip(axes, ["return", "mean_height", "frac_standing"]):
            ax.plot(df["step"], df[col], label=name)
    axes[0].set_ylabel("eval return")
    axes[1].set_ylabel("mean torso height (m)")
    axes[1].axhline(STAND_Z, ls="--", c="k", lw=0.8)
    axes[2].set_ylabel(f"fraction of steps with torso > {STAND_Z} m")
    for ax in axes:
        ax.set_xlabel("env steps")
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig("runs/comparison.png", dpi=120)

    rows = []
    for name, df in runs.items():
        best = df.loc[df["return"].idxmax()]
        last = df.iloc[-1]
        rows.append({"run": name, "steps": int(last["step"]), "last_return": last["return"],
                     "last_mean_z": last["mean_height"], "last_standing": last["frac_standing"],
                     "last_tilt": last.get("final_tilt", float("nan")),
                     "best_return": best["return"], "best_step": int(best["step"])})
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda x: f"{x:,.2f}"))
    print("saved runs/comparison.png")


if __name__ == "__main__":
    main()
