#!/usr/bin/env python3
"""
Plot BigVGAN GTA curves by PARSING THE TEXT LOG (not the giant tensorboard file).

The .log already contains everything we need, written by train.py:
  training:   "Steps: <n>, Gen Loss Total: <x>, Mel Error: <x>, s/b: <x> lr: <x> grad_norm_g: <x>"
  validation: "VALIDATION step <n> <mode>: mel_spec_error=<x> pesq=<x> mrstft=<x>"
Parsing the few-hundred-KB log is instant; the 6 GB tfevents file (mostly audio/
images) is irrelevant for curves. Reads ALL logs/bigvgan_gta_*.log and merges by step.

    python scripts/plot_bigvgan_gta.py /data/.../exp/bigvganGTA/<run_name>

Signals: training mel_error / gen_loss (down/settle), grad_norm_g (spikes = noisy),
validation mel_spec_error (down), pesq (UP), mrstft (down).
"""

import os
import re
import sys
import glob
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

if len(sys.argv) < 2:
    print("Usage: python scripts/plot_bigvgan_gta.py <run_dir>")
    sys.exit(1)

run_dir = sys.argv[1].rstrip("/")
log_files = sorted(glob.glob(os.path.join(run_dir, "logs", "bigvgan_gta_*.log")))
assert log_files, f"no bigvgan_gta_*.log in {run_dir}/logs"

TRAIN_RE = re.compile(
    r"Steps:\s*(\d+),\s*Gen Loss Total:\s*([-\d.]+),\s*Mel Error:\s*([-\d.]+),"
    r"\s*s/b:\s*[-\d.]+\s*lr:\s*([-\d.eE+]+)\s*grad_norm_g:\s*([-\d.]+)")
VAL_RE = re.compile(
    r"VALIDATION step\s*(\d+)\s+(\S+):\s*mel_spec_error=([-\d.]+)"
    r"\s*pesq=([-\d.]+)\s*mrstft=([-\d.]+)")

series = {}  # tag -> list[(step, value)]
def add(tag, step, val):
    series.setdefault(tag, []).append((int(step), float(val)))

for lf in log_files:
    with open(lf, errors="ignore") as f:
        for line in f:
            m = TRAIN_RE.search(line)
            if m:
                step, gen, mel, lr, gnorm = m.groups()
                add("training/gen_loss_total", step, gen)
                add("training/mel_error", step, mel)
                add("training/grad_norm_g", step, gnorm)
                add("training/lr", step, lr)
                continue
            m = VAL_RE.search(line)
            if m:
                step, mode, mse, pesq, mrstft = m.groups()
                mode = mode.rstrip(":")
                add(f"validation_{mode}/mel_spec_error", step, mse)
                add(f"validation_{mode}/pesq", step, pesq)
                add(f"validation_{mode}/mrstft", step, mrstft)

if not series:
    print("no parseable training/validation lines found in the logs.")
    sys.exit(0)

out_dir = os.path.join(run_dir, "curves")
os.makedirs(out_dir, exist_ok=True)
plt.style.use("seaborn-v0_8-darkgrid")
matplotlib.rcParams.update({
    "figure.facecolor": "#1e1e2e", "axes.facecolor": "#1e1e2e",
    "axes.edgecolor": "#cdd6f4", "axes.labelcolor": "#cdd6f4",
    "text.color": "#cdd6f4", "xtick.color": "#cdd6f4", "ytick.color": "#cdd6f4",
    "grid.color": "#45475a", "figure.dpi": 100, "font.size": 11,
})

print(f"parsed {len(log_files)} log file(s); {len(series)} tags:")
for tag in sorted(series):
    pts = sorted(series[tag])
    steps = [s for s, _ in pts]
    vals = [v for _, v in pts]
    color = ("#f38ba8" if tag.startswith("training") else
             "#89b4fa" if "validation" in tag else "#a6e3a1")
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(steps, vals, color=color, linewidth=2,
            marker=("o" if len(vals) <= 30 else None), markersize=4)
    ax.set_title(f"{tag}   (last={vals[-1]:.5f} @ step {steps[-1]})",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Step")
    fig.savefig(os.path.join(out_dir, tag.replace("/", "__") + ".png"),
                bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {tag:38s} last={vals[-1]:.5f}  n={len(vals)}")

print(f"\nwrote {len(series)} plots -> {out_dir}/")
