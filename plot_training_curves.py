#!/usr/bin/env python3
"""
Plot training curves from TensorBoard logs and save them as images.

Usage:
    python plot_training_curves.py /path/to/training/folder

Two subfolders will be created inside the training folder:
  - cell1/  →  one plot per tag (all runs overlaid), named {tag}.png
  - cell2/  →  detailed individual plots for key metrics, named {tag}.png
"""

import os
import sys
import json as _json
import re as _re
import matplotlib
matplotlib.use('Agg')  # non-interactive backend — no display needed
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


# ============================================================
# Parse training folder from command line
# ============================================================
if len(sys.argv) < 2:
    print("Usage: python plot_training_curves.py <training_folder>")
    sys.exit(1)

tF = sys.argv[1].rstrip("/")
TB_DIR = os.path.join(tF, "tensorboard")
assert os.path.isdir(TB_DIR), f"TensorBoard directory not found: {TB_DIR}"

# Create output folders
cell1_dir = os.path.join(tF, "cell1")
cell2_dir = os.path.join(tF, "cell2")
os.makedirs(cell1_dir, exist_ok=True)
os.makedirs(cell2_dir, exist_ok=True)

# ── Style ──────────────────────────────────────────────────────
plt.style.use('seaborn-v0_8-darkgrid')
matplotlib.rcParams.update({
    'figure.facecolor': '#1e1e2e',
    'axes.facecolor':   '#1e1e2e',
    'axes.edgecolor':   '#cdd6f4',
    'axes.labelcolor':  '#cdd6f4',
    'text.color':       '#cdd6f4',
    'xtick.color':      '#cdd6f4',
    'ytick.color':      '#cdd6f4',
    'grid.color':       '#45475a',
    'figure.dpi':       100,
    'font.size':        11,
    'legend.framealpha': 0.3,
})

RUN_COLORS = {
    'train':       '#f38ba8',
    'train_inner': '#fab387',
    'valid':       '#89b4fa',
}

# ── Smoothing utility ────────────────────────────────────────
def smooth(values, weight=0.6):
    """Exponential moving average smoothing (same as TensorBoard)."""
    smoothed = []
    last = values[0] if values else 0
    for v in values:
        s = last * weight + (1 - weight) * v
        smoothed.append(s)
        last = s
    return smoothed

# ── Load all runs ─────────────────────────────────────────────
runs = {}  # run_name -> {tag: (steps, values)}
for run_name in sorted(os.listdir(TB_DIR)):
    run_path = os.path.join(TB_DIR, run_name)
    if not os.path.isdir(run_path):
        continue
    ea = EventAccumulator(run_path)
    ea.Reload()
    scalar_tags = ea.Tags().get('scalars', [])
    if not scalar_tags:
        print(f"  ⚠ No scalar tags found for run '{run_name}'")
        continue
    run_data = {}
    for tag in scalar_tags:
        events = ea.Scalars(tag)
        steps  = [e.step for e in events]
        values = [e.value for e in events]
        run_data[tag] = (steps, values)
    runs[run_name] = run_data
    print(f"  ✓ Loaded run '{run_name}' with {len(scalar_tags)} tags: {scalar_tags}")

# ── Collect all unique tags ──────────────────────────────────
all_tags = set()
for run_data in runs.values():
    all_tags.update(run_data.keys())
all_tags = sorted(all_tags)
print(f"\n📊 Total unique tags: {len(all_tags)}")
print(f"   Tags: {all_tags}")

# ── Parse hydra_train.log for step → epoch mapping ──────────
_log_path = os.path.join(tF, 'hydra_train.log')
_step_epoch_pairs = []  # list of (num_updates, epoch)
if os.path.isfile(_log_path):
    with open(_log_path, 'r') as _f:
        for _line in _f:
            if 'num_updates' in _line and '"epoch"' in _line:
                _m = _re.search(r'\{.*\}', _line)
                if _m:
                    try:
                        _d = _json.loads(_m.group())
                        _step_epoch_pairs.append((int(_d['num_updates']), int(_d['epoch'])))
                    except Exception:
                        pass
    print(f'\n📅 Parsed {len(_step_epoch_pairs)} step→epoch entries from hydra_train.log')
else:
    print(f'\n⚠ hydra_train.log not found at {_log_path}')

# Build interpolation arrays
_step_arr  = np.array([p[0] for p in _step_epoch_pairs], dtype=float) if _step_epoch_pairs else np.array([0.0, 1.0])
_epoch_arr = np.array([p[1] for p in _step_epoch_pairs], dtype=float) if _step_epoch_pairs else np.array([0.0, 0.0])

def step_to_epoch(steps):
    return np.interp(steps, _step_arr, _epoch_arr)

def epoch_to_step(epochs):
    _ue2, _idx2 = np.unique(_epoch_arr[::-1], return_index=True)
    _idx_last = len(_epoch_arr) - 1 - _idx2
    return np.interp(epochs, _epoch_arr[sorted(_idx_last)], _step_arr[sorted(_idx_last)])

# ── Tag ordering (priority first) ────────────────────────────
PRIORITY_TAGS = ['loss', 'best_loss', 'mcd', 'ssim', 'lr']
priority_present = [t for t in PRIORITY_TAGS if t in all_tags]
other_tags = [t for t in all_tags if t not in PRIORITY_TAGS]
ordered_tags = priority_present + other_tags

# ==============================================================
# CELL 1:  One plot per tag (same style as the stacked subplots)
# ==============================================================
print(f"\n🖼  Saving cell1 plots to {cell1_dir}/")
for tag in ordered_tags:
    fig, ax = plt.subplots(figsize=(7, 5))
    for run_name, run_data in runs.items():
        if tag not in run_data:
            continue
        steps, values = run_data[tag]
        color = RUN_COLORS.get(run_name, '#a6e3a1')
        ax.plot(steps, values, alpha=0.9, color=color,
                linewidth=2, label=run_name, marker=('o' if len(values) == 1 else None))
    ax.set_title(tag, fontsize=14, fontweight='bold', color='#cdd6f4')
    ax.set_xlabel('Step')
    ax.legend(loc='best', fontsize=9)
    plt.tight_layout()
    out_path = os.path.join(cell1_dir, f"{tag}.png")
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)
    print(f"    ✓ {tag}.png")

# ==============================================================
# CELL 2:  Detailed individual plots for key metrics
# ==============================================================
KEY_METRICS = ['loss', 'mcd', 'ssim', 'best_loss', 'lr']
key_present = [t for t in KEY_METRICS if t in all_tags]

print(f"\n🖼  Saving cell2 plots to {cell2_dir}/")
for tag in key_present:
    fig, ax = plt.subplots(figsize=(14, 5))
    for run_name, run_data in runs.items():
        if tag not in run_data:
            continue
        steps, values = run_data[tag]
        color = RUN_COLORS.get(run_name, '#a6e3a1')
        ax.plot(steps, values, alpha=0.9, color=color,
                linewidth=2.5, label=f"{run_name} (last={values[-1]:.4f})",
                marker=('o' if len(values) == 1 else None))
    ax.set_title(tag.upper(), fontsize=16, fontweight='bold', color='#f5c2e7')
    ax.set_xlabel('Step', fontsize=12)
    ax.set_ylabel(tag, fontsize=12)
    ax.legend(loc='best', fontsize=11)

    # ── Secondary x-axis: Epoch ────────────────────────────────
    if len(_step_epoch_pairs) > 0:
        secax = ax.secondary_xaxis(-0.18, functions=(step_to_epoch, epoch_to_step))
        secax.set_xlabel('Epoch', fontsize=12, color='#f9e2af')
        secax.tick_params(colors='#f9e2af', labelsize=10)
        secax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=10))

    plt.tight_layout()
    out_path = os.path.join(cell2_dir, f"{tag}.png")
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)
    print(f"    ✓ {tag}.png")

print("\n✅ Done!")
