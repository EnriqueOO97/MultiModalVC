"""
Word error rate across a noise sweep, audio-visual vs audio-only.

The audio-only curve saturates: values above 1.0 mean the transcription has
failed, so the axis is capped at 1.0 and the true values are annotated at the
ceiling instead of stretching the axis to 2.2 and flattening everything else.

Outputs a vector PDF (for the poster) and a PNG (for quick checking).
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.font_manager import findfont, FontProperties

# ---------------------------------------------------------------- output dir
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "outputsForPoster")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------------------- typography
# True Times New Roman (.ttf/.otf) is a proprietary Microsoft font, not bundled with any
# Linux distro or pip package. If you have the file, pass its path via TNR_PATH (env var)
# and it will be registered and used; otherwise this falls back to Liberation Serif, which
# is metric-compatible with Times New Roman (identical glyph widths/kerning -> no reflow),
# though not pixel-identical in letterform.
TNR_PATH = os.environ.get("TNR_PATH", "")
if TNR_PATH and os.path.isfile(TNR_PATH):
    font_manager.fontManager.addfont(TNR_PATH)
    tnr_name = font_manager.FontProperties(fname=TNR_PATH).get_name()
    SERIF_STACK = [tnr_name, "Times New Roman", "Nimbus Roman", "Liberation Serif", "DejaVu Serif"]
else:
    installed_names = {f.name for f in font_manager.fontManager.ttflist}
    if "Times New Roman" not in installed_names:
        print(
            "warning: 'Times New Roman' not installed on this system (set env TNR_PATH to "
            "its .ttf/.otf file to use the real font). Falling back to Liberation Serif "
            "(metric-compatible clone)."
        )
    SERIF_STACK = ["Times New Roman", "Nimbus Roman", "Liberation Serif", "DejaVu Serif"]

# Debug override: force an arbitrary installed font by name, e.g.
#   FONT_OVERRIDE="Droid Sans Mono" python plotsForPoster.py
FONT_OVERRIDE = os.environ.get("FONT_OVERRIDE", "")
if FONT_OVERRIDE:
    SERIF_STACK = [FONT_OVERRIDE] + SERIF_STACK

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": SERIF_STACK,
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "axes.linewidth": 0.8,
    "pdf.fonttype": 42,   # embed real fonts, not outlines - keeps text selectable
    "ps.fonttype": 42,
})

resolved = findfont(FontProperties(family="serif"))
print(f"font in use: {resolved}")

# --------------------------------------------------------------------- data
# "Original" = no noise added. NOT "clean"/"normal": the speech is still pathological,
# it simply has no babble mixed in.
labels = ["Original", "+15 dB", "+10 dB", "+5 dB", "0 dB", "-5 dB"]
x = range(len(labels))

av = [0.180, 0.213, 0.228, 0.279, 0.427, 0.644]
ao = [0.228, 0.304, 0.343, 0.455, 0.978, 1.883]
path = [0.3135, 0.3606, 0.3643, 0.4449, 0.7977, 1.2772]   # pathological speech

CAP = 1.0

COL_AV = "#2A6F7F"
COL_AO = "#C0392B"
COL_PATH = "0.45"


def plot_with_clipping(ax, x, values, color, label, marker_zorder=3, label_row=0):
    drawn = [min(v, CAP) for v in values]
    saturated = [i for i, v in enumerate(values) if v > CAP]
    unsaturated = [i for i in x if i not in saturated]

    ax.plot(x, drawn, color=color, lw=1.8, zorder=marker_zorder,
            clip_on=False, label=label)
    ax.plot([i for i in unsaturated], [drawn[i] for i in unsaturated],
            color=color, lw=0, marker="o", ms=4.5, zorder=marker_zorder,
            clip_on=False)

    for i in saturated:
        ax.plot(i, CAP, marker="x", ms=9, mew=2.2, color=color,
                zorder=marker_zorder + 7, clip_on=False)


# --------------------------------------------------------------------- plot
fig, ax = plt.subplots(figsize=(5.2, 2.4))

plot_with_clipping(ax, x, path, COL_PATH, "Pathological speech", marker_zorder=2, label_row=1)
plot_with_clipping(ax, x, av, COL_AV, "Audio-visual", marker_zorder=3, label_row=0)
plot_with_clipping(ax, x, ao, COL_AO, "Audio-only", marker_zorder=3, label_row=0)

# ---------------------------------------------------------------- axes/limits
ax.set_xticks(list(x))
ax.set_xticklabels(labels)
ax.set_xlim(-0.12, len(labels) - 0.88)   # kills the dead space on the right
ax.set_ylim(0, CAP)

ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
ax.set_ylabel("WER", rotation=90)

ax.grid(axis="y", color="0.9", lw=0.7, zorder=0)
ax.set_axisbelow(True)
for side in ("top", "right"):
    ax.spines[side].set_visible(False)

ax.legend(frameon=False, loc="upper left", handlelength=1.6)

fig.tight_layout(pad=0.4)
fig.savefig(os.path.join(OUTPUT_DIR, "wer_noise_sweep.pdf"), bbox_inches="tight")
fig.savefig(os.path.join(OUTPUT_DIR, "wer_noise_sweep.png"), dpi=300, bbox_inches="tight")
print(f"written: {OUTPUT_DIR}/wer_noise_sweep.pdf / .png")
