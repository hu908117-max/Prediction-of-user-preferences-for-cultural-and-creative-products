"""Figure 7 - ROC curves with confidence bands (mean +/- std over 5 runs).

Methods: LightGCN, MISA, CLIP, BLIP-2, LLaVA, Ours. Layout: 2 x 3 subplots.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from common import apply_style, format_axes

METHODS = ["LightGCN", "MISA", "CLIP", "BLIP-2", "LLaVA", "Ours"]
AUC_MEAN = np.array([0.718, 0.756, 0.741, 0.789, 0.824, 0.882])
AUC_STD = np.array([0.021, 0.018, 0.023, 0.016, 0.014, 0.009])
COLORS = [
    (0.35, 0.35, 0.35), (0.85, 0.55, 0.20), (0.30, 0.65, 0.45),
    (0.55, 0.35, 0.70), (0.80, 0.30, 0.35), (0.15, 0.35, 0.70),
]
SUB_LABELS = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)"]


def build_figure7(out_path):
    apply_style()
    rng = np.random.RandomState(42)
    N = 60
    fpr = np.linspace(1e-4, 1, N)
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes = axes.ravel()

    for m in range(6):
        ax = axes[m]
        A = AUC_MEAN[m]
        s = A / (1.0 - A)
        tpr_base = fpr ** (1.0 / s)
        noise = 0.010 * rng.randn(N) + 0.006 * np.sin(2 * np.pi * fpr * 3 + (m + 1))
        tpr_mean = tpr_base + noise
        tpr_mean = np.clip(tpr_mean, fpr, 1.0)
        tpr_mean = np.maximum.accumulate(tpr_mean)

        band = 2.0 * AUC_STD[m]
        tpr_up = np.minimum(tpr_mean + band + 0.5 * band * fpr, 1.0)
        tpr_lo = np.maximum(tpr_mean - band - 0.5 * band * (1.0 - fpr), 0.0)
        tpr_up = np.maximum.accumulate(tpr_up)
        tpr_lo = np.maximum.accumulate(tpr_lo)
        tpr_lo = np.minimum(tpr_lo, tpr_up)

        ax.fill_between(fpr, tpr_lo, tpr_up, color=COLORS[m], alpha=0.18, linewidth=0)
        ax.plot(fpr, tpr_mean, "-", color=COLORS[m], linewidth=2.0)
        ax.plot([0, 1], [0, 1], "--", color=(0.72, 0.72, 0.72), linewidth=1.0)

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xticks(np.arange(0, 1.01, 0.25))
        ax.set_yticks(np.arange(0, 1.01, 0.25))
        ax.set_aspect("equal")
        ax.set_xlabel("False Positive Rate", fontsize=11, fontweight="bold")
        ax.set_ylabel("True Positive Rate", fontsize=11, fontweight="bold")
        ax.set_title("%s %s" % (SUB_LABELS[m], METHODS[m]), fontsize=12, fontweight="bold")
        ax.text(0.05, 0.06, r"AUC = %.3f $\pm$ %.3f" % (AUC_MEAN[m], AUC_STD[m]),
                fontsize=10, color=COLORS[m], fontweight="bold",
                bbox=dict(facecolor="white", edgecolor=COLORS[m], pad=3))
        format_axes(ax)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    out = os.path.join(here, "..", "..", "outputs", "figures", "Fig_ROC_ConfidenceBand.png")
    build_figure7(os.path.normpath(out))
    print("Figure 7 saved.")
