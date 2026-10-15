"""Figure 6 - Top-K user preference ranking performance comparison.

Methods: LightGCN, MISA, CLIP, BLIP-2, LLaVA, Ours.
Metrics: NDCG@K (K=5,10,15,20), MRR, Hit@5, Hit@10 (mean +/- std over 5 runs).
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from common import apply_style, format_axes

METHODS = ["LightGCN", "MISA", "CLIP", "BLIP-2", "LLaVA", "Ours"]
KVALS = np.array([5, 10, 15, 20])

NDCG_MEAN = np.array([
    [0.298, 0.341, 0.352, 0.367],
    [0.346, 0.382, 0.411, 0.404],
    [0.352, 0.371, 0.396, 0.418],
    [0.397, 0.442, 0.451, 0.470],
    [0.436, 0.448, 0.503, 0.512],
    [0.508, 0.549, 0.567, 0.583],
])
NDCG_STD = np.array([
    [0.028, 0.023, 0.026, 0.022],
    [0.025, 0.019, 0.024, 0.021],
    [0.022, 0.026, 0.020, 0.024],
    [0.024, 0.018, 0.022, 0.017],
    [0.019, 0.016, 0.021, 0.015],
    [0.013, 0.010, 0.012, 0.009],
])

RANK_MEAN = np.array([
    [0.316, 0.348, 0.472],
    [0.372, 0.395, 0.534],
    [0.361, 0.412, 0.518],
    [0.408, 0.461, 0.565],
    [0.449, 0.483, 0.602],
    [0.531, 0.562, 0.671],
])
RANK_STD = np.array([
    [0.026, 0.022, 0.024],
    [0.021, 0.024, 0.019],
    [0.025, 0.019, 0.023],
    [0.022, 0.017, 0.020],
    [0.018, 0.020, 0.016],
    [0.011, 0.009, 0.010],
])

COLORS_M = [
    (0.35, 0.35, 0.35), (0.85, 0.55, 0.20), (0.30, 0.65, 0.45),
    (0.55, 0.35, 0.70), (0.80, 0.30, 0.35), (0.15, 0.35, 0.70),
]
COLORS_K = [(0.2157, 0.4941, 0.7216), (0.8941, 0.4706, 0.2157), (0.3010, 0.7451, 0.4157)]
MARKERS = ["o", "s", "^", "d", "v", "p"]


def build_figure6(out_path):
    apply_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6.2))

    for m in range(6):
        ax1.errorbar(KVALS, NDCG_MEAN[m], yerr=NDCG_STD[m], fmt="-", color=COLORS_M[m],
                     linewidth=1.6, marker=MARKERS[m], markersize=7,
                     markerfacecolor=COLORS_M[m], markeredgecolor="k", capsize=4, zorder=3)
    ax1.set_xticks(KVALS)
    ax1.set_xlim(3, 22)
    ax1.set_ylim(0.25, 0.65)
    ax1.set_xlabel("K", fontsize=12, fontweight="bold")
    ax1.set_ylabel("NDCG@K", fontsize=12, fontweight="bold")
    ax1.set_title("(a) NDCG@K of Different Methods", fontsize=12, fontweight="bold")
    ax1.legend(METHODS, loc="upper center", bbox_to_anchor=(0.5, 1.12), ncol=3,
               frameon=False, fontsize=9)
    format_axes(ax1)

    nbars, ngroups = 3, RANK_MEAN.shape[0]
    group_width = min(0.8, nbars / (nbars + 1.5))
    bar_w = group_width / nbars
    x = np.arange(ngroups)
    for k in range(nbars):
        offsets = (k - (nbars - 1) / 2) * bar_w + x
        ax2.bar(offsets, RANK_MEAN[:, k], width=bar_w * 0.92, color=COLORS_K[k],
                edgecolor=(0.1, 0.1, 0.1), linewidth=0.6, zorder=3)
        eb_x = x - group_width / 2 + (2 * k - 1) * group_width / (2 * nbars)
        ax2.errorbar(eb_x, RANK_MEAN[:, k], RANK_STD[:, k], fmt="k.", linewidth=0.9,
                     capsize=3, zorder=4)
    ax2.set_xticks(x)
    ax2.set_xticklabels(METHODS, rotation=30, ha="right", fontsize=10)
    ax2.set_ylim(0, 0.82)
    ax2.set_yticks(np.arange(0, 0.81, 0.1))
    ax2.set_xlabel("Methods", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Ranking Performance Score", fontsize=12, fontweight="bold")
    ax2.set_title("(b) MRR, Hit@5 and Hit@10 of Different Methods", fontsize=12, fontweight="bold")
    ax2.legend(["MRR", "Hit@5", "Hit@10"], loc="upper center",
               bbox_to_anchor=(0.5, 1.12), ncol=3, frameon=False, fontsize=10)
    format_axes(ax2)

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    out = os.path.join(here, "..", "..", "outputs", "figures", "Fig_TopK_Ranking_Comparison.png")
    build_figure6(os.path.normpath(out))
    print("Figure 6 saved.")
