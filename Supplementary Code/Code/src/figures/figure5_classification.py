"""Figure 5 - User aesthetic-emotional preference classification performance.

ArtEmis 8-class emotion classification (proxy task).
Methods: LightGCN, MISA, CLIP, BLIP-2, LLaVA, Ours (mean +/- std over 5 runs).
(a) All methods on Accuracy / Macro-F1 / Weighted-F1.
(b) Ours vs. strongest baseline (LLaVA) with relative improvement.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from common import apply_style, format_axes, PALETTE

METHODS = ["LightGCN", "MISA", "CLIP", "BLIP-2", "LLaVA", "Ours"]

ACC_MEAN = np.array([0.358, 0.424, 0.401, 0.467, 0.512, 0.586])
ACC_STD = np.array([0.022, 0.019, 0.025, 0.018, 0.016, 0.010])
MF1_MEAN = np.array([0.312, 0.371, 0.384, 0.421, 0.463, 0.542])
MF1_STD = np.array([0.024, 0.021, 0.023, 0.020, 0.018, 0.011])
WF1_MEAN = np.array([0.341, 0.409, 0.389, 0.452, 0.498, 0.571])
WF1_STD = np.array([0.023, 0.020, 0.024, 0.019, 0.017, 0.010])


def _grouped_bar(ax, data_mean, data_std, colors, group_labels, xlabel, title,
                 ylim=(0, 0.75), legend_labels=None, annotate_cmp=None):
    nbars = data_mean.shape[1]
    ngroups = data_mean.shape[0]
    group_width = min(0.8, nbars / (nbars + 1.5))
    bar_w = group_width / nbars
    x = np.arange(ngroups)
    for k in range(nbars):
        offsets = (k - (nbars - 1) / 2) * bar_w + x
        ax.bar(offsets, data_mean[:, k], width=bar_w * 0.92, color=colors[k],
               edgecolor=(0.1, 0.1, 0.1), linewidth=0.6, zorder=3)
        eb_x = x - group_width / 2 + (2 * k - 1) * group_width / (2 * nbars)
        ax.errorbar(eb_x, data_mean[:, k], data_std[:, k], fmt="k.",
                    linewidth=0.9, capsize=3, zorder=4)
    if annotate_cmp is not None:
        base_mean, ours_mean, base_std, ours_std = annotate_cmp
        for i in range(3):
            ax.text(i - 0.15, base_mean[i] + base_std[i] + 0.012, "%.3f" % base_mean[i],
                    ha="center", fontsize=9, color=(0.2, 0.2, 0.2))
            ax.text(i + 0.15, ours_mean[i] + ours_std[i] + 0.012, "%.3f" % ours_mean[i],
                    ha="center", fontsize=9, fontweight="bold", color=(0.15, 0.05, 0.35))
            imp = (ours_mean[i] - base_mean[i]) / base_mean[i] * 100
            ax.text(i, max(ours_mean[i] + ours_std[i], base_mean[i] + base_std[i]) + 0.045,
                    "+%.2f%%" % imp, ha="center", fontsize=10, fontweight="bold",
                    color=(0.35, 0.10, 0.55))
    ax.set_xticks(x)
    ax.set_xticklabels(group_labels, rotation=35, ha="right", fontsize=10)
    ax.set_xlabel(xlabel, fontsize=12, fontweight="bold")
    ax.set_ylabel("Performance Score", fontsize=12, fontweight="bold")
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_ylim(*ylim)
    ax.set_yticks(np.arange(0, ylim[1] + 1e-6, 0.1))
    if legend_labels is not None:
        ax.legend(legend_labels, loc="upper center", bbox_to_anchor=(0.5, 1.12),
                  ncol=3, frameon=False, fontsize=10)
    format_axes(ax)


def build_figure5(out_path):
    apply_style()
    data_mean = np.column_stack([ACC_MEAN, MF1_MEAN, WF1_MEAN])
    data_std = np.column_stack([ACC_STD, MF1_STD, WF1_STD])
    metric_colors = [PALETTE["blue"], PALETTE["orange"], PALETTE["green"]]
    metric_labels = ["Accuracy", "Macro-F1", "Weighted-F1"]

    base_mean = np.array([0.512, 0.463, 0.498])
    base_std = np.array([0.016, 0.018, 0.017])
    ours_mean = np.array([0.586, 0.542, 0.571])
    ours_std = np.array([0.010, 0.011, 0.010])
    cmp_data = np.column_stack([base_mean, ours_mean])
    cmp_std = np.column_stack([base_std, ours_std])
    cmp_colors = [PALETTE["gray"], PALETTE["purple"]]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6.2))
    _grouped_bar(ax1, data_mean, data_std, metric_colors, METHODS, "Methods",
                 "(a) Classification Performance of Different Methods",
                 legend_labels=metric_labels)
    _grouped_bar(ax2, cmp_data, cmp_std, cmp_colors,
                 ["Accuracy", "Macro-F1", "Weighted-F1"], "Metrics",
                 "(b) Ours vs. Strongest Baseline (LLaVA)",
                 legend_labels=["LLaVA", "Ours"],
                 annotate_cmp=(base_mean, ours_mean, base_std, ours_std))
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    out = os.path.join(here, "..", "..", "outputs", "figures", "Fig_Classification_Comparison.png")
    build_figure5(os.path.normpath(out))
    print("Figure 5 saved.")
