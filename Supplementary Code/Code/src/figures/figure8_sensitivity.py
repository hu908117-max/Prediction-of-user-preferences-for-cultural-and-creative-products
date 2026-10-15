"""Figure 8 - Parameter sensitivity analysis.

(a) Temperature coefficient tau, (b) InfoNCE loss weight lambda_1,
(c) Cross-modal consistency loss weight lambda_2.
Metrics: Macro-F1 and AUC (mean +/- std over 5 runs).
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from common import apply_style, format_axes, PALETTE

TAU_VALS = np.array([0.03, 0.05, 0.07, 0.10, 0.15, 0.20])
TAU_F1 = np.array([0.512, 0.528, 0.542, 0.537, 0.521, 0.504])
TAU_F1_S = np.array([0.014, 0.012, 0.011, 0.012, 0.013, 0.015])
TAU_AUC = np.array([0.851, 0.869, 0.882, 0.876, 0.858, 0.840])
TAU_AUC_S = np.array([0.012, 0.010, 0.009, 0.010, 0.011, 0.013])

LAM1_VALS = np.array([0.1, 0.3, 0.5, 0.7, 0.9, 1.1])
LAM1_F1 = np.array([0.508, 0.526, 0.542, 0.535, 0.519, 0.503])
LAM1_F1_S = np.array([0.015, 0.013, 0.011, 0.012, 0.014, 0.016])
LAM1_AUC = np.array([0.848, 0.867, 0.882, 0.873, 0.856, 0.838])
LAM1_AUC_S = np.array([0.013, 0.011, 0.009, 0.010, 0.012, 0.014])

LAM2_VALS = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
LAM2_F1 = np.array([0.514, 0.531, 0.542, 0.536, 0.522, 0.507])
LAM2_F1_S = np.array([0.014, 0.012, 0.011, 0.012, 0.013, 0.015])
LAM2_AUC = np.array([0.852, 0.871, 0.882, 0.875, 0.859, 0.842])
LAM2_AUC_S = np.array([0.012, 0.010, 0.009, 0.010, 0.012, 0.014])


def _panel(ax, vals, f1, f1_s, auc, auc_s, xlabel, title, xlim, optimum):
    c_f1, c_auc = PALETTE["blue"], PALETTE["orange"]
    h1 = ax.errorbar(vals, f1, yerr=f1_s, fmt="-o", color=c_f1, linewidth=1.8, markersize=7,
                     markerfacecolor=c_f1, markeredgecolor="k", capsize=4)
    h2 = ax.errorbar(vals, auc, yerr=auc_s, fmt="-s", color=c_auc, linewidth=1.8, markersize=7,
                     markerfacecolor=c_auc, markeredgecolor="k", capsize=4)
    ax.plot(optimum[0], optimum[1], "p", markersize=13, markerfacecolor=PALETTE["gold"],
            markeredgecolor="k", linewidth=0.8, zorder=5)
    ax.set_xlim(*xlim)
    ax.set_ylim(0.48, 0.92)
    ax.set_yticks(np.arange(0.50, 0.91, 0.10))
    ax.set_xlabel(xlabel, fontsize=11, fontweight="bold")
    ax.set_ylabel("Performance Score", fontsize=11, fontweight="bold")
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend([h1, h2], ["Macro-F1", "AUC"], loc="lower center",
              bbox_to_anchor=(0.5, -0.02), ncol=2, frameon=False, fontsize=9)
    format_axes(ax)


def build_figure8(out_path):
    apply_style()
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.0))
    _panel(axes[0], TAU_VALS, TAU_F1, TAU_F1_S, TAU_AUC, TAU_AUC_S,
           "Temperature coefficient $\\tau$", "(a) Sensitivity to $\\tau$",
           (0.01, 0.22), (0.07, 0.542))
    _panel(axes[1], LAM1_VALS, LAM1_F1, LAM1_F1_S, LAM1_AUC, LAM1_AUC_S,
           "Loss weight $\\lambda_1$", "(b) Sensitivity to $\\lambda_1$",
           (0.0, 1.2), (0.5, 0.542))
    _panel(axes[2], LAM2_VALS, LAM2_F1, LAM2_F1_S, LAM2_AUC, LAM2_AUC_S,
           "Loss weight $\\lambda_2$", "(c) Sensitivity to $\\lambda_2$",
           (0.05, 0.65), (0.3, 0.542))
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    out = os.path.join(here, "..", "..", "outputs", "figures", "Fig_Parameter_Sensitivity.png")
    build_figure8(os.path.normpath(out))
    print("Figure 8 saved.")
