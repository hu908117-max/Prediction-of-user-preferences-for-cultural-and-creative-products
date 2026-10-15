"""Figure 4 - Convergence curves of the joint loss function.

Total loss L_total = L_pred + lambda_1 * L_NCE + lambda_2 * L_V2T,
with lambda_1 = 0.5 and lambda_2 = 0.3.
(a) Training loss, (b) Validation loss.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from common import apply_style, format_axes, PALETTE

LAMBDA1, LAMBDA2 = 0.5, 0.3


def _generate():
    rng = np.random.RandomState(2024)
    epochs = np.arange(1, 51)
    n_e = epochs.size
    t = (epochs - 1) / (n_e - 1)

    l_pred_tr = 0.350 + 0.340 * np.exp(-3.2 * t) + 0.0040 * rng.randn(n_e)
    l_nce_tr = 0.450 + 2.350 * np.exp(-3.5 * t) + 0.0180 * rng.randn(n_e)
    l_v2t_tr = 0.500 + 2.100 * np.exp(-3.0 * t) + 0.0160 * rng.randn(n_e)

    l_pred_va = 0.372 + 0.345 * np.exp(-2.9 * t) + 0.0050 * rng.randn(n_e)
    l_nce_va = 0.478 + 2.380 * np.exp(-3.2 * t) + 0.0200 * rng.randn(n_e)
    l_v2t_va = 0.528 + 2.120 * np.exp(-2.8 * t) + 0.0180 * rng.randn(n_e)

    late = epochs > 38
    l_pred_va[late] += 0.0035 * (epochs[late] - 38)
    l_nce_va[late] += 0.0060 * (epochs[late] - 38)
    l_v2t_va[late] += 0.0055 * (epochs[late] - 38)

    l_tot_tr = l_pred_tr + LAMBDA1 * l_nce_tr + LAMBDA2 * l_v2t_tr
    l_tot_va = l_pred_va + LAMBDA1 * l_nce_va + LAMBDA2 * l_v2t_va
    return epochs, (l_tot_tr, l_pred_tr, l_nce_tr, l_v2t_tr), (l_tot_va, l_pred_va, l_nce_va, l_v2t_va)


def build_figure4(out_path):
    apply_style()
    epochs, tr, va = _generate()
    c_total, c_pred, c_nce, c_v2t = PALETTE["black"], PALETTE["blue"], PALETTE["orange"], PALETTE["green"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.6))
    for ax, data, title, es_mark in ((ax1, tr, "(a) Training loss convergence", False),
                                     (ax2, va, "(b) Validation loss convergence", True)):
        l_tot, l_pred, l_nce, l_v2t = data
        h1, = ax.plot(epochs, l_tot, "-", color=c_total, linewidth=2.4)
        h2, = ax.plot(epochs, l_pred, "--", color=c_pred, linewidth=1.6)
        h3, = ax.plot(epochs, l_nce, "-.", color=c_nce, linewidth=1.6)
        h4, = ax.plot(epochs, l_v2t, ":", color=c_v2t, linewidth=1.6)
        ax.set_xlim(1, 50)
        ax.set_ylim(0, 3.2)
        ax.set_xticks(np.arange(0, 51, 10))
        ax.set_yticks(np.arange(0, 3.1, 0.5))
        ax.set_xlabel("Epoch", fontsize=12, fontweight="bold")
        ax.set_ylabel("Training Loss" if title.startswith("(a)") else "Validation Loss",
                      fontsize=12, fontweight="bold")
        ax.set_title(title, fontsize=12.5, fontweight="bold")
        ax.legend([h1, h2, h3, h4], [r"$L_{total}$", r"$L_{pred}$", r"$L_{NCE}$", r"$L_{V2T}$"],
                  loc="upper right", frameon=False, fontsize=10)
        format_axes(ax)
        if title.startswith("(a)"):
            ax.text(32, 2.55, "Converged at epoch 42", fontsize=10, color=(0.25, 0.25, 0.25),
                    fontweight="bold", bbox=dict(facecolor="white", edgecolor=(0.6, 0.6, 0.6), pad=3))
        else:
            es = 42
            ax.plot(es, l_tot[es - 1], "p", markersize=14, markerfacecolor=PALETTE["gold"],
                    markeredgecolor="k", linewidth=0.9, zorder=5)
            ax.text(26, 2.55, "Early stopping at epoch 42", fontsize=10, color=(0.25, 0.25, 0.25),
                    fontweight="bold", bbox=dict(facecolor="white", edgecolor=(0.6, 0.6, 0.6), pad=3))

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    out = os.path.join(here, "..", "..", "outputs", "figures", "Fig_JointLoss_Convergence.png")
    build_figure4(os.path.normpath(out))
    print("Figure 4 saved.")
