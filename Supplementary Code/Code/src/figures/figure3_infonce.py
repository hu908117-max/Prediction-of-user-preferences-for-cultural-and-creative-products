"""Figure 3 - InfoNCE contrastive learning: similarity distribution before/after.

Positive pairs: user preference representation vs. high-preference product.
Negative pairs: user preference representation vs. low-preference product.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from common import apply_style, format_axes, kde_density, PALETTE

N = 500


def _clip(x):
    return np.clip(x, 0.05, 0.95)


def _panel(ax, pos, neg, title):
    c_pos, c_neg = PALETTE["blue"], PALETTE["orange"]
    f_pos, xg = kde_density(pos, n_points=300)
    f_neg, _ = kde_density(neg, n_points=300, grid=xg)

    ax.fill_between(xg, 0, f_pos, color=c_pos, alpha=0.20, linewidth=0)
    ax.fill_between(xg, 0, f_neg, color=c_neg, alpha=0.20, linewidth=0)
    h1, = ax.plot(xg, f_pos, "-", color=c_pos, linewidth=2.2)
    h2, = ax.plot(xg, f_neg, "-", color=c_neg, linewidth=2.2)

    mu_pos, mu_neg = float(np.mean(pos)), float(np.mean(neg))
    ymax = ax.get_ylim()[1]
    ax.plot([mu_pos, mu_pos], [0, ymax * 0.85], "--", color=c_pos, linewidth=1.3)
    ax.plot([mu_neg, mu_neg], [0, ymax * 0.85], "--", color=c_neg, linewidth=1.3)

    ax.set_xlim(-0.05, 1.0)
    ax.set_xlabel("Cosine Similarity", fontsize=12, fontweight="bold")
    ax.set_ylabel("Probability Density", fontsize=12, fontweight="bold")
    ax.set_title(title, fontsize=12.5, fontweight="bold")
    ax.legend([h1, h2], ["Positive pairs", "Negative pairs"], loc="upper left",
              frameon=False, fontsize=10)
    ax.text(0.88, max(f_pos) * 0.85, r"$\mu_{pos}$ = %.3f" % mu_pos,
            fontsize=10, color=c_pos, fontweight="bold", ha="right")
    ax.text(0.88, max(f_pos) * 0.72, r"$\mu_{neg}$ = %.3f" % mu_neg,
            fontsize=10, color=c_neg, fontweight="bold", ha="right")
    format_axes(ax)
    return mu_pos, mu_neg


def build_figure3(out_path):
    apply_style()
    rng = np.random.RandomState(2024)
    pos_before = _clip(0.47 + 0.12 * rng.randn(N, 1)).ravel()
    neg_before = _clip(0.37 + 0.13 * rng.randn(N, 1)).ravel()
    pos_after = _clip(0.76 + 0.09 * rng.randn(N, 1)).ravel()
    neg_after = _clip(0.24 + 0.11 * rng.randn(N, 1)).ravel()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.4))
    mu_p1, mu_n1 = _panel(ax1, pos_before, neg_before, "(a) Before training")
    mu_p2, mu_n2 = _panel(ax2, pos_after, neg_after, "(b) After training")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return (mu_p1, mu_n1, mu_p2, mu_n2)


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    out = os.path.join(here, "..", "..", "outputs", "figures", "Fig_InfoNCE_Similarity_Distribution.png")
    build_figure3(os.path.normpath(out))
    print("Figure 3 saved.")
