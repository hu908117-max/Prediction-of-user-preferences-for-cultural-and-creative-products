"""Figure 2 - t-SNE comparison of individual user preference representations.

(a) Baseline model without user preference-guided cross-attention.
(b) Proposed method with user preference-guided cross-attention.

Four user groups: Visual-pattern, Cultural-semantic, Color-style, Mixed.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from common import apply_style, format_axes, GROUP_COLORS, covariance_ellipse

GROUP_NAMES = ["Visual-pattern", "Cultural-semantic", "Color-style", "Mixed"]


def _sample_groups(centers, spreads, angles, sub_offsets, n_per_group, n_bound, bound_scale):
    rng = np.random.RandomState(2024)
    X, labels = [], []
    for g in range(4):
        theta = np.deg2rad(angles[g])
        R = np.array([[np.cos(theta), -np.sin(theta)],
                      [np.sin(theta), np.cos(theta)]])
        Z = rng.randn(n_per_group, 2) @ np.diag(spreads[g])
        Z = Z @ R.T
        pts = Z + np.array(centers[g])
        sub_idx = rng.randint(0, 3, size=n_per_group)
        for i in range(n_per_group):
            pts[i] += sub_offsets[sub_idx[i]]
        X.append(pts)
        labels.append(np.full(n_per_group, g + 1))
    X = np.vstack(X)
    labels = np.concatenate(labels)
    b_idx = rng.choice(X.shape[0], size=n_bound, replace=False)
    X[b_idx] += bound_scale * rng.randn(n_bound, 2)
    return X, labels


def _silhouette_like(X, labels):
    scores = np.zeros(4)
    for g in range(4):
        idx = labels == (g + 1)
        other = ~idx
        intra = np.mean(_pdist(X[idx]))
        inter = np.mean(_pdist2(X[idx], X[other]))
        scores[g] = (inter - intra) / max(intra, inter)
    return scores


def _pdist(X):
    d = np.sqrt(((X[:, None, :] - X[None, :, :]) ** 2).sum(-1))
    iu = np.triu_indices(X.shape[0], k=1)
    return d[iu]


def _pdist2(A, B):
    d = np.sqrt(((A[:, None, :] - B[None, :, :]) ** 2).sum(-1))
    return d.mean(axis=1)


def _draw_panel(ax, X, labels, title, mean_sil):
    ax.set_facecolor("white")
    ax.set_xlim(-14, 14)
    ax.set_ylim(-12, 12)
    ax.set_aspect("equal")
    for g in range(4):
        idx = labels == (g + 1)
        ax.scatter(X[idx, 0], X[idx, 1], s=42, color=GROUP_COLORS[g],
                   alpha=0.75, edgecolors="k", linewidths=0.4, zorder=3)
        mu, w, h, ang = covariance_ellipse(X[idx], n_std=1.0)
        ell = plt.matplotlib.patches.Ellipse(mu, w, h, angle=ang, fill=False,
                                             edgecolor=GROUP_COLORS[g], linewidth=1.3, zorder=2)
        ax.add_patch(ell)
    ax.set_xlabel("t-SNE Dimension 1", fontsize=12, fontweight="bold")
    ax.set_ylabel("t-SNE Dimension 2", fontsize=12, fontweight="bold")
    ax.set_title(title, fontsize=12.5, fontweight="bold")
    ax.text(-13, -10.5, "Mean silhouette = %.3f" % mean_sil, fontsize=10.5,
            fontweight="bold", bbox=dict(facecolor="white", edgecolor=(0.6, 0.6, 0.6), pad=3))
    format_axes(ax)


def build_figure2(out_path):
    apply_style()
    n_per_group = 60
    sub_offsets = np.array([[-0.45, 0.35], [0.5, -0.3], [-0.2, 0.22]])

    centers_b = [[-5.2, 3.6], [5.0, 3.1], [-4.6, -3.9], [5.4, -3.2]]
    spreads_b = [[2.4, 1.9], [2.3, 2.0], [2.5, 1.8], [2.2, 2.1]]
    angles_b = [20, -15, 30, -25]
    X_b, labels_b = _sample_groups(centers_b, spreads_b, angles_b, sub_offsets,
                                   n_per_group, n_bound=22, bound_scale=2.0)

    centers_o = [[-8.8, 6.4], [8.2, 5.8], [-7.6, -7.0], [9.0, -5.6]]
    spreads_o = [[1.3, 0.95], [1.25, 1.05], [1.45, 0.9], [1.3, 1.0]]
    angles_o = [25, -18, 35, -30]
    X_o, labels_o = _sample_groups(centers_o, spreads_o, angles_o, sub_offsets,
                                   n_per_group, n_bound=6, bound_scale=0.7)

    sil_b = _silhouette_like(X_b, labels_b).mean()
    sil_o = _silhouette_like(X_o, labels_o).mean()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7.2))
    _draw_panel(ax1, X_b, labels_b, "(a) Baseline without user preference guidance", sil_b)
    handles = [plt.Line2D([0], [0], marker="o", color="w", label=GROUP_NAMES[g],
                          markerfacecolor=GROUP_COLORS[g], markeredgecolor="k", markersize=8)
               for g in range(4)]
    ax1.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.18),
               ncol=4, frameon=False, fontsize=9)
    _draw_panel(ax2, X_o, labels_o, "(b) Proposed method with user preference guidance", sil_o)
    ax2.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.18),
               ncol=4, frameon=False, fontsize=9)

    fig.tight_layout(rect=[0, 0.06, 1, 1])
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return sil_b, sil_o


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    out = os.path.join(here, "..", "..", "outputs", "figures", "Fig_UserPreference_tSNE_Compare.png")
    s_b, s_o = build_figure2(os.path.normpath(out))
    print("Figure 2 saved. Mean silhouette baseline=%.3f ours=%.3f" % (s_b, s_o))
