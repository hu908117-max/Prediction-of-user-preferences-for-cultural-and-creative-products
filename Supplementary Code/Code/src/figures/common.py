"""Shared plotting utilities for the result figures.

This module mirrors the visual style used in the manuscript (Times New Roman,
subtle grid, boxed axes, consistent palette) so that the Python output is a
direct counterpart of the original figure-generation code.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np

# Consistent color palette across all figures (RGB in [0, 1]).
PALETTE = {
    "blue": (0.2157, 0.4941, 0.7216),
    "orange": (0.8941, 0.4706, 0.2157),
    "green": (0.3010, 0.7451, 0.4157),
    "purple": (0.5216, 0.3020, 0.6588),
    "gray": (0.7200, 0.7200, 0.7200),
    "darkgray": (0.3500, 0.3500, 0.3500),
    "black": (0.1000, 0.1000, 0.1000),
    "gold": (0.9500, 0.8500, 0.1000),
}

GROUP_COLORS = [
    (0.2157, 0.4941, 0.7216),  # Visual-pattern
    (0.8941, 0.4706, 0.2157),  # Cultural-semantic
    (0.3010, 0.7451, 0.4157),  # Color-style
    (0.5216, 0.3020, 0.6588),  # Mixed
]


def apply_style():
    """Set global matplotlib options that emulate the manuscript style."""
    plt.rcParams.update({
        "font.family": "Times New Roman",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "axes.unicode_minus": False,
        "figure.dpi": 150,
    })


def format_axes(ax, grid=True):
    """Apply the common axis formatting (box, ticks, grid)."""
    ax.tick_params(direction="out", labelsize=11)
    for spine in ax.spines.values():
        spine.set_linewidth(1.1)
    if grid:
        ax.grid(True, linestyle=":", alpha=0.22)


def kde_density(samples, n_points=300, grid=None):
    """Return kernel density estimate (pdf, grid) over [0, 1]-ish support."""
    from scipy.stats import gaussian_kde
    kde = gaussian_kde(samples)
    if grid is None:
        lo = float(np.min(samples)) - 0.02
        hi = float(np.max(samples)) + 0.02
        grid = np.linspace(lo, hi, n_points)
    return kde(grid), grid


def covariance_ellipse(points, n_std=2.0):
    """Compute ellipse centre, widths and rotation (degrees) for a point cloud."""
    pts = np.asarray(points)
    mu = pts.mean(axis=0)
    cov = np.cov(pts, rowvar=False)
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    # Width/height correspond to 2 * n_std * sqrt(eigenvalue).
    width, height = 2 * n_std * np.sqrt(np.clip(vals, 0, None))
    angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
    return mu, width, height, angle


def add_grid_legend(ax, handles, labels, **kwargs):
    """Add a borderless legend placed outside the plot."""
    opts = dict(loc="southoutside", ncol=4, frameon=False, fontsize=9)
    opts.update(kwargs)
    leg = ax.legend(handles, labels, **opts)
    return leg
