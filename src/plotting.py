"""Shared matplotlib style + helpers for 'money plots' across notebooks."""
import matplotlib.pyplot as plt

STYLE = {
    "figure.figsize": (7.0, 4.0),
    "figure.dpi": 110,
    "savefig.dpi": 140,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "lines.linewidth": 1.6,
    "font.size": 11,
    "legend.frameon": False,
}


def apply_style() -> None:
    """Call this in the first cell of every notebook."""
    plt.rcParams.update(STYLE)


def semilog_convergence(errors, label=None, ax=None, **kw):
    """Plot convergence on semilog-y."""
    ax = ax or plt.gca()
    ax.semilogy(range(len(errors)), errors, label=label, **kw)
    ax.set_xlabel("iteration")
    ax.set_ylabel("error")
    if label:
        ax.legend()
    return ax


def eigenvalue_compare(true_eigs, ritz_eigs, ax=None):
    """Plot top-k true eigenvalues vs Ritz values, sorted descending."""
    import numpy as np
    ax = ax or plt.gca()
    t = np.sort(np.asarray(true_eigs))[::-1]
    r = np.sort(np.asarray(ritz_eigs))[::-1]
    k = min(len(t), len(r))
    ax.plot(range(k), t[:k], "o-", label="true")
    ax.plot(range(k), r[:k], "x--", label="Ritz")
    ax.set_xlabel("index")
    ax.set_ylabel("eigenvalue")
    ax.legend()
    return ax
