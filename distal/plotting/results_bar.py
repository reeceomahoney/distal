"""Grouped bar chart of hardware success rates across methods.

Two tasks (remove pen lid, remove ethernet cable) by three methods (base
Pi0.5, RECAP-style reward, DistAL). Data is hardcoded below — edit the
SUCCESS_RATES table and re-run.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

TASKS = ["Remove pen lid", "Remove ethernet cable"]
METHODS = ["Base Pi0.5", "RECAP-style reward", "DistAL (ours)"]

# rows = methods, cols = tasks. Values are success rates in percent.
SUCCESS_RATES = np.array(
    [
        [42, 46],
        [48, 72],
        [52, 87],
    ]
)

METHOD_COLORS = ["#b8bdc4", "#5b8def", "#f08a3e"]


def main() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 12,
            "axes.titlesize": 14,
            "axes.labelsize": 13,
            "xtick.labelsize": 12,
            "ytick.labelsize": 11,
            "legend.fontsize": 11,
            "axes.linewidth": 0.8,
            "axes.edgecolor": "#333333",
            "xtick.color": "#333333",
            "ytick.color": "#333333",
            "axes.labelcolor": "#1a1a1a",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    n_tasks = len(TASKS)
    n_methods = len(METHODS)
    x = np.arange(n_tasks)
    group_width = 0.78
    bar_width = group_width / n_methods

    fig, ax = plt.subplots(figsize=(8.5, 5), constrained_layout=True)

    for i, (method, color) in enumerate(zip(METHODS, METHOD_COLORS)):
        offsets = x + (i - (n_methods - 1) / 2) * bar_width
        bars = ax.bar(
            offsets,
            SUCCESS_RATES[i],
            width=bar_width * 0.92,
            label=method,
            color=color,
            linewidth=0,
            zorder=3,
        )
        ax.bar_label(
            bars,
            fmt="%.0f",
            padding=3,
            fontsize=10,
            color="#1a1a1a",
            fontweight="medium",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(TASKS)
    ax.set_ylabel("Success rate (%)", labelpad=8)
    ax.set_ylim(0, 100)
    ax.set_yticks(np.arange(0, 101, 20))
    ax.set_yticklabels([f"{v}" for v in np.arange(0, 101, 20)])

    ax.grid(axis="y", color="#d8d8d8", linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)

    for side in ("top", "right", "bottom"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color("#888888")
    ax.axhline(0, color="#888888", linewidth=0.8, zorder=2)
    ax.tick_params(axis="x", length=0, pad=8)
    ax.tick_params(axis="y", length=3, color="#888888")

    ax.margins(x=0.08)

    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.08),
        ncol=n_methods,
        frameon=False,
        handlelength=1.4,
        handletextpad=0.6,
        columnspacing=1.8,
    )
    output_path = Path("outputs/results_bar.pdf")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"saved -> {output_path}")


if __name__ == "__main__":
    main()
