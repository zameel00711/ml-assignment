"""Plot styling and minimal theme helpers."""

import matplotlib.pyplot as plt
import seaborn as sns


def apply_plot_theme() -> None:
    """Sets a clean, publication-ready plot style with readable labels and muted grids."""
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    sns.set_theme(
        style="whitegrid",
        palette="deep",
        rc={
            "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Helvetica", "Arial"],
            "axes.edgecolor": "#CBD5E1",
            "axes.linewidth": 0.8,
            "grid.color": "#E2E8F0",
            "grid.linestyle": "--",
            "grid.alpha": 0.7,
            "figure.dpi": 150,
        },
    )
