"""One chart style for the whole project, so every notebook and the app look alike."""

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"   # categorical slots 1-3, in this order
INK, INK_2, MUTED, GRID = "#0b0b0b", "#52514e", "#c3c2b7", "#e6e5e0"
CATEGORICAL = [BLUE, ORANGE, AQUA]


def apply():
    plt.rcParams.update({
        "figure.figsize": (8, 4.5), "figure.dpi": 110,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
        "axes.edgecolor": INK_2, "axes.labelcolor": INK_2,
        "xtick.color": INK_2, "ytick.color": INK_2,
        "axes.titlelocation": "left", "axes.titleweight": "bold", "axes.titlesize": 12,
        "legend.frameon": False,
    })


def euro(x, _pos=None):
    """Format money for labels: 1_500_000 -> '€1.5M'."""
    if abs(x) >= 1e6:
        return f"€{x / 1e6:.3g}M"
    if abs(x) >= 1e3:
        return f"€{x / 1e3:.3g}K"
    return f"€{x:g}"


EURO = FuncFormatter(euro)
