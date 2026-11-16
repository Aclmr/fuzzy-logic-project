"""Tum grafikler icin ortak, sade stil (renk koru palet sirasi sabit)."""
import matplotlib.pyplot as plt

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
C = {"surface": "#fcfcfb", "text": "#0b0b0b", "text2": "#52514e", "muted": "#8a8984", "grid": "#e4e3df"}


def apply_style():
    plt.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 200, "savefig.bbox": "tight",
        "figure.facecolor": C["surface"], "axes.facecolor": C["surface"],
        "axes.edgecolor": C["grid"], "axes.labelcolor": C["text2"], "axes.titlecolor": C["text"],
        "axes.titlesize": 11, "axes.labelsize": 10, "axes.grid": True,
        "grid.color": C["grid"], "grid.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False,
        "xtick.color": C["text2"], "ytick.color": C["text2"], "font.size": 10,
        "legend.frameon": False, "lines.linewidth": 2,
    })
