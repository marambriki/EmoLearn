"""Distribution des etats DAiSEE (trois fichiers officiels, N = 8571)."""
from pathlib import Path

import matplotlib.pyplot as plt

OUT = Path(__file__).with_name("daisee_distribution.png")

# Concatenation Train + Validation + Test (8571 clips).
etats = ["Ennui", "Engagement", "Confusion", "Frustration"]
counts = {
    "Ennui": [3702, 2656, 1886, 327],
    "Engagement": [61, 440, 4312, 3758],
    "Confusion": [5758, 1994, 720, 99],
    "Frustration": [6629, 1528, 329, 85],
}
colors = {
    "Ennui": "#f07167",
    "Engagement": "#2a9d8f",
    "Confusion": "#457b9d",
    "Frustration": "#f4a261",
}

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Calibri", "Segoe UI", "DejaVu Sans"],
    "axes.unicode_minus": False,
    "figure.dpi": 160,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

fig, axes = plt.subplots(2, 2, figsize=(9.8, 6.8))
niveaux = [0, 1, 2, 3]

for ax, etat in zip(axes.ravel(), etats):
    vals = counts[etat]
    bars = ax.bar(niveaux, vals, color=colors[etat], edgecolor="white", linewidth=0.4)
    ax.set_title(etat)
    ax.set_xticks(niveaux)
    ax.set_xlabel("Niveau (0 = faible, 3 = élevé)")
    ax.set_ylabel("Nombre de clips")
    ax.set_ylim(0, max(vals) * 1.18)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, linestyle=":", alpha=0.45)
    ax.set_axisbelow(True)
    for bar, n in zip(bars, vals):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(vals) * 0.02,
            f"{n}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

fig.suptitle("Distribution des états — DAiSEE (N = 8 571)")
fig.tight_layout()
fig.savefig(OUT, bbox_inches="tight")
print(OUT)
