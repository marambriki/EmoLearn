"""Vocabulaires de Delta : type <- scores, patron <- pattern, urgence <- H."""
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = Path(__file__).resolve().parent / "emolearn_decision_pipeline.png"

C_TEXT = "#1a202c"
C_MUTED = "#4a5568"
ROWS = [
    (
        ("#ebf8ff", "#2b6cb0"),
        "État $\\hat{e}$, $\\hat{n}$ et scores $s$\n(4 états DAiSEE)",
        ("#faf5ff", "#6b46c1"),
        "Type $\\tau$\nA, B, C, D, A+D, D+A, $\\varnothing$",
    ),
    (
        ("#e6fffa", "#2c7a7b"),
        "Pattern\n6 situations",
        ("#fffff0", "#b7791f"),
        "Patron $\\gamma$\nA1–D5 (20)",
    ),
    (
        ("#edf2f7", "#4a5568"),
        "Historique $H$\nclips successifs",
        ("#fff5f5", "#c53030"),
        "Urgence $u$\n4 niveaux",
    ),
]


def _box(ax, x, y, w, h, fc, ec, text):
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.02,rounding_size=0.08",
            facecolor=fc,
            edgecolor=ec,
            linewidth=1.6,
        )
    )
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=12.2,
        fontweight="bold",
        color=C_TEXT,
        linespacing=1.28,
    )


def main():
    fig, ax = plt.subplots(figsize=(10.8, 3.95), dpi=180)
    ax.set_xlim(0, 10.8)
    ax.set_ylim(0, 3.95)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    w, h = 3.75, 0.92
    xl, xr = 0.30, 6.75
    ys = [2.52, 1.40, 0.28]
    for y, ((fc1, ec1), left, (fc2, ec2), right) in zip(ys, ROWS):
        _box(ax, xl, y, w, h, fc1, ec1, left)
        _box(ax, xr, y, w, h, fc2, ec2, right)
        ax.add_patch(
            FancyArrowPatch(
                (xl + w + 0.06, y + h / 2),
                (xr - 0.06, y + h / 2),
                arrowstyle="-|>",
                mutation_scale=12,
                linewidth=1.4,
                color=C_MUTED,
            )
        )

    ax.text(
        5.4,
        3.66,
        r"Décision pédagogique $\Delta$",
        ha="center",
        va="center",
        fontsize=14,
        fontweight="bold",
        color=C_TEXT,
    )
    fig.tight_layout(pad=0.2)
    fig.savefig(OUT, bbox_inches="tight", facecolor="white")
    plt.close()
    print(OUT)


if __name__ == "__main__":
    main()
