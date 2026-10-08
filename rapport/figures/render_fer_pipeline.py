"""Architecture de la perception faciale (section 4.4.1)."""
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = Path(__file__).resolve().parent / "emolearn_fer_pipeline.png"

C_TEXT = "#1a202c"
C_MUTED = "#4a5568"
C_PERC = "#2b6cb0"
C_PERC_BG = "#ebf8ff"
C_PERC_BOX = "#bee3f8"
C_DEC = "#2c7a7b"
C_DEC_BG = "#e6fffa"
C_DEC_BOX = "#b2f5ea"
C_IO = "#f7fafc"
C_IO_STROKE = "#4a5568"


def box(ax, x, y, w, h, text, *, fc, ec, lw=1.6, fs=12, dashed=False):
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.02,rounding_size=0.08",
            facecolor=fc,
            edgecolor=ec,
            linewidth=lw,
            linestyle="--" if dashed else "-",
            zorder=3,
        )
    )
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fs,
        color=C_TEXT,
        fontweight="bold",
        linespacing=1.22,
        zorder=4,
    )
    return x + w / 2, y + h / 2, x, y, w, h


def layer(ax, x, y, w, h, label, color, bg, *, label_right=False):
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.02,rounding_size=0.1",
            facecolor=bg,
            edgecolor=color,
            linewidth=1.2,
            alpha=0.95,
            zorder=1,
        )
    )
    ax.text(
        x + w - 0.20 if label_right else x + 0.20,
        y + h / 2,
        label,
        ha="center",
        va="center",
        fontsize=12,
        fontweight="bold",
        color=color,
        rotation=90,
        zorder=2,
    )


def arrow(ax, p1, p2, color=C_MUTED):
    ax.add_patch(
        FancyArrowPatch(
            p1,
            p2,
            arrowstyle="-|>",
            mutation_scale=11,
            linewidth=1.5,
            color=color,
            shrinkA=1,
            shrinkB=1,
            zorder=2.5,
        )
    )


def elbow(ax, x1, y1, x2, y2, color, first="v"):
    """Orthogonal path: vertical then horizontal, or the reverse."""
    if first == "v":
        ax.plot([x1, x1], [y1, y2], color=color, lw=1.5, zorder=2.4, solid_capstyle="round")
        arrow(ax, (x1, y2), (x2, y2), color)
    else:
        ax.plot([x1, x2], [y1, y1], color=color, lw=1.5, zorder=2.4, solid_capstyle="round")
        arrow(ax, (x2, y1), (x2, y2), color)


def main():
    fig, ax = plt.subplots(figsize=(11.2, 7.6), dpi=220)
    ax.set_xlim(0, 11.2)
    ax.set_ylim(0, 7.6)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    ax.text(
        5.6,
        7.32,
        "Perception faciale : du clip à l'état",
        ha="center",
        va="center",
        fontsize=16,
        fontweight="bold",
        color="#1a365d",
    )

    layer(ax, 0.30, 4.55, 10.60, 2.40, "EXTRACTION", C_PERC, C_PERC_BG)
    c1 = box(
        ax, 0.80, 5.55, 2.10, 0.90,
        "Clip DAiSEE\nvidéo 10 s",
        fc=C_IO, ec=C_IO_STROKE, dashed=True,
    )
    c2 = box(
        ax, 3.20, 5.55, 2.15, 0.90,
        "5 images\nhors bords",
        fc="white", ec=C_PERC,
    )
    c3 = box(
        ax, 5.65, 4.80, 2.50, 1.80,
        "BlazeFace\ncourte portée\n↓ longue portée\n↓ miroir",
        fc=C_PERC_BOX, ec=C_PERC, fs=11,
    )
    c4 = box(
        ax, 8.50, 5.55, 2.05, 0.90,
        "Visage\nmarge 30 %",
        fc="white", ec=C_PERC,
    )
    arrow(ax, (c1[0] + 1.05, c1[1]), (c2[0] - 1.08, c2[1]), C_PERC)
    arrow(ax, (c2[0] + 1.08, c2[1]), (c3[0] - 1.25, 5.55 + 0.20), C_PERC)
    arrow(ax, (c3[0] + 1.25, 5.55 + 0.20), (c4[0] - 1.03, c4[1]), C_PERC)

    layer(ax, 0.30, 2.20, 6.20, 1.95, "TRAITS", C_PERC, C_PERC_BG)
    t1 = box(
        ax, 0.80, 2.48, 2.55, 1.35,
        "FER pré-entraîné\n7 émotions\nfiltre 0,35 + moyenne",
        fc="white", ec=C_PERC, fs=11,
    )
    t2 = box(
        ax, 3.60, 2.48, 2.55, 1.35,
        "Face Landmarker\n10 Action Units\n(approximation FACS)",
        fc="white", ec=C_PERC, fs=11,
    )

    layer(ax, 6.70, 2.20, 4.20, 1.95, "ÉTAT", C_DEC, C_DEC_BG, label_right=True)
    d1 = box(
        ax, 7.00, 2.48, 3.45, 1.35,
        "Random Forest FER+AU\nvecteur 17 → 4 états\n" + r"$\hat{e}_v$, $\hat{n}_v$",
        fc=C_DEC_BOX, ec=C_DEC, fs=11,
    )

    # Visage ↓ then ← T-junction into FER and AU
    jy = 4.35
    ax.plot([c4[0], c4[0]], [c4[1] - 0.45, jy], color=C_PERC, lw=1.5, zorder=2.4)
    ax.plot([t1[0], c4[0]], [jy, jy], color=C_PERC, lw=1.5, zorder=2.4)
    arrow(ax, (t1[0], jy), (t1[0], t1[1] + 0.68), C_PERC)
    arrow(ax, (t2[0], jy), (t2[0], t2[1] + 0.68), C_PERC)

    arrow(ax, (t1[2] + t1[4], t1[1]), (t2[2], t2[1]), C_DEC)
    arrow(ax, (t2[2] + t2[4], t2[1]), (d1[2], d1[1]), C_DEC)

    out = box(
        ax, 0.30, 0.28, 10.60, 0.95,
        "Sortie vers la fusion   ·   repli : mapping heuristique   ·   ViT : référence de frames",
        fc=C_IO, ec=C_IO_STROKE, dashed=True, fs=12,
    )
    arrow(ax, (d1[0], d1[1] - 0.68), (d1[0], out[1] + 0.48), C_DEC)

    fig.tight_layout(pad=0.2)
    fig.savefig(OUT, bbox_inches="tight", facecolor="white")
    print(OUT)


if __name__ == "__main__":
    main()
