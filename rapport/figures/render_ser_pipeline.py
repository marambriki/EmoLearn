"""Architecture de la perception vocale (section 4.4.2)."""
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = Path(__file__).resolve().parent / "emolearn_ser_pipeline.png"

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


def main():
    fig, ax = plt.subplots(figsize=(11.2, 7.6), dpi=220)
    ax.set_xlim(0, 11.2)
    ax.set_ylim(0, 7.6)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    ax.text(
        5.6,
        7.32,
        "Perception vocale : du segment à l'état",
        ha="center",
        va="center",
        fontsize=16,
        fontweight="bold",
        color="#1a365d",
    )

    layer(ax, 0.30, 4.55, 10.60, 2.40, "EXTRACTION", C_PERC, C_PERC_BG)
    c1 = box(
        ax, 0.80, 5.55, 2.20, 0.90,
        "Scénario\nscores DAiSEE",
        fc=C_IO, ec=C_IO_STROKE, dashed=True,
    )
    c2 = box(
        ax, 3.25, 5.55, 2.55, 0.90,
        "Appariement\nfiltre + cosinus",
        fc="white", ec=C_PERC,
    )
    c3 = box(
        ax, 6.10, 5.40, 4.40, 1.20,
        "Énoncé IEMOCAP  ·  dialogue .wav\nsegment $[t_0, t_1]$  ·  16 kHz mono",
        fc=C_PERC_BOX, ec=C_PERC, fs=11,
    )
    arrow(ax, (c1[0] + 1.10, c1[1]), (c2[0] - 1.28, c2[1]), C_PERC)
    arrow(ax, (c2[0] + 1.28, c2[1]), (c3[0] - 2.20, c3[1]), C_PERC)

    layer(ax, 0.30, 2.20, 6.20, 1.95, "TRAITS", C_PERC, C_PERC_BG)
    t1 = box(
        ax, 0.80, 2.48, 2.55, 1.35,
        "WavLM tête DAiSEE\n4 états\n(juge prioritaire)",
        fc="white", ec=C_PERC, fs=11,
    )
    t2 = box(
        ax, 3.60, 2.48, 2.55, 1.35,
        "WavLM ancre ER\nneu / hap / ang / sad\n(projection DAiSEE)",
        fc="white", ec=C_PERC, fs=11,
    )

    layer(ax, 6.70, 2.20, 4.20, 1.95, "ÉTAT", C_DEC, C_DEC_BG, label_right=True)
    d1 = box(
        ax, 7.00, 2.48, 3.45, 1.35,
        "Fusion hybride\nFT prioritaire, $w_{\\mathrm{ancre}}=0{,}22$\n"
        + r"$\hat{e}_a$, $\hat{n}_a$",
        fc=C_DEC_BOX, ec=C_DEC, fs=11,
    )

    jy = 4.35
    ax.plot([c3[0], c3[0]], [c3[1] - 0.60, jy], color=C_PERC, lw=1.5, zorder=2.4)
    ax.plot([t1[0], c3[0]], [jy, jy], color=C_PERC, lw=1.5, zorder=2.4)
    arrow(ax, (t1[0], jy), (t1[0], t1[1] + 0.68), C_PERC)
    arrow(ax, (t2[0], jy), (t2[0], t2[1] + 0.68), C_PERC)

    arrow(ax, (t1[2] + t1[4], t1[1]), (t2[2], t2[1]), C_DEC)
    arrow(ax, (t2[2] + t2[4], t2[1]), (d1[2], d1[1]), C_DEC)

    out = box(
        ax, 0.30, 0.28, 10.60, 0.95,
        "Sortie vers la fusion   ·   repli : heuristique librosa (énergie, ZCR, $F_0$)",
        fc=C_IO, ec=C_IO_STROKE, dashed=True, fs=12,
    )
    arrow(ax, (d1[0], d1[1] - 0.68), (d1[0], out[1] + 0.48), C_DEC)

    fig.tight_layout(pad=0.2)
    fig.savefig(OUT, bbox_inches="tight", facecolor="white")
    print(OUT)


if __name__ == "__main__":
    main()
