"""Render EmoLearn architecture figure for LaTeX (matches .drawio layout)."""
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = Path(__file__).resolve().parent / "emolearn_architecture.drawio.png"

# Palette (professional, print-friendly)
C_TEXT = "#1a202c"
C_MUTED = "#4a5568"
C_PERC = "#2b6cb0"
C_PERC_BG = "#ebf8ff"
C_PERC_BOX = "#bee3f8"
C_DEC = "#2c7a7b"
C_DEC_BG = "#e6fffa"
C_DEC_BOX = "#b2f5ea"
C_GEN = "#b7791f"
C_GEN_BG = "#fffff0"
C_GEN_BOX = "#fefcbf"
C_IO = "#f7fafc"
C_IO_STROKE = "#4a5568"


def box(ax, x, y, w, h, text, *, fc, ec, lw=1.6, fs=12, bold=False, dashed=False):
    patch = FancyBboxPatch(
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
    ax.add_patch(patch)
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fs,
        color=C_TEXT,
        fontweight="bold",
        linespacing=1.25,
        zorder=4,
    )
    return (x + w / 2, y + h / 2, x, y, w, h)


def layer(ax, x, y, w, h, label, color, bg):
    patch = FancyBboxPatch(
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
    ax.add_patch(patch)
    ax.text(
        x + 0.22,
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
            mutation_scale=12,
            linewidth=1.5,
            color=color,
            shrinkA=2,
            shrinkB=2,
            zorder=2.5,
        )
    )


def main():
    fig, ax = plt.subplots(figsize=(11.2, 9.2), dpi=220)
    ax.set_xlim(0, 11.2)
    ax.set_ylim(0, 9.2)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    ax.text(
        5.6,
        8.92,
        "Architecture fonctionnelle en trois couches",
        ha="center",
        va="center",
        fontsize=16,
        fontweight="bold",
        color="#1a365d",
    )

    box(
        ax,
        2.3,
        7.95,
        6.6,
        0.80,
        "Entrée audiovisuelle\nClip vidéo (DAiSEE) + segment audio (IEMOCAP)",
        fc=C_IO,
        ec=C_IO_STROKE,
        lw=1.8,
        fs=12,
        dashed=True,
    )

    # Layers
    layer(ax, 0.45, 5.15, 10.3, 2.50, "PERCEPTION", C_PERC, C_PERC_BG)
    layer(ax, 0.45, 3.25, 10.3, 1.65, "DÉCISION", C_DEC, C_DEC_BG)
    layer(ax, 0.45, 1.40, 10.3, 1.60, "GÉNÉRATION", C_GEN, C_GEN_BG)

    fer = box(
        ax,
        1.15,
        5.95,
        3.15,
        1.40,
        "Module FER (visage)\nExpressions + unités d'action\n4 états",
        fc="white",
        ec=C_PERC,
        lw=1.8,
        fs=12,
    )
    ser = box(
        ax,
        6.90,
        5.95,
        3.15,
        1.40,
        "Module SER (voix)\nEncodeur de parole\n4 états",
        fc="white",
        ec=C_PERC,
        lw=1.8,
        fs=12,
    )
    fusion = box(
        ax,
        2.85,
        5.25,
        5.50,
        0.85,
        "Fusion multimodale\nVote à incertitude",
        fc=C_PERC_BOX,
        ec="#2c5282",
        lw=1.8,
        fs=12,
        bold=True,
    )

    decision = box(
        ax,
        2.35,
        3.42,
        6.50,
        1.20,
        "Moteur de décision\nType (A, B, C, D, A+D, D+A, abstention),\npatron A1–D5 et urgence",
        fc=C_DEC_BOX,
        ec="#234e52",
        lw=1.8,
        fs=12,
        bold=True,
    )

    rag = box(
        ax,
        1.15,
        1.58,
        2.70,
        1.15,
        "Ancrage\ndocumentaire\n(RAG)",
        fc="white",
        ec=C_GEN,
        lw=1.8,
        fs=12,
    )
    llm = box(
        ax,
        4.10,
        1.58,
        3.20,
        1.15,
        "Grand modèle de langue\nPrompt structuré\n+ contexte RAG",
        fc=C_GEN_BOX,
        ec="#975a16",
        lw=1.8,
        fs=12,
        bold=True,
    )
    cours = box(
        ax,
        7.55,
        1.58,
        2.70,
        1.15,
        "Contexte\npédagogique\n(cours)",
        fc="white",
        ec=C_GEN,
        lw=1.8,
        fs=12,
    )

    box(
        ax,
        2.10,
        0.28,
        7.00,
        0.90,
        "Sortie\nContenu pédagogique adaptatif\n+ journal de décision (traçabilité)",
        fc=C_IO,
        ec=C_IO_STROKE,
        lw=1.8,
        fs=12,
        dashed=True,
    )

    arrow(ax, (4.2, 7.95), (fer[0], fer[1] + 0.70), C_MUTED)
    arrow(ax, (7.0, 7.95), (ser[0], ser[1] + 0.70), C_MUTED)
    arrow(ax, (fer[0], fer[1] - 0.70), (fusion[0] - 1.1, fusion[1] + 0.42), C_PERC)
    arrow(ax, (ser[0], ser[1] - 0.70), (fusion[0] + 1.1, fusion[1] + 0.42), C_PERC)
    arrow(ax, (fusion[0], fusion[1] - 0.42), (decision[0], decision[1] + 0.60), C_DEC)
    arrow(ax, (decision[0], decision[1] - 0.60), (llm[0], llm[1] + 0.57), C_GEN)
    arrow(ax, (rag[0] + 1.35, rag[1]), (llm[0] - 1.60, llm[1]), C_GEN)
    arrow(ax, (cours[0] - 1.35, cours[1]), (llm[0] + 1.60, llm[1]), C_GEN)
    arrow(ax, (llm[0], llm[1] - 0.57), (5.6, 1.18), C_MUTED)

    fig.tight_layout(pad=0.25)
    fig.savefig(OUT, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
