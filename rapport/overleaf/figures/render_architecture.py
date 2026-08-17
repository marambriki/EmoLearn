"""Render EmoLearn architecture figure for LaTeX (matches .drawio layout)."""
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = Path(__file__).resolve().parent / "emolearn_architecture.png"

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


def box(ax, x, y, w, h, text, *, fc, ec, lw=1.6, fs=9, bold=False, dashed=False):
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
        fontweight="bold" if bold else "normal",
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
        x + 0.12,
        y + h / 2,
        label,
        ha="center",
        va="center",
        fontsize=8.5,
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
    fig, ax = plt.subplots(figsize=(10.2, 8.4), dpi=220)
    ax.set_xlim(0, 10.2)
    ax.set_ylim(0, 8.4)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    ax.text(
        5.1,
        8.15,
        "EmoLearn — Architecture de bout en bout",
        ha="center",
        va="center",
        fontsize=13,
        fontweight="bold",
        color="#1a365d",
    )

    # I/O
    box(
        ax,
        2.7,
        7.35,
        4.8,
        0.65,
        "Entrée\nClip vidéo (DAiSEE) + segment audio (IEMOCAP)\n+ contexte pédagogique",
        fc=C_IO,
        ec=C_IO_STROKE,
        lw=1.8,
        fs=8,
        dashed=True,
    )

    # Layers
    layer(ax, 0.55, 4.85, 9.1, 2.25, "PERCEPTION", C_PERC, C_PERC_BG)
    layer(ax, 0.55, 3.15, 9.1, 1.45, "DÉCISION", C_DEC, C_DEC_BG)
    layer(ax, 0.55, 1.45, 9.1, 1.45, "GÉNÉRATION", C_GEN, C_GEN_BG)

    fer = box(
        ax,
        1.1,
        5.55,
        2.6,
        1.25,
        "Module FER (visage)\nViT fine-tuné DAiSEE\nVote multi-frames\n4 états",
        fc="white",
        ec=C_PERC,
        lw=1.8,
        fs=8,
    )
    ser = box(
        ax,
        6.5,
        5.55,
        2.6,
        1.25,
        "Module SER (voix)\nwav2vec2 fine-tuné\nDAiSEE (IEMOCAP)\n4 états",
        fc="white",
        ec=C_PERC,
        lw=1.8,
        fs=8,
    )
    fusion = box(
        ax,
        2.85,
        4.90,
        4.5,
        0.72,
        "Fusion multimodale\nSoft : vote pondéré (60 % visage / 40 % voix)\nLegacy : priorité pédagogique  ·  Gate de confiance",
        fc=C_PERC_BOX,
        ec="#2c5282",
        lw=1.8,
        fs=7.5,
        bold=True,
    )

    temp = box(
        ax,
        1.8,
        3.45,
        2.6,
        0.85,
        "Règles temporelles\n(déclenchement)",
        fc="white",
        ec=C_DEC,
        lw=1.8,
        fs=8,
    )
    decision = box(
        ax,
        5.6,
        3.45,
        2.8,
        0.85,
        "Moteur de décision\nÉtat → type de contenu\n+ template (×20)",
        fc=C_DEC_BOX,
        ec="#234e52",
        lw=1.8,
        fs=8,
        bold=True,
    )

    rag = box(
        ax,
        1.8,
        1.65,
        2.8,
        1.0,
        "Module RAG\nChromaDB + embeddings\nancrage documentaire",
        fc="white",
        ec=C_GEN,
        lw=1.8,
        fs=8,
    )
    llm = box(
        ax,
        5.6,
        1.65,
        2.8,
        1.0,
        "LLM (Groq / Llama)\nPrompt structuré\n+ contexte RAG",
        fc=C_GEN_BOX,
        ec="#975a16",
        lw=1.8,
        fs=8,
        bold=True,
    )

    box(
        ax,
        2.4,
        0.35,
        5.4,
        0.75,
        "Sortie\nContenu pédagogique adaptatif + journal de décision (traçabilité)",
        fc=C_IO,
        ec=C_IO_STROKE,
        lw=1.8,
        fs=8,
        dashed=True,
    )

    # Arrows (approximate midpoints)
    arrow(ax, (4.0, 7.35), (fer[0], fer[1] + 0.62), C_MUTED)
    arrow(ax, (6.2, 7.35), (ser[0], ser[1] + 0.62), C_MUTED)
    arrow(ax, (fer[0], fer[1] - 0.62), (fusion[0] - 0.9, fusion[1] + 0.28), C_PERC)
    arrow(ax, (ser[0], ser[1] - 0.62), (fusion[0] + 0.9, fusion[1] + 0.28), C_PERC)
    arrow(ax, (fusion[0] - 0.6, fusion[1] - 0.28), (temp[0], temp[1] + 0.42), C_DEC)
    arrow(ax, (fusion[0] + 0.6, fusion[1] - 0.28), (decision[0], decision[1] + 0.42), C_DEC)
    arrow(ax, (temp[0] + 1.3, temp[1]), (decision[0] - 1.4, decision[1]), C_DEC)
    arrow(ax, (decision[0], decision[1] - 0.42), (llm[0], llm[1] + 0.5), C_GEN)
    arrow(ax, (rag[0] + 1.4, rag[1]), (llm[0] - 1.4, llm[1]), C_GEN)
    arrow(ax, (llm[0], llm[1] - 0.5), (5.1, 1.1), C_MUTED)

    fig.tight_layout(pad=0.25)
    fig.savefig(OUT, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
