"""Render EmoLearn LLM / generation architecture for LaTeX."""
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = Path(__file__).resolve().parent / "emolearn_llm_architecture.png"

C_TEXT = "#1a202c"
C_MUTED = "#4a5568"
C_IN = "#edf2f7"
C_DEC = "#e6fffa"
C_DEC_E = "#2c7a7b"
C_RAG = "#ebf8ff"
C_RAG_E = "#2b6cb0"
C_LLM = "#fffff0"
C_LLM_E = "#b7791f"
C_OUT = "#f7fafc"


def box(ax, x, y, w, h, text, *, fc, ec, lw=1.6, fs=8, bold=False, dashed=False):
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
        fontweight="bold" if bold else "normal",
        linespacing=1.25,
        zorder=4,
    )
    return x + w / 2, y + h / 2


def arrow(ax, p1, p2, color=C_MUTED):
    ax.add_patch(
        FancyArrowPatch(
            p1,
            p2,
            arrowstyle="-|>",
            mutation_scale=11,
            linewidth=1.5,
            color=color,
            shrinkA=3,
            shrinkB=3,
            zorder=2,
        )
    )


def band(ax, x, y, w, h, label, color, bg):
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.02,rounding_size=0.1",
            facecolor=bg,
            edgecolor=color,
            linewidth=1.1,
            zorder=1,
        )
    )
    ax.text(
        x + 0.14,
        y + h / 2,
        label,
        ha="center",
        va="center",
        fontsize=8,
        fontweight="bold",
        color=color,
        rotation=90,
        zorder=2,
    )


def main():
    fig, ax = plt.subplots(figsize=(10.5, 7.6), dpi=220)
    ax.set_xlim(0, 10.5)
    ax.set_ylim(0, 7.6)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    ax.text(
        5.25,
        7.35,
        "EmoLearn — Architecture de génération (LLM)",
        ha="center",
        va="center",
        fontsize=13,
        fontweight="bold",
        color="#1a365d",
    )

    # Inputs
    box(
        ax,
        0.7,
        6.35,
        3.0,
        0.7,
        "État fusionné\n+ scores DAiSEE",
        fc=C_IN,
        ec=C_MUTED,
        dashed=True,
        fs=8,
    )
    box(
        ax,
        3.9,
        6.35,
        2.8,
        0.7,
        "Contexte pédagogique\nconcept / module / niveau",
        fc=C_IN,
        ec=C_MUTED,
        dashed=True,
        fs=8,
    )
    box(
        ax,
        6.9,
        6.35,
        2.8,
        0.7,
        "Règles temporelles\n(déclenchement)",
        fc=C_IN,
        ec=C_MUTED,
        dashed=True,
        fs=8,
    )

    # Decision band
    band(ax, 0.45, 4.55, 9.6, 1.55, "DÉCISION", C_DEC_E, C_DEC)
    box(
        ax,
        1.0,
        4.8,
        3.6,
        1.05,
        "Moteur de décision\nÉtat → type A/B/C/D\n+ stratégie pédagogique",
        fc="white",
        ec=C_DEC_E,
        fs=8,
        bold=True,
    )
    box(
        ax,
        5.2,
        4.8,
        4.2,
        1.05,
        "Sélection template (×20)\nA1–D5  ·  longueur / format / ton\ninstructions spécifiques",
        fc="#b2f5ea",
        ec="#234e52",
        fs=8,
        bold=True,
    )

    # RAG band
    band(ax, 0.45, 2.55, 9.6, 1.75, "RAG", C_RAG_E, C_RAG)
    box(
        ax,
        0.9,
        2.85,
        2.7,
        1.15,
        "Base documentaire\npédagogique\n(définitions, exemples…)",
        fc="white",
        ec=C_RAG_E,
        fs=7.5,
    )
    box(
        ax,
        3.85,
        2.85,
        2.9,
        1.15,
        "Embeddings\nMiniLM multilingue\n+ ChromaDB",
        fc="white",
        ec=C_RAG_E,
        fs=7.5,
    )
    box(
        ax,
        7.0,
        2.85,
        2.6,
        1.15,
        "Retrieval top-k\npassages pertinents\n(ancrage factuel)",
        fc="#bee3f8",
        ec="#2c5282",
        fs=7.5,
        bold=True,
    )

    # LLM band
    band(ax, 0.45, 0.85, 9.6, 1.45, "LLM", C_LLM_E, C_LLM)
    box(
        ax,
        0.9,
        1.05,
        3.5,
        1.05,
        "Prompt structuré\n[Contexte] [État] [Tâche]\n[Contraintes] [Instructions]",
        fc="white",
        ec=C_LLM_E,
        fs=7.5,
    )
    box(
        ax,
        4.7,
        1.05,
        4.9,
        1.05,
        "API Groq  ·  GPT-OSS 20B\nsystem + user  ·  temp. / max_tokens\n(clé GROQ_API_KEY)",
        fc="#fefcbf",
        ec="#975a16",
        fs=8,
        bold=True,
    )

    # Output
    box(
        ax,
        2.4,
        0.12,
        5.7,
        0.55,
        "Sortie : contenu pédagogique adaptatif + sources RAG + journal de décision",
        fc=C_OUT,
        ec=C_MUTED,
        dashed=True,
        fs=8,
    )

    # Arrows (approximate)
    arrow(ax, (2.2, 6.35), (2.2, 5.85), C_MUTED)
    arrow(ax, (5.3, 6.35), (5.3, 5.85), C_MUTED)
    arrow(ax, (8.3, 6.35), (7.2, 5.85), C_MUTED)
    arrow(ax, (2.8, 4.8), (5.2, 5.3), C_DEC_E)
    arrow(ax, (7.3, 4.8), (8.3, 4.0), C_RAG_E)
    arrow(ax, (2.25, 2.85), (3.85, 3.4), C_RAG_E)
    arrow(ax, (5.3, 2.85), (7.0, 3.4), C_RAG_E)
    arrow(ax, (8.3, 2.85), (7.5, 2.1), C_LLM_E)
    arrow(ax, (7.3, 4.8), (2.6, 2.1), C_LLM_E)
    arrow(ax, (2.65, 1.05), (4.7, 1.55), C_LLM_E)
    arrow(ax, (7.15, 1.05), (5.25, 0.67), C_MUTED)

    fig.tight_layout(pad=0.25)
    fig.savefig(OUT, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
