"""Architecture générale d'un LLM (Transformer décodeur) — flux séquentiel."""
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = Path(__file__).resolve().parent / "llm_architecture_generale.png"


def box(ax, x, y, w, h, text, *, fc, ec, fs=8.5, bold=False, dashed=False):
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.02,rounding_size=0.08",
            facecolor=fc,
            edgecolor=ec,
            linewidth=1.7,
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
        fontweight="bold" if bold else "normal",
        color="#1a202c",
        linespacing=1.3,
        zorder=4,
    )


def arrow(ax, x1, y1, x2, y2, color="#4a5568"):
    ax.add_patch(
        FancyArrowPatch(
            (x1, y1),
            (x2, y2),
            arrowstyle="-|>",
            mutation_scale=12,
            linewidth=1.6,
            color=color,
            zorder=2,
        )
    )


def main():
    fig, ax = plt.subplots(figsize=(9.2, 8.4), dpi=220)
    ax.set_xlim(0, 9.2)
    ax.set_ylim(0, 8.4)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    ax.text(
        4.6,
        8.15,
        "Architecture générale d'un LLM (Transformer décodeur)",
        ha="center",
        va="center",
        fontsize=13,
        fontweight="bold",
        color="#1a365d",
    )

    # Flux séquentiel vertical
    box(
        ax, 2.7, 7.35, 3.8, 0.55,
        "Texte d'entrée",
        fc="#f7fafc", ec="#4a5568", dashed=True,
    )
    box(
        ax, 2.3, 6.35, 4.6, 0.7,
        "Tokenisation\nmots / sous-mots → identifiants numériques (IDs)",
        fc="white", ec="#2b6cb0",
    )
    box(
        ax, 2.3, 5.35, 4.6, 0.7,
        "Embeddings + encodage de position\nIDs → vecteurs denses contextualisés en position",
        fc="white", ec="#2b6cb0",
    )

    # Bloc Transformer
    ax.add_patch(
        FancyBboxPatch(
            (1.2, 2.55),
            6.8,
            2.5,
            boxstyle="round,pad=0.02,rounding_size=0.12",
            facecolor="#ebf8ff",
            edgecolor="#2b6cb0",
            linewidth=1.5,
            zorder=1,
        )
    )
    ax.text(
        4.6,
        4.8,
        "Bloc Transformer × N  (couches empilées)",
        ha="center",
        va="center",
        fontsize=10,
        fontweight="bold",
        color="#2b6cb0",
        zorder=2,
    )
    box(
        ax, 1.7, 3.7, 5.8, 0.85,
        "Self-attention multi-têtes\nchaque token contextualise les précédents (causale)",
        fc="white", ec="#2c5282", bold=True, fs=8,
    )
    box(
        ax, 1.7, 2.7, 5.8, 0.8,
        "Réseau feed-forward (MLP)\n+ résidus / normalisation",
        fc="white", ec="#2c5282", bold=True, fs=8,
    )

    box(
        ax, 2.0, 1.7, 5.2, 0.6,
        "Tête de langage (projection → vocabulaire)",
        fc="#fffff0", ec="#b7791f", bold=True,
    )
    box(
        ax, 2.0, 0.85, 5.2, 0.6,
        "Softmax → prochain token  (génération auto-régressive)",
        fc="#fefcbf", ec="#975a16", bold=True, fs=8,
    )
    box(
        ax, 2.9, 0.12, 3.4, 0.5,
        "Texte généré",
        fc="#f7fafc", ec="#4a5568", dashed=True,
    )

    # Flèches verticales
    for y1, y2 in (
        (7.35, 7.05),
        (6.35, 6.05),
        (5.35, 5.05),
        (3.7, 3.5),
        (2.7, 2.3),
        (1.7, 1.45),
        (0.85, 0.62),
    ):
        arrow(ax, 4.6, y1, 4.6, y2)

    # Boucle auto-régressive : token → réinjection en entrée des embeddings
    ax.annotate(
        "",
        xy=(8.0, 5.7),
        xytext=(8.0, 1.15),
        arrowprops=dict(arrowstyle="-", color="#718096", lw=1.4),
    )
    arrow(ax, 8.0, 5.7, 6.9, 5.7, "#718096")
    arrow(ax, 7.2, 1.15, 8.0, 1.15, "#718096")
    ax.text(
        8.2,
        3.4,
        "boucle\nauto-\nrégressive",
        ha="left",
        va="center",
        fontsize=7.5,
        color="#718096",
        style="italic",
    )

    fig.tight_layout(pad=0.3)
    fig.savefig(OUT, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
