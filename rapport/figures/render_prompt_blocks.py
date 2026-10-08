"""Composition du prompt contextuel (chapitre 3)."""
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = Path(__file__).resolve().parent / "emolearn_prompt_blocks.png"

C_TEXT = "#1a202c"
C_MUTED = "#4a5568"
C_SYS = "#edf2f7"
C_SYS_E = "#4a5568"
C_RAG = "#ebf8ff"
C_RAG_E = "#2b6cb0"
C_PED = "#e6fffa"
C_PED_E = "#2c7a7b"
C_AFF = "#faf5ff"
C_AFF_E = "#6b46c1"
C_TASK = "#fffff0"
C_TASK_E = "#b7791f"
C_OUT = "#f7fafc"
C_LOG = "#fffaf0"
C_LOG_E = "#975a16"


def box(ax, x, y, w, h, title, body, *, fc, ec):
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.02,rounding_size=0.08",
            facecolor=fc,
            edgecolor=ec,
            linewidth=1.7,
            zorder=3,
        )
    )
    ax.text(
        x + 0.16,
        y + h - 0.26,
        title,
        ha="left",
        va="center",
        fontsize=12,
        fontweight="bold",
        color=ec,
        zorder=4,
    )
    ax.text(
        x + 0.16,
        y + 0.32,
        body,
        ha="left",
        va="center",
        fontsize=11,
        fontweight="bold",
        color=C_TEXT,
        zorder=4,
    )


def arrow(ax, x1, y1, x2, y2):
    ax.add_patch(
        FancyArrowPatch(
            (x1, y1),
            (x2, y2),
            arrowstyle="-|>",
            mutation_scale=11,
            linewidth=1.4,
            color=C_MUTED,
            zorder=2,
        )
    )


def main():
    fig, ax = plt.subplots(figsize=(10.2, 8.6), dpi=220)
    ax.set_xlim(0, 10.2)
    ax.set_ylim(0, 8.6)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    ax.text(
        5.1,
        8.32,
        "Composition du prompt contextuel",
        ha="center",
        va="center",
        fontsize=16,
        fontweight="bold",
        color="#1a365d",
    )

    xl, w, h = 0.22, 6.55, 0.90
    rows = [
        (6.58, C_RAG, C_RAG_E, "1. Contexte de référence (RAG)",
         "Passages documentaires, ou contexte générique"),
        (5.52, C_PED, C_PED_E, "2. Contexte pédagogique",
         "Sujet, module, concept, niveau, objectif"),
        (4.46, C_AFF, C_AFF_E, "3. État d'engagement",
         "État, intensité, durée, pattern, moment"),
        (3.40, C_TASK, C_TASK_E, "4. Tâche et contraintes",
         "Patron A1–D5, objectif, longueur, format, ton"),
        (2.34, "#fff5f5", "#c53030", "5. Instructions du patron",
         "Structure numérotée A1–D5"),
    ]
    ax.text(
        xl + w / 2,
        7.62,
        "Prompt utilisateur",
        ha="center",
        va="center",
        fontsize=12,
        fontweight="bold",
        color=C_MUTED,
    )
    for i, (y, fc, ec, title, body) in enumerate(rows):
        box(ax, xl, y, w, h, title, body, fc=fc, ec=ec)
        if i < len(rows) - 1:
            arrow(ax, xl + w / 2, y, xl + w / 2, rows[i + 1][0] + h)

    xs, ws, ys, hs = 6.92, 3.05, 4.20, 1.70
    box(
        ax,
        xs,
        ys,
        ws,
        hs,
        "Message système",
        "Expert en pédagogie\nAucun fait inventé",
        fc=C_SYS,
        ec=C_SYS_E,
    )

    llm_x, llm_y, llm_w, llm_h = 0.22, 0.28, 6.55, 1.22
    ax.add_patch(
        FancyBboxPatch(
            (llm_x, llm_y),
            llm_w,
            llm_h,
            boxstyle="round,pad=0.02,rounding_size=0.08",
            facecolor=C_OUT,
            edgecolor=C_MUTED,
            linewidth=1.7,
            linestyle="--",
            zorder=3,
        )
    )
    ax.text(
        llm_x + llm_w / 2,
        llm_y + 0.80,
        "Grand modèle de langue (API)",
        ha="center",
        va="center",
        fontsize=13,
        fontweight="bold",
        color=C_TEXT,
        zorder=4,
    )
    ax.text(
        llm_x + llm_w / 2,
        llm_y + 0.40,
        "Contenu pédagogique adaptatif en français",
        ha="center",
        va="center",
        fontsize=11,
        fontweight="bold",
        color=C_MUTED,
        zorder=4,
    )

    log_x, log_y, log_w, log_h = 6.92, 0.28, 3.05, 1.22
    ax.add_patch(
        FancyBboxPatch(
            (log_x, log_y),
            log_w,
            log_h,
            boxstyle="round,pad=0.02,rounding_size=0.08",
            facecolor=C_LOG,
            edgecolor=C_LOG_E,
            linewidth=1.7,
            linestyle="--",
            zorder=3,
        )
    )
    ax.text(
        log_x + log_w / 2,
        log_y + 0.88,
        "Journal (backend)",
        ha="center",
        va="center",
        fontsize=12,
        fontweight="bold",
        color=C_LOG_E,
        zorder=4,
    )
    ax.text(
        log_x + log_w / 2,
        log_y + 0.42,
        "État, patron, sources,\nnb mots — après l'appel",
        ha="center",
        va="center",
        fontsize=11,
        fontweight="bold",
        color=C_TEXT,
        zorder=4,
    )

    arrow(ax, xl + w / 2, 2.34, llm_x + llm_w / 2, llm_y + llm_h)
    arrow(ax, xs + ws / 2, ys, llm_x + llm_w * 0.82, llm_y + llm_h)

    fig.tight_layout(pad=0.3)
    fig.savefig(OUT, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
