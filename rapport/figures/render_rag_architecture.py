"""Architecture RAG — schéma académique simple (boîtes + flèches)."""
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = Path(__file__).resolve().parent / "rag_architecture_generale.png"

INK = "#111827"
MUTED = "#4b5563"
LINE = "#374151"
TEAL = "#0f766e"
TEAL_BG = "#ecfdf5"
BLUE = "#1d4ed8"
BLUE_BG = "#eff6ff"
GRAY = "#f3f4f6"
WHITE = "#ffffff"
BORDER = "#9ca3af"


def box(ax, x, y, w, h, title, sub=None, *, fc=WHITE, ec=BORDER):
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.02,rounding_size=0.06",
            facecolor=fc,
            edgecolor=ec,
            linewidth=1.5,
            zorder=3,
        )
    )
    cx, cy = x + w / 2, y + h / 2
    if sub:
        ax.text(cx, cy + 0.14, title, ha="center", va="center", fontsize=10,
                fontweight="bold", color=INK, zorder=4)
        ax.text(cx, cy - 0.18, sub, ha="center", va="center", fontsize=8,
                color=MUTED, zorder=4)
    else:
        ax.text(cx, cy, title, ha="center", va="center", fontsize=10,
                fontweight="bold", color=INK, zorder=4)
    return cx, cy


def arrow(ax, p1, p2, *, color=LINE, lw=1.5):
    ax.add_patch(
        FancyArrowPatch(
            p1, p2, arrowstyle="-|>", mutation_scale=12, linewidth=lw,
            color=color, shrinkA=2, shrinkB=2, zorder=2,
        )
    )


def label(ax, x, y, text, *, color=MUTED, fs=8.5, bold=False):
    ax.text(
        x, y, text, ha="center", va="center", fontsize=fs,
        color=color, fontweight="bold" if bold else "normal", zorder=5,
    )


def main():
    fig, ax = plt.subplots(figsize=(10.2, 6.4), dpi=240)
    ax.set_xlim(0, 10.2)
    ax.set_ylim(0, 6.4)
    ax.axis("off")
    fig.patch.set_facecolor(WHITE)

    ax.text(5.1, 6.1, "Architecture du modèle RAG", ha="center", va="center",
            fontsize=13.5, fontweight="bold", color=INK)
    ax.text(5.1, 5.75, "Retrieval-Augmented Generation", ha="center", va="center",
            fontsize=9.5, color=MUTED)

    # --- Indexation (haut) ---
    label(ax, 5.1, 5.35, "1. Indexation (hors ligne)", color=TEAL, fs=9, bold=True)
    box(ax, 1.2, 4.35, 2.8, 0.75, "Documents sources", fc=GRAY)
    box(ax, 4.7, 4.35, 2.4, 0.75, "Chunking +\nembeddings", fc=TEAL_BG, ec=TEAL)
    box(ax, 7.7, 4.35, 2.4, 0.75, "Base vectorielle", fc=TEAL_BG, ec=TEAL)
    arrow(ax, (4.05, 4.72), (4.65, 4.72), color=TEAL)
    arrow(ax, (7.15, 4.72), (7.65, 4.72), color=TEAL)

    # --- Requête (milieu) ---
    label(ax, 5.1, 3.85, "2. Requête et récupération (en ligne)", color=TEAL, fs=9, bold=True)
    box(ax, 1.2, 2.85, 2.8, 0.75, "Question\nutilisateur", fc=GRAY)
    box(ax, 4.7, 2.85, 2.4, 0.75, "Embedding\nde la requête", fc=BLUE_BG, ec=BLUE)
    box(ax, 7.7, 2.85, 2.4, 0.75, "Retrieval\ntop-k", fc=BLUE_BG, ec=BLUE)
    arrow(ax, (4.05, 3.22), (4.65, 3.22), color=BLUE)
    arrow(ax, (7.15, 3.22), (7.65, 3.22), color=BLUE)
    # lien index → retrieval
    arrow(ax, (8.9, 4.35), (8.9, 3.65), color=BORDER, lw=1.25)
    label(ax, 9.35, 4.0, "index", color=MUTED, fs=7.5)

    # Passages
    box(ax, 4.7, 1.85, 5.4, 0.6, "Passages pertinents (contexte documentaire)", fc=GRAY)

    arrow(ax, (8.9, 2.85), (8.9, 2.5), color=BLUE)
    arrow(ax, (2.6, 2.85), (2.6, 1.35), color=BORDER, lw=1.2)

    # --- Génération (bas) ---
    label(ax, 5.1, 1.5, "3. Génération", color=TEAL, fs=9, bold=True)
    box(ax, 1.2, 0.45, 2.8, 0.75, "Prompt enrichi\nquestion + passages", fc=TEAL_BG, ec=TEAL)
    box(ax, 4.7, 0.45, 2.4, 0.75, "LLM", fc=BLUE_BG, ec=BLUE)
    box(ax, 7.7, 0.45, 2.4, 0.75, "Réponse ancrée", fc=GRAY)

    arrow(ax, (5.5, 1.85), (2.6, 1.25), color=TEAL, lw=1.3)
    arrow(ax, (4.05, 0.82), (4.65, 0.82), color=BLUE)
    arrow(ax, (7.15, 0.82), (7.65, 0.82), color=LINE)

    fig.savefig(OUT, dpi=240, bbox_inches="tight", facecolor=WHITE, pad_inches=0.15)
    plt.close(fig)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
