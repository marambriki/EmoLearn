#!/usr/bin/env python3
"""
Analyse exploratoire IEMOCAP — regénère PNG + CSV résumé.
Usage: python scripts/iemocap_exploratory_analysis.py
"""

from __future__ import annotations

import json
import os
import re
import sys

import matplotlib.pyplot as plt
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "backend"))

from daisee_utils import etat_dominant_depuis_scores

DATA_DIR = os.path.join(ROOT, "data")
RESULTS_DIR = os.path.join(ROOT, "results")

IEMOCAP_CSV = os.path.join(DATA_DIR, "iemocap_parsed.csv")
SCENARIOS_CSV = os.path.join(DATA_DIR, "scenarios_50.csv")
TRAIN_LABELS = os.path.join(DATA_DIR, "DAiSEE", "Labels", "TrainLabels.csv")

OUT_ANALYSE_PNG = os.path.join(DATA_DIR, "iemocap_analyse.png")
OUT_COMPARAISON_PNG = os.path.join(DATA_DIR, "comparaison_daisee_iemocap.png")
OUT_SUMMARY_CSV = os.path.join(RESULTS_DIR, "iemocap_exploratory_summary.csv")
OUT_STATS_JSON = os.path.join(RESULTS_DIR, "iemocap_exploratory_stats.json")


def _session_from_id(utterance_id: str) -> str:
    m = re.match(r"(Ses\d+)", str(utterance_id))
    return m.group(1) if m else "unknown"


def charger_iemocap() -> pd.DataFrame:
    if not os.path.exists(IEMOCAP_CSV):
        raise FileNotFoundError(
            f"{IEMOCAP_CSV} introuvable. Exécutez d'abord OS1multi.ipynb ou iemocap_parser."
        )
    df = pd.read_csv(IEMOCAP_CSV)
    df["session"] = df["utterance_id"].apply(_session_from_id)
    return df


def dominant_daisee_scenarios() -> pd.Series:
    df = pd.read_csv(SCENARIOS_CSV)
    df.columns = [c.strip() for c in df.columns]
    counts = {"Boredom": 0, "Confusion": 0, "Engagement": 0, "Frustration": 0}
    for _, row in df.iterrows():
        etat, _, _ = etat_dominant_depuis_scores(
            row["Boredom"], row["Confusion"], row["Engagement"], row["Frustration"]
        )
        counts[etat] += 1
    return pd.Series(counts)


def dominant_daisee_train() -> pd.Series | None:
    if not os.path.exists(TRAIN_LABELS):
        return None
    df = pd.read_csv(TRAIN_LABELS)
    df.columns = [c.strip() for c in df.columns]
    counts = {"Boredom": 0, "Confusion": 0, "Engagement": 0, "Frustration": 0}
    for _, row in df.iterrows():
        etat, _, _ = etat_dominant_depuis_scores(
            row["Boredom"], row["Confusion"], row["Engagement"], row["Frustration"]
        )
        counts[etat] += 1
    return pd.Series(counts)


def build_summary_rows(df: pd.DataFrame) -> list[dict]:
    rows = []
    total = len(df)

    rows.append({"categorie": "global", "metrique": "total_utterances", "valeur": total, "unite": "count"})

    for col in ("valence", "activation", "dominance"):
        rows.append({
            "categorie": "vad_global",
            "metrique": f"{col}_mean",
            "valeur": round(float(df[col].mean()), 3),
            "unite": "1-5",
        })
        rows.append({
            "categorie": "vad_global",
            "metrique": f"{col}_std",
            "valeur": round(float(df[col].std()), 3),
            "unite": "1-5",
        })
        rows.append({
            "categorie": "vad_global",
            "metrique": f"{col}_min",
            "valeur": round(float(df[col].min()), 3),
            "unite": "1-5",
        })
        rows.append({
            "categorie": "vad_global",
            "metrique": f"{col}_max",
            "valeur": round(float(df[col].max()), 3),
            "unite": "1-5",
        })

    rows.append({
        "categorie": "duree",
        "metrique": "duree_moyenne_sec",
        "valeur": round(float(df["duree"].mean()), 3),
        "unite": "secondes",
    })

    for emotion, count in df["emotion"].value_counts().items():
        pct = round(100 * count / total, 2)
        rows.append({
            "categorie": "emotion_iemocap",
            "metrique": str(emotion),
            "valeur": int(count),
            "unite": f"count ({pct}%)",
        })

    for etat, count in df["etat_daisee"].value_counts().items():
        pct = round(100 * count / total, 2)
        rows.append({
            "categorie": "etat_daisee_mappe",
            "metrique": str(etat),
            "valeur": int(count),
            "unite": f"count ({pct}%)",
        })

    for session, grp in df.groupby("session"):
        rows.append({
            "categorie": "par_session",
            "metrique": f"{session}_utterances",
            "valeur": len(grp),
            "unite": "count",
        })

    for etat in ("Boredom", "Confusion", "Engagement", "Frustration"):
        sub = df[df["etat_daisee"] == etat]
        if len(sub) == 0:
            continue
        rows.append({
            "categorie": "vad_par_etat",
            "metrique": f"{etat}_activation_mean",
            "valeur": round(float(sub["activation"].mean()), 3),
            "unite": "1-5",
        })

    mapping_path = os.path.join(DATA_DIR, "mapping_iemocap_daisee.csv")
    if os.path.exists(mapping_path):
        rows.append({
            "categorie": "fichier",
            "metrique": "mapping_iemocap_daisee",
            "valeur": mapping_path,
            "unite": "path",
        })

    return rows


def plot_iemocap_analyse(df: pd.DataFrame, path: str) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Analyse exploratoire IEMOCAP", fontsize=14, fontweight="bold")

    emotion_counts = df["emotion"].value_counts()
    colors1 = ["#FF6B6B", "#FFA07A", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFEAA7", "#DDA0DD", "#778899"]
    axes[0, 0].bar(emotion_counts.index, emotion_counts.values, color=colors1[: len(emotion_counts)])
    axes[0, 0].set_title("Distribution des émotions IEMOCAP", fontweight="bold")
    axes[0, 0].set_xlabel("Émotion")
    axes[0, 0].set_ylabel("Nombre d'utterances")
    axes[0, 0].tick_params(axis="x", rotation=45)
    for i, v in enumerate(emotion_counts.values):
        axes[0, 0].text(i, v + 20, str(v), ha="center", fontsize=9)

    daisee_counts = df["etat_daisee"].value_counts()
    colors2 = ["#FF6B6B", "#45B7D1", "#4ECDC4", "#FFA07A"]
    axes[0, 1].bar(daisee_counts.index, daisee_counts.values, color=colors2[: len(daisee_counts)])
    axes[0, 1].set_title("États DAiSEE mappés depuis IEMOCAP", fontweight="bold")
    axes[0, 1].set_ylabel("Nombre d'utterances")
    for i, (k, v) in enumerate(daisee_counts.items()):
        axes[0, 1].text(i, v + 50, str(v), ha="center", fontsize=10)

    axes[1, 0].hist(df["valence"], bins=20, alpha=0.6, label="Valence", color="#FF6B6B")
    axes[1, 0].hist(df["activation"], bins=20, alpha=0.6, label="Activation", color="#4ECDC4")
    axes[1, 0].hist(df["dominance"], bins=20, alpha=0.6, label="Dominance", color="#45B7D1")
    axes[1, 0].set_title("Distribution VAD — IEMOCAP", fontweight="bold")
    axes[1, 0].set_xlabel("Niveau (1-5)")
    axes[1, 0].set_ylabel("Fréquence")
    axes[1, 0].legend()

    df.boxplot(
        column="activation",
        by="etat_daisee",
        ax=axes[1, 1],
        patch_artist=True,
    )
    axes[1, 1].set_title("Activation vocale par état DAiSEE mappé", fontweight="bold")
    axes[1, 1].set_xlabel("État DAiSEE")
    axes[1, 1].set_ylabel("Activation (1-5)")
    plt.suptitle("")

    plt.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_comparaison_daisee_iemocap(
    df_iemocap: pd.DataFrame,
    daisee_full: pd.Series | None,
    daisee_scenarios: pd.Series,
    path: str,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        "Comparaison exploratoire DAiSEE vs IEMOCAP",
        fontsize=14,
        fontweight="bold",
    )
    couleurs = ["#FF6B6B", "#45B7D1", "#4ECDC4", "#FFA07A"]

    if daisee_full is not None:
        axes[0, 0].bar(daisee_full.index, daisee_full.values, color=couleurs)
        axes[0, 0].set_title("DAiSEE — États dominants\n(dataset TrainLabels)", fontweight="bold")
        for i, v in enumerate(daisee_full.values):
            axes[0, 0].text(i, v + 50, str(v), ha="center", fontsize=9)
    else:
        axes[0, 0].bar(daisee_scenarios.index, daisee_scenarios.values, color=couleurs)
        axes[0, 0].set_title("DAiSEE — États dominants\n(50 scénarios prototype)", fontweight="bold")
        for i, v in enumerate(daisee_scenarios.values):
            axes[0, 0].text(i, v + 0.5, str(v), ha="center", fontsize=10)

    iemocap_etats = df_iemocap["etat_daisee"].value_counts()
    axes[0, 1].bar(iemocap_etats.index, iemocap_etats.values, color=couleurs[: len(iemocap_etats)])
    axes[0, 1].set_title("IEMOCAP — États mappés\n(modalité voix)", fontweight="bold")
    for i, (k, v) in enumerate(iemocap_etats.items()):
        axes[0, 1].text(i, v + 50, str(v), ha="center", fontsize=10)

    etats = ["Boredom", "Confusion", "Engagement", "Frustration"]
    ref = daisee_full if daisee_full is not None else daisee_scenarios
    ref_norm = ref.reindex(etats, fill_value=0)
    ref_norm = ref_norm / ref_norm.sum() * 100 if ref_norm.sum() else ref_norm
    iem_norm = iemocap_etats.reindex(etats, fill_value=0)
    iem_norm = iem_norm / iem_norm.sum() * 100 if iem_norm.sum() else iem_norm

    x = range(len(etats))
    width = 0.35
    axes[1, 0].bar([i - width / 2 for i in x], ref_norm.values, width, label="DAiSEE", color="#4ECDC4")
    axes[1, 0].bar([i + width / 2 for i in x], iem_norm.values, width, label="IEMOCAP", color="#FF6B6B")
    axes[1, 0].set_xticks(list(x))
    axes[1, 0].set_xticklabels(etats, rotation=15)
    axes[1, 0].set_title("Proportions normalisées par état", fontweight="bold")
    axes[1, 0].set_ylabel("%")
    axes[1, 0].legend()

    session_counts = df_iemocap["session"].value_counts().sort_index()
    axes[1, 1].bar(session_counts.index, session_counts.values, color="#45B7D1")
    axes[1, 1].set_title("Utterances IEMOCAP par session", fontweight="bold")
    axes[1, 1].set_xlabel("Session")
    axes[1, 1].tick_params(axis="x", rotation=45)
    for i, v in enumerate(session_counts.values):
        axes[1, 1].text(i, v + 30, str(v), ha="center", fontsize=9)

    plt.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    df = charger_iemocap()
    daisee_scenarios = dominant_daisee_scenarios()
    daisee_train = dominant_daisee_train()

    plot_iemocap_analyse(df, OUT_ANALYSE_PNG)
    plot_comparaison_daisee_iemocap(df, daisee_train, daisee_scenarios, OUT_COMPARAISON_PNG)

    summary_rows = build_summary_rows(df)
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(OUT_SUMMARY_CSV, index=False, encoding="utf-8-sig")

    stats = {
        "total_utterances": len(df),
        "emotions": df["emotion"].value_counts().to_dict(),
        "etats_daisee": df["etat_daisee"].value_counts().to_dict(),
        "vad": {
            "valence_mean": round(float(df["valence"].mean()), 3),
            "activation_mean": round(float(df["activation"].mean()), 3),
            "dominance_mean": round(float(df["dominance"].mean()), 3),
        },
        "sessions": df["session"].value_counts().sort_index().to_dict(),
        "outputs": {
            "iemocap_analyse_png": OUT_ANALYSE_PNG,
            "comparaison_png": OUT_COMPARAISON_PNG,
            "summary_csv": OUT_SUMMARY_CSV,
        },
    }
    with open(OUT_STATS_JSON, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    print(f"IEMOCAP utterances : {len(df)}")
    print(f"PNG analyse        : {OUT_ANALYSE_PNG}")
    print(f"PNG comparaison    : {OUT_COMPARAISON_PNG}")
    print(f"CSV resume         : {OUT_SUMMARY_CSV}")
    print(f"JSON stats         : {OUT_STATS_JSON}")


if __name__ == "__main__":
    main()
