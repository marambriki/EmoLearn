#!/usr/bin/env python3
# ============================================
# Métriques automatiques sur corpus de contenus
# ============================================

from __future__ import annotations

import argparse
import json
import os
import re
import sys

import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RESULTS = os.path.join(ROOT, "results")


def syllables_approx_fr(text: str) -> int:
    """Estimation grossière des syllabes pour lisibilité."""
    text = text.lower()
    text = re.sub(r"[^a-zàâäéèêëïîôùûüç\s]", " ", text)
    vowels = "aeiouyàâäéèêëïîôùûü"
    count = 0
    prev_v = False
    for ch in text:
        is_v = ch in vowels
        if is_v and not prev_v:
            count += 1
        prev_v = is_v
    return max(count, 1)


def kandel_moles_approx(nb_mots: int, nb_syllabes: int) -> float:
    """Indice Kandel-Moles adapté (échelle ~100, plus haut = plus facile)."""
    if nb_mots == 0:
        return 0.0
    ps = nb_syllabes / nb_mots * 100
    pl = 100 / nb_mots * 30  # proxy longueur phrase ~30 mots
    return round(0.5 * ps + 0.5 * pl, 1)


def repetition_rate(text: str) -> float:
    words = re.findall(r"\b\w+\b", text.lower())
    if len(words) < 5:
        return 0.0
    unique = len(set(words))
    return round(1 - unique / len(words), 3)


def analyser_contenu(text: str) -> dict:
    mots = text.split()
    nb_mots = len(mots)
    phrases = [p for p in re.split(r"[.!?]+", text) if p.strip()]
    nb_phrases = max(len(phrases), 1)
    syl = syllables_approx_fr(text)
    return {
        "nb_mots": nb_mots,
        "nb_phrases": nb_phrases,
        "mots_par_phrase": round(nb_mots / nb_phrases, 1),
        "syllabes_approx": syl,
        "kandel_moles_approx": kandel_moles_approx(nb_mots, syl),
        "taux_repetition": repetition_rate(text),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", default=os.path.join(ROOT, "data", "corpus_200.csv"))
    parser.add_argument("--out", default=os.path.join(RESULTS, "corpus_metrics.json"))
    args = parser.parse_args()

    df = pd.read_csv(args.corpus)
    rows = []
    for _, r in df.iterrows():
        m = analyser_contenu(str(r.get("contenu", "")))
        m["clip_id"] = r.get("clip_id")
        m["template_id"] = r.get("template_id")
        m["emotion"] = r.get("emotion")
        rows.append(m)

    metrics_df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    summary = {
        "n": len(metrics_df),
        "mots_moyen": round(metrics_df["nb_mots"].mean(), 1),
        "mots_min": int(metrics_df["nb_mots"].min()),
        "mots_max": int(metrics_df["nb_mots"].max()),
        "kandel_moles_moyen": round(metrics_df["kandel_moles_approx"].mean(), 1),
        "repetition_moyenne": round(metrics_df["taux_repetition"].mean(), 3),
        "par_template": (
            metrics_df.groupby("template_id")["nb_mots"]
            .agg(["mean", "count"])
            .round(1)
            .to_dict()
        ),
    }

    metrics_df.to_csv(args.out.replace(".json", ".csv"), index=False)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
