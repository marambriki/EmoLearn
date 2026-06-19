#!/usr/bin/env python3
"""Exporte les fiches pour validation experte (L1/L2)."""

import os
import random

import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CORPUS = os.path.join(ROOT, "data", "corpus_evaluation_60.csv")
SCENARIOS = os.path.join(ROOT, "data", "scenarios_50.csv")
OUTPUT = os.path.join(ROOT, "results", "validation_expert_fiches.csv")


def main():
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)

    if os.path.exists(CORPUS):
        df = pd.read_csv(CORPUS)
        sample = df
        source = "corpus_evaluation_60"
    else:
        df = pd.read_csv(os.path.join(ROOT, "data", "corpus_200.csv"))
        sample = df.sample(n=min(60, len(df)), random_state=42)
        source = "corpus_200_sample"

    scenarios = pd.read_csv(SCENARIOS)
    rows = []
    for _, row in sample.iterrows():
        sid = int(row.get("scenario_id", 0))
        if sid < len(scenarios):
            sc = scenarios.iloc[sid]
            pattern = sc.get("pattern", "stable")
        else:
            pattern = "—"
        rows.append(
            {
                "fiche_id": len(rows) + 1,
                "source": source,
                "scenario_id": sid,
                "pattern": pattern,
                "template_id": row.get("template_id", ""),
                "etat": row.get("etat", row.get("emotion", "")),
                "type_contenu": row.get("type_contenu", ""),
                "concept": row.get("concept", "les algorithmes de tri"),
                "contenu_extrait": str(row.get("contenu", ""))[:300],
                "C1_coherence": "",
                "C2_pertinence": "",
                "C3_template": "",
                "C4_urgence": "",
                "commentaire_expert": "",
            }
        )

    out = pd.DataFrame(rows)
    out.to_csv(OUTPUT, index=False, encoding="utf-8-sig")
    print(f"Export {len(out)} fiches -> {OUTPUT}")


if __name__ == "__main__":
    main()
