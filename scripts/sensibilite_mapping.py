#!/usr/bin/env python3
# ============================================
# scripts/sensibilite_mapping.py
#
# Tension #2 — Validité de construit du mapping IEMOCAP → DAiSEE.
# Le mapping (émotions conversationnelles → états d'engagement) est une HYPOTHÈSE.
# Ce script mesure sa SENSIBILITÉ : si l'on modifie les correspondances ambiguës,
# combien d'utterances changent d'état DAiSEE, et comment la distribution se déplace.
#
# But : objectiver la "marge d'interprétation" du mapping pour le mémoire,
# en attendant la validation experte (Delphi L1).
#
# Sortie : results/sensibilite_mapping.json
# ============================================

from __future__ import annotations

import json
import os

import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RESULTS = os.path.join(ROOT, "results")

# Mapping de référence (= backend/iemocap_parser.py)
MAPPING_REFERENCE = {
    "neu": "Boredom",
    "fru": "Frustration",
    "ang": "Frustration",
    "sad": "Boredom",
    "hap": "Engagement",
    "exc": "Engagement",
    "sur": "Confusion",
    "dis": "Frustration",
    "fea": "Confusion",
    "xxx": "Boredom",
}

# Variantes plausibles sur les correspondances DISCUTABLES.
# Chaque variante ne modifie que quelques émotions ambiguës ; le reste suit la référence.
VARIANTES = {
    "sad_neutre": {"sad": "Boredom"},  # contrôle (identique)
    "sad_frustration": {"sad": "Frustration"},  # tristesse = mal-être plutôt qu'ennui
    "surprise_engagement": {"sur": "Engagement"},  # surprise = éveil cognitif positif
    "neutre_engagement": {"neu": "Engagement"},  # neutre = état de travail, pas ennui
    "disgust_boredom": {"dis": "Boredom"},  # dégoût = désengagement
}


def appliquer_mapping(emotions: pd.Series, overrides: dict) -> pd.Series:
    table = dict(MAPPING_REFERENCE)
    table.update(overrides)
    return emotions.map(lambda e: table.get(e, "Boredom"))


def main():
    os.makedirs(RESULTS, exist_ok=True)
    df = pd.read_csv(os.path.join(ROOT, "data", "iemocap_parsed.csv"))
    emotions = df["emotion"].astype(str)
    n = len(emotions)

    ref_states = appliquer_mapping(emotions, {})
    ref_dist = ref_states.value_counts().to_dict()

    resultats = {
        "n_utterances": int(n),
        "mapping_reference": MAPPING_REFERENCE,
        "distribution_reference": {k: int(v) for k, v in ref_dist.items()},
        "variantes": {},
    }

    for nom, overrides in VARIANTES.items():
        var_states = appliquer_mapping(emotions, overrides)
        changes = int((var_states != ref_states).sum())
        resultats["variantes"][nom] = {
            "modifie": overrides,
            "utterances_changees": changes,
            "pct_changees": round(100 * changes / n, 1),
            "distribution": {k: int(v) for k, v in var_states.value_counts().to_dict().items()},
        }

    out = os.path.join(RESULTS, "sensibilite_mapping.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(resultats, f, indent=2, ensure_ascii=False)

    print(f"✅ Sensibilité écrite : {out}\n")
    print(f"Distribution référence : {resultats['distribution_reference']}")
    for nom, v in resultats["variantes"].items():
        print(f"  {nom:22s} → {v['pct_changees']:5.1f}% d'utterances changent d'état")


if __name__ == "__main__":
    main()
