#!/usr/bin/env python3
"""
Évalue la pondération continue par incertitude (cas synthétiques).

Sortie : results/fusion_reliability_metrics.json
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "backend"))

from fusion_module import fusionner_modalites

ETATS = ["Boredom", "Confusion", "Engagement", "Frustration"]


def _mk(etat: str, score: float, niveau: int = 2) -> dict:
    probs = {e: 0.05 for e in ETATS}
    probs[etat] = max(score, 0.05)
    s = sum(probs.values())
    probs = {k: v / s for k, v in probs.items()}
    return {
        "etat": etat,
        "emotion": etat.lower(),
        "niveau": niveau,
        "score": score,
        "probs_etats": probs,
    }


CAS = [
    {
        "id": "voix_bruit_sous_ponderee",
        "visage": _mk("Engagement", 0.82),
        "voix": _mk("Frustration", 0.22),
        "gt": "Engagement",
    },
    {
        "id": "visage_bruit_sous_pondere",
        "visage": _mk("Engagement", 0.18),
        "voix": _mk("Confusion", 0.75),
        "gt": "Confusion",
    },
    {
        "id": "vote_engagement",
        "visage": _mk("Engagement", 0.70),
        "voix": _mk("Boredom", 0.65),
        "gt": "Engagement",
    },
    {
        "id": "accord_frustration",
        "visage": _mk("Frustration", 0.70),
        "voix": _mk("Frustration", 0.60),
        "gt": "Frustration",
    },
    {
        "id": "prior_visage_engagement",
        "visage": _mk("Engagement", 0.72),
        "voix": _mk("Boredom", 0.68),
        "gt": "Engagement",
    },
    {
        "id": "incertitude_totale_fru",
        "visage": _mk("Engagement", 0.0),
        "voix": _mk("Frustration", 0.0),
        "gt": "Frustration",
    },
]


def main():
    rows = []
    for cas in CAS:
        fusion = fusionner_modalites(cas["visage"], cas["voix"])
        rows.append(
            {
                "id": cas["id"],
                "gt": cas["gt"],
                "etat": fusion["etat"],
                "ok": fusion["etat"] == cas["gt"],
                "strategie": fusion.get("strategie_fusion"),
            }
        )

    acc = sum(1 for r in rows if r["ok"]) / len(rows)

    report = {
        "mode": "fusion_reliability_synthetic",
        "methode": "ponderation_continue_incertitude",
        "n_cas": len(rows),
        "accuracy": round(acc, 3),
        "cas": rows,
        "baselines_modeles": {
            "fer_full_f1_macro": 0.388,
            "fer_full_accuracy": 0.39,
            "ser_full_best_f1_macro": 0.537,
            "ser_full_best_accuracy": 0.557,
            "sources": [
                "results/fer_finetune_full_metrics.json",
                "results/ser_finetune_full_metrics.json",
            ],
        },
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "note": (
            "Pondération continue : w = prior × confiance, puis renormalisation. "
            "Repli priorité pédagogique seulement si les deux confiances sont nulles."
        ),
    }
    out = os.path.join(ROOT, "results", "fusion_reliability_metrics.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(json.dumps({
        "accuracy": report["accuracy"],
        "n_cas": report["n_cas"],
        "out": out,
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
