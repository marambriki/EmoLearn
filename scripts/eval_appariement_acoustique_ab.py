#!/usr/bin/env python3
# ============================================
# Évaluation A/B — appariement voix IEMOCAP
# A = similarite_profil_vad (sans index acoustique)
# B = similarite_profil_acoustique (avec index)
# ============================================
"""
Usage :
  python scripts/eval_appariement_acoustique_ab.py
  python scripts/eval_appariement_acoustique_ab.py --scenarios data/scenarios_50.csv

Sorties :
  results/appariement_acoustique_ab_metrics.json
  results/appariement_acoustique_ab_detail.csv
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone

import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
BACKEND = os.path.join(ROOT, "backend")
RESULTS = os.path.join(ROOT, "results")
sys.path.insert(0, BACKEND)

from daisee_utils import etat_dominant_depuis_scores, pattern_depuis_scores
from iemocap_appariement import invalider_cache_index, selectionner_par_similarite
from ser_module import charger_iemocap

DEFAULT_SCENARIOS = os.path.join(ROOT, "data", "scenarios_50.csv")
OUT_JSON = os.path.join(RESULTS, "appariement_acoustique_ab_metrics.json")
OUT_CSV = os.path.join(RESULTS, "appariement_acoustique_ab_detail.csv")


def _apparier(df, sid, b, c, e, f, use_acoustic: bool):
    invalider_cache_index()
    row, meta = selectionner_par_similarite(
        df,
        scenario_id=sid,
        boredom=b,
        confusion=c,
        engagement=e,
        frustration=f,
        use_acoustic=use_acoustic,
    )
    return row, meta


def evaluer(scenarios_path: str) -> tuple[dict, pd.DataFrame]:
    if not os.path.isfile(scenarios_path):
        raise FileNotFoundError(scenarios_path)

    df_iemo = charger_iemocap()
    df_sc = pd.read_csv(scenarios_path)
    df_sc.columns = [c.strip() for c in df_sc.columns]

    rows = []
    for sid, row in df_sc.iterrows():
        b = float(row["Boredom"])
        c = float(row["Confusion"])
        e = float(row["Engagement"])
        f = float(row["Frustration"])
        gt, _, _ = etat_dominant_depuis_scores(b, c, e, f)

        row_a, meta_a = _apparier(df_iemo, sid, b, c, e, f, use_acoustic=False)
        row_b, meta_b = _apparier(df_iemo, sid, b, c, e, f, use_acoustic=True)

        uid_a = str(row_a.get("utterance_id", ""))
        uid_b = str(row_b.get("utterance_id", ""))
        sim_a = float(meta_a.get("similarite", 0))
        sim_b = float(meta_b.get("similarite", 0))

        rows.append(
            {
                "scenario_id": sid,
                "ClipID": row.get("ClipID", ""),
                "pattern": pattern_depuis_scores(b, c, e, f),
                "etat_dominant": gt,
                "Boredom": b,
                "Confusion": c,
                "Engagement": e,
                "Frustration": f,
                "utterance_vad": uid_a,
                "utterance_acoustique": uid_b,
                "voix_change": uid_a != uid_b,
                "similarite_vad": sim_a,
                "similarite_acoustique": sim_b,
                "delta_similarite": round(sim_b - sim_a, 4),
                "methode_vad": meta_a.get("methode"),
                "methode_acoustique": meta_b.get("methode"),
                "pool_filtre_vad": meta_a.get("pool_filtre_size"),
                "pool_filtre_acoustique": meta_b.get("pool_filtre_size"),
                "etat_iemocap_vad": meta_a.get("etat_iemocap"),
                "etat_iemocap_acoustique": meta_b.get("etat_iemocap"),
            }
        )

    detail = pd.DataFrame(rows)
    n = len(detail)
    changed = detail["voix_change"].sum()
    delta = detail["delta_similarite"]

    by_etat = {}
    for etat, grp in detail.groupby("etat_dominant"):
        by_etat[etat] = {
            "n": int(len(grp)),
            "voix_change_pct": round(float(grp["voix_change"].mean()), 3),
            "delta_similarite_moyen": round(float(grp["delta_similarite"].mean()), 4),
        }

    metrics = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "scenarios_path": scenarios_path,
        "n_scenarios": n,
        "variante_a": "similarite_profil_vad",
        "variante_b": "similarite_profil_acoustique",
        "voix_change_count": int(changed),
        "voix_change_pct": round(changed / n, 3) if n else 0.0,
        "similarite_vad_moyenne": round(float(detail["similarite_vad"].mean()), 4),
        "similarite_acoustique_moyenne": round(
            float(detail["similarite_acoustique"].mean()), 4,
        ),
        "delta_similarite_moyen": round(float(delta.mean()), 4),
        "delta_similarite_median": round(float(delta.median()), 4),
        "delta_similarite_positif_pct": round(float((delta > 0).mean()), 3),
        "etat_cible_aligne_vad_pct": round(
            float((detail["etat_iemocap_vad"] == detail["etat_dominant"]).mean()), 3,
        ),
        "etat_cible_aligne_acoustique_pct": round(
            float(
                (detail["etat_iemocap_acoustique"] == detail["etat_dominant"]).mean(),
            ),
            3,
        ),
        "par_etat": by_etat,
        "interpretation": (
            "L'index acoustique modifie le choix de voix dans le pool filtré par état ; "
            "l'alignement sur l'état cible reste identique si le filtre est appliqué. "
            "Un delta de similarité positif indique un profil cosinus plus proche du scénario."
        ),
    }
    return metrics, detail


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scenarios",
        default=DEFAULT_SCENARIOS,
        help="CSV scénarios DAiSEE (B/C/E/F par ligne)",
    )
    args = parser.parse_args()

    metrics, detail = evaluer(args.scenarios)
    os.makedirs(RESULTS, exist_ok=True)
    detail.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print(f"A/B appariement — {metrics['n_scenarios']} scénarios")
    print(f"  Voix changées     : {metrics['voix_change_count']} ({metrics['voix_change_pct']:.1%})")
    print(f"  Sim. moy. VAD     : {metrics['similarite_vad_moyenne']}")
    print(f"  Sim. moy. acoust. : {metrics['similarite_acoustique_moyenne']}")
    print(f"  Delta sim. moyen  : {metrics['delta_similarite_moyen']:+.4f}")
    print(f"  Etat aligné (A/B) : {metrics['etat_cible_aligne_vad_pct']:.1%} / {metrics['etat_cible_aligne_acoustique_pct']:.1%}")
    print(f"  -> {OUT_JSON}")
    print(f"  -> {OUT_CSV}")


if __name__ == "__main__":
    main()
