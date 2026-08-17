#!/usr/bin/env python3
# ============================================
# Proxy d'évaluation — pipeline PRODUIT
# Visage DAiSEE (scores scénario / décision) + voix IEMOCAP (appariée)
#
# Pas de vérité terrain multimodale jointe : on compare l'état FUSIONNÉ
# à l'état dominant DAiSEE du scénario (proxy de cohérence pipeline).
#
# Sortie : results/fusion_produit_proxy_metrics.json
# ============================================

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
from fer_module import analyser_visage_depuis_scenario
from ser_module import analyser_voix_depuis_scenario
from fusion_module import fusionner_modalites, fusionner_par_vote

ETATS = ["Boredom", "Confusion", "Engagement", "Frustration"]
ETATS_URGENTS = ["Confusion", "Frustration"]


def _accuracy(y_true, y_pred):
    pairs = [(t, p) for t, p in zip(y_true, y_pred) if t and p]
    if not pairs:
        return 0.0
    return round(sum(1 for t, p in pairs if t == p) / len(pairs), 3)


def _rappel_urgents(y_true, y_pred):
    urg = [(t, p) for t, p in zip(y_true, y_pred) if t in ETATS_URGENTS]
    if not urg:
        return None
    return round(sum(1 for t, p in urg if p in ETATS_URGENTS) / len(urg), 3)


def _confusion(y_true, y_pred):
    mat = {gt: {pr: 0 for pr in ETATS} for gt in ETATS}
    for t, p in zip(y_true, y_pred):
        if t in mat and p in mat[t]:
            mat[t][p] += 1
    return mat


def evaluer(df: pd.DataFrame, limit: int | None = None, use_vote: bool = False):
    y_gt, y_vis, y_voix, y_fus = [], [], [], []
    rows = []
    n = len(df) if limit is None else min(limit, len(df))

    for i in range(n):
        row = df.iloc[i]
        b = int(row["Boredom"])
        c = int(row["Confusion"])
        e = int(row["Engagement"])
        f = int(row["Frustration"])
        gt, _, _ = etat_dominant_depuis_scores(b, c, e, f)
        pattern = pattern_depuis_scores(b, c, e, f)
        scenario = {
            "ClipID": row["ClipID"],
            "Boredom": b,
            "Confusion": c,
            "Engagement": e,
            "Frustration": f,
            "pattern": pattern,
        }

        visage = analyser_visage_depuis_scenario(scenario)
        voix = analyser_voix_depuis_scenario(scenario_id=i, etat_dominant=gt)
        if use_vote:
            fusion = fusionner_par_vote(visage, voix)
        else:
            # Défaut produit = pondération par incertitude (fusionner_modalites)
            fusion = fusionner_modalites(visage, voix)

        y_gt.append(gt)
        y_vis.append(visage.get("etat"))
        y_voix.append(voix.get("etat"))
        y_fus.append(fusion.get("etat"))
        rows.append(
            {
                "scenario_id": i,
                "clip_id": str(row["ClipID"]),
                "gt_etat_dominant": gt,
                "visage_etat": visage.get("etat"),
                "voix_etat": voix.get("etat"),
                "fusion_etat": fusion.get("etat"),
                "conflit": fusion.get("conflit"),
                "strategie_fusion": fusion.get("strategie_fusion"),
            }
        )
        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{n}")

    detail = pd.DataFrame(rows)
    metrics = {
        "mode": "fusion_produit_proxy",
        "note": (
            "Proxy uniquement : GT = état dominant DAiSEE du scénario. "
            "Visage = FER sur clip DAiSEE ; voix = IEMOCAP appariée par état "
            "(pas la même personne). Ce n'est PAS une accuracy multimodale gold."
        ),
        "n": len(detail),
        "strategie_fusion": (
            "vote_pondere_incertitude" if use_vote else "soft_incertitude_defaut"
        ),
        "accuracy_vs_etat_dominant": {
            "visage": _accuracy(y_gt, y_vis),
            "voix": _accuracy(y_gt, y_voix),
            "fusion": _accuracy(y_gt, y_fus),
        },
        "rappel_etats_urgents": {
            "visage": _rappel_urgents(y_gt, y_vis),
            "voix": _rappel_urgents(y_gt, y_voix),
            "fusion": _rappel_urgents(y_gt, y_fus),
        },
        "distribution_gt": detail["gt_etat_dominant"].value_counts().to_dict(),
        "distribution_fusion": detail["fusion_etat"].value_counts().to_dict(),
        "confusion_fusion": _confusion(y_gt, y_fus),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    return detail, metrics


def evaluer_depuis_decisions(decisions_csv: str, scenarios_csv: str):
    """Proxy rapide à partir d'un benchmark déjà calculé (sans ré-inférer FER/SER)."""
    dec = pd.read_csv(decisions_csv)
    sc = pd.read_csv(scenarios_csv)
    sc.columns = [c.strip() for c in sc.columns]
    sc["clip_id"] = sc["ClipID"].astype(str)
    dec["clip_id"] = dec["clip_id"].astype(str)
    etats_cols = ["Boredom", "Engagement", "Confusion", "Frustration"]
    sc["gt_etat_dominant"] = sc[etats_cols].astype(float).idxmax(axis=1)
    m = dec.merge(sc[["clip_id", "gt_etat_dominant"]], on="clip_id", how="inner")
    y_gt = m["gt_etat_dominant"].tolist()
    y_fus = m["etat_fusion"].tolist()
    detail = m.rename(columns={"etat_fusion": "fusion_etat"})[
        ["scenario_id", "clip_id", "gt_etat_dominant", "fusion_etat", "template_id"]
    ]
    return detail, {
        "mode": "fusion_produit_proxy_from_decisions",
        "note": (
            "Proxy depuis results/scenarios_decisions.csv : GT = état dominant DAiSEE, "
            "prédiction = etat_fusion du pipeline produit. Pas d'accuracy multimodale gold."
        ),
        "n": len(detail),
        "strategie_fusion": "pipeline_produit",
        "accuracy_vs_etat_dominant": {
            "visage": None,
            "voix": None,
            "fusion": _accuracy(y_gt, y_fus),
        },
        "rappel_etats_urgents": {
            "visage": None,
            "voix": None,
            "fusion": _rappel_urgents(y_gt, y_fus),
        },
        "distribution_gt": detail["gt_etat_dominant"].value_counts().to_dict(),
        "distribution_fusion": detail["fusion_etat"].value_counts().to_dict(),
        "confusion_fusion": _confusion(y_gt, y_fus),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Proxy métriques fusion produit DAiSEE+IEMOCAP"
    )
    parser.add_argument(
        "--scenarios",
        default=os.path.join(ROOT, "data", "scenarios_50.csv"),
    )
    parser.add_argument(
        "--decisions",
        default=os.path.join(RESULTS, "scenarios_decisions.csv"),
        help="CSV de décisions pour mode rapide --from-decisions",
    )
    parser.add_argument(
        "--from-decisions",
        action="store_true",
        default=True,
        help="Mode rapide (défaut) : utilise scenarios_decisions.csv",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Ré-infère FER+SER+fusion sur chaque scénario (lent)",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--vote",
        action="store_true",
        help="Utiliser fusionner_par_vote (mode --live uniquement)",
    )
    parser.add_argument(
        "--out",
        default=os.path.join(RESULTS, "fusion_produit_proxy_metrics.json"),
    )
    args = parser.parse_args()

    os.makedirs(RESULTS, exist_ok=True)

    if args.live:
        df = pd.read_csv(args.scenarios)
        df.columns = [c.strip() for c in df.columns]
        print(f"Évaluation LIVE proxy sur {len(df)} scénarios…")
        detail, metrics = evaluer(df, limit=args.limit, use_vote=args.vote)
    else:
        if not os.path.isfile(args.decisions):
            raise SystemExit(
                f"Fichier absent : {args.decisions}. "
                "Lancez avec --live ou générez scenarios_decisions.csv."
            )
        print(f"Évaluation proxy depuis {args.decisions}…")
        detail, metrics = evaluer_depuis_decisions(args.decisions, args.scenarios)

    csv_path = args.out.replace(".json", ".csv")
    detail.to_csv(csv_path, index=False, encoding="utf-8-sig")
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print(f"✅ {args.out}")
    print(f"✅ {csv_path}")
    print(json.dumps(metrics["accuracy_vs_etat_dominant"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
