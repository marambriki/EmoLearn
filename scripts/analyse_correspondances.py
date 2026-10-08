#!/usr/bin/env python3
# ============================================
# OS6 — Analyse des correspondances
# Pertinence de l'adaptation : état (GT / fusion) ↔ famille A/B/C/D
# et patron A1–D5, sur le référentiel de 50 scénarios.
#
# Sorties :
#   results/correspondances_adaptation.json
#   results/correspondances_adaptation.csv
#   results/correspondances_memoire.md
# ============================================

from __future__ import annotations

import json
import os
import sys
from collections import Counter
from datetime import datetime, timezone

import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
BACKEND = os.path.join(ROOT, "backend")
RESULTS = os.path.join(ROOT, "results")
sys.path.insert(0, BACKEND)

from daisee_utils import etat_dominant_depuis_scores, pattern_depuis_scores
from decision_engine import ETAT_VERS_CLE, selectionner_strategie, selectionner_template_id

ETATS = ["Boredom", "Confusion", "Engagement", "Frustration"]
FAMILLES = ["A", "B", "C", "D"]

# Correspondance pédagogique canonique (OS2) : un type « attendu » par état.
FAMILLE_CANONIQUE = {
    "Confusion": "A",
    "Boredom": "C",
    "Frustration": "D",
    "Engagement": "D",
}

# Ensemble admissible : le moteur peut légitimement sortir un type voisin.
FAMILLE_ADMISSIBLE = {
    "Confusion": {"A"},
    "Boredom": {"B", "C"},
    "Frustration": {"A", "D"},
    "Engagement": {"B", "D"},
}


def famille_de(type_ou_template: str) -> str:
    s = str(type_ou_template or "").strip().upper()
    if not s or s in ("NAN", "AUCUNE"):
        return "aucune"
    return s[0]


def cramers_v(table: pd.DataFrame) -> float:
    if table.size == 0 or table.values.sum() == 0:
        return 0.0
    n = float(table.values.sum())
    row_s = table.sum(axis=1).values.reshape(-1, 1)
    col_s = table.sum(axis=0).values.reshape(1, -1)
    exp = row_s @ col_s / n
    with_zeros = exp == 0
    chi2 = ((table.values - exp) ** 2 / (exp + with_zeros.astype(float))).sum()
    r, k = table.shape
    denom = n * min(r - 1, k - 1)
    if denom <= 0:
        return 0.0
    return round((chi2 / denom) ** 0.5, 3)


def decision_depuis_etat_seul(etat: str, niveau: int = 2):
    """Contre-factuel : cascade appliquée à l'état fusionné (scores one-hot)."""
    scores = {cle: 0 for cle in ETAT_VERS_CLE.values()}
    cle = ETAT_VERS_CLE.get(etat)
    if cle:
        scores[cle] = int(niveau)
    return decision_depuis_scores(
        scores["boredom"],
        scores["confusion"],
        scores["engagement"],
        scores["frustration"],
    )


def decision_depuis_scores(b, c, e, f):
    etat, niveau, _ = etat_dominant_depuis_scores(b, c, e, f)
    pattern = pattern_depuis_scores(b, c, e, f)
    _, _, strategie, type_regle = selectionner_strategie(b, c, e, f)
    tid = selectionner_template_id(
        type_contenu=type_regle,
        etat=etat,
        niveau=niveau,
        boredom=int(b),
        confusion=int(c),
        engagement=int(e),
        frustration=int(f),
        pattern=pattern,
    )
    return {
        "etat_gt": etat,
        "niveau_gt": int(niveau),
        "pattern_gt": pattern,
        "strategie_gt": strategie,
        "type_regle_gt": type_regle,
        "famille_gt": famille_de(type_regle),
        "template_gt": tid,
    }


def main():
    os.makedirs(RESULTS, exist_ok=True)
    scenarios = pd.read_csv(os.path.join(ROOT, "data", "scenarios_50.csv"))
    scenarios.columns = [c.strip() for c in scenarios.columns]
    decisions = pd.read_csv(os.path.join(RESULTS, "scenarios_decisions.csv"))
    decisions["clip_id"] = decisions["clip_id"].astype(str).str.strip()
    scenarios["ClipID"] = scenarios["ClipID"].astype(str).str.strip()

    rows = []
    for i, sc in scenarios.iterrows():
        clip = sc["ClipID"]
        b, c, e, f = int(sc["Boredom"]), int(sc["Confusion"]), int(sc["Engagement"]), int(sc["Frustration"])
        attendu = decision_depuis_scores(b, c, e, f)
        match = decisions[decisions["clip_id"] == clip]
        if match.empty:
            continue
        obs = match.iloc[0]
        etat_fus = str(obs["etat_fusion"]).strip()
        tid_obs = str(obs["template_id"]).strip()
        fam_obs = famille_de(obs.get("type_contenu", tid_obs))
        fam_can = FAMILLE_CANONIQUE.get(etat_fus, "")
        admissible = FAMILLE_ADMISSIBLE.get(etat_fus, set())
        cf = decision_depuis_etat_seul(etat_fus)
        rows.append(
            {
                "scenario_id": int(obs["scenario_id"]),
                "clip_id": clip,
                "pattern_csv": sc.get("pattern", ""),
                **attendu,
                "etat_fusion": etat_fus,
                "template_observe": tid_obs,
                "famille_observee": fam_obs,
                "famille_canonique_fusion": fam_can,
                "famille_si_fusion_onehot": cf["famille_gt"],
                "template_si_fusion_onehot": cf["template_gt"],
                "concordance_gt_runtime": int(attendu["famille_gt"] == fam_obs),
                "concordance_template_gt_runtime": int(attendu["template_gt"] == tid_obs),
                "concordance_etat_gt_fusion": int(attendu["etat_gt"] == etat_fus),
                "concordance_canonique": int(fam_obs == fam_can) if fam_can else 0,
                "concordance_admissible": int(fam_obs in admissible) if admissible else 0,
                "concordance_fusion_onehot": int(cf["famille_gt"] == fam_obs),
            }
        )

    df = pd.DataFrame(rows)
    n = len(df)
    if n == 0:
        raise SystemExit("Aucune correspondance clip GT ↔ décision runtime.")

    ct_etat_fam = pd.crosstab(df["etat_fusion"], df["famille_observee"]).reindex(
        index=ETATS, columns=FAMILLES, fill_value=0
    )
    ct_gt_obs = pd.crosstab(df["famille_gt"], df["famille_observee"]).reindex(
        index=FAMILLES, columns=FAMILLES, fill_value=0
    )
    ct_etat_gt_fus = pd.crosstab(df["etat_gt"], df["etat_fusion"]).reindex(
        index=ETATS, columns=ETATS, fill_value=0
    )

    templates_obs = Counter(df["template_observe"])
    tous_templates = [f"{p}{i}" for p in FAMILLES for i in range(1, 6)]
    manquants = [t for t in tous_templates if templates_obs.get(t, 0) == 0]

    def taux(col):
        return round(float(df[col].mean()), 3)

    report = {
        "n": n,
        "taux": {
            "concordance_famille_gt_vs_runtime": taux("concordance_gt_runtime"),
            "concordance_template_gt_vs_runtime": taux("concordance_template_gt_runtime"),
            "concordance_etat_gt_vs_fusion": taux("concordance_etat_gt_fusion"),
            "famille_canonique_vs_fusion": taux("concordance_canonique"),
            "famille_admissible_vs_fusion": taux("concordance_admissible"),
            "famille_si_etat_fusion_onehot_vs_journal": taux("concordance_fusion_onehot"),
        },
        "cramers_v": {
            "etat_fusion_x_famille": cramers_v(ct_etat_fam),
            "famille_gt_x_famille_obs": cramers_v(ct_gt_obs),
            "etat_gt_x_etat_fusion": cramers_v(ct_etat_gt_fus),
        },
        "contingence_etat_fusion_famille": ct_etat_fam.to_dict(),
        "contingence_famille_gt_vs_obs": ct_gt_obs.to_dict(),
        "templates_observes": dict(sorted(templates_obs.items())),
        "n_patrons_distincts": len(templates_obs),
        "patrons_non_declenches": manquants,
        "distribution_etat_fusion": df["etat_fusion"].value_counts().to_dict(),
        "distribution_famille_observee": df["famille_observee"].value_counts().to_dict(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "note": (
            "Si la famille GT (cascade sur labels DAiSEE) coincide avec la famille "
            "journalisee, le referentiel de 50 scenarios documente le moteur de "
            "decision sur les scores annotes, pas la cascade appliquee a l'etat fusionne. "
            "La concordance admissible et le contre-factuel one-hot mesurent alors "
            "l'ecart entre etat detecte et type effectivement servi."
        ),
    }

    json_path = os.path.join(RESULTS, "correspondances_adaptation.json")
    csv_path = os.path.join(RESULTS, "correspondances_adaptation.csv")
    md_path = os.path.join(RESULTS, "correspondances_memoire.md")
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    t = report["taux"]
    v = report["cramers_v"]
    lines = [
        "# Analyse des correspondances (OS6)",
        "",
        f"n = {n} scénarios joints (labels DAiSEE × décisions runtime).",
        "",
        "## Taux",
        f"- Famille GT (règles sur scores DAiSEE) vs famille runtime : **{t['concordance_famille_gt_vs_runtime']:.1%}**",
        f"- Patron GT vs patron runtime : **{t['concordance_template_gt_vs_runtime']:.1%}**",
        f"- État dominant GT vs état fusionné : **{t['concordance_etat_gt_vs_fusion']:.1%}**",
        f"- Famille canonique (état fusionné → A/C/D) : **{t['famille_canonique_vs_fusion']:.1%}**",
        f"- Famille admissible pour l'etat fusionne : **{t['famille_admissible_vs_fusion']:.1%}**",
        f"- Famille si cascade one-hot sur l'etat fusionne vs journal : **{t['famille_si_etat_fusion_onehot_vs_journal']:.1%}**",
        "",
        "## Cramér V",
        f"- État fusionné × famille observée : {v['etat_fusion_x_famille']}",
        f"- Famille GT × famille observée : {v['famille_gt_x_famille_obs']}",
        f"- État GT × état fusionné : {v['etat_gt_x_etat_fusion']}",
        "",
        "## Contingence état fusionné × famille",
        ct_etat_fam.to_string(),
        "",
        f"Patrons distincts observés : {report['n_patrons_distincts']}/20. "
        f"Non déclenchés : {', '.join(manquants) or 'aucun'}.",
        "",
        report["note"],
        "",
        f"Généré : {report['timestamp']}",
    ]
    with open(md_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")

    summary = {k: report[k] for k in ("n", "taux", "cramers_v", "n_patrons_distincts", "patrons_non_declenches")}
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("Wrote", json_path)
    print("Wrote", csv_path)
    print("Wrote", md_path)


if __name__ == "__main__":
    main()
