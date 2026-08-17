#!/usr/bin/env python3
# ============================================
# Agrège les scores Likert L1 (mapping) et L2 (contenu)
# depuis les CSV remplis par les experts / enseignants.
#
# Entrées attendues :
#   results/forms/l1_mapping_reponses.csv
#   results/forms/l2_contenu_reponses.csv
#
# Sorties :
#   results/evaluation_experts_mapping.json
#   results/evaluation_experts_contenu.json
#   (+ CSV résumé)
# ============================================

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone

import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RESULTS = os.path.join(ROOT, "results")
FORMS = os.path.join(RESULTS, "forms")

L1_COLS = ["C1_coherence", "C2_pertinence", "C3_template", "C4_urgence"]
L2_COLS = ["Q1_pertinence", "Q2_clarte", "Q3_adequation", "Q4_qualite", "Q5_utilisabilite", "Q6_factualite"]


def _to_num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def agreger_l1(path: str) -> dict:
    df = pd.read_csv(path)
    present = [c for c in L1_COLS if c in df.columns]
    if not present:
        raise SystemExit(f"Colonnes Likert L1 absentes dans {path}")

    for c in present:
        df[c] = _to_num(df[c])

    scored = df.dropna(subset=present, how="all")
    means = {c: round(float(scored[c].mean()), 3) for c in present if scored[c].notna().any()}
    global_mean = round(float(pd.concat([scored[c] for c in present]).mean()), 3)

    by_template = {}
    if "template_id" in scored.columns:
        for tid, g in scored.groupby("template_id"):
            by_template[str(tid)] = {
                c: round(float(g[c].mean()), 3)
                for c in present
                if g[c].notna().any()
            }

    faibles = []
    if scored[present].notna().any().any():
        row_means = scored[present].mean(axis=1)
        low = scored[row_means < 3]
        for _, r in low.iterrows():
            faibles.append(
                {
                    "fiche_id": r.get("fiche_id"),
                    "template_id": r.get("template_id"),
                    "moyenne": round(float(row_means.loc[r.name]), 3),
                    "commentaire": r.get("commentaire_expert", ""),
                }
            )

    return {
        "niveau": "L1_mapping_engagement_adaptation",
        "n_fiches": int(len(df)),
        "n_notees": int(len(scored.dropna(subset=present, how="any"))),
        "moyennes_criteres": means,
        "moyenne_globale": global_mean,
        "par_template": by_template,
        "cas_faibles_lt_3": faibles,
        "source": path,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def agreger_l2(path: str) -> dict:
    df = pd.read_csv(path)
    present = [c for c in L2_COLS if c in df.columns]
    if not present:
        raise SystemExit(f"Colonnes Likert L2 absentes dans {path}")

    for c in present:
        df[c] = _to_num(df[c])

    scored = df.dropna(subset=present, how="all")
    means = {c: round(float(scored[c].mean()), 3) for c in present if scored[c].notna().any()}
    global_mean = round(float(pd.concat([scored[c] for c in present]).mean()), 3)

    by_template = {}
    if "template_id" in scored.columns:
        for tid, g in scored.groupby("template_id"):
            by_template[str(tid)] = {
                "n": int(len(g)),
                **{
                    c: round(float(g[c].mean()), 3)
                    for c in present
                    if g[c].notna().any()
                },
            }

    return {
        "niveau": "L2_qualite_contenu",
        "n_fiches": int(len(df)),
        "n_notees": int(len(scored.dropna(subset=present, how="any"))),
        "moyennes_criteres": means,
        "moyenne_globale": global_mean,
        "par_template": by_template,
        "source": path,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def preparer_formulaires():
    """Crée les CSV vides à remplir si absents."""
    os.makedirs(FORMS, exist_ok=True)
    fiches = os.path.join(RESULTS, "validation_expert_fiches.csv")

    l1_path = os.path.join(FORMS, "l1_mapping_reponses.csv")
    if os.path.isfile(fiches) and not os.path.isfile(l1_path):
        df = pd.read_csv(fiches)
        keep = [
            c
            for c in [
                "fiche_id",
                "scenario_id",
                "pattern",
                "template_id",
                "etat",
                "type_contenu",
                "concept",
                "contenu_extrait",
                "C1_coherence",
                "C2_pertinence",
                "C3_template",
                "C4_urgence",
                "commentaire_expert",
                "expert_id",
            ]
            if c in df.columns or c in ("expert_id",)
        ]
        out = df.copy()
        if "expert_id" not in out.columns:
            out["expert_id"] = ""
        for c in L1_COLS:
            if c not in out.columns:
                out[c] = ""
        out[keep if keep else out.columns].to_csv(l1_path, index=False, encoding="utf-8-sig")
        print(f"Formulaire L1 créé : {l1_path}")
    elif not os.path.isfile(l1_path):
        pd.DataFrame(
            columns=[
                "fiche_id",
                "scenario_id",
                "template_id",
                "etat",
                "expert_id",
                *L1_COLS,
                "commentaire_expert",
            ]
        ).to_csv(l1_path, index=False, encoding="utf-8-sig")
        print(f"Formulaire L1 vide : {l1_path}")

    l2_path = os.path.join(FORMS, "l2_contenu_reponses.csv")
    if not os.path.isfile(l2_path):
        src = os.path.join(ROOT, "data", "corpus_evaluation_60.csv")
        if os.path.isfile(src):
            df = pd.read_csv(src)
            rows = []
            for i, r in df.iterrows():
                rows.append(
                    {
                        "fiche_id": i + 1,
                        "scenario_id": r.get("scenario_id", ""),
                        "template_id": r.get("template_id", ""),
                        "etat": r.get("etat", r.get("emotion", "")),
                        "type_contenu": r.get("type_contenu", ""),
                        "concept": r.get("concept", "les algorithmes de tri"),
                        "contenu_extrait": str(r.get("contenu", ""))[:400],
                        "enseignant_id": "",
                        **{c: "" for c in L2_COLS},
                        "commentaire": "",
                    }
                )
            pd.DataFrame(rows).to_csv(l2_path, index=False, encoding="utf-8-sig")
        else:
            pd.DataFrame(
                columns=[
                    "fiche_id",
                    "template_id",
                    "etat",
                    "enseignant_id",
                    *L2_COLS,
                    "commentaire",
                ]
            ).to_csv(l2_path, index=False, encoding="utf-8-sig")
        print(f"Formulaire L2 créé : {l2_path}")


def main():
    parser = argparse.ArgumentParser(description="Agrège validations expertes L1/L2")
    parser.add_argument("--prepare-forms", action="store_true", help="Crée les CSV à remplir")
    parser.add_argument("--l1", default=os.path.join(FORMS, "l1_mapping_reponses.csv"))
    parser.add_argument("--l2", default=os.path.join(FORMS, "l2_contenu_reponses.csv"))
    args = parser.parse_args()

    os.makedirs(RESULTS, exist_ok=True)
    if args.prepare_forms:
        preparer_formulaires()

    if os.path.isfile(args.l1):
        df = pd.read_csv(args.l1)
        if any(_to_num(df[c]).notna().any() for c in L1_COLS if c in df.columns):
            report = agreger_l1(args.l1)
            out = os.path.join(RESULTS, "evaluation_experts_mapping.json")
            with open(out, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            pd.DataFrame([report["moyennes_criteres"]]).to_csv(
                out.replace(".json", "_resume.csv"), index=False, encoding="utf-8-sig"
            )
            print(f"✅ L1 → {out} (moyenne={report['moyenne_globale']})")
        else:
            print(f"L1 : aucune note remplie dans {args.l1} (formulaire prêt)")
    else:
        print(f"L1 absent — lancez avec --prepare-forms")

    if os.path.isfile(args.l2):
        df = pd.read_csv(args.l2)
        if any(_to_num(df[c]).notna().any() for c in L2_COLS if c in df.columns):
            report = agreger_l2(args.l2)
            out = os.path.join(RESULTS, "evaluation_experts_contenu.json")
            with open(out, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            print(f"✅ L2 → {out} (moyenne={report['moyenne_globale']})")
        else:
            print(f"L2 : aucune note remplie dans {args.l2} (formulaire prêt)")
    else:
        print(f"L2 absent — lancez avec --prepare-forms")


if __name__ == "__main__":
    main()
