#!/usr/bin/env python3
"""Exporte les fiches L1 (mapping, 15 cas) et L2 (contenu, echantillon)."""

from __future__ import annotations

import os

import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CORPUS = os.path.join(ROOT, "data", "corpus_evaluation_60.csv")
SCENARIOS = os.path.join(ROOT, "data", "scenarios_50.csv")
DECISIONS = os.path.join(ROOT, "results", "scenarios_decisions.csv")
CORRESP = os.path.join(ROOT, "results", "correspondances_adaptation.csv")
FORMS = os.path.join(ROOT, "results", "forms")
OUTPUT_L1_FICHES = os.path.join(ROOT, "results", "validation_expert_fiches.csv")


def _scores_vides(path: str, cols: list[str]) -> bool:
    if not os.path.isfile(path):
        return True
    df = pd.read_csv(path)
    present = [c for c in cols if c in df.columns]
    if not present:
        return True
    for c in present:
        if pd.to_numeric(df[c], errors="coerce").notna().any():
            return False
    return True


def exporter_l1(n: int = 15) -> pd.DataFrame:
    if os.path.isfile(CORRESP):
        src = pd.read_csv(CORRESP)
        src["pattern"] = src.get("pattern_gt", src.get("pattern_csv", "stable"))
    else:
        sc = pd.read_csv(SCENARIOS)
        dec = pd.read_csv(DECISIONS)
        sc["ClipID"] = sc["ClipID"].astype(str).str.strip()
        dec["clip_id"] = dec["clip_id"].astype(str).str.strip()
        src = sc.merge(dec, left_on="ClipID", right_on="clip_id", how="inner")
        src["pattern"] = src["pattern_x"] if "pattern_x" in src.columns else src.get("pattern", "stable")
        src["etat_gt"] = ""
        src["template_gt"] = src.get("template_id", "")
        src["template_observe"] = src.get("template_id", "")
        src["etat_fusion"] = src.get("etat_fusion", "")
        src["famille_gt"] = src.get("type_contenu", "")
        src["clip_id"] = src["clip_id"]
        src["scenario_id"] = src.get("scenario_id", range(len(src)))

    parts = []
    for _, g in src.groupby(src["pattern"].astype(str), dropna=False):
        parts.append(g.sample(n=min(3, len(g)), random_state=42))
    sample = pd.concat(parts).drop_duplicates(subset=["clip_id"])
    if len(sample) > n:
        sample = sample.sample(n=n, random_state=42)
    sample = sample.sort_values("scenario_id") if "scenario_id" in sample.columns else sample

    rows = []
    for i, (_, d) in enumerate(sample.iterrows(), start=1):
        rows.append(
            {
                "fiche_id": i,
                "scenario_id": d.get("scenario_id", ""),
                "clip_id": d.get("clip_id", ""),
                "pattern": d.get("pattern", ""),
                "etat_gt": d.get("etat_gt", ""),
                "template_attendu_gt": d.get("template_gt", ""),
                "famille_attendue_gt": d.get("famille_gt", ""),
                "etat_fusion": d.get("etat_fusion", ""),
                "template_observe": d.get("template_observe", d.get("template_id", "")),
                "C1_coherence": "",
                "C2_pertinence": "",
                "C3_template": "",
                "C4_urgence": "",
                "commentaire_expert": "",
                "expert_id": "",
            }
        )
    return pd.DataFrame(rows)


def exporter_l2() -> pd.DataFrame:
    src = CORPUS if os.path.exists(CORPUS) else os.path.join(ROOT, "data", "corpus_200.csv")
    df = pd.read_csv(src)
    rows = []
    for i, row in df.iterrows():
        rows.append(
            {
                "fiche_id": i + 1,
                "scenario_id": row.get("scenario_id", ""),
                "clip_id": row.get("clip_id", ""),
                "template_id": row.get("template_id", ""),
                "etat": row.get("etat", row.get("emotion", "")),
                "type_contenu": row.get("type_contenu", ""),
                "concept": row.get("concept", "les algorithmes de tri"),
                "contenu_extrait": str(row.get("contenu", ""))[:400],
                "enseignant_id": "",
                "Q1_pertinence": "",
                "Q2_clarte": "",
                "Q3_adequation": "",
                "Q4_qualite": "",
                "Q5_utilisabilite": "",
                "Q6_factualite": "",
                "commentaire": "",
            }
        )
    return pd.DataFrame(rows)


def main():
    os.makedirs(FORMS, exist_ok=True)
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)

    l1 = exporter_l1()
    l1.to_csv(OUTPUT_L1_FICHES, index=False, encoding="utf-8-sig")
    l1_form = os.path.join(FORMS, "l1_mapping_reponses.csv")
    if _scores_vides(l1_form, ["C1_coherence", "C2_pertinence", "C3_template", "C4_urgence"]):
        l1.to_csv(l1_form, index=False, encoding="utf-8-sig")
        print(f"L1 mapping : {len(l1)} cas -> {l1_form}")
    else:
        print(f"L1 deja note, formulaire conserve : {l1_form}")

    l2 = exporter_l2()
    l2_form = os.path.join(FORMS, "l2_contenu_reponses.csv")
    if _scores_vides(l2_form, ["Q1_pertinence", "Q4_qualite"]):
        l2.to_csv(l2_form, index=False, encoding="utf-8-sig")
        print(f"L2 contenu : {len(l2)} fiches -> {l2_form}")
    else:
        print(f"L2 deja note, formulaire conserve : {l2_form}")

    l4_form = os.path.join(FORMS, "l4_pilote_reponses.csv")
    if not os.path.isfile(l4_form):
        sus_cols = {f"sus_{i}": "" for i in range(1, 11)}
        l4_rows = []
        for pid in range(1, 31):
            l4_rows.append(
                {
                    "participant_id": pid,
                    "ordre_conditions": "C1_puis_C2" if pid % 2 else "C2_puis_C1",
                    "niveau": "",
                    "familiarite_mooc_1_5": "",
                    "clip_id": "",
                    "condition": "",
                    "utilite_1_7": "",
                    "adequation_emotionnelle_1_7": "",
                    "satisfaction_contenu_1_7": "",
                    **sus_cols,
                    "sus_score": "",
                    "q_ouvert_aide": "",
                    "q_ouvert_gene": "",
                    "q_ouvert_suggestion": "",
                }
            )
        pd.DataFrame(l4_rows).to_csv(l4_form, index=False, encoding="utf-8-sig")
        print(f"L4 pilote : 30 lignes vides -> {l4_form}")


if __name__ == "__main__":
    main()
