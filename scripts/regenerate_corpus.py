#!/usr/bin/env python3
# ============================================
# Régénère les entrées du corpus marquées incohérentes par audit_corpus.py
# Utilise decision_engine + llm_module (templates corrects par scénario)
# ============================================
"""
Usage:
  python scripts/regenerate_corpus.py --dry-run
  python scripts/regenerate_corpus.py --limit 5
  python scripts/regenerate_corpus.py
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
BACKEND = os.path.join(ROOT, "backend")
sys.path.insert(0, BACKEND)

CONCEPT = "les algorithmes de tri"
MODULE = "Algorithmique"


def charger_scenarios():
    path = os.path.join(ROOT, "data", "scenarios_50.csv")
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    df["clip_key"] = df["ClipID"].astype(str).str.strip()
    return df.set_index("clip_key")


def _compute_decision(row_corpus, df_scenarios):
    from fusion_module import fusionner_modalites
    from fer_module import analyser_visage_depuis_scenario
    from ser_module import analyser_voix_depuis_scenario
    from decision_engine import preparer_decision

    clip_id = str(row_corpus["clip_id"]).strip()
    if clip_id not in df_scenarios.index:
        return None, f"scénario introuvable pour {clip_id}"

    scenario = df_scenarios.loc[clip_id]
    if isinstance(scenario, pd.DataFrame):
        scenario = scenario.iloc[0]

    scenario_id = list(df_scenarios.index).index(clip_id)
    pattern = str(scenario.get("pattern", "stable"))

    visage = analyser_visage_depuis_scenario(scenario)
    voix = analyser_voix_depuis_scenario(scenario_id=scenario_id)
    fusion = fusionner_modalites(visage, voix)
    decision = preparer_decision(
        fusion=fusion,
        boredom=int(scenario["Boredom"]),
        confusion=int(scenario["Confusion"]),
        engagement=int(scenario["Engagement"]),
        frustration=int(scenario["Frustration"]),
        pattern=pattern,
    )

    return {
        "clip_id": clip_id,
        "pattern": pattern,
        "decision": decision,
        "old_template": str(row_corpus.get("template_id", "")),
        "frustration": int(scenario["Frustration"]),
    }, None


def regenerer_ligne(
    row_corpus,
    df_scenarios,
    dry_run: bool,
    metadata_only: bool = False,
    utiliser_rag: bool = False,
):
    ctx, err = _compute_decision(row_corpus, df_scenarios)
    if err:
        return None, err

    decision = ctx["decision"]
    new_template = decision["template_id"]

    if dry_run:
        return {
            "clip_id": ctx["clip_id"],
            "old_template": ctx["old_template"],
            "new_template": new_template,
            "etat": decision["etat"],
        }, None

    if metadata_only:
        return {
            "clip_id": ctx["clip_id"],
            "pattern": ctx["pattern"],
            "template_id": new_template,
            "emotion": decision["etat"].lower(),
            "niveau": decision["niveau"],
            "contenu": row_corpus.get("contenu"),
            "nb_mots": row_corpus.get("nb_mots"),
            "old_template": ctx["old_template"],
            "metadata_only": True,
        }, None

    from llm_module import generer_contenu_adaptatif

    resultat = generer_contenu_adaptatif(
        etat=decision["etat"],
        niveau=decision["niveau"],
        concept=CONCEPT,
        template_id=new_template,
        objectif_adaptation=decision["objectif"],
        sujet_cours=f"Informatique — {MODULE}",
        module_actuel=MODULE,
        niveau_apprenant="Intermédiaire",
        duree_etat=2,
        pattern_detecte=ctx["pattern"],
        frustration_niveau=ctx["frustration"],
        utiliser_rag=utiliser_rag,
    )

    return {
        "clip_id": ctx["clip_id"],
        "pattern": ctx["pattern"],
        "template_id": new_template,
        "emotion": decision["etat"].lower(),
        "niveau": decision["niveau"],
        "contenu": resultat["contenu"],
        "nb_mots": resultat["nb_mots"],
        "old_template": ctx["old_template"],
    }, None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", default=os.path.join(ROOT, "data", "corpus_200.csv"))
    parser.add_argument("--audit", default=os.path.join(ROOT, "results", "corpus_audit_report.csv"))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--metadata-only",
        action="store_true",
        help="Met à jour template_id/emotion/niveau sans appeler le LLM",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--delay", type=float, default=1.0, help="Pause entre appels LLM (s)")
    parser.add_argument(
        "--rag",
        action="store_true",
        help="Active le RAG (nécessite sentence_transformers + chromadb)",
    )
    args = parser.parse_args()

    corpus = pd.read_csv(args.corpus)
    audit = pd.read_csv(args.audit)
    df_scenarios = charger_scenarios()

    a_corriger = audit[~audit["ok"].astype(bool)]
    keys = set(zip(a_corriger["clip_id"], a_corriger["template_id"]))

    indices = []
    for i, row in corpus.iterrows():
        if (str(row["clip_id"]), str(row["template_id"])) in keys:
            indices.append(i)

    if args.limit:
        indices = indices[: args.limit]

    mode = "dry-run" if args.dry_run else ("metadata-only" if args.metadata_only else "full")
    print(f"Entrées à régénérer : {len(indices)} (mode={mode})")

    updates = 0
    errors = []

    for i in indices:
        row = corpus.iloc[i]
        print(f"\n[{updates + 1}/{len(indices)}] {row['clip_id']} — {row['template_id']}")

        try:
            result, err = regenerer_ligne(
                row,
                df_scenarios,
                args.dry_run,
                args.metadata_only,
                utiliser_rag=args.rag,
            )
            if err:
                errors.append({"index": i, "error": err})
                print(f"  SKIP: {err}")
                continue

            if args.dry_run:
                print(
                    f"  → {result['old_template']} => {result['new_template']} "
                    f"({result.get('etat', '')})"
                )
            else:
                corpus.at[i, "template_id"] = result["template_id"]
                corpus.at[i, "pattern"] = result["pattern"]
                corpus.at[i, "emotion"] = result["emotion"]
                corpus.at[i, "niveau"] = result["niveau"]
                if not result.get("metadata_only"):
                    corpus.at[i, "contenu"] = result["contenu"]
                    corpus.at[i, "nb_mots"] = result["nb_mots"]
                flag = " [métadonnées]" if result.get("metadata_only") else ""
                print(
                    f"  OK: {result['old_template']} => {result['template_id']}{flag}"
                )
                if not args.metadata_only:
                    time.sleep(args.delay)
            updates += 1
        except Exception as e:
            errors.append({"index": i, "error": str(e)})
            print(f"  ERREUR: {e}")

    plan_path = os.path.join(ROOT, "results", "corpus_regeneration_plan.csv")
    if args.dry_run and updates > 0:
        print(f"\nRelancer sans --dry-run pour appliquer les changements.")

    if not args.dry_run and updates > 0:
        backup = args.corpus.replace(".csv", "_backup.csv")
        pd.read_csv(args.corpus).to_csv(backup, index=False)
        corpus.to_csv(args.corpus, index=False)
        print(f"\nBackup : {backup}")
        print(f"Corpus mis à jour : {args.corpus}")
        print("Relancez : python scripts/audit_corpus.py")

    if errors:
        print(f"\n{len(errors)} erreur(s) :")
        for e in errors[:5]:
            print(f"  - {e}")

    print(f"\nTerminé : {updates} entrée(s) traitée(s).")


if __name__ == "__main__":
    main()
