#!/usr/bin/env python3
# ============================================
# Évaluation du RAG (qualité du retrieval ChromaDB)
# Mesure si le système récupère les BONS documents pour chaque question.
#
# Métriques (standard en recherche d'information) :
#   - Hit Rate @k   : au moins 1 document pertinent dans le top-k
#   - Precision @k  : proportion de documents pertinents parmi ceux récupérés
#   - MRR           : Mean Reciprocal Rank (1 / rang du 1er doc pertinent)
#   - Score moyen   : similarité cosinus moyenne du top-1
#
# Un document est "pertinent" si son concept correspond au concept attendu.
#
# Usage :
#   python scripts/eval_rag.py
#   python scripts/eval_rag.py --k 3
#   python scripts/eval_rag.py --mode query-only   # test robustesse (sans concept)
# ============================================

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone

# Sortie UTF-8 (évite les crashs d'emojis sur terminal Windows cp1252)
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
BACKEND = os.path.join(ROOT, "backend")
RESULTS = os.path.join(ROOT, "results")
sys.path.insert(0, BACKEND)


# ============================================
# Jeu de test : questions réalistes d'apprenants
# Chaque entrée : (question, concept attendu, état émotionnel)
# ============================================
JEU_DE_TEST = [
    # --- Algorithmes de tri ---
    ("Comment fonctionne le tri à bulles ?", "algorithmes de tri", "Confusion"),
    ("Quelle est la différence entre QuickSort et MergeSort ?", "algorithmes de tri", "Confusion"),
    ("Explique le tri par insertion", "algorithmes de tri", "Confusion"),
    ("Quel algorithme de tri est le plus rapide ?", "algorithmes de tri", "Engagement"),

    # --- Complexité algorithmique ---
    ("C'est quoi la notation Big-O ?", "complexité algorithmique", "Confusion"),
    ("Comment mesurer la complexité d'un algorithme ?", "complexité algorithmique", "Confusion"),
    ("Quelle est la complexité d'une recherche dichotomique ?", "complexité algorithmique", "Engagement"),

    # --- Récursion ---
    ("C'est quoi une fonction récursive ?", "récursion", "Confusion"),
    ("Pourquoi une récursion a besoin d'un cas de base ?", "récursion", "Confusion"),
    ("Donne un exemple de récursion", "récursion", "Engagement"),

    # --- Réseaux de neurones ---
    ("Comment fonctionne un réseau de neurones ?", "réseaux de neurones", "Confusion"),
    ("C'est quoi la fonction d'activation ReLU ?", "réseaux de neurones", "Confusion"),
    ("Explique la rétropropagation dans un réseau de neurones", "réseaux de neurones", "Confusion"),

    # --- Structures de données ---
    ("Quelles sont les principales structures de données ?", "structures de données", "Confusion"),
    ("C'est quoi une pile et une file ?", "structures de données", "Confusion"),
    ("Comment fonctionne un arbre binaire de recherche ?", "structures de données", "Engagement"),

    # --- Encouragement (états négatifs) ---
    ("Je suis frustré, je n'arrive pas à comprendre", "encouragement", "Frustration"),
    ("Comment rester motivé quand c'est difficile ?", "encouragement", "Frustration"),
    ("Je m'ennuie, ce cours est trop long", "encouragement", "Boredom"),
]


def evaluer(k: int = 3, mode: str = "realiste"):
    """Évalue le retrieval RAG sur le jeu de test."""
    from rag_module import rechercher_documents

    resultats = []
    for question, concept_attendu, etat in JEU_DE_TEST:
        # Mode "query-only" : on ne fournit PAS le concept (test de robustesse pure)
        concept_arg = "" if mode == "query-only" else concept_attendu

        docs = rechercher_documents(
            query=question,
            concept=concept_arg,
            etat=etat,
            n_results=k,
        )

        # Un doc est pertinent si son concept == concept attendu
        pertinents = [d["metadata"]["concept"] == concept_attendu for d in docs]
        nb_recup = len(docs)
        nb_pertinents = sum(pertinents)

        # Hit@k : au moins un pertinent dans le top-k
        hit = 1 if nb_pertinents > 0 else 0

        # Precision@k : pertinents / récupérés
        precision = (nb_pertinents / nb_recup) if nb_recup else 0.0

        # Reciprocal Rank : 1 / rang du premier pertinent
        rr = 0.0
        for rang, est_pertinent in enumerate(pertinents, start=1):
            if est_pertinent:
                rr = 1.0 / rang
                break

        score_top1 = round(docs[0]["score"], 3) if docs else 0.0

        resultats.append({
            "question": question,
            "concept_attendu": concept_attendu,
            "etat": etat,
            "nb_recuperes": nb_recup,
            "nb_pertinents": nb_pertinents,
            "hit@k": hit,
            "precision@k": round(precision, 3),
            "reciprocal_rank": round(rr, 3),
            "score_top1": score_top1,
            "concepts_recuperes": [d["metadata"]["concept"] for d in docs],
        })

    return resultats


def agreger(resultats, k):
    """Calcule les métriques globales et par concept."""
    n = len(resultats)
    hit_rate = sum(r["hit@k"] for r in resultats) / n
    precision = sum(r["precision@k"] for r in resultats) / n
    mrr = sum(r["reciprocal_rank"] for r in resultats) / n
    score_moyen = sum(r["score_top1"] for r in resultats) / n

    # Par concept
    par_concept = {}
    for r in resultats:
        c = r["concept_attendu"]
        par_concept.setdefault(c, []).append(r)
    concepts_stats = {}
    for c, rs in par_concept.items():
        m = len(rs)
        concepts_stats[c] = {
            "nb_questions": m,
            "hit_rate": round(sum(x["hit@k"] for x in rs) / m, 3),
            "precision@k": round(sum(x["precision@k"] for x in rs) / m, 3),
            "mrr": round(sum(x["reciprocal_rank"] for x in rs) / m, 3),
        }

    return {
        "k": k,
        "nb_questions": n,
        "hit_rate": round(hit_rate, 3),
        "precision@k": round(precision, 3),
        "mrr": round(mrr, 3),
        "score_top1_moyen": round(score_moyen, 3),
        "par_concept": concepts_stats,
    }


def ecrire_markdown(summary, resultats, chemin):
    """Génère un rapport lisible en Markdown."""
    lignes = []
    lignes.append("# Évaluation du RAG — qualité du retrieval\n")
    lignes.append(f"> Généré le {summary['date']} · mode = `{summary['mode']}` · k = {summary['k']}\n")
    lignes.append("## Métriques globales\n")
    lignes.append("| Métrique | Valeur | Interprétation |")
    lignes.append("|---|---|---|")
    lignes.append(f"| **Hit Rate @{summary['k']}** | **{summary['hit_rate']:.0%}** | Au moins 1 bon document trouvé |")
    lignes.append(f"| **Precision @{summary['k']}** | {summary['precision@k']:.0%} | Proportion de documents pertinents |")
    lignes.append(f"| **MRR** | {summary['mrr']:.3f} | Rang moyen du 1er bon document (1.0 = idéal) |")
    lignes.append(f"| **Score top-1 moyen** | {summary['score_top1_moyen']:.3f} | Similarité cosinus moyenne |")
    lignes.append("")
    lignes.append("## Par concept\n")
    lignes.append("| Concept | Questions | Hit Rate | Precision | MRR |")
    lignes.append("|---|---|---|---|---|")
    for c, s in summary["par_concept"].items():
        lignes.append(
            f"| {c} | {s['nb_questions']} | {s['hit_rate']:.0%} | "
            f"{s['precision@k']:.0%} | {s['mrr']:.3f} |"
        )
    lignes.append("")
    lignes.append("## Détail par question\n")
    lignes.append("| Question | Concept attendu | Hit | Prec | RR | Score top-1 |")
    lignes.append("|---|---|---|---|---|---|")
    for r in resultats:
        hit = "✅" if r["hit@k"] else "❌"
        lignes.append(
            f"| {r['question']} | {r['concept_attendu']} | {hit} | "
            f"{r['precision@k']:.2f} | {r['reciprocal_rank']:.2f} | {r['score_top1']:.3f} |"
        )
    lignes.append("")
    with open(chemin, "w", encoding="utf-8") as f:
        f.write("\n".join(lignes))


def main():
    parser = argparse.ArgumentParser(description="Évaluation du retrieval RAG.")
    parser.add_argument("--k", type=int, default=3, help="Nombre de documents récupérés (top-k).")
    parser.add_argument(
        "--mode",
        choices=["realiste", "query-only"],
        default="realiste",
        help="realiste = concept fourni (comme en production) ; query-only = question seule.",
    )
    parser.add_argument("--out", default=os.path.join(RESULTS, "rag_eval.json"))
    parser.add_argument("--out-md", default=os.path.join(RESULTS, "rag_eval.md"))
    args = parser.parse_args()

    os.makedirs(RESULTS, exist_ok=True)

    print(f"Évaluation RAG — mode={args.mode}, k={args.k}, {len(JEU_DE_TEST)} questions...\n")
    resultats = evaluer(k=args.k, mode=args.mode)
    summary = agreger(resultats, args.k)
    summary["mode"] = args.mode
    summary["date"] = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    # Sauvegardes
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "details": resultats}, f, indent=2, ensure_ascii=False)
    ecrire_markdown(summary, resultats, args.out_md)

    # Affichage console
    print("===== MÉTRIQUES GLOBALES =====")
    print(f"  Hit Rate @{args.k}   : {summary['hit_rate']:.0%}")
    print(f"  Precision @{args.k}  : {summary['precision@k']:.0%}")
    print(f"  MRR              : {summary['mrr']:.3f}")
    print(f"  Score top-1 moy. : {summary['score_top1_moyen']:.3f}")
    print("\n===== PAR CONCEPT =====")
    for c, s in summary["par_concept"].items():
        print(f"  {c:28s} hit={s['hit_rate']:.0%}  prec={s['precision@k']:.0%}  mrr={s['mrr']:.2f}")
    print(f"\n✅ Résultats écrits : {args.out}")
    print(f"✅ Rapport Markdown  : {args.out_md}")


if __name__ == "__main__":
    main()
