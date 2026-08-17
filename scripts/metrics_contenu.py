#!/usr/bin/env python3
# ============================================
# Métriques automatiques sur corpus de contenus
# (forme / lisibilité — pas la pertinence pédagogique)
# ============================================

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime, timezone

import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RESULTS = os.path.join(ROOT, "results")

# Cibles indicatives par famille de template (mots)
CIBLES_MOTS = {
    "A": (150, 280),
    "B": (180, 320),
    "C": (200, 360),
    "D": (100, 220),
}

# Mots-clés attendus (heuristique légère) par type
MOTS_CLES_TYPE = {
    "A": ["explication", "étape", "exemple", "comprendre", "notion"],
    "B": ["exercice", "pratique", "essayer", "appliquer", "challenge"],
    "C": ["encourager", "motivation", "progress", "confiance", "continuer"],
    "D": ["approfondir", "aller plus loin", "variante", "bonus", "extension"],
}


def syllables_approx_fr(text: str) -> int:
    """Estimation grossière des syllabes pour lisibilité."""
    text = text.lower()
    text = re.sub(r"[^a-zàâäéèêëïîôùûüç\s]", " ", text)
    vowels = "aeiouyàâäéèêëïîôùûü"
    count = 0
    prev_v = False
    for ch in text:
        is_v = ch in vowels
        if is_v and not prev_v:
            count += 1
        prev_v = is_v
    return max(count, 1)


def kandel_moles_approx(nb_mots: int, nb_syllabes: int) -> float:
    """Indice Kandel-Moles adapté (échelle ~100, plus haut = plus facile)."""
    if nb_mots == 0:
        return 0.0
    ps = nb_syllabes / nb_mots * 100
    pl = 100 / nb_mots * 30
    return round(0.5 * ps + 0.5 * pl, 1)


def repetition_rate(text: str) -> float:
    words = re.findall(r"\b\w+\b", text.lower())
    if len(words) < 5:
        return 0.0
    unique = len(set(words))
    return round(1 - unique / len(words), 3)


def couverture_mots_cles(text: str, type_contenu: str) -> dict:
    keys = MOTS_CLES_TYPE.get(str(type_contenu).upper()[:1], [])
    low = text.lower()
    hits = [k for k in keys if k in low]
    return {
        "mots_cles_attendus": keys,
        "mots_cles_presents": hits,
        "taux_couverture_mots_cles": round(len(hits) / len(keys), 3) if keys else None,
    }


def dans_cible_longueur(nb_mots: int, template_id: str) -> bool | None:
    if not template_id:
        return None
    fam = str(template_id)[0].upper()
    cible = CIBLES_MOTS.get(fam)
    if not cible:
        return None
    return cible[0] <= nb_mots <= cible[1]


def analyser_contenu(text: str, template_id: str = "", type_contenu: str = "") -> dict:
    mots = text.split()
    nb_mots = len(mots)
    phrases = [p for p in re.split(r"[.!?]+", text) if p.strip()]
    nb_phrases = max(len(phrases), 1)
    syl = syllables_approx_fr(text)
    type_c = type_contenu or (str(template_id)[0] if template_id else "")
    cov = couverture_mots_cles(text, type_c)
    return {
        "nb_mots": nb_mots,
        "nb_phrases": nb_phrases,
        "mots_par_phrase": round(nb_mots / nb_phrases, 1),
        "syllabes_approx": syl,
        "kandel_moles_approx": kandel_moles_approx(nb_mots, syl),
        "taux_repetition": repetition_rate(text),
        "dans_cible_longueur": dans_cible_longueur(nb_mots, template_id),
        **cov,
    }


def rapport_memoire(summary: dict) -> str:
    """Texte prêt à coller dans le chapitre évaluation (forme)."""
    lines = [
        "## Métriques automatiques de forme (corpus)",
        "",
        f"- N = {summary['n']} contenus",
        f"- Longueur moyenne : {summary['mots_moyen']} mots "
        f"(min {summary['mots_min']}, max {summary['mots_max']})",
        f"- Lisibilité approx. (Kandel-Moles) : {summary['kandel_moles_moyen']}",
        f"- Taux de répétition moyen : {summary['repetition_moyenne']}",
        f"- Part dans la cible de longueur : {summary.get('pct_dans_cible_longueur', '—')}",
        f"- Couverture mots-clés (heuristique) : {summary.get('couverture_mots_cles_moyenne', '—')}",
        "",
        "> Ces métriques mesurent la **forme**, pas la pertinence pédagogique "
        "(réservée à la validation experte L2).",
        "",
    ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", default=os.path.join(ROOT, "data", "corpus_200.csv"))
    parser.add_argument("--out", default=os.path.join(RESULTS, "corpus_metrics.json"))
    parser.add_argument(
        "--export-md",
        default=os.path.join(RESULTS, "corpus_metrics_memoire.md"),
        help="Export markdown pour le mémoire",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.corpus)
    rows = []
    for _, r in df.iterrows():
        tid = str(r.get("template_id", "") or "")
        typ = str(r.get("type_contenu", "") or (tid[:1] if tid else ""))
        m = analyser_contenu(str(r.get("contenu", "")), template_id=tid, type_contenu=typ)
        m["clip_id"] = r.get("clip_id")
        m["template_id"] = tid
        m["type_contenu"] = typ
        m["emotion"] = r.get("emotion")
        rows.append(m)

    metrics_df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.out) or RESULTS, exist_ok=True)

    cible_series = metrics_df["dans_cible_longueur"].dropna()
    cov_series = metrics_df["taux_couverture_mots_cles"].dropna()

    summary = {
        "n": len(metrics_df),
        "mots_moyen": round(float(metrics_df["nb_mots"].mean()), 1),
        "mots_min": int(metrics_df["nb_mots"].min()),
        "mots_max": int(metrics_df["nb_mots"].max()),
        "mots_median": round(float(metrics_df["nb_mots"].median()), 1),
        "kandel_moles_moyen": round(float(metrics_df["kandel_moles_approx"].mean()), 1),
        "repetition_moyenne": round(float(metrics_df["taux_repetition"].mean()), 3),
        "pct_dans_cible_longueur": (
            round(float(cible_series.mean()), 3) if len(cible_series) else None
        ),
        "couverture_mots_cles_moyenne": (
            round(float(cov_series.mean()), 3) if len(cov_series) else None
        ),
        "par_template": (
            metrics_df.groupby("template_id")["nb_mots"]
            .agg(["mean", "count"])
            .round(1)
            .to_dict()
        ),
        "par_type": (
            metrics_df.groupby("type_contenu")["nb_mots"]
            .agg(["mean", "count"])
            .round(1)
            .to_dict()
            if "type_contenu" in metrics_df.columns
            else {}
        ),
        "cibles_mots": CIBLES_MOTS,
        "note": (
            "Métriques de FORME uniquement. La qualité pédagogique de fond "
            "relève de la validation experte L2 (PROTOCOLE_EVALUATION.md)."
        ),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    metrics_df.to_csv(args.out.replace(".json", ".csv"), index=False, encoding="utf-8-sig")
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    md = rapport_memoire(summary)
    with open(args.export_md, "w", encoding="utf-8") as f:
        f.write(md)

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\n✅ Markdown mémoire : {args.export_md}")


if __name__ == "__main__":
    main()
