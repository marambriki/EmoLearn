#!/usr/bin/env python3
# ============================================
# Audit qualité du corpus généré + échantillon évaluation (60 ex.)
# ============================================
"""
Usage:
  python scripts/audit_corpus.py
  python scripts/audit_corpus.py --corpus data/corpus_200.csv --out results/
"""

from __future__ import annotations

import argparse
import os
import re
import sys

import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
BACKEND = os.path.join(ROOT, "backend")
sys.path.insert(0, BACKEND)

from prompts.templates import TEMPLATE_META, get_meta_template

RESULTS_DIR = os.path.join(ROOT, "results")

# Mots-clés attendus / interdits par famille de template
KEYWORDS_POSITIVE = re.compile(
    r"\b(bravo|félicit|felicit|fier|excellent|exceptionnel|réussi|reussi|parfait)\b",
    re.I,
)
KEYWORDS_NEGATIVE_SUPPORT = re.compile(
    r"\b(difficile|normal|pause|déculpabil|deculpabil|persévér|persever|encourage)\b",
    re.I,
)
KEYWORDS_EXERCISE = re.compile(
    r"\b(exercice|problème|probleme|étape|etape|consigne|justifiez|implément)\b",
    re.I,
)
KEYWORDS_EXPLAIN = re.compile(
    r"\b(concept|reformul|analogi|étape|etape|compréhension|comprehension)\b",
    re.I,
)

# Émotions attendues par préfixe template (souples)
TEMPLATE_EMOTION_HINTS = {
    "A": {"confusion", "critique"},
    "B": {"boredom", "ennui", "engagement", "engagement_faible"},
    "C": {"boredom", "ennui", "engagement"},
    "D": {"frustration", "engagement", "boredom", "confusion", "critique"},
}


def _parse_longueur_cible(template_id: str) -> tuple[int, int]:
    meta = get_meta_template(template_id)
    text = meta.get("longueur", "200 mots")
    nums = [int(n) for n in re.findall(r"\d+", text)]
    if len(nums) >= 2:
        return nums[0], nums[1]
    if len(nums) == 1:
        w = nums[0]
        return int(w * 0.7), int(w * 1.3)
    return 120, 320


def _emotion_ok(template_id: str, emotion: str, pattern: str) -> bool:
    prefix = template_id[0] if template_id else "A"
    hints = TEMPLATE_EMOTION_HINTS.get(prefix, set())
    em = str(emotion).lower().strip()
    pat = str(pattern).lower().strip()

    if prefix == "A" and ("confusion" in em or pat in ("confusion", "critique")):
        return True
    if prefix == "B" and ("boredom" in em or "ennui" in pat or "engagement" in em):
        return True
    if prefix == "C" and ("boredom" in em or "ennui" in pat or "engagement" in em):
        return True
    if prefix == "D":
        return True
    return em in hints or pat in hints


def _contenu_coherent(template_id: str, emotion: str, contenu: str) -> list[str]:
    """Retourne liste de problèmes détectés."""
    issues = []
    em = str(emotion).lower()
    tid = template_id or ""

    if tid in ("D1", "D2", "D3") and KEYWORDS_POSITIVE.search(contenu) and "difficile" not in contenu.lower():
        if "frustration" in em or "confusion" in em:
            issues.append("ton_trop_positif_pour_frustration")

    if tid.startswith("A") and not KEYWORDS_EXPLAIN.search(contenu):
        issues.append("manque_marqueurs_explication")

    if tid.startswith("C") and not KEYWORDS_EXERCISE.search(contenu):
        issues.append("manque_marqueurs_exercice")

    if tid == "D4" and "frustration" in em and KEYWORDS_POSITIVE.search(contenu):
        issues.append("D4_sur_frustration_avec_felicitations")

    if tid in ("A1", "A2", "A3") and KEYWORDS_POSITIVE.search(contenu) and not KEYWORDS_NEGATIVE_SUPPORT.search(contenu):
        if "confusion" in em:
            issues.append("confusion_sans_ton_rassurant")

    return issues


def auditer_ligne(row) -> dict:
    template_id = str(row.get("template_id", "")).strip()
    emotion = str(row.get("emotion", "")).lower()
    pattern = str(row.get("pattern", "")).lower()
    contenu = str(row.get("contenu", "") or "")
    nb_mots = len(contenu.split())

    min_w, max_w = _parse_longueur_cible(template_id)
    issues = _contenu_coherent(template_id, emotion, contenu)

    if not _emotion_ok(template_id, emotion, pattern):
        issues.append("emotion_template_suspect")

    if nb_mots < min_w * 0.5:
        issues.append("trop_court")
    elif nb_mots > max_w * 1.8:
        issues.append("trop_long")

    if template_id not in TEMPLATE_META:
        issues.append("template_inconnu")

    score = max(0, 100 - 25 * len(issues))
    return {
        "clip_id": row.get("clip_id"),
        "pattern": row.get("pattern"),
        "template_id": template_id,
        "emotion": emotion,
        "niveau": row.get("niveau"),
        "nb_mots": nb_mots,
        "longueur_attendue": f"{min_w}-{max_w}",
        "issues": "|".join(issues) if issues else "",
        "n_issues": len(issues),
        "quality_score": score,
        "ok": len(issues) == 0,
    }


def construire_echantillon_evaluation(df_audit: pd.DataFrame, n_total: int = 60) -> pd.DataFrame:
    """15 exemples par famille A/B/C/D (via préfixe template)."""
    ok_df = df_audit[df_audit["ok"]].copy()
    if ok_df.empty:
        ok_df = df_audit.sort_values("quality_score", ascending=False)

    per_family = n_total // 4
    samples = []

    for prefix in ("A", "B", "C", "D"):
        subset = ok_df[ok_df["template_id"].str.startswith(prefix, na=False)]
        if subset.empty:
            subset = df_audit[df_audit["template_id"].str.startswith(prefix, na=False)]
        n = min(per_family, len(subset))
        if n > 0:
            samples.append(subset.sample(n=n, random_state=42))

    if not samples:
        return df_audit.head(n_total)

    out = pd.concat(samples).drop_duplicates(subset=["clip_id", "template_id"])
    return out.head(n_total)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", default=os.path.join(ROOT, "data", "corpus_200.csv"))
    parser.add_argument("--out", default=RESULTS_DIR)
    parser.add_argument("--evaluation-size", type=int, default=60)
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    if not os.path.exists(args.corpus):
        alt = os.path.join(ROOT, "data", "corpus_contenus.csv")
        if os.path.exists(alt):
            args.corpus = alt
        else:
            raise FileNotFoundError(f"Corpus introuvable: {args.corpus}")

    corpus = pd.read_csv(args.corpus)
    audits = [auditer_ligne(row) for _, row in corpus.iterrows()]
    df_audit = pd.DataFrame(audits)

    audit_path = os.path.join(args.out, "corpus_audit_report.csv")
    df_audit.to_csv(audit_path, index=False)

    merged = pd.concat(
        [corpus.reset_index(drop=True), df_audit[["issues", "quality_score", "ok", "n_issues"]]],
        axis=1,
    )

    eval_df = construire_echantillon_evaluation(merged, n_total=args.evaluation_size)
    eval_path = os.path.join(ROOT, "data", "corpus_evaluation_60.csv")
    eval_df.to_csv(eval_path, index=False)

    summary = {
        "corpus_source": args.corpus,
        "n_total": len(df_audit),
        "n_ok": int(df_audit["ok"].sum()),
        "n_with_issues": int((df_audit["n_issues"] > 0).sum()),
        "pct_ok": round(100 * df_audit["ok"].mean(), 1),
        "top_issues": (
            df_audit[df_audit["issues"] != ""]["issues"]
            .str.split("|")
            .explode()
            .value_counts()
            .head(10)
            .to_dict()
        ),
        "audit_csv": audit_path,
        "evaluation_csv": eval_path,
        "n_evaluation_sample": len(eval_df),
    }

    summary_path = os.path.join(args.out, "corpus_audit_summary.json")
    import json

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("=== AUDIT CORPUS ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
