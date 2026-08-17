#!/usr/bin/env python3
# ============================================
# BLEU / ROUGE / BERTScore — similarité LLM
# ============================================
#
# Usage (corpus réel, mode par défaut) :
#   python scripts/metrics_llm_similarity.py
#   python scripts/metrics_llm_similarity.py --skip-bertscore
#
# Mode gold (références experts) :
#   python scripts/metrics_llm_similarity.py --mode gold \
#       --references data/references_contenu.csv
#
# Smoke test :
#   python scripts/metrics_llm_similarity.py --demo
#
# Dépendances optionnelles :
#   pip install sacrebleu rouge-score bert-score
# ============================================

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone

import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RESULTS = os.path.join(ROOT, "results")
DATA = os.path.join(ROOT, "data")

CANDIDATE_COLS = ("contenu", "contenu_genere", "candidate", "text", "hypothese")
REFERENCE_COLS = ("contenu_reference", "reference", "contenu_ref", "ref", "gold")


def _require(pkg: str, pip_name: str | None = None):
    try:
        return __import__(pkg)
    except ImportError as exc:
        name = pip_name or pkg
        raise SystemExit(
            f"Package manquant : {pkg}. Installez avec : pip install {name}"
        ) from exc


def _pick_col(df: pd.DataFrame, candidates: tuple[str, ...], label: str) -> str:
    for col in candidates:
        if col in df.columns:
            return col
    raise SystemExit(
        f"Colonne {label} introuvable. Attendu l'un de : {', '.join(candidates)}. "
        f"Colonnes présentes : {list(df.columns)}"
    )


def _join_keys(cand: pd.DataFrame, refs: pd.DataFrame) -> list[str]:
    """Choisit les clés de jointure les plus précises disponibles des deux côtés."""
    if "clip_id" in cand.columns and "clip_id" in refs.columns and "template_id" in cand.columns and "template_id" in refs.columns:
        return ["clip_id", "template_id"]
    if "id" in cand.columns and "id" in refs.columns:
        return ["id"]
    if "template_id" in cand.columns and "template_id" in refs.columns:
        return ["template_id"]
    raise SystemExit(
        "Impossible de joindre corpus et références. "
        "Fournissez clip_id+template_id, ou id, ou template_id des deux côtés."
    )


def tokenize_fr(text: str) -> list[str]:
    """Tokenisation légère FR (fallback si sacrebleu indisponible pour un usage interne)."""
    import re

    text = (text or "").lower().strip()
    return re.findall(r"\b[\w'-]+\b", text, flags=re.UNICODE)


def _ngrams(tokens: list[str], n: int) -> list[tuple[str, ...]]:
    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def _bleu_sentence_simple(hyp: str, ref: str, max_n: int = 4) -> float:
    """BLEU-4 sentence-level minimal (précision + brevity penalty), score [0, 1]."""
    import math
    from collections import Counter

    hyp_t = tokenize_fr(hyp)
    ref_t = tokenize_fr(ref)
    if not hyp_t or not ref_t:
        return 0.0

    precisions = []
    for n in range(1, max_n + 1):
        hyp_ng = Counter(_ngrams(hyp_t, n))
        ref_ng = Counter(_ngrams(ref_t, n))
        overlap = sum((hyp_ng & ref_ng).values())
        total = sum(hyp_ng.values())
        if total == 0:
            precisions.append(0.0)
        else:
            # lissage add-1 pour éviter log(0) sur phrases courtes
            precisions.append((overlap + 1) / (total + 1))

    bp = 1.0 if len(hyp_t) >= len(ref_t) else math.exp(1 - len(ref_t) / len(hyp_t))
    log_avg = sum(math.log(p) for p in precisions) / max_n
    return round(bp * math.exp(log_avg), 4)


def score_bleu(candidates: list[str], references: list[str]) -> list[float]:
    """BLEU-4 sentence-level, score dans [0, 1]. Préfère sacrebleu, sinon fallback."""
    try:
        import sacrebleu

        scores = []
        for hyp, ref in zip(candidates, references):
            blu = sacrebleu.sentence_bleu(hyp, [ref], tokenize="intl")
            scores.append(round(blu.score / 100.0, 4))
        return scores
    except ImportError:
        print("WARN sacrebleu absent — BLEU calculé en fallback local")
        return [
            _bleu_sentence_simple(hyp, ref)
            for hyp, ref in zip(candidates, references)
        ]


def score_rouge(candidates: list[str], references: list[str]) -> dict[str, list[float]]:
    """ROUGE-1 / ROUGE-2 / ROUGE-L (F1). Préfère rouge-score, sinon fallback local."""
    try:
        from rouge_score import rouge_scorer as rs

        scorer = rs.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
        out = {"rouge1_f": [], "rouge2_f": [], "rougeL_f": []}
        for hyp, ref in zip(candidates, references):
            s = scorer.score(ref, hyp)
            out["rouge1_f"].append(round(s["rouge1"].fmeasure, 4))
            out["rouge2_f"].append(round(s["rouge2"].fmeasure, 4))
            out["rougeL_f"].append(round(s["rougeL"].fmeasure, 4))
        return out
    except ImportError:
        print("WARN rouge-score absent — ROUGE calculé en fallback local")
        return _score_rouge_simple(candidates, references)


def _f1_overlap(hyp_items: list, ref_items: list) -> float:
    from collections import Counter

    if not hyp_items or not ref_items:
        return 0.0
    hyp_c, ref_c = Counter(hyp_items), Counter(ref_items)
    overlap = sum((hyp_c & ref_c).values())
    precision = overlap / sum(hyp_c.values())
    recall = overlap / sum(ref_c.values())
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _lcs_length(a: list[str], b: list[str]) -> int:
    """Longueur de la plus longue sous-séquence commune (ROUGE-L)."""
    m, n = len(a), len(b)
    if m == 0 or n == 0:
        return 0
    # optimisation espace O(min(m,n))
    if m < n:
        a, b, m, n = b, a, n, m
    prev = [0] * (n + 1)
    for i in range(1, m + 1):
        cur = [0] * (n + 1)
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                cur[j] = prev[j - 1] + 1
            else:
                cur[j] = max(prev[j], cur[j - 1])
        prev = cur
    return prev[n]


def _score_rouge_simple(
    candidates: list[str], references: list[str]
) -> dict[str, list[float]]:
    out = {"rouge1_f": [], "rouge2_f": [], "rougeL_f": []}
    for hyp, ref in zip(candidates, references):
        ht, rt = tokenize_fr(hyp), tokenize_fr(ref)
        r1 = _f1_overlap(ht, rt)
        r2 = _f1_overlap(_ngrams(ht, 2), _ngrams(rt, 2))
        lcs = _lcs_length(ht, rt)
        if ht and rt:
            p = lcs / len(ht)
            r = lcs / len(rt)
            rl = 2 * p * r / (p + r) if (p + r) else 0.0
        else:
            rl = 0.0
        out["rouge1_f"].append(round(r1, 4))
        out["rouge2_f"].append(round(r2, 4))
        out["rougeL_f"].append(round(rl, 4))
    return out


def score_bertscore(
    candidates: list[str],
    references: list[str],
    model_type: str = "bert-base-multilingual-cased",
    batch_size: int = 8,
) -> dict[str, list[float]]:
    """BERTScore P/R/F1 (embeddings contextuels)."""
    _require("bert_score", "bert-score")
    from bert_score import score as bert_score_fn

    P, R, F1 = bert_score_fn(
        candidates,
        references,
        model_type=model_type,
        lang="fr",
        verbose=True,
        batch_size=batch_size,
        rescale_with_baseline=False,
    )
    return {
        "bertscore_p": [round(float(x), 4) for x in P],
        "bertscore_r": [round(float(x), 4) for x in R],
        "bertscore_f1": [round(float(x), 4) for x in F1],
    }


def build_demo_pairs() -> pd.DataFrame:
    """Paires synthétiques pour smoke test (sans fichier gold)."""
    rows = [
        {
            "clip_id": "demo1",
            "template_id": "A1",
            "contenu": (
                "Un algorithme de tri réorganise une liste. "
                "Le tri à bulles compare les voisins et échange si besoin. "
                "Imaginez ranger des livres du plus petit au plus grand."
            ),
            "contenu_reference": (
                "Un algorithme de tri réorganise les éléments d'une liste. "
                "Le tri à bulles compare les éléments voisins et les échange. "
                "Pensez à classer des livres par taille croissante."
            ),
        },
        {
            "clip_id": "demo2",
            "template_id": "D1",
            "contenu": (
                "Ce concept est difficile pour beaucoup d'étudiants. "
                "Vous avez déjà progressé ; continuez avec de petits exercices."
            ),
            "contenu_reference": (
                "Beaucoup d'apprenants trouvent ce passage complexe. "
                "Rappelez-vous vos acquis et avancez par micro-objectifs."
            ),
        },
        {
            "clip_id": "demo3",
            "template_id": "B1",
            "contenu": "Les piles LIFO servent à l'historique du navigateur.",
            "contenu_reference": (
                "Une pile (LIFO) stocke le dernier élément ajouté en premier sorti ; "
                "exemple : bouton retour d'un navigateur web."
            ),
        },
    ]
    return pd.DataFrame(rows)


def align_pairs(corpus: pd.DataFrame, references: pd.DataFrame) -> pd.DataFrame:
    cand_col = _pick_col(corpus, CANDIDATE_COLS, "candidat")
    ref_col = _pick_col(references, REFERENCE_COLS, "référence")
    keys = _join_keys(corpus, references)

    left = corpus.copy()
    right = references.copy()
    left[cand_col] = left[cand_col].fillna("").astype(str).str.strip()
    right[ref_col] = right[ref_col].fillna("").astype(str).str.strip()

    # Dédupliquer les refs sur les clés (garder la première)
    right = right.drop_duplicates(subset=keys, keep="first")

    merged = left.merge(
        right[keys + [ref_col]],
        on=keys,
        how="inner",
        suffixes=("", "_ref"),
    )
    merged = merged.rename(columns={cand_col: "candidate", ref_col: "reference"})
    merged = merged[
        (merged["candidate"].str.len() > 0) & (merged["reference"].str.len() > 0)
    ]
    if merged.empty:
        raise SystemExit(
            "Aucune paire candidat/référence après jointure. "
            f"Clés utilisées : {keys}. Vérifiez que les identifiants correspondent."
        )
    return merged.reset_index(drop=True)


def aggregate(detail: pd.DataFrame, metric_cols: list[str]) -> dict:
    summary: dict = {
        "n": len(detail),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "note": (
            "Similarité lexicale/sémantique vs référence gold. "
            "Ne mesure PAS la qualité pédagogique (réservée à L2 humaine)."
        ),
        "global": {},
        "par_template": {},
        "par_type": {},
    }
    for col in metric_cols:
        if col in detail.columns:
            summary["global"][col] = {
                "mean": round(float(detail[col].mean()), 4),
                "std": round(float(detail[col].std(ddof=0)), 4),
                "min": round(float(detail[col].min()), 4),
                "max": round(float(detail[col].max()), 4),
            }

    if "template_id" in detail.columns:
        for tid, g in detail.groupby("template_id"):
            summary["par_template"][str(tid)] = {
                col: round(float(g[col].mean()), 4)
                for col in metric_cols
                if col in g.columns
            }

    if "template_id" in detail.columns:
        detail = detail.copy()
        detail["type_contenu"] = detail["template_id"].astype(str).str[0]
        for typ, g in detail.groupby("type_contenu"):
            summary["par_type"][str(typ)] = {
                col: round(float(g[col].mean()), 4)
                for col in metric_cols
                if col in g.columns
            }
    return summary


def rapport_memoire(summary: dict) -> str:
    g = summary.get("global", {})
    kind = summary.get("reference_kind", "unknown")
    lines = [
        "## Métriques de similarité LLM (BLEU / ROUGE / BERTScore)",
        "",
        f"- N = {summary['n']} paires candidat / référence",
        f"- Type de référence : `{kind}`",
    ]
    if kind == "auto_median_template":
        lines.append(
            "- Mode auto : pour chaque template, la référence est le contenu "
            "de longueur médiane du corpus (cohérence interne, pas gold expert)."
        )
    mapping = [
        ("bleu", "BLEU-4"),
        ("rouge1_f", "ROUGE-1 F1"),
        ("rouge2_f", "ROUGE-2 F1"),
        ("rougeL_f", "ROUGE-L F1"),
        ("bertscore_f1", "BERTScore F1"),
    ]
    for key, label in mapping:
        if key in g:
            lines.append(f"- {label} moyen : {g[key]['mean']} (σ={g[key]['std']})")
    lines += [
        "",
        "> Ces scores mesurent la **proximité à une référence**, pas la pertinence "
        "pédagogique (validation experte L2). BERTScore tolère mieux les paraphrases "
        "que BLEU/ROUGE.",
        "",
    ]
    return "\n".join(lines)


def compute_metrics(
    pairs: pd.DataFrame,
    *,
    skip_bertscore: bool = False,
    bert_model: str = "bert-base-multilingual-cased",
    batch_size: int = 8,
) -> tuple[pd.DataFrame, list[str]]:
    candidates = pairs["candidate"].tolist()
    references = pairs["reference"].tolist()
    detail = pairs.copy()
    metric_cols: list[str] = []

    print(f"Calcul BLEU sur {len(candidates)} paires…")
    detail["bleu"] = score_bleu(candidates, references)
    metric_cols.append("bleu")

    print("Calcul ROUGE-1/2/L…")
    rouge = score_rouge(candidates, references)
    for k, vals in rouge.items():
        detail[k] = vals
        metric_cols.append(k)

    if not skip_bertscore:
        print(f"Calcul BERTScore ({bert_model})…")
        bert = score_bertscore(
            candidates, references, model_type=bert_model, batch_size=batch_size
        )
        for k, vals in bert.items():
            detail[k] = vals
            metric_cols.append(k)
    else:
        print("BERTScore ignoré (--skip-bertscore).")

    return detail, metric_cols


def export_scaffold_from_corpus(corpus_path: str, out_path: str, limit: int | None = None):
    """Génère un CSV de références à remplir (colonne contenu_reference vide)."""
    df = pd.read_csv(corpus_path)
    cols = [c for c in ("clip_id", "template_id", "emotion", "contenu") if c in df.columns]
    out = df[cols].copy()
    if limit:
        out = out.head(limit)
    out = out.rename(columns={"contenu": "contenu_candidat_apercu"})
    out["contenu_reference"] = ""
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    out.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"OK Scaffold références : {out_path} ({len(out)} lignes à compléter)")


def build_pairs_from_corpus(corpus: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Construit des paires candidat/référence à partir du corpus réel.

    Pour chaque template_id, on choisit comme référence le contenu dont la
    longueur (nb de mots) est la plus proche de la médiane du template.
    Les autres contenus du même template sont les candidats.

    Ce n'est PAS une référence experte (gold) : c'est une mesure de
    cohérence interne du corpus (variabilité des générations d'un même template).
    """
    if "contenu" not in corpus.columns or "template_id" not in corpus.columns:
        raise SystemExit(
            "Le corpus doit contenir les colonnes 'contenu' et 'template_id'."
        )

    df = corpus.copy()
    df["contenu"] = df["contenu"].fillna("").astype(str).str.strip()
    df = df[df["contenu"].str.len() > 0].copy()
    df["template_id"] = df["template_id"].astype(str)
    df["_nb_mots"] = df["contenu"].str.split().str.len()

    ref_rows = []
    pair_rows = []

    for tid, g in df.groupby("template_id", sort=True):
        g = g.reset_index(drop=True)
        if len(g) < 2:
            # Un seul exemple : on ne peut pas former de paire utile
            continue
        median_len = float(g["_nb_mots"].median())
        # Index du prototype (longueur la plus proche de la médiane)
        ref_idx = (g["_nb_mots"] - median_len).abs().idxmin()
        ref_row = g.loc[ref_idx]
        ref_text = ref_row["contenu"]

        ref_rows.append(
            {
                "template_id": tid,
                "clip_id_reference": ref_row.get("clip_id"),
                "nb_mots_reference": int(ref_row["_nb_mots"]),
                "contenu_reference": ref_text,
            }
        )

        for i, row in g.iterrows():
            if i == ref_idx:
                continue
            pair_rows.append(
                {
                    "clip_id": row.get("clip_id"),
                    "template_id": tid,
                    "emotion": row.get("emotion"),
                    "pattern": row.get("pattern"),
                    "candidate": row["contenu"],
                    "reference": ref_text,
                    "clip_id_reference": ref_row.get("clip_id"),
                    "mode_reference": "auto_median_template",
                }
            )

    if not pair_rows:
        raise SystemExit(
            "Impossible de construire des paires : chaque template doit avoir "
            "au moins 2 contenus dans le corpus."
        )

    pairs = pd.DataFrame(pair_rows)
    refs_table = pd.DataFrame(ref_rows)
    return pairs, refs_table


def main():
    parser = argparse.ArgumentParser(
        description="BLEU / ROUGE / BERTScore sur corpus LLM (corpus_200 par défaut)"
    )
    parser.add_argument(
        "--corpus",
        default=os.path.join(DATA, "corpus_200.csv"),
        help="CSV des contenus générés (défaut: data/corpus_200.csv)",
    )
    parser.add_argument(
        "--mode",
        choices=("auto", "gold", "demo"),
        default="auto",
        help=(
            "auto = références construites depuis le corpus (défaut) ; "
            "gold = fichier experts ; demo = 3 paires synthétiques"
        ),
    )
    parser.add_argument(
        "--references",
        default=os.path.join(DATA, "references_contenu.csv"),
        help="CSV des références gold (uniquement avec --mode gold)",
    )
    parser.add_argument(
        "--out",
        default=os.path.join(RESULTS, "llm_similarity_metrics.json"),
        help="Résumé JSON",
    )
    parser.add_argument(
        "--export-md",
        default=os.path.join(RESULTS, "llm_similarity_memoire.md"),
        help="Export markdown pour le mémoire",
    )
    parser.add_argument(
        "--save-auto-refs",
        default=os.path.join(DATA, "references_contenu_auto.csv"),
        help="Où sauver les références auto choisies (mode auto)",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Alias de --mode demo",
    )
    parser.add_argument(
        "--scaffold",
        action="store_true",
        help="Génère un CSV de références vide à partir du corpus, puis quitte",
    )
    parser.add_argument(
        "--scaffold-out",
        default=os.path.join(DATA, "references_contenu.csv"),
        help="Chemin du scaffold",
    )
    parser.add_argument("--scaffold-limit", type=int, default=None)
    parser.add_argument(
        "--skip-bertscore",
        action="store_true",
        help="Ne calcule que BLEU + ROUGE (plus rapide)",
    )
    parser.add_argument(
        "--bert-model",
        default="bert-base-multilingual-cased",
        help="Modèle HuggingFace pour BERTScore",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None, help="Limiter le nb de paires")
    args = parser.parse_args()

    if args.demo:
        args.mode = "demo"

    if args.scaffold:
        export_scaffold_from_corpus(args.corpus, args.scaffold_out, args.scaffold_limit)
        return

    refs_table = None
    if args.mode == "demo":
        pairs = build_demo_pairs()
        pairs = pairs.rename(
            columns={"contenu": "candidate", "contenu_reference": "reference"}
        )
        print("Mode demo : 3 paires synthetiques")
    elif args.mode == "gold":
        if not os.path.isfile(args.references):
            raise SystemExit(
                f"Fichier references introuvable : {args.references}\n"
                "Creez-le avec :\n"
                f"  python scripts/metrics_llm_similarity.py --scaffold\n"
                "puis remplissez contenu_reference.\n"
                "Sinon utilisez le mode auto (defaut) :\n"
                "  python scripts/metrics_llm_similarity.py"
            )
        corpus = pd.read_csv(args.corpus)
        refs = pd.read_csv(args.references)
        # Vérifier que les références ne sont pas vides
        ref_col = _pick_col(refs, REFERENCE_COLS, "référence")
        n_filled = refs[ref_col].fillna("").astype(str).str.strip().ne("").sum()
        if n_filled == 0:
            raise SystemExit(
                f"Le fichier {args.references} n'a aucune reference remplie.\n"
                "Utilisez le mode auto sur le corpus :\n"
                "  python scripts/metrics_llm_similarity.py"
            )
        pairs = align_pairs(corpus, refs)
        keep = [
            c
            for c in ("clip_id", "template_id", "emotion", "pattern")
            if c in pairs.columns
        ]
        pairs = pairs[keep + ["candidate", "reference"]]
        pairs["mode_reference"] = "gold"
        print(f"Mode gold : {len(pairs)} paires (corpus={args.corpus})")
    else:
        # mode auto — corpus réel
        if not os.path.isfile(args.corpus):
            raise SystemExit(f"Corpus introuvable : {args.corpus}")
        corpus = pd.read_csv(args.corpus)
        pairs, refs_table = build_pairs_from_corpus(corpus)
        os.makedirs(os.path.dirname(args.save_auto_refs) or DATA, exist_ok=True)
        refs_table.to_csv(args.save_auto_refs, index=False, encoding="utf-8-sig")
        print(
            f"Mode auto : corpus={args.corpus} | "
            f"{len(pairs)} candidats vs {len(refs_table)} references (1 par template)"
        )
        print(f"References auto sauvees : {args.save_auto_refs}")

    if args.limit:
        pairs = pairs.head(args.limit)

    detail, metric_cols = compute_metrics(
        pairs,
        skip_bertscore=args.skip_bertscore,
        bert_model=args.bert_model,
        batch_size=args.batch_size,
    )
    summary = aggregate(detail, metric_cols)

    # Clarifier dans le résumé / mémoire
    if args.mode == "auto":
        summary["note"] = (
            "Mode AUTO : référence = contenu médian par template_id dans le corpus. "
            "Mesure la cohérence interne des générations, PAS une qualité vs expert. "
            "La pertinence pédagogique reste à la validation L2."
        )
        summary["reference_kind"] = "auto_median_template"
    elif args.mode == "gold":
        summary["reference_kind"] = "expert_gold"
    else:
        summary["reference_kind"] = "demo"

    summary["config"] = {
        "mode": args.mode,
        "skip_bertscore": args.skip_bertscore,
        "bert_model": None if args.skip_bertscore else args.bert_model,
        "corpus": args.corpus if args.mode != "demo" else "demo",
        "references": (
            args.references
            if args.mode == "gold"
            else (args.save_auto_refs if args.mode == "auto" else "demo")
        ),
        "n_pairs": len(detail),
        "n_templates_ref": int(refs_table["template_id"].nunique())
        if refs_table is not None
        else None,
    }

    os.makedirs(os.path.dirname(args.out) or RESULTS, exist_ok=True)
    detail_csv = args.out.replace(".json", ".csv")
    detail.to_csv(detail_csv, index=False, encoding="utf-8-sig")
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    md = rapport_memoire(summary)
    with open(args.export_md, "w", encoding="utf-8") as f:
        f.write(md)

    print(json.dumps(summary["global"], indent=2, ensure_ascii=False))
    print(f"\nOK Detail : {detail_csv}")
    print(f"OK Resume : {args.out}")
    print(f"OK Markdown memoire : {args.export_md}")


if __name__ == "__main__":
    main()