#!/usr/bin/env python3
# ============================================
# Évaluation FER sur labels / frames DAiSEE (OS2 / QR1)
# ============================================
"""
Usage:
  python scripts/eval_fer_daisee.py labels --split train --sample 2000
  python scripts/eval_fer_daisee.py fer --frames-root "C:/path/to/DAiSEE/Data" --sample 100
  python scripts/eval_fer_daisee.py scenarios

Produit: results/fer_metrics.json (+ results/fer_confusion.png si matplotlib dispo)
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone

import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
BACKEND = os.path.join(ROOT, "backend")
sys.path.insert(0, BACKEND)

from daisee_utils import charger_labels_daisee, etat_depuis_ligne, normaliser_clip_id
from fer_mapping import mapping_fer_daisee

RESULTS_DIR = os.path.join(ROOT, "results")
ETATS = ["Boredom", "Confusion", "Engagement", "Frustration"]


def _ensure_results_dir():
    os.makedirs(RESULTS_DIR, exist_ok=True)


def _metrics_classification(y_true, y_pred, labels):
    """Accuracy, precision/recall/F1 macro, matrice de confusion."""
    idx = {l: i for i, l in enumerate(labels)}
    n = len(labels)
    matrix = [[0] * n for _ in range(n)]

    correct = 0
    for t, p in zip(y_true, y_pred):
        if t in idx and p in idx:
            matrix[idx[t]][idx[p]] += 1
            if t == p:
                correct += 1

    total = len(y_true)
    accuracy = correct / total if total else 0.0

    precisions, recalls, f1s = [], [], []
    for i, lab in enumerate(labels):
        tp = matrix[i][i]
        fp = sum(matrix[r][i] for r in range(n)) - tp
        fn = sum(matrix[i][c] for c in range(n)) - tp
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        precisions.append(prec)
        recalls.append(rec)
        f1s.append(f1)

    return {
        "accuracy": round(accuracy, 4),
        "precision_macro": round(sum(precisions) / n, 4) if n else 0,
        "recall_macro": round(sum(recalls) / n, 4) if n else 0,
        "f1_macro": round(sum(f1s) / n, 4) if n else 0,
        "confusion_matrix": {
            "labels": labels,
            "counts": matrix,
        },
        "per_class": {
            labels[i]: {
                "precision": round(precisions[i], 4),
                "recall": round(recalls[i], 4),
                "f1": round(f1s[i], 4),
            }
            for i in range(n)
        },
        "n_samples": total,
    }


def cmd_labels_analysis(split: str, sample: int | None, seed: int):
    """Analyse descriptive des labels DAiSEE (baseline sans FER)."""
    df = charger_labels_daisee(split=split)
    if sample and sample < len(df):
        df = df.sample(n=sample, random_state=seed)

    etats = []
    for _, row in df.iterrows():
        etat, _, _ = etat_depuis_ligne(row)
        etats.append(etat)

    dist = Counter(etats)
    co_occur = defaultdict(int)
    for _, row in df.iterrows():
        if int(row["Confusion"]) >= 2 and int(row["Frustration"]) >= 2:
            co_occur["confusion+frustration"] += 1
        if int(row["Engagement"]) == 0:
            co_occur["engagement_zero"] += 1
        if int(row["Boredom"]) >= 2:
            co_occur["boredom_eleve"] += 1

    report = {
        "mode": "labels_analysis",
        "split": split,
        "n_clips": len(df),
        "distribution_etat_dominant": dict(dist),
        "co_occurrences": dict(co_occur),
        "mapping_fer_vers_daisee": mapping_fer_daisee,
        "note": (
            "Analyse des ground truth DAiSEE. Pour métriques FER réelles, "
            "fournir --frames-root avec frames extraites (ffmpeg)."
        ),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    return report


def _trouver_frame(frames_root: str, clip_id: str) -> str | None:
    """Cherche une image jpg/png pour un ClipID DAiSEE."""
    base = os.path.splitext(normaliser_clip_id(clip_id))[0]
    extensions = (".jpg", ".jpeg", ".png")

    for dirpath, _, filenames in os.walk(frames_root):
        for fname in filenames:
            low = fname.lower()
            if not low.endswith(extensions):
                continue
            stem = os.path.splitext(fname)[0]
            if stem == base or stem.startswith(base) or base in stem:
                return os.path.join(dirpath, fname)
    return None


def cmd_fer_evaluation(frames_root: str, sample: int, seed: int, labels_split: str):
    """Compare prédictions FER+mapping vs état dominant DAiSEE."""
    from fer_module import analyser_visage

    if not os.path.isdir(frames_root):
        raise FileNotFoundError(f"Dossier introuvable: {frames_root}")

    df = charger_labels_daisee(split=labels_split)
    random.seed(seed)
    rows = df.sample(n=min(sample, len(df)), random_state=seed).to_dict("records")

    y_true, y_pred = [], []
    latencies = []
    skipped = []

    for row in rows:
        clip_id = normaliser_clip_id(row["ClipID"])
        frame_path = _trouver_frame(frames_root, clip_id)
        if not frame_path:
            skipped.append(clip_id)
            continue

        etat_gt, _, _ = etat_depuis_ligne(pd.Series(row))

        t0 = time.perf_counter()
        result = analyser_visage(frame_path)
        latencies.append(time.perf_counter() - t0)

        if result.get("erreur"):
            skipped.append(clip_id)
            continue

        y_true.append(etat_gt)
        y_pred.append(result["etat"])

    if not y_true:
        return {
            "mode": "fer_evaluation",
            "error": "Aucune frame trouvée pour les clips échantillonnés.",
            "frames_root": frames_root,
            "skipped": skipped[:20],
            "hint": (
                "Structure attendue: .../Data/Training/user/video/CLIPID1.jpg "
                "(voir data/DAiSEE/extractFrames.py)"
            ),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    metrics = _metrics_classification(y_true, y_pred, ETATS)
    metrics["mode"] = "fer_evaluation"
    metrics["frames_root"] = frames_root
    metrics["labels_split"] = labels_split
    metrics["n_skipped"] = len(skipped)
    metrics["latency_mean_sec"] = round(sum(latencies) / len(latencies), 4)
    metrics["latency_p95_sec"] = round(sorted(latencies)[int(0.95 * len(latencies)) - 1], 4)
    metrics["model"] = "dima806/facial_emotions_image_detection"
    metrics["mapping"] = mapping_fer_daisee
    metrics["timestamp"] = datetime.now(timezone.utc).isoformat()

    _maybe_plot_confusion(metrics, os.path.join(RESULTS_DIR, "fer_confusion.png"))
    return metrics


def cmd_scenarios_benchmark():
    """Évalue la cohérence décision sur les 50 scénarios (fusion + templates)."""
    from fusion_module import fusionner_modalites
    from fer_module import analyser_visage_depuis_scenario  # noqa: F401 — charge FER lazy
    from ser_module import analyser_voix_depuis_scenario
    from decision_engine import preparer_decision

    path = os.path.join(ROOT, "data", "scenarios_50.csv")
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]

    rows_out = []
    for i, scenario in df.iterrows():
        visage = analyser_visage_depuis_scenario(scenario)
        voix = analyser_voix_depuis_scenario(scenario_id=i)
        fusion = fusionner_modalites(visage, voix)
        decision = preparer_decision(
            fusion=fusion,
            boredom=int(scenario["Boredom"]),
            confusion=int(scenario["Confusion"]),
            engagement=int(scenario["Engagement"]),
            frustration=int(scenario["Frustration"]),
            pattern=str(scenario.get("pattern", "stable")),
        )
        rows_out.append(
            {
                "scenario_id": i,
                "clip_id": scenario["ClipID"],
                "pattern": scenario.get("pattern"),
                "etat_fusion": decision["etat"],
                "template_id": decision["template_id"],
                "type_contenu": decision["type_contenu"],
            }
        )

    out_path = os.path.join(RESULTS_DIR, "scenarios_decisions.csv")
    pd.DataFrame(rows_out).to_csv(out_path, index=False)

    template_dist = Counter(r["template_id"] for r in rows_out)
    return {
        "mode": "scenarios_benchmark",
        "n_scenarios": len(rows_out),
        "template_distribution": dict(template_dist),
        "output_csv": out_path,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def _maybe_plot_confusion(metrics: dict, out_path: str):
    try:
        import matplotlib.pyplot as plt
        import numpy as np

        labels = metrics["confusion_matrix"]["labels"]
        counts = np.array(metrics["confusion_matrix"]["counts"])

        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(counts, cmap="Blues")
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_yticklabels(labels)
        ax.set_xlabel("Prédit (FER → DAiSEE)")
        ax.set_ylabel("Vérité terrain (DAiSEE)")
        ax.set_title(
            f"Matrice de confusion FER (acc={metrics.get('accuracy', 0):.2%})"
        )

        for i in range(len(labels)):
            for j in range(len(labels)):
                ax.text(j, i, counts[i, j], ha="center", va="center", color="black")

        plt.colorbar(im)
        plt.tight_layout()
        plt.savefig(out_path, dpi=120)
        plt.close()
        metrics["confusion_plot"] = out_path
    except ImportError:
        metrics["confusion_plot"] = None


def main():
    parser = argparse.ArgumentParser(description="Évaluation FER / DAiSEE — EmoLearn")
    sub = parser.add_subparsers(dest="command", required=True)

    p_labels = sub.add_parser("labels", help="Analyse distribution labels DAiSEE")
    p_labels.add_argument("--split", default="train", choices=["train", "validation", "test", "all"])
    p_labels.add_argument("--sample", type=int, default=None)
    p_labels.add_argument("--seed", type=int, default=42)

    p_fer = sub.add_parser("fer", help="Évaluation FER sur frames extraites")
    p_fer.add_argument("--frames-root", required=True, help="Racine des frames DAiSEE (dossier Data)")
    p_fer.add_argument("--sample", type=int, default=100)
    p_fer.add_argument("--seed", type=int, default=42)
    p_fer.add_argument("--labels-split", default="train")

    sub.add_parser("scenarios", help="Benchmark décisions sur scenarios_50.csv")

    args = parser.parse_args()
    _ensure_results_dir()

    if args.command == "labels":
        report = cmd_labels_analysis(args.split, args.sample, args.seed)
        out = os.path.join(RESULTS_DIR, "fer_labels_analysis.json")
    elif args.command == "fer":
        report = cmd_fer_evaluation(
            args.frames_root, args.sample, args.seed, args.labels_split
        )
        out = os.path.join(RESULTS_DIR, "fer_metrics.json")
    else:
        report = cmd_scenarios_benchmark()
        out = os.path.join(RESULTS_DIR, "scenarios_benchmark.json")

    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\nRapport sauvegardé: {out}")


if __name__ == "__main__":
    main()
