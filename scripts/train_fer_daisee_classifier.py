#!/usr/bin/env python3
# ============================================
# Entraînement classifieur sklearn : FER → DAiSEE
# ============================================
"""
Usage:
  python scripts/train_fer_daisee_classifier.py --from-cache --classifier rf
  python scripts/train_fer_daisee_classifier.py --sample 800 --stratified --classifier rf
  python scripts/train_fer_daisee_classifier.py --from-cache --oversample --classifier rf

Produit:
  models/fer_daisee_clf.joblib
  results/fer_classifier_metrics.json
  results/fer_features_cache.csv (cache des features extraites)

Étape 1 (rééquilibrage) :
  --stratified  : échantillonne un quota par état DAiSEE (évite 75% Engagement)
  --oversample  : duplique les classes rares sur le TRAIN uniquement

Note: l'algo reste Random Forest (ou logreg) ; GridSearchCV choisit ses hyperparamètres.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from datetime import datetime, timezone

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
BACKEND = os.path.join(ROOT, "backend")
sys.path.insert(0, BACKEND)

from daisee_utils import charger_labels_daisee, etat_depuis_ligne, normaliser_clip_id
from daisee_video import extract_face_frames, resolve_daisee_video_path
from fer_classifier import (
    DEFAULT_AU_MODEL_PATH,
    DEFAULT_MODEL_PATH,
    ETATS_DAISEE,
    FER_FEATURE_ORDER,
    mapping_fer_daisee,
    scores_to_vector,
)
from au_module import AU_FEATURE_ORDER, au_to_vector, extraire_au

RESULTS_DIR = os.path.join(ROOT, "results")
MODELS_DIR = os.path.join(ROOT, "models")
DATASET_ROOT = os.path.join(ROOT, "data", "DAiSEE", "DataSet")


def resolve_daisee_video(clip_id: str, dataset_root: str = DATASET_ROOT) -> str | None:
    """Résolution rapide via l'index partagé (backend/daisee_video.py)."""
    path = resolve_daisee_video_path(normaliser_clip_id(clip_id))
    return path or None


def extraire_frame_visage(clip_id: str, video_path: str):
    """
    Retourne une image PIL RGB (visage détecté, frame du milieu) ou None.
    Utilise la MÊME extraction (cascades OpenCV) que fer_module.py en
    production, pour que le classifieur soit entraîné sur une distribution
    cohérente avec l'inférence réelle.
    """
    images, _meta = extract_face_frames(clip_id, n_frames=1, avi_path=video_path)
    return images[0] if images else None


def _annoter_etat_gt(df: pd.DataFrame) -> pd.DataFrame:
    """Ajoute la colonne etat_gt (état dominant DAiSEE) à un DataFrame labels."""
    etats = []
    for _, row in df.iterrows():
        etat, _, _ = etat_depuis_ligne(row)
        etats.append(etat)
    out = df.copy()
    out["etat_gt"] = etats
    return out


def echantillonner_labels(
    df: pd.DataFrame,
    sample: int,
    seed: int,
    stratified: bool,
) -> pd.DataFrame:
    """
    Tirage de clips pour l'extraction FER.
    - stratified=False : tirage aléatoire global (comportement historique)
    - stratified=True  : quota ~égal par état dominant (réduit le biais Engagement)
    """
    if len(df) == 0:
        return df
    if not stratified:
        return df.sample(n=min(sample, len(df)), random_state=seed).reset_index(drop=True)

    labeled = _annoter_etat_gt(df)
    classes = [c for c in ETATS_DAISEE if c in set(labeled["etat_gt"])]
    if not classes:
        return labeled.sample(n=min(sample, len(labeled)), random_state=seed).reset_index(
            drop=True
        )

    per_class = max(1, sample // len(classes))
    rng = np.random.RandomState(seed)
    parts = []
    for etat in classes:
        pool = labeled[labeled["etat_gt"] == etat]
        n = min(per_class, len(pool))
        if n == 0:
            continue
        idx = rng.choice(pool.index.to_numpy(), size=n, replace=False)
        parts.append(pool.loc[idx])
        print(f"  stratifié {etat}: {n}/{len(pool)}")

    if not parts:
        return labeled.head(0)

    out = pd.concat(parts, ignore_index=True)
    # Si le total < sample (classes rares épuisées), compléter avec le reste
    if len(out) < sample:
        pris = set(out["ClipID"].astype(str))
        reste = labeled[~labeled["ClipID"].astype(str).isin(pris)]
        manque = min(sample - len(out), len(reste))
        if manque > 0:
            out = pd.concat(
                [out, reste.sample(n=manque, random_state=seed)],
                ignore_index=True,
            )
            print(f"  stratifié complément: +{manque} (total={len(out)})")
    return out.reset_index(drop=True)


def oversample_train(X, y, seed: int):
    """
    Oversampling aléatoire des classes minoritaires jusqu'à la taille
    de la classe majoritaire (TRAIN uniquement — pas de fuite test).
    Retourne (X_res, y_res, stats).
    """
    rng = np.random.RandomState(seed)
    y = np.asarray(y)
    X = np.asarray(X)
    counts = Counter(y.tolist())
    if len(counts) < 2:
        return X, y, {"enabled": False, "reason": "une seule classe"}

    target = max(counts.values())
    X_parts = [X]
    y_parts = [y]
    added = {}
    for etat, n in counts.items():
        manque = target - n
        if manque <= 0:
            added[etat] = 0
            continue
        idx = np.where(y == etat)[0]
        choix = rng.choice(idx, size=manque, replace=True)
        X_parts.append(X[choix])
        y_parts.append(y[choix])
        added[etat] = int(manque)

    X_res = np.concatenate(X_parts, axis=0)
    y_res = np.concatenate(y_parts, axis=0)
    # Mélange pour éviter un bloc de classes à la fin
    perm = rng.permutation(len(y_res))
    stats = {
        "enabled": True,
        "avant": dict(counts),
        "ajoutes": added,
        "apres": dict(Counter(y_res[perm].tolist())),
        "n_train_avant": int(len(y)),
        "n_train_apres": int(len(y_res)),
    }
    return X_res[perm], y_res[perm], stats


def extraire_features(
    sample: int,
    seed: int,
    labels_split: str,
    dataset_root: str,
    cache_csv: str | None,
    stratified: bool = False,
    with_au: bool = False,
):
    # Features FER+AU : toujours les 7 émotions pretrained (pas le ViT DAiSEE)
    if with_au:
        from fer_offline import analyser_visage_image_pretrained as analyser_fn
    else:
        from fer_offline import analyser_visage_image as analyser_fn

    df = charger_labels_daisee(split=labels_split)
    print(
        f"Échantillonnage labels (sample={sample}, stratified={stratified}, au={with_au})…"
    )
    df = echantillonner_labels(df, sample=sample, seed=seed, stratified=stratified)

    rows = []
    rejets = []
    for i, row in df.iterrows():
        clip_id = normaliser_clip_id(row["ClipID"])
        etat_gt, niveau_gt, _ = etat_depuis_ligne(row)
        video = resolve_daisee_video_path(clip_id)
        if not video:
            rejets.append({"clip_id": clip_id, "code": "video_absente", "detail": ""})
            continue
        images, meta = extract_face_frames(clip_id, n_frames=1, avi_path=video)
        if not images:
            rejets.append(
                {
                    "clip_id": clip_id,
                    "code": meta.get("erreur") or "aucune_image",
                    "detail": "",
                }
            )
            continue
        image = images[0]
        fer = analyser_fn(image)
        if fer.get("erreur") or not fer.get("tous_scores"):
            rejets.append(
                {
                    "clip_id": clip_id,
                    "code": "fer_echec",
                    "detail": str(fer.get("erreur") or "scores_vides"),
                }
            )
            continue
        vec = scores_to_vector(fer["tous_scores"])
        heur = mapping_fer_daisee.get(
            fer["emotion"], {"etat": "Engagement", "niveau": 1}
        )
        rec = {
            "clip_id": clip_id,
            "etat_gt": etat_gt,
            "niveau_gt": niveau_gt,
            "fer_emotion": fer["emotion"],
            "etat_heuristique": heur["etat"],
            **{f"fer_{lab}": float(vec[j]) for j, lab in enumerate(FER_FEATURE_ORDER)},
        }
        if with_au:
            au = extraire_au(image)
            au_vec = au_to_vector(au)
            for j, name in enumerate(AU_FEATURE_ORDER):
                rec[f"au_{name}"] = float(au_vec[j])
        rows.append(rec)
        if (len(rows) % 25) == 0:
            print(f"  features: {len(rows)} extraites ({len(rejets)} écartés)…")

    feats = pd.DataFrame(rows)
    if cache_csv:
        os.makedirs(os.path.dirname(cache_csv) or ".", exist_ok=True)
        rejets_csv = os.path.splitext(cache_csv)[0] + "_rejets.csv"
        pd.DataFrame(rejets, columns=["clip_id", "code", "detail"]).to_csv(
            rejets_csv, index=False
        )
        print(
            f"Écartés: {len(rejets)} ({dict(Counter(r['code'] for r in rejets))}) "
            f"→ {rejets_csv}"
        )
        if len(feats):
            feats.to_csv(cache_csv, index=False)
            print(f"Cache features → {cache_csv}")
    dist = dict(Counter(feats["etat_gt"].tolist())) if len(feats) else {}
    print(f"Features OK: {len(feats)} | écartés: {len(rejets)} | dist={dist}")
    return feats


def _metrics(y_true, y_pred, labels):
    from sklearn.metrics import (
        accuracy_score,
        classification_report,
        confusion_matrix,
        f1_score,
    )

    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "f1_macro": round(float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)), 4),
        "confusion_matrix": {
            "labels": list(labels),
            "counts": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        },
        "report": classification_report(
            y_true, y_pred, labels=labels, zero_division=0, output_dict=True
        ),
    }


def entrainer(
    feats: pd.DataFrame,
    classifier: str,
    seed: int,
    out_model: str,
    use_grid_search: bool = True,
    scoring: str = "f1_macro",
    oversample: bool = False,
    with_au: bool = False,
):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    import joblib

    feature_cols = [f"fer_{lab}" for lab in FER_FEATURE_ORDER]
    feature_names = list(FER_FEATURE_ORDER)
    if with_au:
        au_cols = [f"au_{name}" for name in AU_FEATURE_ORDER]
        missing = [c for c in au_cols if c not in feats.columns]
        if missing:
            raise SystemExit(
                f"Cache sans colonnes AU ({missing[:3]}…). "
                "Relancer sans --from-cache avec --with-au."
            )
        feature_cols = feature_cols + au_cols
        feature_names = feature_names + list(AU_FEATURE_ORDER)

    X = feats[feature_cols].to_numpy(dtype=np.float64)
    y = feats["etat_gt"].to_numpy()
    y_heur = feats["etat_heuristique"].to_numpy()

    X_train, X_test, y_train, y_test, heur_train, heur_test = train_test_split(
        X,
        y,
        y_heur,
        test_size=0.25,
        random_state=seed,
        stratify=y if len(set(y)) > 1 else None,
    )

    oversample_stats = {"enabled": False}
    if oversample:
        X_train, y_train, oversample_stats = oversample_train(X_train, y_train, seed)
        print(
            f"Oversampling TRAIN: {oversample_stats.get('n_train_avant')} → "
            f"{oversample_stats.get('n_train_apres')} | apres={oversample_stats.get('apres')}"
        )

    baseline = _metrics(y_test, heur_test, list(ETATS_DAISEE))

    best_params = None
    cv_best_score = None
    search_used = False

    rf_weight = "balanced_subsample" if oversample else "balanced"

    if classifier == "rf":
        pipe = Pipeline(
            [
                (
                    "clf",
                    RandomForestClassifier(
                        class_weight=rf_weight,
                        random_state=seed,
                        n_jobs=-1,
                    ),
                )
            ]
        )
        param_grid = {
            "clf__n_estimators": [100, 200, 300],
            "clf__max_depth": [4, 8, None],
            "clf__min_samples_leaf": [1, 2],
        }
    else:
        pipe = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=2000,
                        class_weight="balanced",
                        random_state=seed,
                    ),
                ),
            ]
        )
        param_grid = {
            "clf__C": [0.1, 1.0, 10.0],
        }

    class_counts = Counter(y_train.tolist())
    n_splits = min(3, min(class_counts.values())) if class_counts else 0

    if use_grid_search and n_splits >= 2:
        print(
            f"GridSearchCV sur {classifier} "
            f"(cv={n_splits}, grille={param_grid}, au={with_au})"
        )
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        search = GridSearchCV(
            pipe,
            param_grid=param_grid,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
            refit=True,
        )
        search.fit(X_train, y_train)
        pipe = search.best_estimator_
        best_params = search.best_params_
        cv_best_score = float(search.best_score_)
        search_used = True
        print(f"Meilleurs paramètres: {best_params} (cv_score={cv_best_score:.4f})")
    else:
        if classifier == "rf":
            pipe.set_params(clf__n_estimators=200, clf__max_depth=8)
            best_params = {"clf__n_estimators": 200, "clf__max_depth": 8}
        else:
            pipe.set_params(clf__C=1.0)
            best_params = {"clf__C": 1.0}
        reason = "désactivé" if not use_grid_search else "trop peu d'exemples/classe"
        print(f"GridSearch non utilisé ({reason}) → params fixes {best_params}")
        pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)
    learned = _metrics(y_test, y_pred, list(ETATS_DAISEE))
    gain = round(learned["accuracy"] - baseline["accuracy"], 4)
    gain_f1_macro = round(learned["f1_macro"] - baseline["f1_macro"], 4)
    beats_baseline = bool(gain > 0 and gain_f1_macro > 0)

    os.makedirs(os.path.dirname(out_model) or ".", exist_ok=True)
    pack = {
        "model": pipe,
        "feature_names": feature_names,
        "include_au": with_au,
        "au_names": list(AU_FEATURE_ORDER) if with_au else [],
        "classes_": list(getattr(pipe.named_steps["clf"], "classes_", sorted(set(y)))),
        "classifier": classifier,
        "best_params": best_params,
        "grid_search": search_used,
        "cv_best_score": cv_best_score,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "test_accuracy": learned["accuracy"],
        "baseline_test_accuracy": baseline["accuracy"],
        "test_f1_macro": learned["f1_macro"],
        "baseline_test_f1_macro": baseline["f1_macro"],
        "beats_baseline": beats_baseline,
        "oversample": oversample_stats,
    }
    joblib.dump(pack, out_model)

    report = {
        "mode": "fer_daisee_classifier_au" if with_au else "fer_daisee_classifier",
        "with_au": with_au,
        "classifier": classifier,
        "grid_search": search_used,
        "best_params": best_params,
        "cv_best_score": cv_best_score,
        "n_samples": int(len(feats)),
        "distribution_gt": dict(Counter(y.tolist())),
        "oversample": oversample_stats,
        "baseline_heuristique_test": baseline,
        "learned_test": learned,
        "gain_accuracy": gain,
        "gain_f1_macro": gain_f1_macro,
        "beats_baseline": beats_baseline,
        "recommandation": "learned" if beats_baseline else "heuristique",
        "model_path": out_model,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "note": (
            "Algo = Random Forest (ou logreg). "
            + ("Features = FER 7 + AU MediaPipe. " if with_au else "")
            + "GridSearchCV + options --stratified / --oversample."
        ),
    }
    return report


def main():
    parser = argparse.ArgumentParser(description="Train FER→DAiSEE sklearn classifier")
    parser.add_argument(
        "--sample",
        type=int,
        default=800,
        help="Nb clips max (défaut 800 — mieux pour classes rares)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--labels-split",
        default="all",
        choices=["train", "validation", "test", "all"],
        help="Split labels DAiSEE (défaut all pour maximiser les rares)",
    )
    parser.add_argument("--dataset-root", default=DATASET_ROOT)
    parser.add_argument("--classifier", default="rf", choices=["logreg", "rf"])
    parser.add_argument("--features-csv", default=None)
    parser.add_argument("--from-cache", action="store_true", help="Réutilise le CSV de features sans ré-extraire")
    parser.add_argument(
        "--with-au",
        action="store_true",
        help="Ajoute Action Units (MediaPipe) aux features → models/fer_daisee_clf_au.joblib",
    )
    parser.add_argument(
        "--stratified",
        action="store_true",
        help="Échantillonnage stratifié par état DAiSEE (quota ~égal / classe)",
    )
    parser.add_argument(
        "--oversample",
        action="store_true",
        help="Oversampling des classes rares sur le TRAIN uniquement",
    )
    parser.add_argument(
        "--no-grid-search",
        action="store_true",
        help="Désactive GridSearchCV (params fixes n_estimators=200, max_depth=8)",
    )
    parser.add_argument(
        "--scoring",
        default="f1_macro",
        choices=["accuracy", "f1_macro"],
        help="Score GridSearch (f1_macro recommandé sur DAiSEE déséquilibré)",
    )
    parser.add_argument("--out-model", default=None)
    parser.add_argument("--out-metrics", default=None)
    args = parser.parse_args()

    if args.features_csv is None:
        args.features_csv = os.path.join(
            RESULTS_DIR,
            "fer_features_cache_au.csv" if args.with_au else "fer_features_cache.csv",
        )
    if args.out_model is None:
        args.out_model = DEFAULT_AU_MODEL_PATH if args.with_au else DEFAULT_MODEL_PATH
    if args.out_metrics is None:
        args.out_metrics = os.path.join(
            RESULTS_DIR,
            "fer_classifier_au_metrics.json"
            if args.with_au
            else "fer_classifier_metrics.json",
        )

    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

    if args.from_cache and os.path.isfile(args.features_csv):
        print(f"Chargement cache: {args.features_csv}")
        feats = pd.read_csv(args.features_csv)
        if args.stratified:
            print(
                "Note: --stratified ignore avec --from-cache "
                "(le cache a déjà été extrait). Relancer sans --from-cache "
                "pour un nouvel échantillonnage stratifié."
            )
    else:
        print("Extraction FER depuis vidéos DAiSEE…")
        feats = extraire_features(
            sample=args.sample,
            seed=args.seed,
            labels_split=args.labels_split,
            dataset_root=args.dataset_root,
            cache_csv=args.features_csv,
            stratified=args.stratified,
            with_au=args.with_au,
        )

    if len(feats) < 20:
        raise SystemExit(
            f"Trop peu d'échantillons ({len(feats)}). Vérifiez data/DAiSEE/DataSet/."
        )

    report = entrainer(
        feats,
        args.classifier,
        args.seed,
        args.out_model,
        use_grid_search=not args.no_grid_search,
        scoring=args.scoring,
        oversample=args.oversample,
        with_au=args.with_au,
    )
    report["grid_scoring"] = args.scoring
    report["sample_arg"] = args.sample
    report["labels_split"] = args.labels_split
    report["stratified"] = bool(args.stratified and not args.from_cache)
    report["oversample_flag"] = args.oversample
    with open(args.out_metrics, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    per_class = {
        etat: round(float(report["learned_test"]["report"].get(etat, {}).get("f1-score", 0)), 3)
        for etat in ETATS_DAISEE
    }
    summary = {
        "with_au": args.with_au,
        "baseline_acc": report["baseline_heuristique_test"]["accuracy"],
        "learned_acc": report["learned_test"]["accuracy"],
        "baseline_f1_macro": report["baseline_heuristique_test"]["f1_macro"],
        "learned_f1_macro": report["learned_test"]["f1_macro"],
        "gain_acc": report["gain_accuracy"],
        "gain_f1": report["gain_f1_macro"],
        "f1_par_classe": per_class,
        "oversample": report.get("oversample", {}).get("enabled"),
        "recommandation": report["recommandation"],
        "model": args.out_model,
        "metrics": args.out_metrics,
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
