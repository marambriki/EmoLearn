"""Tests des modules d'entraînement léger (sans GPU / sans données lourdes)."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from fer_classifier import (
    FER_FEATURE_ORDER,
    appliquer_mapping,
    scores_to_vector,
)


def test_scores_to_vector_ordre_fixe():
    vec = scores_to_vector({"happy": 0.7, "neutral": 0.2})
    assert vec.shape == (len(FER_FEATURE_ORDER),)
    assert abs(vec[FER_FEATURE_ORDER.index("happy")] - 0.7) < 1e-9
    assert abs(vec[FER_FEATURE_ORDER.index("angry")]) < 1e-9


def test_appliquer_mapping_heuristique(monkeypatch):
    monkeypatch.setenv("EMOLEARN_FER_MAPPING", "heuristique")
    # Recharge le mode
    import fer_classifier as fc

    fc._FER_MAPPING_MODE = "heuristique"
    etat, niveau, source, probs = fc.appliquer_mapping("surprise", {"surprise": 0.9})
    assert etat == "Confusion"
    assert source == "heuristique"
    assert probs is None


def test_classifieur_sklearn_roundtrip(tmp_path, monkeypatch):
    from sklearn.linear_model import LogisticRegression
    import joblib
    import fer_classifier as fc

    rng = np.random.RandomState(0)
    X = rng.rand(40, len(FER_FEATURE_ORDER))
    y = np.array(["Boredom", "Confusion", "Engagement", "Frustration"] * 10)
    clf = LogisticRegression(max_iter=500).fit(X, y)
    path = tmp_path / "clf.joblib"
    joblib.dump({"model": clf, "feature_names": list(FER_FEATURE_ORDER), "classes_": list(clf.classes_)}, path)

    monkeypatch.setenv("EMOLEARN_FER_MAPPING", "learned")
    fc._FER_MAPPING_MODE = "learned"
    fc._pack = None
    fc._load_attempted = False
    fc.DEFAULT_MODEL_PATH = str(path)

    scores = {lab: float(X[0, i]) for i, lab in enumerate(FER_FEATURE_ORDER)}
    etat, niveau, source, probs = fc.appliquer_mapping("happy", scores)
    assert etat in ("Boredom", "Confusion", "Engagement", "Frustration")
    assert source == "classifieur_sklearn"
    assert probs is not None
    assert abs(sum(probs.values()) - 1.0) < 1e-5


def test_ser_finetuned_path_helper():
    from ser_module import ser_finetuned_disponible

    assert ser_finetuned_disponible(os.path.join("no", "such", "dir")) is False


def test_fer_finetuned_path_helper():
    from fer_offline import fer_finetuned_disponible, fer_runtime_mode

    assert fer_finetuned_disponible(os.path.join("no", "such", "dir")) is False
    assert fer_runtime_mode() in ("finetuned", "pretrained")


def test_fer_module_agrege_finetuned_sans_remapping(monkeypatch):
    """Le vote multi-frames fine-tuné ne doit PAS repasser par mapping 7→4."""
    import fer_module as fm
    from PIL import Image

    monkeypatch.setattr(fm, "fer_runtime_mode", lambda: "finetuned")
    monkeypatch.setattr(fm, "model_au_disponible", lambda: False)
    monkeypatch.setattr(
        fm,
        "extract_face_frames",
        lambda clip_id, n_frames=5: (
            [Image.new("RGB", (64, 64), color=(120, 90, 70))],
            {"erreur": None, "path": "fake.avi"},
        ),
    )
    monkeypatch.setattr(
        fm,
        "analyser_visage_image",
        lambda img: {
            "modalite": "visage_fer",
            "emotion": "Frustration",
            "score": 0.82,
            "etat": "Frustration",
            "niveau": 2,
            "tous_scores": {
                "Boredom": 0.05,
                "Confusion": 0.08,
                "Engagement": 0.05,
                "Frustration": 0.82,
            },
            "probs_etats": {
                "Boredom": 0.05,
                "Confusion": 0.08,
                "Engagement": 0.05,
                "Frustration": 0.82,
            },
            "mapping_source": "fer_vit_finetuned",
            "erreur": None,
        },
    )

    out = fm.analyser_visage_depuis_scenario({"ClipID": "demo.avi"}, n_frames=1)
    assert out["erreur"] is None
    assert out["etat"] == "Frustration"
    assert out["mapping_source"] == "fer_vit_finetuned"
    assert out["fer_mode"] == "finetuned"
    assert out["probs_etats"]["Frustration"] > 0.7


def test_oversample_train_equilibre_classes():
    """Oversampling TRAIN : chaque classe atteint la taille de la majoritaire."""
    import importlib.util

    path = os.path.join(
        os.path.dirname(__file__), "..", "scripts", "train_fer_daisee_classifier.py"
    )
    spec = importlib.util.spec_from_file_location("train_fer", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    X = np.arange(10).reshape(10, 1).astype(float)
    y = np.array(
        ["Engagement"] * 5 + ["Boredom"] * 3 + ["Confusion"] * 1 + ["Frustration"] * 1
    )
    X_res, y_res, stats = mod.oversample_train(X, y, seed=0)
    assert stats["enabled"] is True
    assert stats["n_train_apres"] == 5 * 4
    counts = {e: int((y_res == e).sum()) for e in set(y)}
    assert all(c == 5 for c in counts.values())


def test_echantillonner_labels_stratified_quota():
    import importlib.util
    import pandas as pd

    path = os.path.join(
        os.path.dirname(__file__), "..", "scripts", "train_fer_daisee_classifier.py"
    )
    spec = importlib.util.spec_from_file_location("train_fer", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    rows = []
    for i in range(40):
        rows.append(
            {
                "ClipID": f"c{i}.avi",
                "Boredom": 3 if i < 10 else 0,
                "Engagement": 3 if 10 <= i < 30 else 0,
                "Confusion": 3 if 30 <= i < 35 else 0,
                "Frustration": 3 if i >= 35 else 0,
            }
        )
    df = pd.DataFrame(rows)
    out = mod.echantillonner_labels(df, sample=20, seed=0, stratified=True)
    assert len(out) <= 20
    # Au moins 3 états représentés si le pool le permet
    assert out["etat_gt"].nunique() >= 3
