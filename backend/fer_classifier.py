# ============================================
# backend/fer_classifier.py
# Classifieur sklearn : features FER (7) [+ AU] → états DAiSEE (4)
# ============================================

from __future__ import annotations

import os
from typing import Any

import numpy as np

from au_module import AU_FEATURE_ORDER, au_to_vector

# Mapping émotions FER (7 classes) → états DAiSEE (4 classes)
mapping_fer_daisee = {
    "happy": {"etat": "Engagement", "niveau": 3},
    "neutral": {"etat": "Boredom", "niveau": 1},
    "sad": {"etat": "Boredom", "niveau": 2},
    "angry": {"etat": "Frustration", "niveau": 3},
    "fear": {"etat": "Confusion", "niveau": 3},
    "surprise": {"etat": "Confusion", "niveau": 2},
    "disgust": {"etat": "Frustration", "niveau": 2},
}

FER_FEATURE_ORDER = (
    "angry",
    "disgust",
    "fear",
    "happy",
    "neutral",
    "sad",
    "surprise",
)

ETATS_DAISEE = ("Boredom", "Confusion", "Engagement", "Frustration")

DEFAULT_MODEL_PATH = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "models", "fer_daisee_clf.joblib")
)
DEFAULT_AU_MODEL_PATH = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "models", "fer_daisee_clf_au.joblib")
)

# Défaut = "auto" : classifieur entraîné SEULEMENT s'il bat réellement l'heuristique
_FER_MAPPING_MODE = os.getenv("EMOLEARN_FER_MAPPING", "auto").lower()
_pack: dict[str, Any] | None = None
_pack_au: dict[str, Any] | None = None
_load_attempted = False
_load_au_attempted = False


def scores_to_vector(tous_scores: dict | None) -> np.ndarray:
    """Vecteur ordonné des 7 scores FER (0.0 si label absent)."""
    scores = tous_scores or {}
    return np.asarray(
        [float(scores.get(lab, 0.0)) for lab in FER_FEATURE_ORDER],
        dtype=np.float64,
    )


def features_fer_au(tous_scores: dict | None, au_scores: dict | None) -> np.ndarray:
    """Concatène FER (7) + AU (10) pour le classifieur enrichi."""
    return np.concatenate(
        [scores_to_vector(tous_scores), au_to_vector(au_scores)],
        axis=0,
    )


def model_path_default() -> str:
    return DEFAULT_MODEL_PATH


def model_au_path_default() -> str:
    return DEFAULT_AU_MODEL_PATH


def model_disponible(path: str | None = None) -> bool:
    return os.path.isfile(path or DEFAULT_MODEL_PATH)


def model_au_disponible(path: str | None = None) -> bool:
    return os.path.isfile(path or DEFAULT_AU_MODEL_PATH)


def charger_classifieur(path: str | None = None, force: bool = False):
    """Charge le pack joblib {model, feature_names, classes_, metrics}."""
    global _pack, _load_attempted
    if _pack is not None and not force:
        return _pack
    if _load_attempted and not force:
        return None

    cible = path or DEFAULT_MODEL_PATH
    _load_attempted = True
    if not os.path.isfile(cible):
        return None

    import joblib

    _pack = joblib.load(cible)
    print(f"Classifieur FER→DAiSEE chargé : {cible}")
    return _pack


def charger_classifieur_au(path: str | None = None, force: bool = False):
    """Charge le classifieur FER+AU si présent."""
    global _pack_au, _load_au_attempted
    if _pack_au is not None and not force:
        return _pack_au
    if _load_au_attempted and not force:
        return None

    cible = path or DEFAULT_AU_MODEL_PATH
    _load_au_attempted = True
    if not os.path.isfile(cible):
        return None

    import joblib

    _pack_au = joblib.load(cible)
    print(f"Classifieur FER+AU→DAiSEE chargé : {cible}")
    return _pack_au


def _predire_avec_pack(pack, x: np.ndarray):
    model = pack["model"]
    x = x.reshape(1, -1)
    etat = str(model.predict(x)[0])

    proba_dict = {e: 0.0 for e in ETATS_DAISEE}
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(x)[0]
        classes = list(getattr(model, "classes_", pack.get("classes_", [])))
        for cls, p in zip(classes, probs):
            if cls in proba_dict:
                proba_dict[cls] = float(p)
        niveau = max(1, min(3, int(round(max(probs) * 3))))
    else:
        niveau = 2
        proba_dict[etat] = 1.0

    return etat, niveau, proba_dict


def predire_etat(tous_scores: dict, path: str | None = None, au_scores: dict | None = None):
    """
    Prédit l'état DAiSEE depuis les scores FER (+ AU si modèle AU présent).
    Retourne (etat, niveau, proba_dict) ou None si modèle indisponible.
    """
    if model_au_disponible():
        pack_au = charger_classifieur_au()
        if pack_au and pack_au.get("beats_baseline", True) is not False:
            x = features_fer_au(tous_scores, au_scores)
            return _predire_avec_pack(pack_au, x)

    pack = charger_classifieur(path)
    if not pack:
        return None
    x = scores_to_vector(tous_scores)
    return _predire_avec_pack(pack, x)


def mapping_mode_actif() -> str:
    mode = _FER_MAPPING_MODE
    if mode == "auto":
        if model_au_disponible():
            pack = charger_classifieur_au()
            if pack and pack.get("beats_baseline") is not False:
                return "learned"
        if not model_disponible():
            return "heuristique"
        pack = charger_classifieur()
        if pack and pack.get("beats_baseline") is False:
            return "heuristique"
        return "learned"
    if mode in ("learned", "learn", "sklearn", "classifieur"):
        if model_au_disponible():
            return "learned"
        return "learned" if model_disponible() else "heuristique"
    return "heuristique"


def appliquer_mapping(
    emotion: str,
    tous_scores: dict | None,
    au_scores: dict | None = None,
):
    """
    Applique le mapping appris (FER ou FER+AU) ou heuristique.
    Retourne (etat, niveau, source, probs_etats|None).
    """
    mode = mapping_mode_actif()
    if mode == "learned" and tous_scores:
        pred = predire_etat(tous_scores, au_scores=au_scores)
        if pred is not None:
            etat, niveau, probs = pred
            source = (
                "classifieur_sklearn_au"
                if model_au_disponible()
                else "classifieur_sklearn"
            )
            return etat, niveau, source, probs

    heur = mapping_fer_daisee.get(emotion, {"etat": "Engagement", "niveau": 1})
    return heur["etat"], heur["niveau"], "heuristique", None
