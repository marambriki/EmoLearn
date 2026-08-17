# ============================================
# backend/fer_offline.py
# FER image HuggingFace
# - pretrained : dima806 (7 émotions) + mapping FER→DAiSEE
# - finetuned  : ViT fine-tuné DAiSEE (4 états) si models/fer_vit_daisee/
# ============================================

import os

from PIL import Image

from fer_classifier import appliquer_mapping

_fer_model = None
_fer_mode_loaded = None
_fer_pretrained = None

ETATS_DAISEE = ("Boredom", "Confusion", "Engagement", "Frustration")
_MODELS = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "models"))
# Préfère le checkpoint full s'il existe (évite conflit d'écriture avec l'API)
FINETUNED_DIR = next(
    (
        d
        for d in (
            os.path.join(_MODELS, "fer_vit_daisee_ckpt"),
            os.path.join(_MODELS, "fer_vit_daisee"),
        )
        if os.path.isdir(d) and os.path.isfile(os.path.join(d, "config.json"))
    ),
    os.path.join(_MODELS, "fer_vit_daisee"),
)
BASE_MODEL = "dima806/facial_emotions_image_detection"


def fer_finetuned_disponible(path: str | None = None) -> bool:
    cible = path or FINETUNED_DIR
    return os.path.isdir(cible) and (
        os.path.isfile(os.path.join(cible, "config.json"))
        or os.path.isfile(os.path.join(cible, "model.safetensors"))
        or os.path.isfile(os.path.join(cible, "pytorch_model.bin"))
    )


def fer_runtime_mode() -> str:
    """
    finetuned (défaut produit) | pretrained | auto
    EMOLEARN_FER_MODE=finetuned|pretrained|auto

    Par défaut on force le ViT fine-tuné DAiSEE dès qu'il est présent
    (models/fer_vit_daisee[_ckpt]/). Fallback pretrained seulement si absent.
    """
    raw = os.getenv("EMOLEARN_FER_MODE", "finetuned").lower().strip()
    if raw in ("pretrained", "base", "hf", "heuristique"):
        return "pretrained"
    # finetuned | auto | ft | vit | daisee
    return "finetuned" if fer_finetuned_disponible() else "pretrained"


def get_fer_model():
    """Charge le pipeline HuggingFace à la demande (finetuned ou pretrained)."""
    global _fer_model, _fer_mode_loaded
    mode = fer_runtime_mode()
    if _fer_model is not None and _fer_mode_loaded == mode:
        return _fer_model

    from transformers import pipeline

    if mode == "finetuned":
        print(f"Chargement modèle FER fine-tuné DAiSEE ({FINETUNED_DIR})...")
        _fer_model = pipeline(
            "image-classification",
            model=FINETUNED_DIR,
            device=-1,
        )
        print("Modèle FER fine-tuné chargé.")
    else:
        print("Chargement modèle FER (pretrained HuggingFace)...")
        _fer_model = pipeline(
            "image-classification",
            model=BASE_MODEL,
            device=-1,
        )
        print("Modèle FER offline chargé.")
    _fer_mode_loaded = mode
    return _fer_model


def _result_from_pipeline(result, runtime_mode: str):
    emotion_dominante = result[0]["label"]
    score_confiance = float(result[0]["score"])
    tous_scores = {r["label"]: float(r["score"]) for r in result}

    # Fine-tune direct DAiSEE : le label EST l'état
    if runtime_mode == "finetuned" and emotion_dominante in ETATS_DAISEE:
        niveau = max(1, min(3, int(round(score_confiance * 3))))
        probs = {e: float(tous_scores.get(e, 0.0)) for e in ETATS_DAISEE}
        return {
            "modalite": "visage_fer",
            "emotion": emotion_dominante,
            "score": score_confiance,
            "etat": emotion_dominante,
            "niveau": niveau,
            "tous_scores": tous_scores,
            "mapping_source": "fer_vit_finetuned",
            "probs_etats": probs,
            "erreur": None,
        }

    etat, niveau, mapping_source, probs_etats = appliquer_mapping(
        emotion_dominante, tous_scores
    )
    out = {
        "modalite": "visage_fer",
        "emotion": emotion_dominante,
        "score": score_confiance,
        "etat": etat,
        "niveau": niveau,
        "tous_scores": tous_scores,
        "mapping_source": mapping_source,
        "erreur": None,
    }
    if probs_etats:
        out["probs_etats"] = probs_etats
    return out


def get_fer_pretrained():
    """Pipeline 7 émotions (indépendant du mode finetuned) — pour features FER+AU."""
    global _fer_pretrained
    if _fer_pretrained is None:
        from transformers import pipeline

        print(f"Chargement FER pretrained ({BASE_MODEL}) pour features AU…")
        _fer_pretrained = pipeline(
            "image-classification",
            model=BASE_MODEL,
            device=-1,
        )
    return _fer_pretrained


def analyser_visage_image(image):
    """Analyse une image PIL déjà ouverte."""
    try:
        mode = fer_runtime_mode()
        result = get_fer_model()(image)
        return _result_from_pipeline(result, mode)
    except Exception as e:
        return {
            "modalite": "visage_fer",
            "emotion": "neutral",
            "score": 0.5,
            "etat": "Boredom",
            "niveau": 1,
            "tous_scores": {},
            "erreur": str(e),
        }


def analyser_visage_image_pretrained(image):
    """Force les 7 scores émotionnels (mapping FER→DAiSEE / features AU)."""
    try:
        result = get_fer_pretrained()(image)
        return _result_from_pipeline(result, "pretrained")
    except Exception as e:
        return {
            "modalite": "visage_fer",
            "emotion": "neutral",
            "score": 0.5,
            "etat": "Boredom",
            "niveau": 1,
            "tous_scores": {},
            "erreur": str(e),
        }


def analyser_visage(image_path):
    """Analyse l'émotion depuis un chemin d'image (scripts offline)."""
    try:
        image = Image.open(image_path)
        return analyser_visage_image(image)
    except Exception as e:
        return {
            "modalite": "visage_fer",
            "emotion": "neutral",
            "score": 0.5,
            "etat": "Boredom",
            "niveau": 1,
            "tous_scores": {},
            "erreur": str(e),
        }
