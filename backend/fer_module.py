# ============================================
# backend/fer_module.py
# Module FER — Reconnaissance émotions faciales
# Basé sur DAiSEE + modèle HuggingFace
# ============================================

from PIL import Image
import os

from daisee_utils import etat_dominant_depuis_scores
from fer_mapping import mapping_fer_daisee

# ===== CHARGEMENT LAZY DU MODÈLE FER =====
_fer_model = None


def get_fer_model():
    """Charge le pipeline HuggingFace à la demande."""
    global _fer_model
    if _fer_model is None:
        from transformers import pipeline

        print("Chargement modèle FER...")
        _fer_model = pipeline(
            "image-classification",
            model="dima806/facial_emotions_image_detection",
            device=-1,
        )
        print("Modèle FER chargé.")
    return _fer_model

def analyser_visage_image(image):
    """Analyse une image PIL déjà ouverte."""
    try:
        result = get_fer_model()(image)
        emotion_dominante = result[0]["label"]
        score_confiance = result[0]["score"]
        etat_daisee = mapping_fer_daisee.get(
            emotion_dominante,
            {"etat": "Engagement", "niveau": 1},
        )
        return {
            "modalite": "visage_fer",
            "emotion": emotion_dominante,
            "score": score_confiance,
            "etat": etat_daisee["etat"],
            "niveau": etat_daisee["niveau"],
            "tous_scores": {r["label"]: r["score"] for r in result},
            "erreur": None,
        }
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
    """
    Analyse l'émotion depuis une image de visage
    Input  : chemin vers l'image
    Output : dictionnaire avec émotion, score, état DAiSEE
    """
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

def analyser_visage_depuis_scenario(scenario_row):
    """
    Analyse l'émotion depuis les données DAiSEE
    (simulation sans vraie image)
    Input  : ligne du DataFrame DAiSEE
    Output : dictionnaire avec état émotionnel
    """
    etat_dominant, niveau_dominant, etats = etat_dominant_depuis_scores(
        int(scenario_row["Boredom"]),
        int(scenario_row["Confusion"]),
        int(scenario_row["Engagement"]),
        int(scenario_row["Frustration"]),
    )

    return {
        "modalite": "visage_daisee",
        "emotion": etat_dominant.lower(),
        "score": niveau_dominant / 3.0,
        "etat": etat_dominant,
        "niveau": niveau_dominant,
        "tous_scores": etats,
        "erreur": None,
    }