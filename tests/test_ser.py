"""Tests module SER IEMOCAP (signal audio .wav réel, obligatoire)."""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from ser_module import (
    analyser_voix_depuis_scenario,
    fusionner_probs_ser,
    get_statistiques_iemocap,
    selectionner_utterance_iemocap,
    charger_iemocap,
)


def test_statistiques_iemocap():
    stats = get_statistiques_iemocap()
    assert stats["total"] > 9000
    assert "distribution" in stats
    # Couverture .wav attendue = 100% (corpus complet présent localement)
    assert stats["wav_disponibles"] == stats["total"]
    assert stats["wav_manquants"] == 0


def test_voix_audio_toujours_actif():
    """L'audio réel est désormais la SEULE voie : plus de mode CSV/annotations."""
    result = analyser_voix_depuis_scenario(
        scenario_id=10,
        etat_dominant="Engagement",
        methode="heuristique",
    )
    assert result["erreur"] is None
    assert "audio" in result["source"].lower()
    assert result["utterance_id"]
    assert result.get("wav_path")
    assert result["appariement"]["methode"].startswith("emotion_dominante")
    assert result["appariement"]["etat_cible"] == "Engagement"


def test_appariement_emotion_dominante():
    """Scénario Frustration → utterance IEMOCAP appariée sur l'état Frustration."""
    result = analyser_voix_depuis_scenario(
        scenario_id=30,
        etat_dominant="Frustration",
        methode="heuristique",
    )
    assert result["appariement"]["methode"].startswith("emotion_dominante")
    assert result["appariement"]["etat_cible"] == "Frustration"
    assert result["appariement"]["etat_iemocap"] == "Frustration"


def test_selectionner_utterance_reproductible():
    df = charger_iemocap()
    _, meta1 = selectionner_utterance_iemocap(df, scenario_id=5, etat_dominant="Boredom")
    _, meta2 = selectionner_utterance_iemocap(df, scenario_id=5, etat_dominant="Boredom")
    assert meta1["index_pool"] == meta2["index_pool"]
    assert meta1["methode"] == "emotion_dominante"


def test_voix_audio_mode_heuristique():
    # Forcer heuristique pour rester déterministe sans charger le fine-tuné
    result = analyser_voix_depuis_scenario(
        scenario_id=42, methode="heuristique"
    )
    assert result["erreur"] is None
    assert result["source"] == "IEMOCAP audio (SER acoustique)"
    assert result.get("features_acoustiques")
    assert result.get("wav_path")
    assert result.get("ser_methode") == "heuristique"


def test_fusion_ser_corrige_hap_vs_confusion():
    """SUPERB confiant Engagement doit battre un FT qui dit Confusion (cas hap)."""
    p_ft = {
        "Boredom": 0.10,
        "Confusion": 0.55,
        "Engagement": 0.20,
        "Frustration": 0.15,
    }
    p_superb = {
        "Boredom": 0.05,
        "Confusion": 0.0,
        "Engagement": 0.80,
        "Frustration": 0.15,
    }
    etat, score, probs, meta = fusionner_probs_ser(p_ft, p_superb)
    assert etat == "Engagement"
    assert probs["Engagement"] > probs["Confusion"]
    assert meta["mode"] == "superb_prioritaire_vs_confusion"


def test_fusion_ser_ft_seul_sans_superb():
    p_ft = {
        "Boredom": 0.1,
        "Confusion": 0.2,
        "Engagement": 0.6,
        "Frustration": 0.1,
    }
    etat, score, probs, meta = fusionner_probs_ser(p_ft, None)
    assert etat == "Engagement"
    assert meta["mode"] == "ft_seul"
