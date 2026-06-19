"""Tests module SER IEMOCAP (annotations + audio .wav)."""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from ser_module import (
    analyser_voix_depuis_scenario,
    get_statistiques_iemocap,
    selectionner_utterance_iemocap,
    charger_iemocap,
)


def test_statistiques_iemocap():
    stats = get_statistiques_iemocap()
    assert stats["total"] > 9000
    assert "distribution" in stats


def test_voix_csv_mode():
    result = analyser_voix_depuis_scenario(
        scenario_id=10,
        etat_dominant="Engagement",
        utiliser_audio=False,
    )
    assert result["erreur"] is None
    assert "annotations" in result["source"].lower()
    assert result["utterance_id"]
    assert result["appariement"]["methode"].startswith("emotion_dominante")
    assert result["appariement"]["etat_cible"] == "Engagement"


def test_appariement_emotion_dominante():
    """Scénario Frustration → utterance IEMOCAP Frustration (plus modulo ligne 30 = Boredom)."""
    df = charger_iemocap()
    row_modulo = df.iloc[30]
    assert row_modulo["etat_daisee"] == "Boredom"

    result = analyser_voix_depuis_scenario(
        scenario_id=30,
        etat_dominant="Frustration",
        utiliser_audio=False,
    )
    assert result["appariement"]["methode"] == "emotion_dominante"
    assert result["appariement"]["etat_cible"] == "Frustration"
    # En mode CSV, l'état voix suit l'annotation IEMOCAP sélectionnée
    assert result["etat"] == "Frustration"
    assert result["utterance_id"] != row_modulo["utterance_id"]


def test_selectionner_utterance_reproductible():
    df = charger_iemocap()
    _, meta1 = selectionner_utterance_iemocap(df, scenario_id=5, etat_dominant="Boredom")
    _, meta2 = selectionner_utterance_iemocap(df, scenario_id=5, etat_dominant="Boredom")
    assert meta1["index_pool"] == meta2["index_pool"]
    assert meta1["methode"] == "emotion_dominante"


def test_voix_audio_mode():
    result = analyser_voix_depuis_scenario(scenario_id=42, utiliser_audio=True)
    assert result["erreur"] is None
    assert result["source"] in (
        "IEMOCAP audio (SER acoustique)",
        "IEMOCAP annotations (fallback, .wav absent)",
        "IEMOCAP annotations (fallback, segment audio)",
    )
    if result["source"] == "IEMOCAP audio (SER acoustique)":
        assert result.get("features_acoustiques")
        assert result.get("wav_path")
