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
from iemocap_appariement import selectionner_par_similarite, invalider_cache_index


def test_statistiques_iemocap():
    stats = get_statistiques_iemocap()
    assert stats["total"] > 9000
    assert "distribution" in stats
    assert stats["wav_disponibles"] == stats["total"]
    assert stats["wav_manquants"] == 0


def test_voix_audio_toujours_actif():
    result = analyser_voix_depuis_scenario(
        scenario_id=10,
        scores_daisee={
            "boredom": 0,
            "confusion": 0,
            "engagement": 3,
            "frustration": 0,
        },
        methode="heuristique",
    )
    assert result["erreur"] is None
    assert "audio" in result["source"].lower()
    assert result["utterance_id"]
    assert result.get("wav_path")
    assert result["appariement"]["methode"].find("similarite_profil") >= 0
    assert "similarite" in result["appariement"]


def test_appariement_similarite_reproductible():
    df = charger_iemocap()
    invalider_cache_index()
    _, meta1 = selectionner_par_similarite(
        df, 5, boredom=1, confusion=0, engagement=2, frustration=0
    )
    invalider_cache_index()
    _, meta2 = selectionner_par_similarite(
        df, 5, boredom=1, confusion=0, engagement=2, frustration=0
    )
    assert meta1["index_pool"] == meta2["index_pool"]
    assert meta1["similarite"] == meta2["similarite"]


def test_selectionner_utterance_reproductible():
    df = charger_iemocap()
    invalider_cache_index()
    _, meta1 = selectionner_utterance_iemocap(
        df,
        scenario_id=5,
        scores_daisee={"boredom": 2, "confusion": 0, "engagement": 0, "frustration": 0},
    )
    invalider_cache_index()
    _, meta2 = selectionner_utterance_iemocap(
        df,
        scenario_id=5,
        scores_daisee={"boredom": 2, "confusion": 0, "engagement": 0, "frustration": 0},
    )
    assert meta1["index_pool"] == meta2["index_pool"]
    assert "similarite_profil" in meta1["methode"]


def test_voix_audio_mode_heuristique():
    result = analyser_voix_depuis_scenario(
        scenario_id=42,
        scores_daisee={"boredom": 0, "confusion": 1, "engagement": 1, "frustration": 0},
        methode="heuristique",
    )
    assert result["erreur"] is None
    assert result["source"] == "IEMOCAP audio (SER acoustique)"
    assert result.get("features_acoustiques")
    assert result.get("wav_path")
    assert result.get("ser_methode") == "heuristique"


def test_fusion_ser_corrige_hap_vs_confusion():
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
    assert meta["mode"].startswith("anchor_vs_confusion")


def test_fusion_ser_engagement_vs_frustration_ft():
    """FT dit Engagement (rire), ancre sur-interprète en Frustration (ang)."""
    p_ft = {
        "Boredom": 0.05,
        "Confusion": 0.15,
        "Engagement": 0.43,
        "Frustration": 0.37,
    }
    p_anchor = {
        "Boredom": 0.05,
        "Confusion": 0.0,
        "Engagement": 0.12,
        "Frustration": 0.83,
    }
    etat, score, probs, meta = fusionner_probs_ser(p_ft, p_anchor)
    assert etat == "Engagement"
    assert meta["mode"] == "ft_vs_anchor_ang_sur_rire"


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
