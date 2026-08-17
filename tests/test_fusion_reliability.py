"""Tests de la fusion par pondération continue d'incertitude (mode soft)."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from fusion_module import fusionner_par_vote, fusionner_modalites  # noqa: E402


def _fer(etat: str, score: float):
    return {
        "etat": etat,
        "niveau": 2,
        "score": score,
        "source": "DAiSEE",
        "confiance_visage": score,
        "methode": "test",
        "probas_4classes": {etat: score},
    }


def _ser(etat: str, score: float):
    return {
        "etat": etat,
        "niveau": 2,
        "score": score,
        "source": "IEMOCAP",
        "confiance_voix": score,
        "methode": "test",
        "probas_4classes": {etat: score},
    }


def test_accord_engagement_confiance_haute():
    fusion = fusionner_par_vote(_fer("Engagement", 0.90), _ser("Engagement", 0.85))
    assert fusion["etat"] == "Engagement"
    assert fusion["strategie_fusion"] == "vote_pondere_incertitude"
    assert fusion["gate"]["methode"] == "incertitude"


def test_voix_plus_confiante_peut_gagner_malgre_prior_60_40():
    """0.60×0.20 vs 0.40×0.90 → voix 0.36 > visage 0.12 → Frustration."""
    fusion = fusionner_par_vote(_fer("Engagement", 0.20), _ser("Frustration", 0.90))
    assert fusion["etat"] == "Frustration"
    assert fusion["gate"]["poids_effectif"]["voix"] > fusion["gate"]["poids_effectif"]["visage"]


def test_visage_plus_confiant_domine_quand_voix_incertaine():
    fusion = fusionner_par_vote(_fer("Boredom", 0.85), _ser("Frustration", 0.25))
    assert fusion["etat"] == "Boredom"
    assert fusion["gate"]["poids_effectif"]["visage"] > fusion["gate"]["poids_effectif"]["voix"]


def test_deux_confiances_nulles_priorite_pedagogique():
    fusion = fusionner_par_vote(_fer("Engagement", 0.0), _ser("Frustration", 0.0))
    assert fusion["etat"] == "Frustration"
    assert fusion["strategie_fusion"] == "priorite_pedagogique_incertitude_totale"


def test_voix_absente_reste_unimodal():
    fusion = fusionner_par_vote(_fer("Confusion", 0.80), None)
    assert fusion["etat"] == "Confusion"
    assert fusion["strategie_fusion"] == "vote_incertitude_unimodal_visage"


def test_fusionner_modalites_appelle_incertitude():
    fusion = fusionner_modalites(_fer("Engagement", 0.70), _ser("Engagement", 0.65))
    assert fusion["strategie_fusion"] == "vote_pondere_incertitude"
