"""Tests appariement IEMOCAP par similarité."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from iemocap_appariement import (
    invalider_cache_index,
    selectionner_par_similarite,
    vecteur_profil_scenario,
    vecteur_profil_iemocap,
)
from ser_module import charger_iemocap


def test_vecteur_scenario_normalise():
    v = vecteur_profil_scenario(0, 0, 3, 0)
    assert v[3] == 0.0  # frustration
    assert v[2] == 1.0  # engagement max
    assert abs(v[4] - 0.75) < 1e-9  # valence centroïde Engagement (Baker)
    assert abs(v[5] - 0.50) < 1e-9  # activation centroïde Engagement
    assert 0.0 <= v[6] <= 1.0


def test_similarite_engagement_prefere_hap():
    df = charger_iemocap()
    invalider_cache_index()
    row, meta = selectionner_par_similarite(
        df,
        scenario_id=2,
        boredom=0,
        confusion=0,
        engagement=3,
        frustration=0,
    )
    assert meta["similarite"] >= 0.5
    assert meta["filtre_etat_applique"] is True
    assert meta["etat_cible"] == "Engagement"
    assert row.get("etat_daisee") == "Engagement"
    assert "similarite_profil" in meta["methode"]


def test_similarite_frustration_restreint_etat():
    df = charger_iemocap()
    invalider_cache_index()
    row, meta = selectionner_par_similarite(
        df,
        scenario_id=30,
        boredom=0,
        confusion=0,
        engagement=0,
        frustration=3,
    )
    assert meta["etat_cible"] == "Frustration"
    assert row.get("etat_daisee") == "Frustration"
    assert meta["filtre_etat_applique"] is True


def test_similarite_boredom_restreint_etat():
    df = charger_iemocap()
    invalider_cache_index()
    row, meta = selectionner_par_similarite(
        df,
        scenario_id=10,
        boredom=2,
        confusion=0,
        engagement=0,
        frustration=0,
    )
    assert meta["etat_cible"] == "Boredom"
    assert row.get("etat_daisee") == "Boredom"


def test_profil_iemocap_dimensions():
    row = {
        "etat_daisee": "Engagement",
        "niveau_daisee": 3,
        "valence": 4.0,
        "activation": 4.5,
        "dominance": 3.5,
        "duree": 2.5,
    }
    v = vecteur_profil_iemocap(row)
    assert len(v) == 12
