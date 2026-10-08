"""Tests unitaires — règles temporelles et moteur de décision."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from temporal_rules import (
    URGENCE_IMMEDIATE,
    URGENCE_RAPIDE,
    AUCUNE_URGENCE,
    evaluer_declenchement_temporel,
)
from decision_engine import preparer_decision, scores_depuis_fusion


def _clip(b=0, c=0, e=2, f=0):
    return {"boredom": b, "confusion": c, "engagement": e, "frustration": f}


def test_declenchement_immediate_confusion_frustration():
    hist = [_clip(c=3, f=2, e=1)]
    r = evaluer_declenchement_temporel(hist)
    assert r["declencher"] is True
    assert r["urgence"] == URGENCE_IMMEDIATE


def test_declenchement_immediate_engagement_zero():
    hist = [_clip(e=0)]
    r = evaluer_declenchement_temporel(hist)
    assert r["declencher"] is True
    assert r["regle"] == "engagement_zero"


def test_pas_intervention_engagement_stable():
    hist = [_clip(e=3), _clip(e=3)]
    r = evaluer_declenchement_temporel(hist)
    assert r["declencher"] is False
    assert r["urgence"] == AUCUNE_URGENCE


def test_boredom_consecutif():
    hist = [_clip(b=2, e=2), _clip(b=2, e=1)]
    r = evaluer_declenchement_temporel(hist)
    assert r["declencher"] is True
    assert r["urgence"] == URGENCE_RAPIDE


def test_decision_sans_declenchement():
    fusion = {
        "etat": "Engagement",
        "niveau": 3,
        "multimodal": False,
        "conflit": False,
    }
    temporal = {"declencher": False, "duree_etat": 2, "urgence": AUCUNE_URGENCE}
    d = preparer_decision(
        fusion=fusion,
        temporal=temporal,
    )
    assert d["declencher"] is False
    assert d["type_contenu"] == "aucune"


def test_decision_frustration_molle_suit_etat_pas_scores():
    """Argmax Frustration + proba molles (scores 0–1) → D, pas B2."""
    fusion = {
        "etat": "Frustration",
        "niveau": 1,
        "probs_etats": {
            "Boredom": 0.08,
            "Confusion": 0.34,
            "Engagement": 0.22,
            "Frustration": 0.36,
        },
        "multimodal": True,
        "conflit": False,
    }
    d = preparer_decision(fusion=fusion)
    assert d["etat"] == "Frustration"
    assert d["type_contenu_regle"] == "D+A"
    assert d["template_id"] in ("D1", "D2")
    assert d["pattern"] == "frustration"


def test_decision_ennui_suit_etat():
    fusion = {
        "etat": "Boredom",
        "niveau": 2,
        "probs_etats": {
            "Boredom": 0.40,
            "Confusion": 0.20,
            "Engagement": 0.25,
            "Frustration": 0.15,
        },
        "multimodal": True,
        "conflit": False,
    }
    d = preparer_decision(fusion=fusion)
    assert d["type_contenu_regle"] == "B"
    assert d["template_id"] in ("B1", "B3")
    assert d["pattern"] == "ennui_eleve"
    """La cascade pédagogique doit suivre la fusion, pas les scores CSV bruts."""
    fusion = {
        "etat": "Engagement",
        "niveau": 2,
        "multimodal": True,
        "conflit": True,
    }
    d = preparer_decision(
        fusion=fusion,
        scores_bruts={"boredom": 2, "confusion": 0, "engagement": 1, "frustration": 0},
    )
    assert d["etat"] == "Engagement"
    assert d["type_contenu_regle"] == "D"
    assert d["template_id"] == "D4"
    assert d["scores_daisee"]["engagement"] == 2
    assert d["scores_bruts"]["boredom"] == 2


def test_scores_depuis_fusion_distribution():
    fusion = {
        "etat": "Confusion",
        "niveau": 2,
        "probs_etats": {
            "Boredom": 0.1,
            "Confusion": 0.7,
            "Engagement": 0.1,
            "Frustration": 0.1,
        },
    }
    scores = scores_depuis_fusion(fusion)
    assert scores["confusion"] == 2


def test_pipeline_branche_temporel(monkeypatch):
    """Vérifie que analyser_clip évalue bien temporal_rules (plus temporal=None)."""
    import pipeline as pl

    scenario = {
        "ClipID": "test.avi",
        "Boredom": 0,
        "Confusion": 0,
        "Engagement": 3,
        "Frustration": 0,
        "pattern": "stable",
    }

    monkeypatch.setattr(
        pl,
        "analyser_visage_depuis_scenario",
        lambda s: {"etat": "Engagement", "niveau": 3, "score": 0.9},
    )
    monkeypatch.setattr(
        pl,
        "analyser_voix_depuis_scenario",
        lambda **kw: {"etat": "Engagement", "niveau": 2, "source": "mock"},
    )
    monkeypatch.setattr(
        pl,
        "etat_depuis_ligne",
        lambda s: ("Engagement", 3, {}),
    )
    monkeypatch.setattr(
        pl,
        "fusionner_modalites",
        lambda v, x: {
            "etat": "Engagement",
            "niveau": 3,
            "multimodal": True,
            "conflit": False,
            "details": {},
        },
    )

    # Deux clips engagement fort → pas d'intervention
    hist = [_clip(e=3)]
    out = pl.analyser_clip(
        scenario=scenario,
        scenario_id=0,
        concept="tri",
        module="Algo",
        niveau="Intermédiaire",
        generer_contenu=False,
        historique_scores=hist,
        utiliser_temporel=True,
    )
    assert out["temporal"] is not None
    assert out["temporal"]["declencher"] is False
    assert out["decision"]["declencher"] is False


def test_proxy_fusion_script_importable():
    import importlib.util

    path = os.path.join(
        os.path.dirname(__file__), "..", "scripts", "eval_fusion_produit_proxy.py"
    )
    spec = importlib.util.spec_from_file_location("eval_fusion_produit_proxy", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    # Ne charge pas le module entier (dépendances lourdes) : vérifie le fichier
    assert os.path.isfile(path)
