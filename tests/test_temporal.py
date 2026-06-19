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
from decision_engine import preparer_decision


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
        boredom=0,
        confusion=0,
        engagement=3,
        frustration=0,
        temporal=temporal,
    )
    assert d["declencher"] is False
    assert d["type_contenu"] == "aucune"
