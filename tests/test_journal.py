"""Le journal doit expliquer l'état final, pas seulement le nommer."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from decision_engine import preparer_decision
from pipeline import construire_journal


def _fusion_frustration():
    return {
        "etat": "Frustration",
        "niveau": 2,
        "probs_etats": {
            "Boredom": 0.05,
            "Confusion": 0.05,
            "Engagement": 0.25,
            "Frustration": 0.65,
        },
        "multimodal": True,
        "conflit": False,
        "type_conflit": "accord_soft",
        "strategie_fusion": "vote_pondere_incertitude",
        "explication_fusion": "Pondération → Frustration",
    }


def test_journal_explique_pourquoi_frustration():
    fusion = _fusion_frustration()
    temporal = {
        "declencher": True,
        "regle": "seuils_instantanes",
        "urgence": "rapide",
        "duree_etat": 1,
    }
    decision = preparer_decision(fusion, temporal=temporal)
    analyse = {
        "result_visage": {
            "etat": "Frustration",
            "score": 0.81,
            "erreur": None,
            "n_frames_analysees": 4,
        },
        "result_voix": {
            "etat": "Confusion",
            "score": 0.42,
            "erreur": None,
            "ser_methode": "learned",
        },
        "fusion": fusion,
        "decision": decision,
        "temporal": temporal,
        "resultat_llm": {
            "rag_utilise": False,
            "rag_erreur": "index_absent",
            "skipped": False,
        },
    }
    journal = construire_journal(analyse)
    par_etape = {ligne["etape"]: ligne for ligne in journal}

    assert par_etape["visage"]["code"] == "FER_OK"
    assert "Frustration" in par_etape["visage"]["message"]
    assert par_etape["decision"]["code"] == "STRAT_FRU_GE_2"
    assert "Frustration" in par_etape["decision"]["message"]
    assert par_etape["template"]["code"].startswith("TPL_")
    assert par_etape["generation"]["code"] == "RAG_ERR"
    assert par_etape["generation"]["statut"] == "erreur"


def test_journal_garde_le_code_erreur_modalite():
    fusion = _fusion_frustration()
    decision = preparer_decision(fusion)
    analyse = {
        "result_visage": {"erreur": "video_indisponible:clip42", "etat": None},
        "result_voix": {"erreur": "audio_indisponible:utt9", "etat": None},
        "fusion": fusion,
        "decision": decision,
        "resultat_llm": None,
    }
    journal = construire_journal(analyse)
    par_etape = {ligne["etape"]: ligne for ligne in journal}
    assert par_etape["visage"]["code"] == "video_indisponible:clip42"
    assert par_etape["visage"]["statut"] == "erreur"
    assert par_etape["voix"]["code"] == "audio_indisponible:utt9"
    assert par_etape["generation"]["code"] == "GEN_ABSENT"
