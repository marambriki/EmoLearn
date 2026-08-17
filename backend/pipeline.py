# ============================================
# Pipeline d'analyse unifié — clip de simulation
# ============================================

import json
import os
from datetime import datetime, timezone

from fer_module import analyser_visage_depuis_scenario
from ser_module import analyser_voix_depuis_scenario
from fusion_module import fusionner_modalites
from decision_engine import preparer_decision, scores_depuis_fusion
from llm_module import generer_contenu_adaptatif
from daisee_utils import etat_depuis_ligne
from temporal_rules import evaluer_declenchement_temporel

LOG_DIR = os.path.join(os.path.dirname(__file__), "../logs")
LOG_FILE = os.path.join(LOG_DIR, "decisions.jsonl")


def enregistrer_decision(
    scenario_id,
    clip_id,
    fusion,
    decision,
    resultat_llm=None,
    extra=None,
):
    """Ajoute une ligne JSON au journal decisions.jsonl."""
    os.makedirs(LOG_DIR, exist_ok=True)

    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "scenario_id": scenario_id,
        "clip_id": clip_id,
        "fusion": {
            "etat": fusion.get("etat"),
            "niveau": fusion.get("niveau"),
            "multimodal": fusion.get("multimodal"),
            "conflit": fusion.get("conflit"),
            "type_conflit": fusion.get("type_conflit"),
        },
        "decision": {
            "etat": decision.get("etat"),
            "niveau": decision.get("niveau"),
            "template_id": decision.get("template_id"),
            "type_contenu": decision.get("type_contenu"),
            "strategie": decision.get("strategie"),
            "scores_daisee": decision.get("scores_daisee"),
        },
    }

    if resultat_llm:
        entry["generation"] = {
            "nb_mots": resultat_llm.get("nb_mots"),
            "rag_utilise": resultat_llm.get("rag_utilise"),
            "nb_docs_rag": resultat_llm.get("nb_docs_rag"),
        }

    if extra:
        entry["extra"] = extra

    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    return LOG_FILE


def scores_depuis_scenario(scenario):
    return {
        "boredom": int(scenario["Boredom"]),
        "confusion": int(scenario["Confusion"]),
        "engagement": int(scenario["Engagement"]),
        "frustration": int(scenario["Frustration"]),
    }


def analyser_clip(
    scenario,
    scenario_id,
    concept,
    module,
    niveau,
    moment_cours="Milieu du module",
    generer_contenu=True,
    historique_scores=None,
    utiliser_temporel=True,
):
    """
    Analyse un clip de simulation :
    visage = vidéo DAiSEE réelle + ViT FER fine-tuné (4 états), vote multi-frames
    voix   = audio IEMOCAP .wav + wav2vec2 SER fine-tuné (défaut: learned)

    historique_scores : clips précédents (dicts boredom/confusion/engagement/frustration).
    Le clip courant est ajouté avant l'évaluation temporelle.
    """
    pattern = str(scenario.get("pattern", "stable"))
    scores_bruts = scores_depuis_scenario(scenario)

    result_visage = analyser_visage_depuis_scenario(scenario)

    etat_dominant, _, _ = etat_depuis_ligne(scenario)
    result_voix = analyser_voix_depuis_scenario(
        scenario_id=scenario_id,
        etat_dominant=etat_dominant,
        methode="learned",
    )
    fusion = fusionner_modalites(result_visage, result_voix)

    hist = list(historique_scores or [])
    hist.append(scores_depuis_fusion(fusion))
    temporal = evaluer_declenchement_temporel(hist) if utiliser_temporel else None

    decision = preparer_decision(
        fusion=fusion,
        duree_etat=len(hist),
        temporal=temporal,
        scores_bruts=scores_bruts,
    )

    resultat_llm = None
    if generer_contenu and decision.get("declencher", True):
        resultat_llm = generer_contenu_adaptatif(
            etat=decision["etat"],
            niveau=decision["niveau"],
            concept=concept,
            template_id=decision["template_id"],
            objectif_adaptation=decision["objectif"],
            sujet_cours=f"Informatique — {module}",
            module_actuel=module,
            niveau_apprenant=niveau,
            objectif_pedagogique=f"Comprendre {concept}",
            duree_etat=decision["duree_etat"],
            pattern_detecte=decision.get("pattern", pattern),
            moment_cours=moment_cours,
            frustration_niveau=decision["scores_daisee"]["frustration"],
        )
    elif not decision.get("declencher", True):
        resultat_llm = {
            "contenu": (
                "Aucune adaptation nécessaire pour ce clip : l'engagement est dans une "
                "zone acceptable. Poursuite du parcours normal."
            ),
            "nb_mots": 0,
            "rag_utilise": False,
            "nb_docs_rag": 0,
            "sources_rag": [],
            "skipped": True,
        }

    return {
        "clip_id": str(scenario["ClipID"]),
        "pattern": decision.get("pattern", pattern),
        "mode": "simulation",
        "scores": scores_bruts,
        "scores_fusion": decision["scores_daisee"],
        "result_visage": result_visage,
        "result_voix": result_voix,
        "fusion": fusion,
        "decision": decision,
        "temporal": temporal,
        "resultat_llm": resultat_llm,
    }


def formater_reponse_api(analyse, scenario_id=None):
    """Formate la sortie pipeline pour l'API JSON."""
    decision = analyse["decision"]
    fusion = analyse["fusion"]
    llm = analyse["resultat_llm"] or {}

    return {
        "scenario_id": scenario_id,
        "clip_id": analyse["clip_id"],
        "pattern": analyse.get("decision", {}).get("pattern", analyse["pattern"]),
        "mode": "simulation",
        "emotion": decision["etat"],
        "niveau": decision["niveau"],
        "strategie": decision["strategie"],
        "type_contenu": decision["type_contenu"],
        "type_contenu_regle": decision["type_contenu_regle"],
        "template_id": decision["template_id"],
        "description": decision["description"],
        "objectif": decision["objectif"],
        "contenu": llm.get("contenu", ""),
        "nb_mots": llm.get("nb_mots", 0),
        "contenu_skipped": llm.get("skipped", False),
        "multimodal": fusion.get("multimodal", False),
        "modalite_dominante": fusion.get("modalite_dominante"),
        "details_fusion": fusion.get("details"),
        "scores_bruts": analyse.get("scores"),
        "scores_fusion": decision["scores_daisee"],
        "visage": {
            "etat": analyse.get("result_visage", {}).get("etat"),
            "emotion": analyse.get("result_visage", {}).get("emotion"),
            "niveau": analyse.get("result_visage", {}).get("niveau"),
            "score": analyse.get("result_visage", {}).get("score"),
            "fer_mode": analyse.get("result_visage", {}).get("fer_mode"),
            "mapping_source": analyse.get("result_visage", {}).get("mapping_source"),
            "n_frames_analysees": analyse.get("result_visage", {}).get("n_frames_analysees"),
            "clip_id": analyse.get("result_visage", {}).get("clip_id"),
            "erreur": analyse.get("result_visage", {}).get("erreur"),
        },
        "voix": {
            "source": analyse.get("result_voix", {}).get("source"),
            "ser_methode": analyse.get("result_voix", {}).get("ser_methode"),
            "utterance_id": analyse.get("result_voix", {}).get("utterance_id"),
            "emotion": analyse.get("result_voix", {}).get("emotion"),
            "etat": analyse.get("result_voix", {}).get("etat"),
            "niveau": analyse.get("result_voix", {}).get("niveau"),
            "score": analyse.get("result_voix", {}).get("score"),
            "wav_path": analyse.get("result_voix", {}).get("wav_path"),
            "appariement": analyse.get("result_voix", {}).get("appariement"),
            "erreur": analyse.get("result_voix", {}).get("erreur"),
        },
        "decision": {
            "scores_daisee": decision["scores_daisee"],
            "scores_bruts": decision.get("scores_bruts"),
            "pattern": decision.get("pattern"),
            "fusion_meta": decision["fusion"],
            "declencher": decision.get("declencher", True),
            "temporal": decision.get("temporal") or analyse.get("temporal"),
        },
        "temporal": analyse.get("temporal") or decision.get("temporal"),
        "rag_utilise": llm.get("rag_utilise", False),
        "nb_docs_rag": llm.get("nb_docs_rag", 0),
        "sources_rag": llm.get("sources_rag", []),
        "rag_erreur": llm.get("rag_erreur"),
    }
