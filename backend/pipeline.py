# ============================================
# Pipeline d'analyse unifié — clip / séquence / FER
# ============================================

from fer_module import analyser_visage, analyser_visage_depuis_scenario
from ser_module import analyser_voix_depuis_scenario
from fusion_module import fusionner_modalites
from decision_engine import preparer_decision
from llm_module import generer_contenu_adaptatif
from temporal_rules import AUCUNE_URGENCE
from daisee_utils import etat_depuis_ligne


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
    mode="simulation",
    image_path=None,
    temporal=None,
    generer_contenu=True,
    utiliser_audio=None,
):
    """
    Analyse un clip DAiSEE (simulation ou FER image) + voix IEMOCAP + décision + LLM.
    """
    pattern = str(scenario.get("pattern", "stable"))
    scores = scores_depuis_scenario(scenario)
    duree_etat = temporal["duree_etat"] if temporal else 1

    if mode == "fer" and image_path:
        result_visage = analyser_visage(image_path)
        result_visage["modalite"] = "visage_fer"
    else:
        result_visage = analyser_visage_depuis_scenario(scenario)

    etat_dominant, _, _ = etat_depuis_ligne(scenario)
    result_voix = analyser_voix_depuis_scenario(
        scenario_id=scenario_id,
        etat_dominant=etat_dominant,
        utiliser_audio=utiliser_audio,
    )
    fusion = fusionner_modalites(result_visage, result_voix)

    decision = preparer_decision(
        fusion=fusion,
        boredom=scores["boredom"],
        confusion=scores["confusion"],
        engagement=scores["engagement"],
        frustration=scores["frustration"],
        pattern=pattern,
        duree_etat=duree_etat,
        temporal=temporal,
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
            pattern_detecte=pattern,
            moment_cours=moment_cours,
            frustration_niveau=scores["frustration"],
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
        "pattern": pattern,
        "mode": mode,
        "scores": scores,
        "result_visage": result_visage,
        "result_voix": result_voix,
        "fusion": fusion,
        "decision": decision,
        "resultat_llm": resultat_llm,
        "temporal": temporal or decision.get("temporal"),
    }


def formater_reponse_api(analyse, scenario_id=None, session_id=None, clip_index=None):
    """Formate la sortie pipeline pour l'API JSON."""
    decision = analyse["decision"]
    fusion = analyse["fusion"]
    llm = analyse["resultat_llm"] or {}

    return {
        "scenario_id": scenario_id,
        "session_id": session_id,
        "clip_index": clip_index,
        "clip_id": analyse["clip_id"],
        "pattern": analyse["pattern"],
        "mode": analyse["mode"],
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
        "voix": {
            "source": analyse.get("result_voix", {}).get("source"),
            "utterance_id": analyse.get("result_voix", {}).get("utterance_id"),
            "emotion": analyse.get("result_voix", {}).get("emotion"),
            "etat": analyse.get("result_voix", {}).get("etat"),
            "wav_path": analyse.get("result_voix", {}).get("wav_path"),
            "features_acoustiques": analyse.get("result_voix", {}).get("features_acoustiques"),
            "appariement": analyse.get("result_voix", {}).get("appariement"),
        },
        "decision": {
            "scores_daisee": decision["scores_daisee"],
            "fusion_meta": decision["fusion"],
            "temporal": decision.get("temporal"),
            "declencher": decision.get("declencher", True),
        },
        "rag_utilise": llm.get("rag_utilise", False),
        "nb_docs_rag": llm.get("nb_docs_rag", 0),
        "sources_rag": llm.get("sources_rag", []),
        "rag_erreur": llm.get("rag_erreur"),
    }
