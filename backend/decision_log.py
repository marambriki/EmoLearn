# ============================================
# Journal de traçabilité des décisions (soutenance / debug)
# ============================================

import json
import os
from datetime import datetime, timezone

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
    """
    Ajoute une ligne JSON au journal decisions.jsonl.
    """
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
