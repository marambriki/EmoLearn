# ============================================
# EmoLearn — Moteur de décision unifié
# Source unique : fusion multimodale + règles DAiSEE → stratégie + template
# ============================================

from adaptation import selectionner_strategie, get_description_strategie
from prompts.templates import get_meta_template


DESCRIPTIONS_TEMPLATE = {
    "A1": "Explication alternative — Confusion modérée",
    "A2": "Micro-explication — Confusion intense",
    "A3": "Aide de déblocage — Confusion + frustration",
    "A4": "Explication de fond — Confusion récurrente",
    "A5": "Explication narrative — Concept abstrait",
    "B1": "Exemple concret — Ennui modéré",
    "B2": "Cas pratique — Engagement faible",
    "B3": "Mise en pratique — Ennui après théorie",
    "B4": "Application réelle — Contextualisation",
    "B5": "Exemple interdisciplinaire",
    "C1": "Exercice avancé — Ennui élevé (maîtrise)",
    "C2": "Exercice de consolidation",
    "C3": "Exercice créatif",
    "C4": "Mini-projet",
    "C5": "Exercice comparatif",
    "D1": "Encouragement — Frustration modérée",
    "D2": "Soutien — Frustration élevée",
    "D3": "Remotivation — Découragement",
    "D4": "Renforcement positif — Bon engagement",
    "D5": "Félicitation — Après déblocage",
}


def selectionner_template_id(
    type_contenu,
    etat,
    niveau,
    boredom,
    confusion,
    engagement,
    frustration,
    pattern="stable",
):
    """
    Choisit un template parmi les 20 selon le type de contenu et le contexte DAiSEE.
    """
    if type_contenu == "A+D":
        return "A3"
    if type_contenu == "D+A":
        return "D2" if frustration >= 3 else "D1"
    if type_contenu == "A":
        if niveau >= 3:
            return "A2"
        if confusion >= 2 and pattern in ("confusion", "critique"):
            return "A1"
        return "A5"
    if type_contenu == "B":
        if engagement == 0 or pattern == "engagement_faible":
            return "B2"
        if boredom >= 2 and pattern == "ennui_eleve":
            return "B3"
        return "B1"
    if type_contenu == "C":
        if boredom >= 3 or pattern == "ennui_eleve":
            return "C1"
        if engagement >= 2:
            return "C2"
        return "C1"
    if type_contenu == "D":
        if etat == "Frustration" and niveau >= 3:
            return "D2"
        if etat == "Engagement" or engagement >= 2:
            return "D4"
        return "D1"
    if type_contenu == "aucune":
        return "D4"
    return "A1"


def preparer_decision(
    fusion,
    boredom,
    confusion,
    engagement,
    frustration,
    pattern="stable",
    duree_etat=2,
    temporal=None,
):
    """
    Construit la décision pédagogique complète à partir de la fusion et des scores DAiSEE.

    Returns:
        dict avec etat, niveau, strategie, type_contenu, template_id, objectif, description, meta
    """
    etat = fusion["etat"]
    niveau = int(fusion["niveau"])

    declencher = True
    if temporal is not None:
        declencher = bool(temporal.get("declencher", True))
        duree_etat = int(temporal.get("duree_etat", duree_etat))

    _emotion_label, _niveau_raw, strategie, type_contenu = selectionner_strategie(
        boredom, confusion, engagement, frustration
    )

    if not declencher:
        type_contenu = "aucune"
        strategie = "Continuation normale — pas d'intervention"
        template_id = "D4"
        meta = get_meta_template(template_id)
        return {
            "etat": etat,
            "niveau": niveau,
            "strategie": strategie,
            "type_contenu": "aucune",
            "type_contenu_regle": "aucune",
            "template_id": template_id,
            "objectif": meta["objectif"],
            "description": "Pas d'adaptation — engagement stable",
            "longueur": meta["longueur"],
            "format": meta["format"],
            "ton": meta["ton"],
            "duree_etat": duree_etat,
            "declencher": False,
            "temporal": temporal,
            "scores_daisee": {
                "boredom": boredom,
                "confusion": confusion,
                "engagement": engagement,
                "frustration": frustration,
            },
            "fusion": {
                "multimodal": fusion.get("multimodal", False),
                "conflit": fusion.get("conflit", False),
                "type_conflit": fusion.get("type_conflit"),
                "strategie_fusion": fusion.get("strategie_fusion"),
            },
        }

    template_id = selectionner_template_id(
        type_contenu=type_contenu,
        etat=etat,
        niveau=niveau,
        boredom=boredom,
        confusion=confusion,
        engagement=engagement,
        frustration=frustration,
        pattern=pattern,
    )

    meta = get_meta_template(template_id)
    description = DESCRIPTIONS_TEMPLATE.get(
        template_id,
        get_description_strategie(meta["type"]),
    )

    return {
        "etat": etat,
        "niveau": niveau,
        "strategie": strategie,
        "type_contenu": meta["type"],
        "type_contenu_regle": type_contenu,
        "template_id": template_id,
        "objectif": meta["objectif"],
        "description": description,
        "longueur": meta["longueur"],
        "format": meta["format"],
        "ton": meta["ton"],
        "duree_etat": duree_etat,
        "declencher": declencher,
        "temporal": temporal,
        "scores_daisee": {
            "boredom": boredom,
            "confusion": confusion,
            "engagement": engagement,
            "frustration": frustration,
        },
        "fusion": {
            "multimodal": fusion.get("multimodal", False),
            "conflit": fusion.get("conflit", False),
            "type_conflit": fusion.get("type_conflit"),
            "strategie_fusion": fusion.get("strategie_fusion"),
        },
    }
