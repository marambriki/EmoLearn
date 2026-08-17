# ============================================
# EmoLearn — Moteur de décision unifié
# Source unique : fusion multimodale + règles DAiSEE → stratégie + template
# ============================================

from daisee_utils import pattern_depuis_scores
from prompts.templates import get_meta_template

ETAT_VERS_CLE = {
    "Boredom": "boredom",
    "Confusion": "confusion",
    "Engagement": "engagement",
    "Frustration": "frustration",
}


def scores_depuis_fusion(fusion):
    """
    Dérive les 4 scores DAiSEE à partir de l'état fusionné.

    Si la fusion expose une distribution ``probs_etats`` (vote pondéré),
    chaque probabilité est projetée sur l'échelle 0–3. Sinon, l'état retenu
    reçoit le niveau fusionné et les autres dimensions valent 0.
    """
    probs = fusion.get("probs_etats")
    if probs:
        return {
            "boredom": int(round(float(probs.get("Boredom", 0)) * 3)),
            "confusion": int(round(float(probs.get("Confusion", 0)) * 3)),
            "engagement": int(round(float(probs.get("Engagement", 0)) * 3)),
            "frustration": int(round(float(probs.get("Frustration", 0)) * 3)),
        }

    etat = fusion["etat"]
    niveau = int(fusion["niveau"])
    scores = {cle: 0 for cle in ETAT_VERS_CLE.values()}
    scores[ETAT_VERS_CLE[etat]] = niveau
    return scores


def selectionner_strategie(boredom, confusion, engagement, frustration):
    if confusion >= 3 and frustration >= 2:
        return ("confusion+frustration", 3, "Aide de déblocage immédiate", "A+D")
    if engagement == 0:
        return ("engagement", 0, "Réengagement urgent", "B")
    if frustration >= 2:
        return ("frustration", frustration, "Encouragement + simplification", "D+A")
    if confusion >= 2:
        return ("confusion", confusion, "Explication alternative", "A")
    if boredom >= 2:
        return ("boredom", boredom, "Exercice avancé", "C")
    if engagement <= 1:
        return ("engagement", engagement, "Exemple concret stimulant", "B")
    if boredom == 1:
        return ("boredom", boredom, "Application pratique", "B")
    if engagement >= 2:
        return ("engagement", engagement, "Renforcement positif", "D")
    return ("neutre", 0, "Continuation normale", "aucune")


def get_description_strategie(type_contenu):
    descriptions = {
        "A": "Explication alternative — Reformulation du concept",
        "B": "Exemple concret — Cas pratique et application réelle",
        "C": "Exercice progressif — Challenge adapté au niveau",
        "D": "Encouragement motivationnel — Support émotionnel",
        "A+D": "Aide de déblocage — Explication + Encouragement",
        "D+A": "Soutien guidé — Encouragement + Simplification",
        "aucune": "Pas d'adaptation nécessaire",
    }
    return descriptions.get(type_contenu, "Type inconnu")


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
    duree_etat=2,
    temporal=None,
    scores_bruts=None,
):
    """
    Construit la décision pédagogique à partir de l'état fusionné.

    La cascade A/B/C/D et le choix de template s'appuient sur les scores
    dérivés de la fusion (aligné chapitre 4 du mémoire), pas sur les
    annotations DAiSEE brutes du scénario.
    """
    etat = fusion["etat"]
    niveau = int(fusion["niveau"])
    scores = scores_depuis_fusion(fusion)
    boredom = scores["boredom"]
    confusion = scores["confusion"]
    engagement = scores["engagement"]
    frustration = scores["frustration"]
    pattern = pattern_depuis_scores(boredom, confusion, engagement, frustration)

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
            "scores_daisee": scores,
            "scores_bruts": scores_bruts,
            "pattern": pattern,
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
        "scores_daisee": scores,
        "scores_bruts": scores_bruts,
        "pattern": pattern,
        "fusion": {
            "multimodal": fusion.get("multimodal", False),
            "conflit": fusion.get("conflit", False),
            "type_conflit": fusion.get("type_conflit"),
            "strategie_fusion": fusion.get("strategie_fusion"),
        },
    }
