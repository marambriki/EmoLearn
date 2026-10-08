# ============================================
# EmoLearn — Moteur de décision unifié
# Source unique : fusion multimodale + règles DAiSEE → stratégie + template
# ============================================

from daisee_utils import PATTERN_PAR_ETAT
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


def expliquer_strategie(
    boredom,
    confusion,
    engagement,
    frustration,
    etat=None,
    niveau=None,
):
    """Branche réellement empruntée, avec un code stable pour le journal.

    Si ``etat`` (fusion) est fourni, la famille A/B/C/D suit cet argmax.
    Les scores 0–3 (proba × 3) restent au journal : trop souvent tous à 0–1
    pour piloter les seuils ≥ 2.
    """
    if confusion >= 3 and frustration >= 2:
        return {
            "cle": "confusion+frustration",
            "niveau": 3,
            "strategie": "Aide de déblocage immédiate",
            "type": "A+D",
            "code": "STRAT_CONF3_FRU2",
            "pourquoi": (
                f"Confusion {confusion}/3 ≥ 3 et frustration {frustration}/3 ≥ 2"
            ),
        }

    niv = int(niveau) if niveau is not None else None

    if etat == "Frustration":
        return {
            "cle": "frustration",
            "niveau": niv if niv is not None else frustration,
            "strategie": "Encouragement + simplification",
            "type": "D+A",
            "code": "STRAT_FRU_GE_2" if (niv or frustration) >= 2 else "STRAT_FUSION_FRU",
            "pourquoi": f"État fusionné Frustration (niveau {(niv if niv is not None else frustration)}/3)",
        }
    if etat == "Confusion":
        return {
            "cle": "confusion",
            "niveau": niv if niv is not None else confusion,
            "strategie": "Explication alternative",
            "type": "A",
            "code": "STRAT_CONF_GE_2" if (niv or confusion) >= 2 else "STRAT_FUSION_CONF",
            "pourquoi": f"État fusionné Confusion (niveau {(niv if niv is not None else confusion)}/3)",
        }
    if etat == "Boredom":
        return {
            "cle": "boredom",
            "niveau": niv if niv is not None else boredom,
            "strategie": "Exemple concret — relance (ennui)",
            "type": "B",
            "code": "STRAT_FUSION_BOR",
            "pourquoi": f"État fusionné Ennui (niveau {(niv if niv is not None else boredom)}/3)",
        }
    if etat == "Engagement":
        if niv is not None and niv <= 1:
            return {
                "cle": "engagement",
                "niveau": niv,
                "strategie": "Exemple concret stimulant",
                "type": "B",
                "code": "STRAT_ENG_LE_1",
                "pourquoi": f"État fusionné Engagement, niveau {niv}/3 ≤ 1",
            }
        return {
            "cle": "engagement",
            "niveau": niv if niv is not None else engagement,
            "strategie": "Renforcement positif",
            "type": "D",
            "code": "STRAT_ENG_GE_2",
            "pourquoi": f"État fusionné Engagement (niveau {(niv if niv is not None else engagement)}/3)",
        }

    if engagement == 0:
        return {
            "cle": "engagement",
            "niveau": 0,
            "strategie": "Réengagement urgent",
            "type": "B",
            "code": "STRAT_ENG_0",
            "pourquoi": "Engagement = 0",
        }
    if frustration >= 2:
        return {
            "cle": "frustration",
            "niveau": frustration,
            "strategie": "Encouragement + simplification",
            "type": "D+A",
            "code": "STRAT_FRU_GE_2",
            "pourquoi": f"Frustration {frustration}/3 ≥ 2",
        }
    if confusion >= 2:
        return {
            "cle": "confusion",
            "niveau": confusion,
            "strategie": "Explication alternative",
            "type": "A",
            "code": "STRAT_CONF_GE_2",
            "pourquoi": f"Confusion {confusion}/3 ≥ 2",
        }
    if boredom >= 2:
        return {
            "cle": "boredom",
            "niveau": boredom,
            "strategie": "Exercice avancé",
            "type": "C",
            "code": "STRAT_BOR_GE_2",
            "pourquoi": f"Ennui {boredom}/3 ≥ 2",
        }
    if engagement <= 1:
        return {
            "cle": "engagement",
            "niveau": engagement,
            "strategie": "Exemple concret stimulant",
            "type": "B",
            "code": "STRAT_ENG_LE_1",
            "pourquoi": f"Engagement {engagement}/3 ≤ 1",
        }
    if boredom == 1:
        return {
            "cle": "boredom",
            "niveau": boredom,
            "strategie": "Application pratique",
            "type": "B",
            "code": "STRAT_BOR_1",
            "pourquoi": "Ennui = 1",
        }
    if engagement >= 2:
        return {
            "cle": "engagement",
            "niveau": engagement,
            "strategie": "Renforcement positif",
            "type": "D",
            "code": "STRAT_ENG_GE_2",
            "pourquoi": f"Engagement {engagement}/3 ≥ 2",
        }
    return {
        "cle": "neutre",
        "niveau": 0,
        "strategie": "Continuation normale",
        "type": "aucune",
        "code": "STRAT_NEUTRE",
        "pourquoi": "Aucun seuil pédagogique atteint",
    }


def selectionner_strategie(boredom, confusion, engagement, frustration):
    branche = expliquer_strategie(boredom, confusion, engagement, frustration)
    return (branche["cle"], branche["niveau"], branche["strategie"], branche["type"])


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


def expliquer_template(
    type_contenu,
    etat,
    niveau,
    boredom,
    confusion,
    engagement,
    frustration,
    pattern="stable",
):
    """Patron retenu et la condition qui l'a choisi."""
    if type_contenu == "A+D":
        return (
            "A3",
            "TPL_A3",
            "Type A+D : confusion et frustration ensemble",
        )
    if type_contenu == "D+A":
        if niveau >= 3 or frustration >= 3:
            return ("D2", "TPL_D2", f"Frustration, niveau {niveau}/3")
        return ("D1", "TPL_D1", f"Frustration, niveau {niveau}/3")
    if type_contenu == "A":
        if niveau >= 3:
            return ("A2", "TPL_A2", f"Confusion intense, niveau {niveau}/3")
        if etat == "Confusion" or (
            confusion >= 2 and pattern in ("confusion", "critique")
        ):
            return (
                "A1",
                "TPL_A1",
                f"Confusion fusionnée, niveau {niveau}/3",
            )
        return ("A5", "TPL_A5", f"Confusion, pattern {pattern}")
    if type_contenu == "B":
        if etat == "Boredom":
            if niveau >= 2:
                return ("B3", "TPL_B3", f"Ennui fusionné, niveau {niveau}/3")
            return ("B1", "TPL_B1", f"Ennui fusionné, niveau {niveau}/3")
        if engagement == 0 or pattern == "engagement_faible":
            return (
                "B2",
                "TPL_B2",
                f"Engagement {engagement}/3, pattern {pattern}",
            )
        if boredom >= 2 and pattern == "ennui_eleve":
            return ("B3", "TPL_B3", f"Ennui {boredom}/3, pattern ennui_eleve")
        return ("B1", "TPL_B1", f"Pattern {pattern}")
    if type_contenu == "C":
        if boredom >= 3 or pattern == "ennui_eleve":
            return (
                "C1",
                "TPL_C1",
                f"Ennui {boredom}/3, pattern {pattern}",
            )
        if engagement >= 2:
            return ("C2", "TPL_C2", f"Engagement {engagement}/3 ≥ 2")
        return ("C1", "TPL_C1", f"Pattern {pattern}")
    if type_contenu == "D":
        if etat == "Frustration" and niveau >= 3:
            return ("D2", "TPL_D2", f"Frustration, niveau {niveau}/3 ≥ 3")
        if etat == "Engagement" or engagement >= 2:
            return ("D4", "TPL_D4", f"Engagement {engagement}/3")
        return ("D1", "TPL_D1", f"État {etat}, niveau {niveau}/3")
    if type_contenu == "aucune":
        return ("D4", "TPL_AUCUNE", "Pas d'adaptation : patron de repli non servi")
    return ("A1", "TPL_DEFAUT", f"Type {type_contenu} sans branche dédiée")


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
    template_id, _code, _pourquoi = expliquer_template(
        type_contenu,
        etat,
        niveau,
        boredom,
        confusion,
        engagement,
        frustration,
        pattern,
    )
    return template_id


def preparer_decision(
    fusion,
    duree_etat=2,
    temporal=None,
    scores_bruts=None,
):
    """
    Construit la décision pédagogique à partir de l'état fusionné.

    La cascade A/B/C/D suit l'état fusionné (argmax), pas les scores
    recollés proba×3 (souvent 0–1 → faux « engagement faible »).
    Les scores projetés restent dans le journal. Pas les annotations CSV.
    """
    etat = fusion["etat"]
    niveau = int(fusion["niveau"])
    scores = scores_depuis_fusion(fusion)
    boredom = scores["boredom"]
    confusion = scores["confusion"]
    engagement = scores["engagement"]
    frustration = scores["frustration"]
    pattern = PATTERN_PAR_ETAT.get(etat, "stable")

    declencher = True
    if temporal is not None:
        declencher = bool(temporal.get("declencher", True))
        duree_etat = int(temporal.get("duree_etat", duree_etat))

    branche = expliquer_strategie(
        boredom,
        confusion,
        engagement,
        frustration,
        etat=etat,
        niveau=niveau,
    )
    strategie = branche["strategie"]
    type_contenu = branche["type"]

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
            "trace": {
                "regle_code": branche["code"],
                "regle_pourquoi": branche["pourquoi"],
                "regle_ecrasee": True,
                "template_code": "TPL_AUCUNE",
                "template_pourquoi": (
                    f"Règle temporelle {temporal.get('regle') if temporal else '—'} "
                    ": pas d'intervention"
                ),
            },
            "fusion": {
                "multimodal": fusion.get("multimodal", False),
                "conflit": fusion.get("conflit", False),
                "type_conflit": fusion.get("type_conflit"),
                "strategie_fusion": fusion.get("strategie_fusion"),
            },
        }

    template_id, template_code, template_pourquoi = expliquer_template(
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
        "trace": {
            "regle_code": branche["code"],
            "regle_pourquoi": branche["pourquoi"],
            "regle_ecrasee": False,
            "template_code": template_code,
            "template_pourquoi": template_pourquoi,
        },
        "fusion": {
            "multimodal": fusion.get("multimodal", False),
            "conflit": fusion.get("conflit", False),
            "type_conflit": fusion.get("type_conflit"),
            "strategie_fusion": fusion.get("strategie_fusion"),
        },
    }
