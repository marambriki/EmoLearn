# ============================================
# backend/fusion_module.py
# Module de fusion multimodale avec gestion
# des conflits DAiSEE + IEMOCAP
# ============================================

# ===== POIDS DE FUSION =====
POIDS_VISAGE = 0.60
POIDS_VOIX   = 0.40

# ===== PRIORITÉ PÉDAGOGIQUE =====
# Plus le score est élevé, plus l'état est prioritaire
# En cas de conflit, on prend l'état le plus urgent
priorite_pedagogique = {
    'Frustration': 4,  # Priorité maximale
    'Confusion'  : 3,  # Priorité haute
    'Boredom'    : 2,  # Priorité moyenne
    'Engagement' : 1   # Priorité basse
}

# ===== SEUIL DE CONFLIT =====
# Si la différence de niveaux > SEUIL → conflit détecté
SEUIL_CONFLIT = 1

def detecter_conflit(result_visage, result_voix):
    """
    Détecte si les deux modalités sont en conflit
    Retourne : (en_conflit, type_conflit, severite)
    """
    if not result_voix or result_voix.get('erreur'):
        return False, "pas_de_voix", 0

    etat_visage  = result_visage['etat']
    etat_voix    = result_voix['etat']
    niveau_visage = result_visage['niveau']
    niveau_voix   = result_voix['niveau']

    # Cas 1 — Même état
    if etat_visage == etat_voix:
        difference = abs(niveau_visage - niveau_voix)
        if difference == 0:
            return False, "accord_parfait", 0
        elif difference <= SEUIL_CONFLIT:
            return False, "accord_partiel", difference
        else:
            return True, "conflit_niveau", difference

    # Cas 2 — États différents
    else:
        severite = abs(
            priorite_pedagogique[etat_visage] -
            priorite_pedagogique[etat_voix]
        )
        if severite <= 1:
            return True, "conflit_mineur", severite
        elif severite == 2:
            return True, "conflit_modere", severite
        else:
            return True, "conflit_majeur", severite


def resoudre_conflit(result_visage, result_voix, type_conflit, severite):
    """
    Résout le conflit selon la stratégie pédagogique
    3 stratégies selon la sévérité du conflit
    """

    etat_visage   = result_visage['etat']
    etat_voix     = result_voix['etat']
    niveau_visage = result_visage['niveau']
    niveau_voix   = result_voix['niveau']
    score_visage  = result_visage['score']
    score_voix    = result_voix['score']

    # ===== STRATÉGIE 1 — Conflit de niveau (même état) =====
    if type_conflit == "conflit_niveau":
        # Moyenne pondérée des niveaux
        niveau_final = round(
            niveau_visage * POIDS_VISAGE +
            niveau_voix   * POIDS_VOIX
        )
        return {
            'etat'      : etat_visage,
            'niveau'    : niveau_final,
            'strategie' : 'moyenne_ponderee',
            'explication': f"Même état ({etat_visage}) mais niveaux différents ({niveau_visage} vs {niveau_voix}) → moyenne pondérée = {niveau_final}"
        }

    # ===== STRATÉGIE 2 — Conflit mineur (états proches) =====
    elif type_conflit == "conflit_mineur":
        # Prendre l'état avec le score de confiance le plus élevé
        if score_visage >= score_voix:
            etat_final   = etat_visage
            niveau_final = niveau_visage
            raison = f"Score visage ({score_visage:.2f}) > Score voix ({score_voix:.2f})"
        else:
            etat_final   = etat_voix
            niveau_final = niveau_voix
            raison = f"Score voix ({score_voix:.2f}) > Score visage ({score_visage:.2f})"

        return {
            'etat'      : etat_final,
            'niveau'    : niveau_final,
            'strategie' : 'confiance_maximale',
            'explication': f"Conflit mineur entre {etat_visage} et {etat_voix} → {raison}"
        }

    # ===== STRATÉGIE 3 — Conflit modéré ou majeur =====
    else:
        # Priorité pédagogique — état le plus urgent
        priorite_visage = priorite_pedagogique[etat_visage]
        priorite_voix   = priorite_pedagogique[etat_voix]

        if priorite_visage >= priorite_voix:
            etat_final   = etat_visage
            niveau_final = niveau_visage
            raison = f"{etat_visage} (priorité {priorite_visage}) > {etat_voix} (priorité {priorite_voix})"
        else:
            etat_final   = etat_voix
            niveau_final = niveau_voix
            raison = f"{etat_voix} (priorité {priorite_voix}) > {etat_visage} (priorité {priorite_visage})"

        return {
            'etat'      : etat_final,
            'niveau'    : niveau_final,
            'strategie' : 'priorite_pedagogique',
            'explication': f"Conflit {type_conflit} → priorité pédagogique : {raison}"
        }


def fusionner_modalites(result_visage, result_voix):
    """
    Fonction principale de fusion multimodale
    Gère tous les cas : accord, conflit mineur, conflit majeur
    """

    # ===== CAS 1 — Voix non disponible =====
    if not result_voix or result_voix.get('erreur'):
        return {
            'etat'               : result_visage['etat'],
            'emotion'            : result_visage['emotion'],
            'niveau'             : result_visage['niveau'],
            'score_global'       : round(result_visage['score'], 3),
            'modalite_dominante' : 'visage',
            'conflit'            : False,
            'type_conflit'       : 'pas_de_voix',
            'strategie_fusion'   : 'unimodal',
            'explication_fusion' : 'Mode unimodal — voix non disponible',
            'details'            : {
                'visage': {
                    'etat'   : result_visage['etat'],
                    'niveau' : result_visage['niveau'],
                    'score'  : round(result_visage['score'], 3),
                    'poids'  : 1.0
                },
                'voix': None
            },
            'multimodal': False
        }

    # ===== CAS 2 — Les deux modalités disponibles =====
    en_conflit, type_conflit, severite = detecter_conflit(
        result_visage, result_voix
    )

    if not en_conflit:
        # ===== Pas de conflit — fusion normale =====
        niveau_fusionne = round(
            result_visage['niveau'] * POIDS_VISAGE +
            result_voix['niveau']   * POIDS_VOIX
        )
        score_global = (
            result_visage['score'] * POIDS_VISAGE +
            result_voix['score']   * POIDS_VOIX
        )

        if result_visage['score'] >= result_voix['score']:
            etat_final    = result_visage['etat']
            emotion_finale = result_visage['emotion']
            modalite_dom  = 'visage'
        else:
            etat_final    = result_voix['etat']
            emotion_finale = result_voix['emotion']
            modalite_dom  = 'voix'

        explication = f"Accord {type_conflit} → fusion pondérée normale (60% visage + 40% voix)"

        return {
            'etat'               : etat_final,
            'emotion'            : emotion_finale,
            'niveau'             : niveau_fusionne,
            'score_global'       : round(score_global, 3),
            'modalite_dominante' : modalite_dom,
            'conflit'            : False,
            'type_conflit'       : type_conflit,
            'strategie_fusion'   : 'moyenne_ponderee',
            'explication_fusion' : explication,
            'details'            : {
                'visage': {
                    'etat'   : result_visage['etat'],
                    'niveau' : result_visage['niveau'],
                    'score'  : round(result_visage['score'], 3),
                    'poids'  : POIDS_VISAGE
                },
                'voix': {
                    'etat'   : result_voix['etat'],
                    'niveau' : result_voix['niveau'],
                    'score'  : round(result_voix['score'], 3),
                    'poids'  : POIDS_VOIX
                }
            },
            'multimodal': True
        }

    else:
        # ===== Conflit détecté — résolution =====
        resolution = resoudre_conflit(
            result_visage, result_voix,
            type_conflit, severite
        )

        score_global = (
            result_visage['score'] * POIDS_VISAGE +
            result_voix['score']   * POIDS_VOIX
        )

        return {
            'etat'               : resolution['etat'],
            'emotion'            : resolution['etat'].lower(),
            'niveau'             : resolution['niveau'],
            'score_global'       : round(score_global, 3),
            'modalite_dominante' : 'resolution_conflit',
            'conflit'            : True,
            'type_conflit'       : type_conflit,
            'strategie_fusion'   : resolution['strategie'],
            'explication_fusion' : resolution['explication'],
            'details'            : {
                'visage': {
                    'etat'   : result_visage['etat'],
                    'niveau' : result_visage['niveau'],
                    'score'  : round(result_visage['score'], 3),
                    'poids'  : POIDS_VISAGE
                },
                'voix': {
                    'etat'   : result_voix['etat'],
                    'niveau' : result_voix['niveau'],
                    'score'  : round(result_voix['score'], 3),
                    'poids'  : POIDS_VOIX
                }
            },
            'multimodal': True
        }


def expliquer_fusion(resultat_fusion):
    """
    Génère une explication lisible de la fusion
    """
    conflit_emoji = "⚠️" if resultat_fusion['conflit'] else "✅"

    if resultat_fusion['multimodal']:
        details = resultat_fusion['details']
        return f"""
=== FUSION MULTIMODALE ===
Visage (DAiSEE)  : {details['visage']['etat']} niveau {details['visage']['niveau']}/3 (score {details['visage']['score']*100:.1f}%)
Voix   (IEMOCAP) : {details['voix']['etat']} niveau {details['voix']['niveau']}/3 (score {details['voix']['score']*100:.1f}%)

{conflit_emoji} Conflit : {'OUI — ' + resultat_fusion['type_conflit'] if resultat_fusion['conflit'] else 'NON'}
Stratégie : {resultat_fusion['strategie_fusion']}
Explication : {resultat_fusion['explication_fusion']}

→ État final : {resultat_fusion['etat']} niveau {resultat_fusion['niveau']}/3
"""
    else:
        return f"""
=== MODE UNIMODAL ===
→ État final : {resultat_fusion['etat']} niveau {resultat_fusion['niveau']}/3
→ Raison : {resultat_fusion['explication_fusion']}
"""

print("✅ Module de fusion multimodale avec gestion des conflits chargé !")