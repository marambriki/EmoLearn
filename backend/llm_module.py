# ============================================
# backend/llm_module.py
# ============================================

from groq import Groq
from dotenv import load_dotenv
import os

load_dotenv()
client = Groq(api_key=os.getenv('GROQ_API_KEY'))

instructions_specifiques = {
    "A": """- Reformuler le concept avec des mots plus simples
- Utiliser une analogie concrète de la vie quotidienne
- Décomposer en étapes claires
- Terminer par une question de vérification""",
    "B": """- Partir d'une situation réelle et actuelle
- Montrer comment le concept résout un problème concret
- Inclure des données/chiffres réels si possible
- Terminer par une question engageante""",
    "C": """- Poser un problème complexe nécessitant le concept
- Ajouter une contrainte supplémentaire
- Demander de justifier la solution
- Proposer une extension bonus""",
    "D": """- Valider la difficulté de manière empathique
- Rappeler les progrès déjà réalisés
- Proposer une perspective positive
- Terminer par une phrase motivante"""
}

config_types = {
    "A": {"longueur": "200 mots", "format": "Paragraphe avec étapes", "ton": "Encourageant et rassurant"},
    "B": {"longueur": "200 mots", "format": "Paragraphe engageant",   "ton": "Dynamique et stimulant"},
    "C": {"longueur": "250 mots", "format": "Exercice structuré",     "ton": "Challengeant et dynamique"},
    "D": {"longueur": "150 mots", "format": "Message motivant",       "ton": "Empathique et bienveillant"}
}

def selectionner_type_contenu(etat, niveau):
    if etat == 'Confusion'   and niveau >= 2: return 'A', 'Clarifier'
    if etat == 'Boredom'     and niveau >= 2: return 'C', 'Stimuler'
    if etat == 'Frustration' and niveau >= 2: return 'D', 'Encourager'
    if etat == 'Engagement'  and niveau <= 1: return 'B', 'Réengager'
    return 'D', 'Renforcer'

def generer_contenu_adaptatif(
    etat,
    niveau,
    concept,
    sujet_cours          = "Informatique",
    module_actuel        = "Module en cours",
    niveau_apprenant     = "Intermédiaire",
    objectif_pedagogique = "Comprendre le concept",
    duree_etat           = 2,
    pattern_detecte      = "stable",
    moment_cours         = "Milieu du module"
):
    type_contenu, objectif_adaptation = selectionner_type_contenu(etat, niveau)
    config = config_types.get(type_contenu, config_types["D"])

    prompt = f"""
[CONTEXTE PÉDAGOGIQUE]
Sujet du cours        : {sujet_cours}
Module actuel         : {module_actuel}
Concept en cours      : {concept}
Niveau apprenant      : {niveau_apprenant}
Objectif pédagogique  : {objectif_pedagogique}

[ÉTAT D'ENGAGEMENT DE L'APPRENANT]
État dominant         : {etat}
Niveau d'intensité    : {niveau}/3
Durée de l'état       : {duree_etat} clips ({duree_etat*10} secondes)
Pattern détecté       : {pattern_detecte}
Moment du cours       : {moment_cours}

[TÂCHE DE GÉNÉRATION]
Type de contenu       : Type {type_contenu}
Objectif adaptation   : {objectif_adaptation}

[CONTRAINTES]
- Longueur            : {config['longueur']}
- Format              : {config['format']}
- Ton                 : {config['ton']}
- Niveau de langage   : Adapté au niveau {niveau_apprenant}

[INSTRUCTIONS SPÉCIFIQUES]
{instructions_specifiques.get(type_contenu, instructions_specifiques["D"])}

Génère maintenant le contenu adaptatif en français.
"""

    response = client.chat.completions.create(
        model    = "llama-3.1-8b-instant",
        messages = [
            {
                "role"   : "system",
                "content": "Tu es un expert en pédagogie. Tu génères du contenu adaptatif en français."
            },
            {
                "role"   : "user",
                "content": prompt
            }
        ],
        max_tokens  = 600,
        temperature = 0.7
    )

    contenu = response.choices[0].message.content
    return {
        'etat'        : etat,
        'niveau'      : niveau,
        'type_contenu': type_contenu,
        'objectif'    : objectif_adaptation,
        'concept'     : concept,
        'contenu'     : contenu,
        'nb_mots'     : len(contenu.split()),
        'ton'         : config['ton'],
        'format'      : config['format']
    }

print("✅ Module LLM OS4 chargé !")