# ============================================
# backend/llm_module.py
# OS4 — Module LLM + RAG + 20 templates
# ============================================

import os
import time

from dotenv import load_dotenv
from groq import Groq, RateLimitError

load_dotenv()
client = Groq(api_key=os.getenv('GROQ_API_KEY'))

from prompts.templates import get_meta_template, construire_instructions_template


def generer_contenu_adaptatif(
    etat,
    niveau,
    concept,
    template_id,
    objectif_adaptation=None,
    sujet_cours="Informatique",
    module_actuel="Module en cours",
    niveau_apprenant="Intermédiaire",
    objectif_pedagogique="Comprendre le concept",
    duree_etat=2,
    pattern_detecte="stable",
    moment_cours="Milieu du module",
    frustration_niveau=None,
    utiliser_rag=True,
):
    """
    Génère du contenu adaptatif via le template_id (A1–D5) et le contexte RAG.
    """
    meta = get_meta_template(template_id)
    type_contenu = meta["type"]
    objectif = objectif_adaptation or meta["objectif"]

    contexte_rag = ""
    documents_trouves = []
    rag_erreur = None
    if utiliser_rag:
        try:
            from rag_module import rechercher_documents, construire_contexte_rag

            query = f"Expliquer {concept} pour étudiant avec {etat}"
            documents_trouves = rechercher_documents(
                query=query,
                concept=concept,
                etat=etat,
                n_results=3,
            )
            contexte_rag = construire_contexte_rag(
                documents_trouves,
                concept=concept,
                etat=etat,
            )
            print(f"RAG : {len(documents_trouves)} documents pertinents trouvés")
        except Exception as exc:
            rag_erreur = str(exc)
            print(f"⚠️ RAG indisponible : {exc}")

    instructions_template = construire_instructions_template(
        template_id=template_id,
        concept=concept,
        niveau=niveau,
        duree=duree_etat,
        module=module_actuel,
        frustration_niveau=frustration_niveau,
    )

    prompt = f"""
{contexte_rag}

[CONTEXTE PÉDAGOGIQUE]
Sujet du cours        : {sujet_cours}
Module actuel         : {module_actuel}
Concept en cours      : {concept}
Niveau apprenant      : {niveau_apprenant}
Objectif pédagogique  : {objectif_pedagogique}

[ÉTAT D'ENGAGEMENT DE L'APPRENANT]
État dominant         : {etat}
Niveau d'intensité    : {niveau}/3
Durée de l'état       : {duree_etat} clips ({duree_etat * 10} secondes)
Pattern détecté       : {pattern_detecte}
Moment du cours       : {moment_cours}

[TÂCHE DE GÉNÉRATION]
Template              : {template_id} (Type {type_contenu})
Objectif adaptation   : {objectif}

[CONTRAINTES]
- Longueur            : {meta['longueur']}
- Format              : {meta['format']}
- Ton                 : {meta['ton']}
- Niveau de langage   : Adapté au niveau {niveau_apprenant}
- IMPORTANT           : Basez-vous sur le contexte de référence
                        fourni ci-dessus pour éviter les erreurs factuelles

[INSTRUCTIONS SPÉCIFIQUES — {template_id}]
{instructions_template}

Génère maintenant le contenu adaptatif en français.
En te basant sur le contexte de référence fourni,
génère un contenu pédagogique précis et sans hallucinations.
"""

    messages = [
        {
            "role": "system",
            "content": """Tu es un expert en pédagogie et ingénierie de formation.
Tu génères du contenu pédagogique adaptatif en français.
RÈGLE IMPORTANTE : Utilise UNIQUEMENT les informations du contexte de référence fourni.
Ne pas inventer de faits, de chiffres ou d'exemples non présents dans le contexte.
Si le contexte ne contient pas l'information, dis-le clairement.""",
        },
        {"role": "user", "content": prompt},
    ]

    response = None
    for tentative in range(4):
        try:
            response = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=messages,
                max_tokens=600,
                temperature=0.3,
            )
            break
        except RateLimitError:
            if tentative >= 3:
                raise
            attente = 2 + tentative * 3
            print(f"⚠️ Limite Groq (429) — nouvelle tentative dans {attente}s...")
            time.sleep(attente)

    contenu = response.choices[0].message.content
    return {
        "etat": etat,
        "niveau": niveau,
        "type_contenu": type_contenu,
        "template_id": template_id,
        "objectif": objectif,
        "concept": concept,
        "contenu": contenu,
        "nb_mots": len(contenu.split()),
        "ton": meta["ton"],
        "format": meta["format"],
        "longueur": meta["longueur"],
        "rag_utilise": utiliser_rag and rag_erreur is None,
        "nb_docs_rag": len(documents_trouves),
        "sources_rag": [d["metadata"]["source"] for d in documents_trouves],
        "rag_erreur": rag_erreur,
    }


print("✅ Module LLM + RAG + templates chargé !")
