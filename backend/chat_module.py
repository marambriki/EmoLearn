# ============================================
# backend/chat_module.py
# Tuteur conversationnel affect-aware (Groq + RAG)
# ============================================

import os
import time

from dotenv import load_dotenv
from groq import Groq, RateLimitError

from groq_config import extraire_texte, options_completion

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

from prompts.templates import get_meta_template

TON_PAR_ETAT = {
    "Boredom": "Dynamique et concret — propose des exemples vivants pour relancer l'intérêt.",
    "Confusion": "Patient et décomposé — reformule, utilise des analogies, vérifie la compréhension.",
    "Frustration": "Empathique et rassurant — normalise la difficulté, débloque sans culpabiliser.",
    "Engagement": "Stimulant et approfondissant — propose des défis adaptés au niveau.",
}

CONSIGNE_PAR_ETAT = {
    "Boredom": (
        "Relance l'attention : lien avec un cas réel, anecdote ou comparaison surprenante. "
        "Évite les longues définitions ; va droit au « pourquoi c'est utile »."
    ),
    "Confusion": (
        "Décompose en 2–3 micro-étapes numérotées. Termine par une question de vérification "
        "du type « Est-ce que l'étape X est claire ? »."
    ),
    "Frustration": (
        "Valide l'effort (« c'est une étape difficile pour beaucoup »). "
        "Propose UNE action concrète et simple pour débloquer, sans surcharge."
    ),
    "Engagement": (
        "Approfondis avec nuance ou un mini-défi. "
        "Encourage la réflexion critique plutôt que la répétition."
    ),
}

TEMPERATURE_PAR_ETAT = {
    "Boredom": 0.55,
    "Confusion": 0.25,
    "Frustration": 0.35,
    "Engagement": 0.45,
}

MOTS_cles_reponse_longue = (
    "détail", "detail", "exemple", "exercice", "explique", "approfond",
    "complet", "étape", "etape", "démonstration", "demonstration",
)


def _contexte_rag(message, concept, etat, utiliser_rag):
    if not utiliser_rag:
        return "", [], None
    try:
        from rag_module import rechercher_documents, construire_contexte_rag

        query = f"{message} — {concept} — {etat}"
        docs = rechercher_documents(query=query, concept=concept, etat=etat, n_results=3)
        ctx = construire_contexte_rag(docs, concept=concept, etat=etat)
        sources = [d["metadata"]["source"] for d in docs]
        return ctx, sources, None
    except Exception as exc:
        return "", [], str(exc)


def _construire_system_prompt(
    etat,
    niveau,
    template_id,
    strategie,
    concept,
    module,
    niveau_apprenant,
    objectif_pedagogique="",
):
    meta = get_meta_template(template_id)
    ton_etat = TON_PAR_ETAT.get(etat, meta["ton"])
    consigne = CONSIGNE_PAR_ETAT.get(etat, meta["objectif"])
    objectif = objectif_pedagogique.strip() or meta["objectif"]
    return f"""Tu es EmoTutor, un tuteur pédagogique adaptatif intégré à EmoLearn.

RÈGLES ABSOLUES :
- Réponds UNIQUEMENT en français.
- Longueur par défaut : 3 à 8 phrases. Si l'apprenant demande un exemple, un exercice ou plus de détails, tu peux aller jusqu'à 12 phrases.
- Adapte ton style à l'état émotionnel détecté de l'apprenant.
- Utilise le contexte de référence RAG quand il est fourni ; cite les idées clés sans copier mot pour mot.
- N'invente pas de faits techniques absents du contexte ou du concept en cours.
- Si tu ne sais pas, dis-le clairement et propose une piste de réflexion ou un prérequis à revoir.
- Reste focalisé sur le concept en cours ; ne change pas de sujet sans demande explicite.
- Ne répète pas mot pour mot le contenu adaptatif déjà envoyé ; complète ou clarifie.

[PROFIL APPRENANT — issu de l'analyse multimodale DAiSEE + IEMOCAP]
État dominant     : {etat} (intensité {niveau}/3)
Stratégie active  : {strategie or "—"}
Template          : {template_id} (Type {meta['type']})
Objectif pédago   : {objectif}
Ton attendu       : {meta['ton']}
Consigne émotion  : {ton_etat}
Adaptation        : {consigne}

[CONTEXTE PÉDAGOGIQUE]
Module            : {module}
Concept           : {concept}
Niveau apprenant  : {niveau_apprenant}
"""


def _estimer_max_tokens(message):
    """Allonge la réponse si l'apprenant demande explicitement plus de matière."""
    msg = message.lower()
    if any(mot in msg for mot in MOTS_cles_reponse_longue):
        return 650
    if len(message) > 120:
        return 500
    return 400


def repondre_tuteur(
    message,
    historique=None,
    etat="Engagement",
    niveau=2,
    template_id="A1",
    strategie="",
    concept="le concept",
    module="Module en cours",
    niveau_apprenant="Intermédiaire",
    objectif_pedagogique="",
    contenu_initial=None,
    utiliser_rag=True,
):
    """
    Génère une réponse du tuteur conversationnel en tenant compte de l'état affectif,
    de l'historique et du RAG.
    """
    historique = historique or []
    contexte_rag, sources_rag, rag_erreur = _contexte_rag(
        message, concept, etat, utiliser_rag
    )

    system = _construire_system_prompt(
        etat,
        niveau,
        template_id,
        strategie,
        concept,
        module,
        niveau_apprenant,
        objectif_pedagogique=objectif_pedagogique,
    )
    temperature = TEMPERATURE_PAR_ETAT.get(etat, 0.4)
    max_tokens = _estimer_max_tokens(message)

    messages = [{"role": "system", "content": system}]

    if contenu_initial and not historique:
        messages.append({
            "role": "assistant",
            "content": (
                f"[Contenu adaptatif généré pour l'apprenant — {template_id}]\n"
                f"{contenu_initial}"
            ),
        })

    for msg in historique[-10:]:
        role = msg.get("role", "user")
        if role in ("user", "assistant"):
            messages.append({"role": role, "content": msg.get("content", "")})

    user_content = message
    if contexte_rag:
        user_content = f"{contexte_rag}\n\n[QUESTION DE L'APPRENANT]\n{message}"

    messages.append({"role": "user", "content": user_content})

    response = None
    for tentative in range(4):
        try:
            response = client.chat.completions.create(
                messages=messages,
                **options_completion(max_tokens, temperature),
            )
            break
        except RateLimitError:
            if tentative >= 3:
                raise
            time.sleep(2 + tentative * 3)

    reponse = extraire_texte(response.choices[0].message)
    return {
        "reponse": reponse,
        "etat": etat,
        "template_id": template_id,
        "rag_utilise": bool(contexte_rag),
        "sources_rag": sources_rag,
        "rag_erreur": rag_erreur,
    }
