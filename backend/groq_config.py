# ============================================
# backend/groq_config.py
# Modèle Groq + extraction du texte (GPT-OSS = modèle à raisonnement)
# ============================================

import os

from dotenv import load_dotenv

load_dotenv()

# Groq a retiré llama-3.1-8b-instant le 16/08/2026.
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")


def extraire_texte(message) -> str:
    """Texte visible d'une réponse Groq (content, sinon reasoning)."""
    contenu = getattr(message, "content", None) or ""
    if isinstance(contenu, list):
        parties = []
        for part in contenu:
            if isinstance(part, str):
                parties.append(part)
            elif isinstance(part, dict):
                parties.append(str(part.get("text") or part.get("content") or ""))
            else:
                parties.append(str(getattr(part, "text", "") or ""))
        contenu = "".join(parties)
    contenu = str(contenu).strip()
    if contenu:
        return contenu
    for attr in ("reasoning", "reasoning_content"):
        extra = getattr(message, attr, None)
        if extra:
            return str(extra).strip()
    return ""


def options_completion(max_tokens: int, temperature: float) -> dict:
    """
    GPT-OSS consomme max_tokens en raisonnement interne.
    Sans budget suffisant, message.content est vide (cas vu dans l'UI).
    """
    opts = {
        "model": GROQ_MODEL,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if "gpt-oss" in GROQ_MODEL.lower():
        opts["max_tokens"] = max(int(max_tokens), 2048)
        opts["reasoning_effort"] = "low"
    return opts
