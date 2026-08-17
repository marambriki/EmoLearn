# ============================================
# Utilitaires DAiSEE partagés (labels, état dominant)
# ============================================

import os

import pandas as pd

ETATS_DAISEE = ("Boredom", "Confusion", "Engagement", "Frustration")

PATTERN_PAR_ETAT = {
    "Boredom": "ennui_eleve",
    "Confusion": "confusion",
    "Frustration": "frustration",
    "Engagement": "stable",
}

# Départage explicite des ÉGALITÉS de score (priorité pédagogique) :
# en cas d'ex æquo, l'état le plus URGENT l'emporte afin de ne jamais masquer
# un apprenant en difficulté derrière un score d'engagement équivalent.
# Frustration > Confusion > Boredom > Engagement (rang 0 = plus prioritaire).
PRIORITE_ETAT = {
    "Frustration": 0,
    "Confusion": 1,
    "Boredom": 2,
    "Engagement": 3,
}


def etat_dominant_depuis_scores(boredom, confusion, engagement, frustration):
    """
    Retourne l'état dominant et son niveau (ground truth DAiSEE).

    En cas d'égalité de score (ex. Ennui 2 = Engagement 2), le départage n'est
    plus arbitraire (ordre du dict) mais suit une PRIORITÉ PÉDAGOGIQUE explicite :
    l'état le plus urgent gagne (Frustration > Confusion > Boredom > Engagement).
    """
    etats = {
        "Boredom": int(boredom),
        "Confusion": int(confusion),
        "Engagement": int(engagement),
        "Frustration": int(frustration),
    }
    # Trie par score décroissant, puis par priorité pédagogique croissante (rang).
    etat_dominant = max(etats, key=lambda e: (etats[e], -PRIORITE_ETAT[e]))
    niveau = etats[etat_dominant]
    return etat_dominant, niveau, etats


def pattern_depuis_scores(boredom, confusion, engagement, frustration):
    """
    Déduit le pattern pédagogique depuis les scores DAiSEE,
    aligné sur l'état dominant (évite ex. « ennui élevé » + Engagement dominant).
    """
    b, c, e, f = int(boredom), int(confusion), int(engagement), int(frustration)
    etat, _, _ = etat_dominant_depuis_scores(b, c, e, f)
    scores = [b, c, e, f]

    if sum(1 for v in scores if v >= 2) >= 3:
        return "critique"

    if e <= 1 and f < 2 and c < 2:
        return "engagement_faible"

    return PATTERN_PAR_ETAT.get(etat, "stable")


def pattern_depuis_ligne(row):
    """Pattern cohérent depuis une ligne scénario."""
    cols = {c.strip(): c for c in row.index} if hasattr(row, "index") else {}

    def _get(name):
        key = cols.get(name, name)
        return row[key] if key in row.index else row[name]

    return pattern_depuis_scores(
        _get("Boredom"),
        _get("Confusion"),
        _get("Engagement"),
        _get("Frustration"),
    )


def etat_depuis_ligne(row):
    """Extrait l'état dominant depuis une ligne DataFrame ou Series."""
    cols = {c.strip(): c for c in row.index} if hasattr(row, "index") else {}
    def _get(name):
        key = cols.get(name, name)
        return row[key] if key in row.index else row[name]

    return etat_dominant_depuis_scores(
        _get("Boredom"),
        _get("Confusion"),
        _get("Engagement"),
        _get("Frustration"),
    )


def charger_labels_daisee(split="train", data_dir=None):
    """
    Charge les labels DAiSEE.
    split: train | validation | test | all
    """
    if data_dir is None:
        data_dir = os.path.join(os.path.dirname(__file__), "../data/DAiSEE/Labels")

    fichiers = {
        "train": "TrainLabels.csv",
        "validation": "ValidationLabels.csv",
        "test": "TestLabels.csv",
        "all": "AllLabels.csv",
    }
    if split not in fichiers:
        raise ValueError(f"split inconnu: {split}")

    path = os.path.join(data_dir, fichiers[split])
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    return df


def normaliser_clip_id(clip_id):
    return str(clip_id).strip()
