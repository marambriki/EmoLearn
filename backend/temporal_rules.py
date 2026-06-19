# ============================================
# Règles de déclenchement temporel (OS1 / mémoire)
# ============================================

URGENCE_IMMEDIATE = "immediate"
URGENCE_RAPIDE = "rapide"
URGENCE_DIFFEREE = "differee"
AUCUNE_URGENCE = "aucune"


def _consecutifs(historique, predicate, min_count):
    """Compte combien de clips consécutifs (depuis la fin) satisfont predicate."""
    count = 0
    for clip in reversed(historique):
        if predicate(clip):
            count += 1
        else:
            break
    return count


def _clip_scores(clip):
    return (
        int(clip["boredom"]),
        int(clip["confusion"]),
        int(clip["engagement"]),
        int(clip["frustration"]),
    )


def evaluer_declenchement_temporel(historique):
    """
    Évalue si une adaptation doit être déclenchée selon l'historique de clips.

    historique: liste de dicts avec boredom, confusion, engagement, frustration
    Returns: dict urgence, declencher, regle, duree_etat, clips_analyses
    """
    if not historique:
        return {
            "urgence": AUCUNE_URGENCE,
            "declencher": False,
            "regle": "historique_vide",
            "duree_etat": 0,
            "clips_analyses": 0,
        }

    current = historique[-1]
    b, c, e, f = _clip_scores(current)
    n = len(historique)

    # 1. Déclenchement immédiat (1 clip)
    if c >= 3 and f >= 2:
        return _result(URGENCE_IMMEDIATE, True, "confusion_3_frustration_2", n)
    if e == 0:
        return _result(URGENCE_IMMEDIATE, True, "engagement_zero", n)

    # 2. Déclenchement rapide (2-3 clips consécutifs)
    if _consecutifs(historique, lambda x: int(x["boredom"]) >= 2, 1) >= 2:
        return _result(URGENCE_RAPIDE, True, "boredom_ge_2_consecutif", n)
    if _consecutifs(historique, lambda x: int(x["confusion"]) >= 2, 1) >= 2:
        return _result(URGENCE_RAPIDE, True, "confusion_ge_2_consecutif", n)
    if _consecutifs(historique, lambda x: int(x["frustration"]) >= 2, 1) >= 2:
        return _result(URGENCE_RAPIDE, True, "frustration_ge_2_consecutif", n)

    # 3. Déclenchement différé (4-5 clips)
    if n >= 4 and all(int(x["engagement"]) <= 1 for x in historique[-4:]):
        return _result(URGENCE_DIFFEREE, True, "engagement_faible_stable_4clips", n)
    if n >= 4 and all(int(x["boredom"]) == 1 for x in historique[-4:]):
        return _result(URGENCE_DIFFEREE, True, "boredom_1_prolonge_4clips", n)

    # 4. Pas d'intervention
    if e >= 2 and n >= 2 and all(int(x["engagement"]) >= 2 for x in historique[-2:]):
        return _result(AUCUNE_URGENCE, False, "engagement_stable", n)

    # Déclenchement par scores instantanés (clip unique ou fallback)
    if c >= 2 or f >= 2 or b >= 2 or e <= 1:
        duree = _consecutifs(
            historique,
            lambda x: max(_clip_scores(x)[1], _clip_scores(x)[3], _clip_scores(x)[0])
            >= 2
            or _clip_scores(x)[2] <= 1,
            1,
        )
        return _result(URGENCE_RAPIDE, True, "seuils_instantanes", max(duree, 1))

    return _result(AUCUNE_URGENCE, False, "zones_acceptables", n)


def _result(urgence, declencher, regle, clips_analyses):
    return {
        "urgence": urgence,
        "declencher": declencher,
        "regle": regle,
        "duree_etat": clips_analyses,
        "clips_analyses": clips_analyses,
    }
