# ============================================
# backend/fusion_module.py
# Fusion multimodale : pondération continue
# par incertitude (late fusion)
# ============================================

# Poids de départ (prior) visage / voix
POIDS_VISAGE = 0.60
POIDS_VOIX = 0.40

# Repli uniquement si les deux confiances sont nulles
priorite_pedagogique = {
    "Frustration": 4,
    "Confusion": 3,
    "Boredom": 2,
    "Engagement": 1,
}

ETATS_DAISEE = ["Boredom", "Confusion", "Engagement", "Frustration"]


def _onehot_probs(etat: str, score: float) -> dict:
    probs = {e: 0.0 for e in ETATS_DAISEE}
    if etat in probs:
        probs[etat] = max(float(score), 0.01)
    return probs


def _probs_depuis_result(result: dict | None) -> dict | None:
    if not result or result.get("erreur"):
        return None
    pv = result.get("probs_etats")
    if pv:
        return {e: float(pv.get(e, 0.0)) for e in ETATS_DAISEE}
    etat = result.get("etat")
    if etat in ETATS_DAISEE:
        return _onehot_probs(etat, float(result.get("score", 0.5)))
    return None


def _fallback_priorite(result_visage, result_voix):
    """Les deux confiances sont nulles → état le plus urgent disponible."""
    candidats = []
    for r in (result_visage, result_voix):
        if r and not r.get("erreur") and r.get("etat") in priorite_pedagogique:
            candidats.append(r)
    if not candidats:
        return {
            "etat": "Engagement",
            "niveau": 1,
            "strategie": "indetermine",
            "explication": "Aucune modalité disponible — fallback Engagement",
        }
    best = max(candidats, key=lambda r: priorite_pedagogique[r["etat"]])
    return {
        "etat": best["etat"],
        "niveau": best.get("niveau", 1),
        "strategie": "priorite_pedagogique_incertitude_totale",
        "explication": (
            f"Confiances nulles → repli priorité pédagogique ({best['etat']})"
        ),
    }


def _confiance_modalite(result) -> float:
    """Confiance dans [0, 1] ; 0 si modalité absente ou en erreur."""
    if not result or result.get("erreur"):
        return 0.0
    return max(0.0, min(1.0, float(result.get("score", 0.0) or 0.0)))


def fusionner_par_vote(
    result_visage,
    result_voix,
    poids_visage=None,
    seuil_confiance=None,
):
    """
    Late fusion à pondération continue par incertitude.

    Poids effectifs (puis renormalisés) :
      w_visage = prior_visage × confiance_visage
      w_voix   = prior_voix   × confiance_voix
    Si les deux confiances sont nulles → priorité pédagogique.

    `seuil_confiance` est ignoré (compatibilité d'appel).
    """
    if poids_visage is None:
        poids_visage = POIDS_VISAGE
    prior_v = float(poids_visage)
    prior_x = 1.0 - prior_v

    c_v = _confiance_modalite(result_visage)
    c_x = _confiance_modalite(result_voix)
    pv = _probs_depuis_result(result_visage)
    pvx = _probs_depuis_result(result_voix)

    w_v_raw = prior_v * c_v if pv is not None else 0.0
    w_x_raw = prior_x * c_x if pvx is not None else 0.0
    total = w_v_raw + w_x_raw

    score_v = round(c_v, 3) if result_visage else None
    score_x = (
        round(c_x, 3)
        if result_voix and not result_voix.get("erreur")
        else None
    )
    gate = {
        "methode": "incertitude",
        "prior_visage": prior_v,
        "confiance_visage": score_v,
        "confiance_voix": score_x,
        "seuil": None,
        "visage_fiable": pv is not None and c_v > 0.0,
        "voix_fiable": pvx is not None and c_x > 0.0,
        "score_visage": score_v,
        "score_voix": score_x,
    }

    def _details(w_v, w_x):
        voix_block = None
        if result_voix:
            voix_block = {
                "etat": result_voix.get("etat")
                if not result_voix.get("erreur")
                else None,
                "niveau": result_voix.get("niveau")
                if not result_voix.get("erreur")
                else None,
                "score": score_x,
                "poids": round(w_x, 3),
                "fiable": pvx is not None and c_x > 0.0,
            }
        return {
            "visage": {
                "etat": result_visage.get("etat") if result_visage else None,
                "niveau": result_visage.get("niveau") if result_visage else None,
                "score": score_v,
                "poids": round(w_v, 3),
                "fiable": pv is not None and c_v > 0.0,
            },
            "voix": voix_block,
        }

    if total < 1e-6:
        fb = _fallback_priorite(result_visage, result_voix)
        probs = _onehot_probs(fb["etat"], 0.5)
        return {
            "etat": fb["etat"],
            "emotion": fb["etat"].lower(),
            "niveau": fb["niveau"],
            "score_global": 0.0,
            "modalite_dominante": "abstention",
            "conflit": True,
            "type_conflit": "basse_confiance",
            "strategie_fusion": fb["strategie"],
            "explication_fusion": (
                "Confiances nulles → repli priorité pédagogique. " + fb["explication"]
            ),
            "probs_etats": probs,
            "gate": gate,
            "details": _details(0.0, 0.0),
            "multimodal": False,
        }

    w_v = w_v_raw / total
    w_x = w_x_raw / total
    gate["poids_effectif_visage"] = round(w_v, 3)
    gate["poids_effectif_voix"] = round(w_x, 3)
    gate["poids_effectif"] = {
        "visage": round(w_v, 3),
        "voix": round(w_x, 3),
    }

    zeros = {e: 0.0 for e in ETATS_DAISEE}
    p_face = pv or zeros
    p_voix = pvx or zeros
    combined = {
        e: w_v * p_face.get(e, 0.0) + w_x * p_voix.get(e, 0.0) for e in ETATS_DAISEE
    }
    etat_final = max(combined, key=combined.get)

    n_v = float(result_visage["niveau"]) if result_visage and pv is not None else 0.0
    n_x = float(result_voix["niveau"]) if result_voix and pvx is not None else 0.0
    niveau_final = int(round(n_v * w_v + n_x * w_x)) or 1

    etat_v = result_visage.get("etat") if result_visage else None
    etat_x = (
        result_voix.get("etat")
        if result_voix and not result_voix.get("erreur")
        else None
    )
    accord = bool(etat_v and etat_x and etat_v == etat_x)
    multimodal = w_v > 1e-6 and w_x > 1e-6
    dominante = "visage" if w_v >= w_x else "voix"

    if multimodal:
        type_conflit = "accord_soft" if accord else "soft_fusion"
        strategie = "vote_pondere_incertitude"
        expl = (
            f"Pondération par incertitude : prior {int(prior_v*100)}/{int(prior_x*100)}, "
            f"effectif {int(round(w_v*100))}% visage (c={c_v:.2f}) + "
            f"{int(round(w_x*100))}% voix (c={c_x:.2f}) → {etat_final}"
        )
    elif w_v >= w_x:
        type_conflit = "pas_de_voix" if pvx is None else "voix_poids_nul"
        strategie = "vote_incertitude_unimodal_visage"
        expl = "Voix absente ou confiance nulle → état depuis le visage"
    else:
        type_conflit = "visage_poids_nul"
        strategie = "vote_incertitude_unimodal_voix"
        expl = "Visage absent ou confiance nulle → état depuis la voix"

    return {
        "etat": etat_final,
        "emotion": etat_final.lower(),
        "niveau": niveau_final,
        "score_global": round(float(combined[etat_final]), 3),
        "modalite_dominante": dominante,
        "conflit": not accord if multimodal else (pvx is not None and pv is not None),
        "type_conflit": type_conflit,
        "strategie_fusion": strategie,
        "explication_fusion": expl,
        "probs_etats": combined,
        "gate": gate,
        "details": _details(w_v, w_x),
        "multimodal": multimodal,
    }


def fusionner_modalites(result_visage, result_voix, strategie=None, mode=None):
    """
    Fusion produit : pondération continue par incertitude.

    `strategie` et `mode` sont ignorés (compatibilité d'appel des scripts).
    """
    return fusionner_par_vote(result_visage, result_voix)


def agreger_probs_etats(tous_scores, mapping_emotion_etat):
    """
    Agrège une distribution de probabilités sur des ÉMOTIONS vers les 4 ÉTATS DAiSEE.
    """
    probs = {e: 0.0 for e in ETATS_DAISEE}
    for emo, p in (tous_scores or {}).items():
        cible = mapping_emotion_etat.get(emo)
        etat = cible.get("etat") if isinstance(cible, dict) else cible
        if etat in probs:
            probs[etat] += float(p)
    return probs


def expliquer_fusion(resultat_fusion):
    """Génère une explication lisible de la fusion."""
    conflit_emoji = "⚠️" if resultat_fusion["conflit"] else "✅"

    if resultat_fusion["multimodal"]:
        details = resultat_fusion["details"]
        return f"""
=== FUSION MULTIMODALE ===
Visage (DAiSEE)  : {details['visage']['etat']} niveau {details['visage']['niveau']}/3 (score {details['visage']['score']*100:.1f}%)
Voix   (IEMOCAP) : {details['voix']['etat']} niveau {details['voix']['niveau']}/3 (score {details['voix']['score']*100:.1f}%)

{conflit_emoji} Désaccord : {'OUI — ' + resultat_fusion['type_conflit'] if resultat_fusion['conflit'] else 'NON'}
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


print("Module de fusion multimodale (ponderation par incertitude) charge !")
