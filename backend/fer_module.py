# ============================================
# backend/fer_module.py
# Perception visage — analyse RÉELLE de la vidéo DAiSEE
# (multi-frames + FER + AU optionnel)
# ============================================

import os

from au_module import au_actif, extraire_au, moyenner_au
from daisee_video import extract_face_frames
from fer_offline import (
    ETATS_DAISEE,
    analyser_visage_image,
    analyser_visage_image_pretrained,
    fer_runtime_mode,
)
from fer_classifier import (
    appliquer_mapping,
    model_au_disponible,
)

N_FRAMES_DEFAUT = 5
# Frames dont la proba max < seuil sont rejetées avant moyenne
SEUIL_FRAME_FER = float(os.getenv("EMOLEARN_SEUIL_FRAME_FER", "0.35"))


def _confiance_frame(result: dict) -> float:
    dist = result.get("probs_etats") or result.get("tous_scores") or {}
    if not dist:
        return float(result.get("score", 0.0) or 0.0)
    return float(max(dist.values()))


def _agreger_scores_frames(resultats_frames, seuil_frame: float | None = None):
    """
    Moyenne les distributions sur les frames valides et assez confiantes.
    Retourne (moyennes, n_valides, n_rejetees) ou None.
    """
    seuil = SEUIL_FRAME_FER if seuil_frame is None else float(seuil_frame)
    sums = {}
    n_valides = 0
    n_rejetees = 0
    for r in resultats_frames:
        if r.get("erreur"):
            n_rejetees += 1
            continue
        if _confiance_frame(r) < seuil:
            n_rejetees += 1
            continue
        dist = r.get("probs_etats") or r.get("tous_scores") or {}
        for label, proba in dist.items():
            sums[label] = sums.get(label, 0.0) + float(proba)
        n_valides += 1

    if n_valides == 0:
        return None
    return (
        {label: total / n_valides for label, total in sums.items()},
        n_valides,
        n_rejetees,
    )


def _agreger_fer7_frames(images, indices_ok=None):
    """Moyenne des 7 scores pretrained (pour classifieur FER+AU)."""
    sums = {}
    n = 0
    for i, img in enumerate(images):
        if indices_ok is not None and i not in indices_ok:
            continue
        r = analyser_visage_image_pretrained(img)
        if r.get("erreur") or not r.get("tous_scores"):
            continue
        for lab, p in r["tous_scores"].items():
            sums[lab] = sums.get(lab, 0.0) + float(p)
        n += 1
    if n == 0:
        return None
    return {lab: total / n for lab, total in sums.items()}


def analyser_visage_depuis_scenario(scenario_row, n_frames=N_FRAMES_DEFAUT):
    """
    Analyse l'émotion faciale depuis la VRAIE vidéo DAiSEE du scénario :
      1. ClipID → chemin vidéo réel
      2. Extraction de n_frames visages
      3. FER fine-tuné / pretrained
      4. Filtre frames basse confiance puis moyenne des probabilités
      5. AU (MediaPipe blendshapes) + mapping FER+AU si modèle présent
    """
    clip_id = str(scenario_row["ClipID"])
    images, meta_video = extract_face_frames(clip_id, n_frames=n_frames)

    if not images:
        return {
            "modalite": "visage_daisee",
            "emotion": None,
            "score": 0.0,
            "etat": "Boredom",
            "niveau": 1,
            "tous_scores": {},
            "erreur": f"video_indisponible:{meta_video.get('erreur')}",
            "clip_id": clip_id,
            "video_meta": meta_video,
            "fer_mode": fer_runtime_mode(),
        }

    resultats_frames = [analyser_visage_image(img) for img in images]
    agrege = _agreger_scores_frames(resultats_frames)

    fallback_filtre = False
    if agrege is None:
        agrege_soft = _agreger_scores_frames(resultats_frames, seuil_frame=0.0)
        if agrege_soft is None:
            return {
                "modalite": "visage_daisee",
                "emotion": None,
                "score": 0.0,
                "etat": "Boredom",
                "niveau": 1,
                "tous_scores": {},
                "erreur": "fer_echec_toutes_frames",
                "clip_id": clip_id,
                "video_meta": meta_video,
                "fer_mode": fer_runtime_mode(),
            }
        moyennes, n_valides, n_rejetees = agrege_soft
        fallback_filtre = True
    else:
        moyennes, n_valides, n_rejetees = agrege

    # AU moyen sur les frames acceptées (seulement si classifieur AU présent)
    au_scores = None
    use_au = au_actif() and model_au_disponible()
    if use_au:
        aus = []
        for i, (img, r) in enumerate(zip(images, resultats_frames)):
            if r.get("erreur"):
                continue
            if not fallback_filtre and _confiance_frame(r) < SEUIL_FRAME_FER:
                continue
            aus.append(extraire_au(img))
        if aus:
            au_scores = moyenner_au(aus)

    emotion_dominante = max(moyennes, key=moyennes.get)
    score_confiance = float(moyennes[emotion_dominante])
    mode = fer_runtime_mode()

    # Chemin FER+AU : 7 scores pretrained + AU → RF
    if use_au and au_scores is not None:
        fer7 = _agreger_fer7_frames(images)
        if fer7:
            emo7 = max(fer7, key=fer7.get)
            etat, niveau, mapping_source, probs_etats = appliquer_mapping(
                emo7, fer7, au_scores=au_scores
            )
            result = {
                "modalite": "visage_daisee",
                "emotion": emo7,
                "score": round(float(fer7[emo7]), 3),
                "etat": etat,
                "niveau": niveau,
                "tous_scores": {label: round(v, 4) for label, v in fer7.items()},
                "mapping_source": mapping_source,
                "fer_mode": mode,
                "au_actif": True,
                "au_scores": {k: round(v, 4) for k, v in au_scores.items()},
                "erreur": None,
                "clip_id": clip_id,
                "n_frames_analysees": n_valides,
                "n_frames_rejetees": n_rejetees,
                "seuil_frame": SEUIL_FRAME_FER,
                "filtre_fallback": fallback_filtre,
                "video_meta": meta_video,
                "fer_vit_ref": {
                    "emotion": emotion_dominante,
                    "score": round(score_confiance, 3),
                    "tous_scores": {
                        label: round(v, 4) for label, v in moyennes.items()
                    },
                },
            }
            if probs_etats:
                result["probs_etats"] = probs_etats
            return result

    if mode == "finetuned" and emotion_dominante in ETATS_DAISEE:
        probs_etats = {e: float(moyennes.get(e, 0.0)) for e in ETATS_DAISEE}
        niveau = max(1, min(3, int(round(score_confiance * 3))))
        out = {
            "modalite": "visage_daisee",
            "emotion": emotion_dominante,
            "score": round(score_confiance, 3),
            "etat": emotion_dominante,
            "niveau": niveau,
            "tous_scores": {label: round(v, 4) for label, v in moyennes.items()},
            "probs_etats": probs_etats,
            "mapping_source": "fer_vit_finetuned",
            "fer_mode": "finetuned",
            "au_actif": False,
            "erreur": None,
            "clip_id": clip_id,
            "n_frames_analysees": n_valides,
            "n_frames_rejetees": n_rejetees,
            "seuil_frame": SEUIL_FRAME_FER,
            "filtre_fallback": fallback_filtre,
            "video_meta": meta_video,
        }
        if au_scores:
            out["au_scores"] = {k: round(v, 4) for k, v in au_scores.items()}
        return out

    etat, niveau, mapping_source, probs_etats = appliquer_mapping(
        emotion_dominante, moyennes, au_scores=au_scores
    )

    result = {
        "modalite": "visage_daisee",
        "emotion": emotion_dominante,
        "score": round(score_confiance, 3),
        "etat": etat,
        "niveau": niveau,
        "tous_scores": {label: round(v, 4) for label, v in moyennes.items()},
        "mapping_source": mapping_source,
        "fer_mode": mode,
        "au_actif": bool(au_scores and "au" in mapping_source),
        "erreur": None,
        "clip_id": clip_id,
        "n_frames_analysees": n_valides,
        "n_frames_rejetees": n_rejetees,
        "seuil_frame": SEUIL_FRAME_FER,
        "filtre_fallback": fallback_filtre,
        "video_meta": meta_video,
    }
    if au_scores:
        result["au_scores"] = {k: round(v, 4) for k, v in au_scores.items()}
    if probs_etats:
        result["probs_etats"] = probs_etats
    return result


print("Module FER DAiSEE (filtre frames + fine-tune + AU) charge !")
