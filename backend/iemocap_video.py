# ============================================
# backend/iemocap_video.py
# Modalité VISAGE IEMOCAP — sous-corpus audiovisuel ALIGNÉ
# Extrait le visage du locuteur depuis les vidéos dyadiques IEMOCAP
# (visage + voix du même locuteur, synchronisés → Option A).
#
# Disposition vidéo (720x480, dyadique) :
#   - Acteur porteur des marqueurs mocap (lettre du dialogue SesXX[F/M]) → moitié GAUCHE
#   - Partenaire → moitié DROITE
#   - locuteur (suffixe _F### / _M### de l'utterance) :
#       speaker == marker_gender -> GAUCHE, sinon -> DROITE
#
# Remarque données : seule Session1 contient le dossier avi/DivX dans ce dépôt.
# ============================================

import os
import re

IEMOCAP_PATH = os.path.join(
    os.path.dirname(__file__),
    "../data/IEMOCAP_full_release",
)

FRAME_WIDTH = 720
FRAME_HEIGHT = 480
HALF_WIDTH = FRAME_WIDTH // 2



def dialogue_id_from_utterance(utterance_id: str) -> str:
    """Ses01F_impro01_F000 -> Ses01F_impro01 (gère _F### ET _M###)."""
    return re.sub(r"_[FM]\d+$", "", str(utterance_id))


def session_num_from_utterance(utterance_id: str):
    match = re.match(r"Ses(\d+)", str(utterance_id))
    return int(match.group(1)) if match else None


def marker_gender_from_utterance(utterance_id: str):
    """Acteur porteur des marqueurs = lettre F/M juste après SesNN."""
    match = re.match(r"Ses\d+([FM])", str(utterance_id))
    return match.group(1) if match else None


def speaker_gender_from_utterance(utterance_id: str):
    """Locuteur = lettre F/M du suffixe _F###/_M###."""
    match = re.search(r"_([FM])\d+$", str(utterance_id))
    return match.group(1) if match else None


def speaker_side(utterance_id: str) -> str:
    """
    Retourne 'left' ou 'right' selon que le locuteur porte (ou non) les marqueurs.
    speaker == marker -> 'left' ; sinon -> 'right'.
    """
    marker = marker_gender_from_utterance(utterance_id)
    speaker = speaker_gender_from_utterance(utterance_id)
    if marker is None or speaker is None:
        return "left"
    return "left" if speaker == marker else "right"


def resolve_avi_path(utterance_id: str, session_num=None) -> str:
    """Chemin du .avi dialogue contenant l'utterance."""
    session = session_num or session_num_from_utterance(utterance_id)
    dialogue = dialogue_id_from_utterance(utterance_id)
    if session is None:
        return ""
    return os.path.normpath(
        os.path.join(
            IEMOCAP_PATH,
            f"Session{session}",
            "dialog",
            "avi",
            "DivX",
            f"{dialogue}.avi",
        )
    )


def avi_exists(utterance_id: str, session_num=None) -> bool:
    path = resolve_avi_path(utterance_id, session_num)
    return bool(path) and os.path.isfile(path)


def _crop_speaker_half(frame, side: str):
    """Découpe la moitié gauche/droite correspondant au locuteur."""
    h, w = frame.shape[:2]
    mid = w // 2
    if side == "left":
        return frame[:, :mid]
    return frame[:, mid:]


def _detect_largest_face(bgr_half, margin: float = 0.3):
    """Détecte le plus grand visage (MediaPipe) et retourne (crop BGR, methode)."""
    from face_detect import crop_largest_face

    return crop_largest_face(bgr_half, margin=margin)


def extract_face_image(
    utterance_id: str,
    start_time: float,
    end_time: float,
    avi_path: str = None,
    return_debug: bool = False,
):
    """
    Extrait une image PIL du visage du LOCUTEUR au milieu du segment [start, end].

    Retourne :
      - PIL.Image (RGB) du visage, ou None si indisponible
      - si return_debug=True : (pil_image, meta)
    """
    import cv2
    import numpy as np
    from PIL import Image

    path = avi_path or resolve_avi_path(utterance_id)
    meta = {
        "utterance_id": utterance_id,
        "avi_path": path,
        "side": speaker_side(utterance_id),
        "face_detected": False,
        "detection_methode": None,
        "timestamp_sec": None,
        "erreur": None,
    }

    if not path or not os.path.isfile(path):
        meta["erreur"] = "avi_absent"
        return (None, meta) if return_debug else None

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        meta["erreur"] = "avi_illisible"
        cap.release()
        return (None, meta) if return_debug else None

    midpoint = (float(start_time) + float(end_time)) / 2.0
    meta["timestamp_sec"] = round(midpoint, 3)
    cap.set(cv2.CAP_PROP_POS_MSEC, midpoint * 1000.0)
    ok, frame = cap.read()
    cap.release()

    if not ok or frame is None:
        meta["erreur"] = "frame_illisible"
        return (None, meta) if return_debug else None

    half = _crop_speaker_half(frame, meta["side"])
    face, methode = _detect_largest_face(half)

    if face is not None and face.size > 0:
        meta["face_detected"] = True
        meta["detection_methode"] = methode
        crop_bgr = face
    else:
        # Repli : demi-image entière (le pipeline FER peut tenter quand même)
        meta["detection_methode"] = "demi_image_fallback"
        crop_bgr = half

    rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    return (pil, meta) if return_debug else pil


def extract_face_frames(utterance_id, start_time, end_time, n_frames=5, avi_path=None):
    """
    Extrait jusqu'à `n_frames` visages du LOCUTEUR, répartis uniformément sur le
    segment [start, end] (vote multi-frames pour stabiliser le FER).

    Retourne (liste_images_PIL, meta) où meta agrège le nombre de visages détectés.
    Ouvre la vidéo UNE seule fois.
    """
    import cv2
    from PIL import Image

    path = avi_path or resolve_avi_path(utterance_id)
    side = speaker_side(utterance_id)
    meta = {
        "utterance_id": utterance_id,
        "side": side,
        "n_demande": n_frames,
        "n_frames": 0,
        "n_visages_detectes": 0,
        "methodes": [],
        "erreur": None,
    }

    if not path or not os.path.isfile(path):
        meta["erreur"] = "avi_absent"
        return [], meta

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        meta["erreur"] = "avi_illisible"
        cap.release()
        return [], meta

    start = float(start_time)
    end = max(float(end_time), start + 0.05)
    if n_frames <= 1:
        timestamps = [(start + end) / 2.0]
    else:
        # Points internes (on évite les bords exacts du segment)
        step = (end - start) / (n_frames + 1)
        timestamps = [start + step * (i + 1) for i in range(n_frames)]

    images = []
    for ts in timestamps:
        cap.set(cv2.CAP_PROP_POS_MSEC, ts * 1000.0)
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        half = _crop_speaker_half(frame, side)
        face, methode = _detect_largest_face(half)
        if face is not None and face.size > 0:
            crop_bgr = face
            meta["n_visages_detectes"] += 1
            meta["methodes"].append(methode)
        else:
            crop_bgr = half
            meta["methodes"].append("demi_image_fallback")
        images.append(Image.fromarray(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)))

    cap.release()
    meta["n_frames"] = len(images)
    return images, meta
