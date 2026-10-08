# ============================================
# backend/daisee_video.py
# Résolution des vidéos DAiSEE réelles (ClipID -> chemin .avi/.mp4)
# + extraction de visage (multi-frames) pour le FER en production.
#
# data/DAiSEE/DataSet/{Train,Validation,Test}/<sujet>/<ClipID_sans_ext>/<ClipID>
# ============================================

import os

DAISEE_DATASET_ROOT = os.path.join(
    os.path.dirname(__file__),
    "../data/DAiSEE/DataSet",
)

SPLITS = ("Train", "Validation", "Test")
EXTENSIONS = (".avi", ".mp4")

_index = None  # {ClipID (avec extension): chemin absolu}


def _construire_index(dataset_root: str = DAISEE_DATASET_ROOT) -> dict:
    """Scanne une seule fois les 3 splits et indexe tous les clips par ClipID."""
    index = {}
    for split in SPLITS:
        base = os.path.join(dataset_root, split)
        if not os.path.isdir(base):
            continue
        for sujet in os.listdir(base):
            sujet_path = os.path.join(base, sujet)
            if not os.path.isdir(sujet_path):
                continue
            for clip_dir in os.listdir(sujet_path):
                clip_path = os.path.join(sujet_path, clip_dir)
                if not os.path.isdir(clip_path):
                    continue
                for fname in os.listdir(clip_path):
                    if fname.lower().endswith(EXTENSIONS):
                        index[fname] = os.path.join(clip_path, fname)
    return index


def get_index(force: bool = False) -> dict:
    global _index
    if _index is None or force:
        _index = _construire_index()
        print(f"Index vidéos DAiSEE construit : {len(_index)} clips.")
    return _index


def resolve_daisee_video_path(clip_id: str) -> str:
    """Chemin absolu de la vidéo DAiSEE pour un ClipID (ex. '4000182015.avi')."""
    clip_id = str(clip_id).strip()
    index = get_index()
    if clip_id in index:
        return index[clip_id]
    # tolérance : ClipID fourni sans extension
    for ext in EXTENSIONS:
        candidat = f"{clip_id}{ext}"
        if candidat in index:
            return index[candidat]
    return ""


def daisee_video_exists(clip_id: str) -> bool:
    path = resolve_daisee_video_path(clip_id)
    return bool(path) and os.path.isfile(path)


def _read_preview_frame(cap):
    """Lit une frame lisible (certains .avi DAiSEE ne seekent pas au milieu)."""
    import cv2

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    positions = []
    if total > 4:
        positions.append(total // 2)
    positions.extend([0, 1, 2, 5, 10])

    for pos in positions:
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(pos)))
        ok, frame = cap.read()
        if ok and frame is not None and getattr(frame, "size", 0) > 0:
            return frame

    # Repli : lecture séquentielle sans seek
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    for _ in range(15):
        ok, frame = cap.read()
        if ok and frame is not None and getattr(frame, "size", 0) > 0:
            return frame
    return None


def extract_preview_jpeg(clip_id: str, quality: int = 80) -> bytes | None:
    """Extrait une frame JPEG pour l'affiche du lecteur UI."""
    import cv2

    path = resolve_daisee_video_path(clip_id)
    if not path:
        return None
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return None
    try:
        frame = _read_preview_frame(cap)
    finally:
        cap.release()
    if frame is None:
        return None
    h, w = frame.shape[:2]
    if w > 640:
        scale = 640 / w
        frame = cv2.resize(frame, (640, int(h * scale)))
    ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    return buf.tobytes() if ok else None


def iter_mjpeg_stream(
    clip_id: str,
    max_fps: float = 8.0,
    loop: bool = False,
    max_seconds: float = 12.0,
):
    """
    Générateur MJPEG pour prévisualisation navigateur (<img src=...>).
    Durée limitée pour ne pas bloquer le worker API.
    """
    import time

    import cv2

    path = resolve_daisee_video_path(clip_id)
    if not path:
        return
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return
    delay = 1.0 / max(1.0, max_fps)
    started = time.time()
    try:
        while True:
            if time.time() - started > max_seconds:
                break
            ok, frame = cap.read()
            if not ok or frame is None:
                if not loop:
                    break
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            h, w = frame.shape[:2]
            if w > 640:
                scale = 640 / w
                frame = cv2.resize(frame, (640, int(h * scale)))
            ok, buf = cv2.imencode(
                ".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70]
            )
            if not ok:
                continue
            payload = buf.tobytes()
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + payload + b"\r\n"
            )
            time.sleep(delay)
    finally:
        cap.release()


def _crop_largest_face(bgr_frame, margin: float = 0.3):
    """Retourne (crop BGR, methode) du plus grand visage, ou (None, None)."""
    from face_detect import crop_largest_face

    return crop_largest_face(bgr_frame, margin=margin)


def extract_face_frames(clip_id: str, n_frames: int = 5, avi_path: str = None):
    """
    Extrait jusqu'à `n_frames` visages (répartis uniformément sur la vidéo,
    vote multi-frames pour stabiliser le FER) depuis la VRAIE vidéo DAiSEE.

    Retourne (liste_images_PIL, meta).
    """
    import cv2
    from PIL import Image

    clip_id = str(clip_id).strip()
    path = avi_path or resolve_daisee_video_path(clip_id)
    meta = {
        "clip_id": clip_id,
        "video_path": path,
        "n_demande": n_frames,
        "n_frames": 0,
        "n_visages_detectes": 0,
        "methodes": [],
        "erreur": None,
    }

    if not path or not os.path.isfile(path):
        meta["erreur"] = "video_absente"
        return [], meta

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        meta["erreur"] = "video_illisible"
        cap.release()
        return [], meta

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if total_frames <= 0:
        meta["erreur"] = "aucune_image"
        cap.release()
        return [], meta

    n = max(1, n_frames)
    if n == 1:
        positions = [total_frames // 2]
    else:
        step = total_frames / (n + 1)
        positions = [int(step * (i + 1)) for i in range(n)]

    images = []
    for pos in positions:
        cap.set(cv2.CAP_PROP_POS_FRAMES, min(pos, total_frames - 1))
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        face, methode = _crop_largest_face(frame)
        if face is not None and face.size > 0:
            crop_bgr = face
            meta["n_visages_detectes"] += 1
            meta["methodes"].append(methode)
        else:
            crop_bgr = frame
            meta["methodes"].append("image_entiere_fallback")
        images.append(Image.fromarray(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)))

    cap.release()
    meta["n_frames"] = len(images)
    if not images and not meta["erreur"]:
        meta["erreur"] = "aucune_image"
    return images, meta
