# ============================================
# backend/au_module.py
# Action Units (proxy) via MediaPipe Face Landmarker blendshapes
# Complément FER — pas un remplacement.
# ============================================
"""
Extrait un vecteur AU pédagogique à partir des blendshapes MediaPipe
(ARKit-like). Mapping approximatif FACS :

  AU1/2  brow raise     ← browInnerUp, browOuterUp*
  AU4    brow lower     ← browDown*
  AU5    upper lid raise← eyeWide*
  AU6/7  cheek / squint ← eyeSquint*, cheekSquint*
  AU9    nose wrinkle   ← noseSneer*
  AU12   lip corner pull← mouthSmile*
  AU15   lip corner dep.← mouthFrown*
  AU17   chin raiser    ← mouthShrugLower, mouthPress*
  AU20   lip stretcher  ← mouthStretch*
  AU25/26 jaw drop      ← jawOpen, mouthOpen
"""

from __future__ import annotations

import os
import threading
import urllib.request

import numpy as np

_lock = threading.Lock()
_landmarker = None

_MODELS_DIR = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "models", "mediapipe")
)
_MODEL_FILE = "face_landmarker_v2_with_blendshapes.task"
_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)

# Vecteur AU compact (ordre fixe pour sklearn)
AU_FEATURE_ORDER = (
    "AU1_2",
    "AU4",
    "AU5",
    "AU6_7",
    "AU9",
    "AU12",
    "AU15",
    "AU17",
    "AU20",
    "AU25_26",
)

# blendshape name → contribution AU (moyenne si plusieurs)
_BLEND_TO_AU = {
    "browInnerUp": "AU1_2",
    "browOuterUpLeft": "AU1_2",
    "browOuterUpRight": "AU1_2",
    "browDownLeft": "AU4",
    "browDownRight": "AU4",
    "eyeWideLeft": "AU5",
    "eyeWideRight": "AU5",
    "eyeSquintLeft": "AU6_7",
    "eyeSquintRight": "AU6_7",
    "cheekSquintLeft": "AU6_7",
    "cheekSquintRight": "AU6_7",
    "noseSneerLeft": "AU9",
    "noseSneerRight": "AU9",
    "mouthSmileLeft": "AU12",
    "mouthSmileRight": "AU12",
    "mouthFrownLeft": "AU15",
    "mouthFrownRight": "AU15",
    "mouthShrugLower": "AU17",
    "mouthPressLeft": "AU17",
    "mouthPressRight": "AU17",
    "mouthStretchLeft": "AU20",
    "mouthStretchRight": "AU20",
    "jawOpen": "AU25_26",
    "mouthOpen": "AU25_26",
}

# AU toujours actif en runtime (complément FER obligatoire si le modèle est présent).
def au_actif() -> bool:
    return True


def _ensure_model() -> str:
    os.makedirs(_MODELS_DIR, exist_ok=True)
    path = os.path.join(_MODELS_DIR, _MODEL_FILE)
    if os.path.isfile(path) and os.path.getsize(path) > 1000:
        return path
    # Certains mirrors utilisent face_landmarker.task
    alt = os.path.join(_MODELS_DIR, "face_landmarker.task")
    if os.path.isfile(alt) and os.path.getsize(alt) > 1000:
        return alt
    print(f"Téléchargement Face Landmarker → {path}…")
    try:
        urllib.request.urlretrieve(_MODEL_URL, path)
    except Exception:
        urllib.request.urlretrieve(_MODEL_URL, alt)
        return alt
    return path


def get_face_landmarker():
    global _landmarker
    with _lock:
        if _landmarker is None:
            import mediapipe as mp

            model_path = _ensure_model()
            options = mp.tasks.vision.FaceLandmarkerOptions(
                base_options=mp.tasks.BaseOptions(model_asset_path=model_path),
                running_mode=mp.tasks.vision.RunningMode.IMAGE,
                num_faces=1,
                output_face_blendshapes=True,
                output_facial_transformation_matrixes=False,
            )
            _landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(options)
        return _landmarker


def _empty_au() -> dict:
    return {k: 0.0 for k in AU_FEATURE_ORDER}


def _pil_to_rgb_np(image):
    from PIL import Image
    import numpy as np

    if isinstance(image, Image.Image):
        return np.ascontiguousarray(image.convert("RGB"))
    arr = np.asarray(image)
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    return np.ascontiguousarray(arr.astype(np.uint8))


def blendshapes_to_au(blendshapes) -> dict:
    """Agrège les catégories MediaPipe → vecteur AU pédagogique."""
    buckets: dict[str, list[float]] = {k: [] for k in AU_FEATURE_ORDER}
    for cat in blendshapes or []:
        name = getattr(cat, "category_name", None) or getattr(cat, "name", "")
        score = float(getattr(cat, "score", 0.0))
        au = _BLEND_TO_AU.get(name)
        if au:
            buckets[au].append(score)
    out = _empty_au()
    for au, vals in buckets.items():
        if vals:
            out[au] = float(np.mean(vals))
    return out


def extraire_au(image) -> dict:
    """
    Extrait les AU depuis une image PIL/ndarray RGB (visage croppé de préférence).
    Retourne dict AU_FEATURE_ORDER → float [0,1]. En cas d'échec → zéros.
    """
    try:
        import mediapipe as mp

        rgb = _pil_to_rgb_np(image)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        landmarker = get_face_landmarker()
        with _lock:
            result = landmarker.detect(mp_image)
        if not result.face_blendshapes:
            return _empty_au()
        return blendshapes_to_au(result.face_blendshapes[0])
    except Exception as exc:
        print(f"⚠ AU extraction échouée: {exc}")
        return _empty_au()


def au_to_vector(au_scores: dict | None) -> np.ndarray:
    scores = au_scores or {}
    return np.asarray(
        [float(scores.get(k, 0.0)) for k in AU_FEATURE_ORDER],
        dtype=np.float64,
    )


def moyenner_au(liste_au: list[dict]) -> dict:
    if not liste_au:
        return _empty_au()
    out = _empty_au()
    for k in AU_FEATURE_ORDER:
        vals = [float(d.get(k, 0.0)) for d in liste_au]
        out[k] = float(np.mean(vals)) if vals else 0.0
    return out
