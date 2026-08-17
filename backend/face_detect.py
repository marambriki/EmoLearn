# ============================================
# backend/face_detect.py
# Détection de visages MediaPipe FaceDetector (BlazeFace).
# Remplace les cascades Haar OpenCV (frontal + profil).
# ============================================

import os
import threading
import urllib.request

_lock = threading.Lock()
_short = None
_full = None

MIN_FACE = 40

_MODELS_DIR = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "models", "mediapipe")
)

_MODEL_URLS = {
    "short": (
        "blaze_face_short_range.tflite",
        "https://storage.googleapis.com/mediapipe-models/face_detector/"
        "blaze_face_short_range/float16/1/blaze_face_short_range.tflite",
    ),
    "full": (
        "blaze_face_full_range.tflite",
        "https://storage.googleapis.com/mediapipe-models/face_detector/"
        "blaze_face_full_range/float16/1/blaze_face_full_range.tflite",
    ),
}


def _ensure_model(kind: str) -> str:
    filename, url = _MODEL_URLS[kind]
    os.makedirs(_MODELS_DIR, exist_ok=True)
    path = os.path.join(_MODELS_DIR, filename)
    if os.path.isfile(path) and os.path.getsize(path) > 1000:
        return path
    urllib.request.urlretrieve(url, path)
    return path


def _make_detector(kind: str):
    import mediapipe as mp

    model_path = _ensure_model(kind)
    options = mp.tasks.vision.FaceDetectorOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=model_path),
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
        min_detection_confidence=0.5,
        min_suppression_threshold=0.3,
    )
    return mp.tasks.vision.FaceDetector.create_from_options(options)


def _get_detectors():
    """Lazy-init : courte portée (webcam) + longue portée (plan plus large)."""
    global _short, _full
    with _lock:
        if _short is None:
            _short = _make_detector("short")
            _full = _make_detector("full")
        return _short, _full


def _to_mp_image(rgb):
    import mediapipe as mp
    import numpy as np

    rgb = np.ascontiguousarray(rgb)
    return mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)


def _best_box(result, width: int, height: int):
    """Plus grande boîte en pixels, en ignorant les visages trop petits."""
    if result is None or not result.detections:
        return None
    best = None
    best_area = 0
    for det in result.detections:
        bb = det.bounding_box
        x = max(0, int(bb.origin_x))
        y = max(0, int(bb.origin_y))
        w = min(int(bb.width), width - x)
        h = min(int(bb.height), height - y)
        if w < MIN_FACE or h < MIN_FACE:
            continue
        area = w * h
        if area > best_area:
            best_area = area
            best = (x, y, w, h)
    return best


def _detect(detector, mp_image):
    with _lock:
        return detector.detect(mp_image)


def detect_face_box(bgr_frame):
    """
    Détecte le plus grand visage dans une image BGR.

    Ordre : courte portée, puis longue portée, puis image miroir
    (équivalent du profil Haar pour une tête tournée).

    Retourne (x, y, w, h, methode) ou None.
    """
    import cv2

    if bgr_frame is None or getattr(bgr_frame, "size", 0) == 0:
        return None

    height, width = bgr_frame.shape[:2]
    rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
    mp_image = _to_mp_image(rgb)
    short, full = _get_detectors()

    box = _best_box(_detect(short, mp_image), width, height)
    if box:
        return (*box, "mediapipe")

    box = _best_box(_detect(full, mp_image), width, height)
    if box:
        return (*box, "mediapipe_far")

    flipped = cv2.flip(rgb, 1)
    mp_flipped = _to_mp_image(flipped)
    box = _best_box(_detect(short, mp_flipped), width, height)
    if not box:
        box = _best_box(_detect(full, mp_flipped), width, height)
    if box:
        x, y, w, h = box
        x = width - x - w
        return x, y, w, h, "mediapipe_miroir"

    return None


def crop_largest_face(bgr_frame, margin: float = 0.3):
    """Retourne (crop BGR, methode) du plus grand visage, ou (None, None)."""
    box = detect_face_box(bgr_frame)
    if box is None:
        return None, None

    x, y, fw, fh, methode = box
    mx = int(fw * margin)
    my = int(fh * margin)
    h, w = bgr_frame.shape[:2]
    x0 = max(0, x - mx)
    y0 = max(0, y - my)
    x1 = min(w, x + fw + mx)
    y1 = min(h, y + fh + my)
    crop = bgr_frame[y0:y1, x0:x1]
    if crop is None or getattr(crop, "size", 0) == 0:
        return None, None
    return crop, methode
