# ============================================
# backend/iemocap_audio.py
# Résolution chemins .wav IEMOCAP + extraction segments
# ============================================

import os
import re

IEMOCAP_PATH = os.path.join(
    os.path.dirname(__file__),
    "../data/IEMOCAP_full_release",
)


def dialogue_id_from_utterance(utterance_id: str) -> str:
    """Ses01F_impro01_F000 → Ses01F_impro01"""
    return re.sub(r"_F\d+$", "", str(utterance_id))


def session_num_from_utterance(utterance_id: str) -> int | None:
    match = re.match(r"Ses(\d+)", str(utterance_id))
    return int(match.group(1)) if match else None


def resolve_wav_path(utterance_id: str, session_num: int | None = None) -> str:
    """Chemin du fichier dialogue .wav contenant l'utterance."""
    session = session_num or session_num_from_utterance(utterance_id)
    dialogue = dialogue_id_from_utterance(utterance_id)
    if session is None:
        return ""
    return os.path.normpath(
        os.path.join(
            IEMOCAP_PATH,
            f"Session{session}",
            "dialog",
            "wav",
            f"{dialogue}.wav",
        )
    )


def wav_exists(utterance_id: str, session_num: int | None = None) -> bool:
    path = resolve_wav_path(utterance_id, session_num)
    return bool(path) and os.path.isfile(path)


def enrich_row_wav_paths(row: dict) -> dict:
    """Ajoute session_num, dialogue_id, wav_path, wav_exists à une ligne."""
    utterance_id = row.get("utterance_id", "")
    session_num = session_num_from_utterance(utterance_id)
    wav_path = resolve_wav_path(utterance_id, session_num)
    return {
        **row,
        "session_num": session_num,
        "dialogue_id": dialogue_id_from_utterance(utterance_id),
        "wav_path": wav_path,
        "wav_exists": os.path.isfile(wav_path) if wav_path else False,
    }


def charger_segment_wav(
    utterance_id: str,
    start_time: float,
    end_time: float,
    sr: int = 16000,
    wav_path: str | None = None,
):
    """
    Charge le segment audio [start_time, end_time] depuis le .wav dialogue.
    Retourne (y, sr) ou (None, sr) si indisponible.
    """
    try:
        import librosa
    except ImportError:
        return None, sr

    path = wav_path or resolve_wav_path(utterance_id)
    if not path or not os.path.isfile(path):
        return None, sr

    duree = max(float(end_time) - float(start_time), 0.05)
    offset = max(float(start_time), 0.0)

    try:
        y, loaded_sr = librosa.load(
            path,
            sr=sr,
            mono=True,
            offset=offset,
            duration=duree,
        )
        if y is None or len(y) == 0:
            return None, sr
        return y, loaded_sr
    except Exception:
        return None, sr
