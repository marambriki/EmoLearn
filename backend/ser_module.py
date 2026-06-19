# ============================================
# backend/ser_module.py
# Module SER — IEMOCAP (annotations CSV + audio .wav)
# ============================================

import os
import sys

import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(__file__))
from iemocap_audio import (
    charger_segment_wav,
    enrich_row_wav_paths,
    resolve_wav_path,
    wav_exists,
)
from iemocap_parser import charger_iemocap_complet

# ===== MAPPING IEMOCAP → DAISEE =====
mapping_ser_daisee = {
    "neu": {"etat": "Boredom", "niveau": 1},
    "fru": {"etat": "Frustration", "niveau": 3},
    "ang": {"etat": "Frustration", "niveau": 3},
    "sad": {"etat": "Boredom", "niveau": 2},
    "hap": {"etat": "Engagement", "niveau": 3},
    "exc": {"etat": "Engagement", "niveau": 3},
    "sur": {"etat": "Confusion", "niveau": 2},
    "dis": {"etat": "Frustration", "niveau": 2},
    "fea": {"etat": "Confusion", "niveau": 3},
    "xxx": {"etat": "Boredom", "niveau": 1},
}

_df_iemocap = None
SER_AUDIO_DEFAULT = os.getenv("EMOLearn_SER_AUDIO", "0").lower() in (
    "1",
    "true",
    "yes",
    "on",
)


def _ser_audio_active(utiliser_audio=None) -> bool:
    if utiliser_audio is not None:
        return bool(utiliser_audio)
    return SER_AUDIO_DEFAULT


def charger_iemocap():
    global _df_iemocap
    if _df_iemocap is not None:
        return _df_iemocap

    csv_path = os.path.join(
        os.path.dirname(__file__),
        "../data/iemocap_parsed.csv",
    )

    if os.path.exists(csv_path):
        _df_iemocap = pd.read_csv(csv_path)
        if "wav_path" not in _df_iemocap.columns:
            _df_iemocap = _enrichir_colonnes_wav(_df_iemocap)
        print(f"✅ IEMOCAP chargé depuis CSV : {len(_df_iemocap)} utterances")
    else:
        print("Parsing IEMOCAP complet...")
        _df_iemocap = charger_iemocap_complet()
        _df_iemocap = _enrichir_colonnes_wav(_df_iemocap)

    return _df_iemocap


def _enrichir_colonnes_wav(df: pd.DataFrame) -> pd.DataFrame:
    enriched = [enrich_row_wav_paths(row.to_dict()) for _, row in df.iterrows()]
    return pd.DataFrame(enriched)


def selectionner_utterance_iemocap(
    df,
    scenario_id=None,
    etat_dominant=None,
    prefer_wav=False,
):
    """
    Sélectionne une utterance IEMOCAP pour un scénario DAiSEE.

    Priorité :
    1. Filtrer par etat_daisee == émotion dominante du scénario
    2. Si prefer_wav : préférer les utterances avec .wav disponible
    3. Index reproductible : scenario_id % len(pool)
    """
    sid = 0 if scenario_id is None else int(scenario_id)
    pool = df
    methode = "modulo_global"
    etat_cible = etat_dominant

    if etat_dominant:
        subset = df[df["etat_daisee"] == etat_dominant]
        if len(subset) > 0:
            pool = subset
            methode = "emotion_dominante"
        else:
            methode = "modulo_global_fallback"

    if prefer_wav and "wav_exists" in pool.columns:
        wav_pool = pool[pool["wav_exists"]]
        if len(wav_pool) > 0:
            pool = wav_pool
            if methode == "emotion_dominante":
                methode = "emotion_dominante_wav"
            elif methode == "modulo_global":
                methode = "modulo_global_wav"

    idx = sid % len(pool)
    row = pool.iloc[idx]

    meta = {
        "methode": methode,
        "etat_cible": etat_cible,
        "etat_iemocap": row["etat_daisee"],
        "pool_size": int(len(pool)),
        "index_pool": int(idx),
        "scenario_id": sid,
    }
    return row.to_dict(), meta


def _resultat_depuis_row(row, source: str, emotion_override=None, features=None, appariement=None):
    emotion = emotion_override or row["emotion"]
    activation = float(row["activation"])
    valence = float(row["valence"])
    dominance = float(row["dominance"])

    etat_daisee = mapping_ser_daisee.get(
        emotion,
        {"etat": "Boredom", "niveau": 1},
    )
    score = (activation - 1) / 4.0

    result = {
        "modalite": "voix_iemocap",
        "emotion": emotion,
        "score": round(score, 3),
        "etat": etat_daisee["etat"],
        "niveau": etat_daisee["niveau"],
        "valence": valence,
        "activation": activation,
        "dominance": dominance,
        "utterance_id": row["utterance_id"],
        "wav_path": row.get("wav_path") or resolve_wav_path(row["utterance_id"]),
        "segment": {
            "start_time": float(row["start_time"]),
            "end_time": float(row["end_time"]),
            "duree": float(row.get("duree", row["end_time"] - row["start_time"])),
        },
        "erreur": None,
        "source": source,
    }
    if features:
        result["features_acoustiques"] = features
    if appariement:
        result["appariement"] = appariement
    return result


def _predire_emotion_acoustique(y, sr):
    """
    SER léger par features acoustiques (librosa).
    Retourne (emotion_code, valence, activation, dominance, features_dict).
    """
    import librosa

    if y is None or len(y) < max(int(sr * 0.05), 1):
        return "neu", 2.5, 2.5, 2.5, {"erreur": "segment_trop_court"}

    rms = float(np.mean(librosa.feature.rms(y=y)))
    zcr = float(np.mean(librosa.feature.zero_crossing_rate(y=y)))
    centroid = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))

    f0, _, _ = librosa.pyin(y, fmin=75, fmax=400, sr=sr)
    f0_valid = f0[~np.isnan(f0)] if f0 is not None else np.array([])
    pitch_mean = float(np.mean(f0_valid)) if len(f0_valid) > 0 else 150.0
    pitch_std = float(np.std(f0_valid)) if len(f0_valid) > 0 else 10.0

    energy = min(1.0, rms * 15)
    brightness = min(1.0, centroid / 4000)
    pitch_norm = min(1.0, max(0.0, (pitch_mean - 80) / 250))
    variability = min(1.0, pitch_std / 80)

    activation = float(np.clip(1 + 4 * (0.5 * energy + 0.3 * min(zcr * 10, 1) + 0.2 * variability), 1, 5))
    valence = float(np.clip(1 + 4 * (0.4 * pitch_norm + 0.3 * brightness + 0.3 * (1 - energy * 0.5)), 1, 5))
    dominance = float(np.clip(1 + 4 * (0.5 * energy + 0.3 * brightness + 0.2 * (1 - variability)), 1, 5))

    if energy > 0.6 and valence > 3.5:
        emotion = "exc" if variability > 0.4 else "hap"
    elif energy > 0.55 and valence < 2.8:
        emotion = "ang" if variability > 0.35 else "fru"
    elif energy < 0.25:
        emotion = "sad" if valence < 2.5 else "neu"
    elif zcr > 0.08 and energy > 0.35:
        emotion = "sur" if variability > 0.3 else "fea"
    else:
        emotion = "neu"

    features = {
        "rms": round(rms, 4),
        "zcr": round(zcr, 4),
        "spectral_centroid": round(centroid, 1),
        "pitch_mean": round(pitch_mean, 1),
        "pitch_std": round(pitch_std, 1),
        "energy_norm": round(energy, 3),
        "brightness_norm": round(brightness, 3),
    }
    return emotion, valence, activation, dominance, features


def _analyser_voix_audio(row, appariement=None):
    wav_path = row.get("wav_path") or resolve_wav_path(row["utterance_id"])
    if not wav_path or not os.path.isfile(wav_path):
        return None

    y, sr = charger_segment_wav(
        row["utterance_id"],
        row["start_time"],
        row["end_time"],
        wav_path=wav_path,
    )
    if y is None:
        return None

    emotion, valence, activation, dominance, features = _predire_emotion_acoustique(y, sr)
    row_audio = row.copy()
    row_audio["emotion"] = emotion
    row_audio["valence"] = valence
    row_audio["activation"] = activation
    row_audio["dominance"] = dominance
    row_audio["wav_path"] = wav_path

    return _resultat_depuis_row(
        row_audio,
        source="IEMOCAP audio (SER acoustique)",
        emotion_override=emotion,
        features=features,
        appariement=appariement,
    )


def analyser_voix_depuis_scenario(
    scenario_id=None,
    etat_dominant=None,
    utiliser_audio=None,
):
    """
    Retourne l'état émotionnel vocal depuis IEMOCAP.
    - etat_dominant : état DAiSEE du scénario (appariement sémantique)
    - utiliser_audio=True : segment .wav + SER acoustique
    - sinon : annotations CSV (fallback automatique si .wav absent)
    """
    df = charger_iemocap()
    prefer_wav = _ser_audio_active(utiliser_audio)
    row, appariement = selectionner_utterance_iemocap(
        df,
        scenario_id=scenario_id,
        etat_dominant=etat_dominant,
        prefer_wav=prefer_wav,
    )

    if _ser_audio_active(utiliser_audio):
        result_audio = _analyser_voix_audio(row, appariement=appariement)
        if result_audio is not None:
            result_audio["annotation_reference"] = {
                "emotion": row["emotion"],
                "etat_daisee": row.get("etat_daisee"),
                "valence": float(row["valence"]),
                "activation": float(row["activation"]),
            }
            return result_audio

        if not wav_exists(row["utterance_id"]):
            fallback = _resultat_depuis_row(
                row,
                source="IEMOCAP annotations (fallback, .wav absent)",
                appariement=appariement,
            )
            fallback["audio_demande"] = True
            return fallback

        fallback = _resultat_depuis_row(
            row,
            source="IEMOCAP annotations (fallback, segment audio)",
            appariement=appariement,
        )
        fallback["audio_demande"] = True
        return fallback

    return _resultat_depuis_row(
        row,
        source="IEMOCAP annotations (CSV)",
        appariement=appariement,
    )


def get_statistiques_iemocap():
    df = charger_iemocap()
    stats = {
        "total": len(df),
        "distribution": df["emotion"].value_counts().to_dict(),
        "etats_daisee": df["etat_daisee"].value_counts().to_dict(),
        "activation_mean": float(df["activation"].mean()),
        "valence_mean": float(df["valence"].mean()),
        "dominance_mean": float(df["dominance"].mean()),
    }
    if "wav_exists" in df.columns:
        stats["wav_disponibles"] = int(df["wav_exists"].sum())
        stats["wav_manquants"] = int((~df["wav_exists"]).sum())
    return stats


print("✅ Module SER IEMOCAP réel chargé !")
