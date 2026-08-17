# ============================================
# backend/ser_module.py
# Module SER — IEMOCAP (signal audio .wav réel, OBLIGATOIRE)
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
)
from iemocap_parser import charger_iemocap_complet, mapping_iemocap_daisee

# Alias canonique (source unique : iemocap_parser.mapping_iemocap_daisee)
mapping_ser_daisee = mapping_iemocap_daisee

# ===== SER APPRIS (wav2vec2 fine-tuné DAiSEE + SUPERB ER hybride) =====
# Le fine-tune DAiSEE (4 états) peut confondre hap→Confusion.
# En production on fusionne avec le SUPERB ER d'origine (neu/hap/ang/sad),
# meilleur sur les émotions conversationnelles classiques.
_MODELS = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "models"))
_SER_FINETUNED_DIR = next(
    (
        d
        for d in (
            os.path.join(_MODELS, "ser_wav2vec2_daisee_full"),
            os.path.join(_MODELS, "ser_wav2vec2_daisee"),
        )
        if os.path.isfile(os.path.join(d, "config.json"))
    ),
    os.path.join(_MODELS, "ser_wav2vec2_daisee"),
)
_SER_SUPERB_ID = os.getenv("EMOLEARN_SER_SUPERB_MODEL", "superb/wav2vec2-base-superb-er")
_modele_ser_ft = None
_modele_ser_superb = None

_df_iemocap = None
ETATS_DAISEE_SER = ("Boredom", "Confusion", "Engagement", "Frustration")

# Méthode SER audio : learned (hybride FT+SUPERB, défaut) | heuristique
SER_METHODE_DEFAUT = os.getenv("EMOLEARN_SER_METHODE", "learned").lower()
if SER_METHODE_DEFAUT in ("finetuned", "ft", "modele", "model", "learn"):
    SER_METHODE_DEFAUT = "learned"

# Hybride SUPERB+FT activé par défaut (désactiver : EMOLEARN_SER_HYBRID=0)
_SER_HYBRID = os.getenv("EMOLEARN_SER_HYBRID", "1").strip().lower() not in (
    "0",
    "false",
    "off",
    "no",
)


def _empty_probs_etats():
    return {e: 0.0 for e in ETATS_DAISEE_SER}


def _normaliser_probs(probs: dict) -> dict:
    out = {e: float(probs.get(e, 0.0)) for e in ETATS_DAISEE_SER}
    total = sum(out.values())
    if total <= 1e-12:
        out["Boredom"] = 1.0
        return out
    return {e: v / total for e, v in out.items()}


def ser_finetuned_disponible(path: str | None = None) -> bool:
    cible = path or _SER_FINETUNED_DIR
    return os.path.isfile(os.path.join(cible, "config.json"))


def get_modele_ser_finetuned():
    """Charge (lazy) le modèle fine-tuné local (4 états DAiSEE, incl. Confusion)."""
    global _modele_ser_ft
    if _modele_ser_ft is None:
        if not ser_finetuned_disponible():
            return None
        from transformers import pipeline

        print(f"Chargement SER fine-tuné ({_SER_FINETUNED_DIR})...")
        _modele_ser_ft = pipeline(
            "audio-classification",
            model=_SER_FINETUNED_DIR,
            device=-1,
        )
        print("SER fine-tuné chargé.")
    return _modele_ser_ft


def get_modele_ser_superb():
    """Charge (lazy) SUPERB ER (neu/hap/ang/sad) — ancre émotionnelle conversationnelle."""
    global _modele_ser_superb
    if _modele_ser_superb is None:
        from transformers import pipeline

        print(f"Chargement SER SUPERB ER ({_SER_SUPERB_ID})...")
        _modele_ser_superb = pipeline(
            "audio-classification",
            model=_SER_SUPERB_ID,
            device=-1,
        )
        print("SER SUPERB ER chargé.")
    return _modele_ser_superb


def ser_methode_active(methode: str | None = None) -> str:
    """
    learned (défaut) → hybride fine-tuné DAiSEE + SUPERB ER (fallback heuristique)
    heuristique      → features librosa + règles
    """
    m = (methode or SER_METHODE_DEFAUT).lower()
    if m in ("learned", "learn", "finetuned", "ft", "modele", "model", "hybrid", "hybride"):
        return "learned" if ser_finetuned_disponible() else "heuristique"
    return "heuristique"


def _probs_depuis_pipeline(preds, label_map=None) -> dict:
    """Agrège scores pipeline → distribution DAiSEE (optionnellement via mapping émotion)."""
    probs = _empty_probs_etats()
    for p in preds or []:
        lab = str(p.get("label", ""))
        score = float(p.get("score", 0.0))
        if label_map is not None:
            mapped = label_map.get(lab)
            if mapped is None and lab in mapping_ser_daisee:
                mapped = mapping_ser_daisee[lab]["etat"]
            if mapped in probs:
                probs[mapped] += score
        elif lab in probs:
            probs[lab] += score
    return _normaliser_probs(probs)


def fusionner_probs_ser(p_ft: dict, p_superb: dict | None, p_heur: dict | None = None):
    """
    Fusion adaptative FT DAiSEE + SUPERB ER (+ heuristique faible).

    Corrige le biais fréquent fine-tune hap→Confusion : si SUPERB est confiant
    sur Engagement/Frustration, on lui donne plus de poids.
    Retourne (etat, score, probs, meta).
    """
    p_ft = _normaliser_probs(p_ft or _empty_probs_etats())
    if not p_superb:
        etat = max(p_ft, key=p_ft.get)
        return etat, float(p_ft[etat]), p_ft, {"mode": "ft_seul", "w_superb": 0.0}

    p_er = _normaliser_probs(p_superb)
    er_top = max(p_er, key=p_er.get)
    er_score = float(p_er[er_top])
    ft_top = max(p_ft, key=p_ft.get)
    ft_score = float(p_ft[ft_top])

    # Poids SUPERB : plus fort si émotion claire (hap/ang) et FT tire vers Confusion
    if er_top in ("Engagement", "Frustration") and er_score >= 0.40 and ft_top == "Confusion":
        w_er = 0.78
        mode = "superb_prioritaire_vs_confusion"
    elif er_top in ("Engagement", "Frustration") and er_score >= 0.50:
        w_er = 0.62
        mode = "superb_confiant"
    elif er_score >= 0.45:
        w_er = 0.52
        mode = "blend_equilibre"
    else:
        w_er = 0.35
        mode = "ft_prioritaire"

    combined = {
        e: w_er * p_er.get(e, 0.0) + (1.0 - w_er) * p_ft.get(e, 0.0)
        for e in ETATS_DAISEE_SER
    }

    # Confusion : SUPERB n'en a pas — conserver une part FT (et heuristique)
    # pour ne pas perdre sur/fea, tout en évitant de sur-pondérer Confusion
    # quand SUPERB affirme hap/ang.
    if er_top in ("Engagement", "Frustration") and er_score >= 0.40:
        combined["Confusion"] *= 0.45

    if p_heur and ft_score < 0.38 and er_score < 0.40:
        p_h = _normaliser_probs(p_heur)
        for e in ETATS_DAISEE_SER:
            combined[e] = 0.85 * combined[e] + 0.15 * p_h.get(e, 0.0)
        mode = mode + "+heur"

    combined = _normaliser_probs(combined)
    etat = max(combined, key=combined.get)
    meta = {
        "mode": mode,
        "w_superb": round(w_er, 3),
        "ft_top": ft_top,
        "ft_score": round(ft_score, 3),
        "superb_top": er_top,
        "superb_score": round(er_score, 3),
    }
    return etat, float(combined[etat]), combined, meta


def _predire_emotion_finetuned(y, sr):
    """
    SER appris : fine-tuné DAiSEE, éventuellement fusionné avec SUPERB ER.
    Retourne (etat, score, probs_etats) ou None si modèle indisponible.
    """
    empty = _empty_probs_etats()
    if y is None or len(y) < max(int(sr * 0.05), 1):
        return "Boredom", 0.0, {**empty, "Boredom": 1.0}

    pipe_ft = get_modele_ser_finetuned()
    if pipe_ft is None:
        return None

    audio = {"raw": np.asarray(y, dtype=np.float32), "sampling_rate": int(sr)}
    preds_ft = pipe_ft(audio)
    p_ft = _probs_depuis_pipeline(preds_ft)

    p_superb = None
    if _SER_HYBRID:
        try:
            pipe_er = get_modele_ser_superb()
            preds_er = pipe_er(audio)
            # label_map={} → mappe neu/hap/ang/sad via mapping_ser_daisee
            p_superb = _probs_depuis_pipeline(preds_er, label_map={})
        except Exception as exc:
            print(f"⚠ SER SUPERB indisponible ({exc}) — FT seul.")
            p_superb = None

    # Repli heuristique léger uniquement si les deux modèles sont peu confiants
    p_heur = None
    if p_superb is not None:
        er_s = max(p_superb.values())
        ft_s = max(p_ft.values())
        if ft_s < 0.38 and er_s < 0.40:
            emo, *_rest = _predire_emotion_acoustique(y, sr)
            mapped = mapping_ser_daisee.get(emo, {}).get("etat", "Boredom")
            p_heur = {e: 0.0 for e in ETATS_DAISEE_SER}
            p_heur[mapped] = 0.55
            p_heur = _normaliser_probs(p_heur)

    etat, score, probs, meta = fusionner_probs_ser(p_ft, p_superb, p_heur)
    _predire_emotion_finetuned._last_meta = {
        **meta,
        "probs_ft": {k: round(v, 3) for k, v in p_ft.items()},
        "probs_superb": (
            {k: round(v, 3) for k, v in p_superb.items()} if p_superb else None
        ),
    }
    return etat, score, probs


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


def _onehot_etats(etat, score=1.0):
    """Distribution d'états « dure » (one-hot pondéré) pour la fusion par vote."""
    probs = {"Boredom": 0.0, "Confusion": 0.0, "Engagement": 0.0, "Frustration": 0.0}
    if etat in probs:
        probs[etat] = float(score)
    return probs


def _resultat_depuis_row(row, source: str, emotion_override=None, features=None,
                          appariement=None, probs_etats=None):
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
        "probs_etats": probs_etats or _onehot_etats(etat_daisee["etat"], max(score, 0.5)),
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

    # Ordre : valence haute → hap/exc avant sur/fea (évite Confusion sur voix joyeuses)
    if valence >= 3.4 and energy >= 0.35:
        emotion = "exc" if (energy > 0.65 and variability > 0.4) else "hap"
    elif energy > 0.55 and valence < 2.8:
        emotion = "ang" if variability > 0.35 else "fru"
    elif energy < 0.25:
        emotion = "sad" if valence < 2.5 else "neu"
    elif zcr > 0.09 and energy > 0.40 and valence < 3.2:
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


def _analyser_voix_audio(row, appariement=None, methode=None):
    methode = ser_methode_active(methode)
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

    # ===== SER APPRIS (hybride fine-tuné DAiSEE + SUPERB ER) =====
    if methode == "learned":
        pred = _predire_emotion_finetuned(y, sr)
        if pred is not None:
            etat, score, probs_etats = pred
            niveau = max(1, min(3, int(round(float(score) * 3))))
            ser_meta = getattr(_predire_emotion_finetuned, "_last_meta", {}) or {}
            hybrid = bool(ser_meta.get("w_superb", 0))
            return {
                "modalite": "voix_iemocap",
                "emotion": etat,
                "score": round(float(score), 3),
                "etat": etat,
                "niveau": niveau,
                "probs_etats": probs_etats,
                "valence": float(row["valence"]),
                "activation": float(row["activation"]),
                "dominance": float(row["dominance"]),
                "utterance_id": row["utterance_id"],
                "wav_path": wav_path,
                "segment": {
                    "start_time": float(row["start_time"]),
                    "end_time": float(row["end_time"]),
                    "duree": float(
                        row.get("duree", float(row["end_time"]) - float(row["start_time"]))
                    ),
                },
                "erreur": None,
                "source": (
                    "IEMOCAP audio (SER hybride SUPERB+DAiSEE)"
                    if hybrid
                    else "IEMOCAP audio (SER fine-tuné DAiSEE)"
                ),
                "ser_methode": "learned",
                "ser_details": ser_meta,
                "appariement": appariement,
            }

    # ===== SER HEURISTIQUE (features acoustiques librosa) =====
    emotion, valence, activation, dominance, features = _predire_emotion_acoustique(y, sr)
    row_audio = row.copy()
    row_audio["emotion"] = emotion
    row_audio["valence"] = valence
    row_audio["activation"] = activation
    row_audio["dominance"] = dominance
    row_audio["wav_path"] = wav_path

    result = _resultat_depuis_row(
        row_audio,
        source="IEMOCAP audio (SER acoustique)",
        emotion_override=emotion,
        features=features,
        appariement=appariement,
    )
    result["ser_methode"] = "heuristique"
    return result


def analyser_voix_depuis_scenario(
    scenario_id=None,
    etat_dominant=None,
    methode=None,
):
    """
    Retourne l'état émotionnel vocal depuis IEMOCAP — TOUJOURS depuis le
    signal audio réel (.wav), SER learned (fine-tuné DAiSEE) ou heuristique
    (librosa) selon `methode`. Le mode "annotations CSV seules" a été retiré :
    la couverture .wav du corpus IEMOCAP est de 100% (10 039/10 039), l'audio
    n'est donc plus une option mais une entrée obligatoire du pipeline.

    - etat_dominant : état DAiSEE du scénario (appariement sémantique)
    """
    df = charger_iemocap()
    row, appariement = selectionner_utterance_iemocap(
        df,
        scenario_id=scenario_id,
        etat_dominant=etat_dominant,
        prefer_wav=True,
    )

    result_audio = _analyser_voix_audio(row, appariement=appariement, methode=methode)
    if result_audio is not None:
        result_audio["annotation_reference"] = {
            "emotion": row["emotion"],
            "etat_daisee": row.get("etat_daisee"),
            "valence": float(row["valence"]),
            "activation": float(row["activation"]),
        }
        return result_audio

    # Cas limite (ne devrait pas se produire : couverture .wav = 100%) :
    # signalé explicitement, PAS de repli silencieux sur les annotations CSV.
    return {
        "modalite": "voix_iemocap",
        "emotion": None,
        "score": 0.0,
        "etat": "Boredom",
        "niveau": 1,
        "erreur": f"audio_indisponible:{row.get('utterance_id')}",
        "utterance_id": row.get("utterance_id"),
        "wav_path": row.get("wav_path") or resolve_wav_path(row["utterance_id"]),
        "appariement": appariement,
    }


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


def apercu_audio_scenario(scenario_id, etat_dominant=None):
    """Métadonnées de l'utterance IEMOCAP appariée (sans lancer le SER)."""
    df = charger_iemocap()
    row, appariement = selectionner_utterance_iemocap(
        df,
        scenario_id=scenario_id,
        etat_dominant=etat_dominant,
        prefer_wav=True,
    )
    wav_path = row.get("wav_path") or resolve_wav_path(row["utterance_id"])
    disponible = bool(wav_path) and os.path.isfile(wav_path)
    return {
        "scenario_id": int(scenario_id) if scenario_id is not None else 0,
        "utterance_id": row.get("utterance_id"),
        "emotion_iemocap": row.get("emotion"),
        "etat_daisee": row.get("etat_daisee"),
        "valence": float(row["valence"]),
        "activation": float(row["activation"]),
        "dominance": float(row["dominance"]),
        "start_time": float(row["start_time"]),
        "end_time": float(row["end_time"]),
        "duree": float(row.get("duree", row["end_time"] - row["start_time"])),
        "audio_disponible": disponible,
        "appariement": appariement,
    }


print("✅ Module SER IEMOCAP réel chargé !")
