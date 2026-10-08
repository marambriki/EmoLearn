# ============================================
# backend/ser_module.py
# Module SER — IEMOCAP (signal audio .wav réel, OBLIGATOIRE)
# WavLM fine-tuné DAiSEE + ancre WavLM IEMOCAP ER (hybride)
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
from iemocap_appariement import selectionner_par_similarite

mapping_ser_daisee = mapping_iemocap_daisee

# ===== SER APPRIS (WavLM fine-tuné DAiSEE + ancre IEMOCAP ER hybride) =====
_MODELS = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "models"))
_SER_DIR_HEAD = os.path.join(_MODELS, "ser_wavlm_daisee")
_SER_DIR_FULL = os.path.join(_MODELS, "ser_wavlm_daisee_full")


def _model_dir_ok(path: str) -> bool:
    return os.path.isfile(os.path.join(path, "config.json"))


def _resolve_ser_finetuned_dir() -> str:
    """
    Checkpoint WavLM DAiSEE pour le runtime.
    Défaut : tête seule (meilleur F1 macro mesuré ~0,43 vs ~0,40 full sur val).
    EMOLEARN_SER_USE_FULL=1 → force ser_wavlm_daisee_full si présent.
    EMOLEARN_SER_MODEL_DIR=/chemin → override explicite.
    """
    override = os.getenv("EMOLEARN_SER_MODEL_DIR", "").strip()
    if override and _model_dir_ok(override):
        return override
    use_full = os.getenv("EMOLEARN_SER_USE_FULL", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "full",
    )
    if use_full and _model_dir_ok(_SER_DIR_FULL):
        return _SER_DIR_FULL
    if _model_dir_ok(_SER_DIR_HEAD):
        return _SER_DIR_HEAD
    if _model_dir_ok(_SER_DIR_FULL):
        return _SER_DIR_FULL
    return _SER_DIR_HEAD


_SER_FINETUNED_DIR = _resolve_ser_finetuned_dir()
_SER_ANCHOR_DIR = next(
    (
        d
        for d in (os.path.join(_MODELS, "ser_wavlm_iemocap_er"),)
        if os.path.isfile(os.path.join(d, "config.json"))
    ),
    os.path.join(_MODELS, "ser_wavlm_iemocap_er"),
)
# HF fallback si pas de checkpoint local (ancien SUPERB wav2vec2 — déprécié)
_SER_ANCHOR_HF = os.getenv("EMOLEARN_SER_ANCHOR_MODEL", "")

_modele_ser_ft = None
_modele_ser_anchor = None

_df_iemocap = None
ETATS_DAISEE_SER = ("Boredom", "Confusion", "Engagement", "Frustration")

SER_METHODE_DEFAUT = os.getenv("EMOLEARN_SER_METHODE", "learned").lower()
if SER_METHODE_DEFAUT in ("finetuned", "ft", "modele", "model", "learn"):
    SER_METHODE_DEFAUT = "learned"

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
    return _model_dir_ok(path or _SER_FINETUNED_DIR)


def ser_finetune_mode(path: str | None = None) -> str:
    """head | full | unknown — lu depuis emolearn_meta.json si présent."""
    cible = path or _SER_FINETUNED_DIR
    meta_path = os.path.join(cible, "emolearn_meta.json")
    if os.path.isfile(meta_path):
        try:
            import json

            with open(meta_path, encoding="utf-8") as f:
                meta = json.load(f)
            if meta.get("unfreeze_all"):
                return "full"
            if meta.get("unfreeze_last"):
                return "last_layer"
            return "head"
        except Exception:
            pass
    if cible.rstrip("/\\").endswith("_full"):
        return "full"
    return "unknown"


def ser_anchor_disponible() -> bool:
    if _model_dir_ok(_SER_ANCHOR_DIR):
        return True
    return bool(_SER_ANCHOR_HF)


def get_modele_ser_finetuned():
    """Charge (lazy) WavLM fine-tuné local (4 états DAiSEE, incl. Confusion)."""
    global _modele_ser_ft
    if _modele_ser_ft is None:
        if not ser_finetuned_disponible():
            return None
        from transformers import pipeline

        print(f"Chargement SER WavLM fine-tuné ({_SER_FINETUNED_DIR})...")
        _modele_ser_ft = pipeline(
            "audio-classification",
            model=_SER_FINETUNED_DIR,
            device=-1,
        )
        print("SER WavLM fine-tuné chargé.")
    return _modele_ser_ft


def get_modele_ser_anchor():
    """Charge (lazy) ancre IEMOCAP ER (neu/hap/ang/sad) — WavLM fine-tuné local."""
    global _modele_ser_anchor
    if _modele_ser_anchor is None:
        from transformers import pipeline

        if _model_dir_ok(_SER_ANCHOR_DIR):
            print(f"Chargement SER ancre IEMOCAP ER ({_SER_ANCHOR_DIR})...")
            _modele_ser_anchor = pipeline(
                "audio-classification",
                model=_SER_ANCHOR_DIR,
                device=-1,
            )
        elif _SER_ANCHOR_HF:
            print(f"Chargement SER ancre HF ({_SER_ANCHOR_HF})...")
            _modele_ser_anchor = pipeline(
                "audio-classification",
                model=_SER_ANCHOR_HF,
                device=-1,
            )
        else:
            return None
        print("SER ancre IEMOCAP ER chargé.")
    return _modele_ser_anchor


def ser_methode_active(methode: str | None = None) -> str:
    """
    learned (défaut) → hybride WavLM FT DAiSEE + ancre IEMOCAP ER
    heuristique      → features librosa + règles
    """
    m = (methode or SER_METHODE_DEFAUT).lower()
    if m in ("learned", "learn", "finetuned", "ft", "modele", "model", "hybrid", "hybride"):
        return "learned" if ser_finetuned_disponible() else "heuristique"
    return "heuristique"


def _probs_depuis_pipeline(preds, label_map=None) -> dict:
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


def fusionner_probs_ser(p_ft: dict, p_anchor: dict | None, p_heur: dict | None = None):
    """
    Fusion WavLM FT DAiSEE (prioritaire) + ancre IEMOCAP ER (correction ciblée).

    L'ancre sert surtout à corriger le biais Confusion du FT.
    Si le FT dit Engagement (ex. rire) et l'ancre sur-interprète en Frustration
    (ang), on garde le FT — l'ancre ne doit pas écraser une voix positive.
    """
    p_ft = _normaliser_probs(p_ft or _empty_probs_etats())
    ft_top = max(p_ft, key=p_ft.get)
    ft_score = float(p_ft[ft_top])

    if not p_anchor:
        combined = dict(p_ft)
        mode = "ft_seul"
        w_er = 0.0
    else:
        p_er = _normaliser_probs(p_anchor)
        er_top = max(p_er, key=p_er.get)
        er_score = float(p_er[er_top])

        w_er = 0.22
        mode = "ft_prioritaire"

        if ft_top == "Confusion" and er_top in ("Engagement", "Frustration") and er_score >= 0.40:
            w_er = 0.78
            mode = "anchor_vs_confusion"
        elif ft_top == "Engagement" and er_top == "Frustration" and ft_score >= 0.33:
            w_er = 0.0
            mode = "ft_vs_anchor_ang_sur_rire"
        elif ft_top == "Frustration" and er_top == "Engagement" and ft_score >= 0.38:
            w_er = 0.18
            mode = "ft_frustration_vs_anchor_hap"
        elif ft_score < 0.32 and er_score >= 0.48:
            w_er = 0.50
            mode = "anchor_si_ft_incertain"

        combined = {
            e: w_er * p_er.get(e, 0.0) + (1.0 - w_er) * p_ft.get(e, 0.0)
            for e in ETATS_DAISEE_SER
        }

        if ft_top == "Confusion" and er_top in ("Engagement", "Frustration") and er_score >= 0.40:
            combined["Confusion"] *= 0.45

        er_top = max(p_er, key=p_er.get)
        er_score = float(p_er[er_top])

    if p_heur:
        p_h = _normaliser_probs(p_heur)
        h_top = max(p_h, key=p_h.get)
        h_score = float(p_h[h_top])
        if h_top == "Engagement" and h_score >= 0.50 and ft_top == "Engagement":
            w_er = min(w_er, 0.08) if p_anchor else 0.0
            mode = mode + "+heur_hap"
            for e in ETATS_DAISEE_SER:
                combined[e] = 0.82 * combined.get(e, p_ft.get(e, 0.0)) + 0.18 * p_h.get(e, 0.0)
        elif ft_score < 0.35 and h_score >= 0.50:
            for e in ETATS_DAISEE_SER:
                combined[e] = 0.80 * combined.get(e, p_ft.get(e, 0.0)) + 0.20 * p_h.get(e, 0.0)
            mode = mode + "+heur"

    combined = _normaliser_probs(combined)
    etat = max(combined, key=combined.get)
    meta = {
        "mode": mode,
        "w_anchor": round(w_er, 3),
        "w_superb": round(w_er, 3),
        "ft_top": ft_top,
        "ft_score": round(ft_score, 3),
    }
    if p_anchor:
        meta["anchor_top"] = er_top
        meta["anchor_score"] = round(er_score, 3)
        meta["superb_top"] = er_top
        meta["superb_score"] = round(er_score, 3)
    return etat, float(combined[etat]), combined, meta


def _predire_emotion_finetuned(y, sr):
    empty = _empty_probs_etats()
    if y is None or len(y) < max(int(sr * 0.05), 1):
        return "Boredom", 0.0, {**empty, "Boredom": 1.0}

    pipe_ft = get_modele_ser_finetuned()
    if pipe_ft is None:
        return None

    audio = {"raw": np.asarray(y, dtype=np.float32), "sampling_rate": int(sr)}
    preds_ft = pipe_ft(audio)
    p_ft = _probs_depuis_pipeline(preds_ft)

    p_anchor = None
    if _SER_HYBRID and ser_anchor_disponible():
        try:
            pipe_er = get_modele_ser_anchor()
            if pipe_er is not None:
                preds_er = pipe_er(audio)
                p_anchor = _probs_depuis_pipeline(preds_er, label_map={})
        except Exception as exc:
            print(f"⚠ SER ancre IEMOCAP indisponible ({exc}) — FT seul.")
            p_anchor = None

    emo_heur, *_rest = _predire_emotion_acoustique(y, sr)
    etat_heur = mapping_ser_daisee.get(emo_heur, {}).get("etat", "Boredom")
    p_heur = {e: 0.0 for e in ETATS_DAISEE_SER}
    if emo_heur in ("hap", "exc"):
        p_heur["Engagement"] = 0.62
    elif emo_heur in ("ang", "fru", "dis"):
        p_heur["Frustration"] = 0.58
    elif emo_heur in ("sur", "fea"):
        p_heur["Confusion"] = 0.55
    elif emo_heur in ("neu", "sad"):
        p_heur["Boredom"] = 0.52
    else:
        p_heur[etat_heur] = 0.45
    p_heur = _normaliser_probs(p_heur)

    etat, score, probs, meta = fusionner_probs_ser(p_ft, p_anchor, p_heur)
    _predire_emotion_finetuned._last_meta = {
        **meta,
        "backbone": "wavlm",
        "heur_emotion": emo_heur,
        "probs_ft": {k: round(v, 3) for k, v in p_ft.items()},
        "probs_heur": {k: round(v, 3) for k, v in p_heur.items()},
        "probs_anchor": (
            {k: round(v, 3) for k, v in p_anchor.items()} if p_anchor else None
        ),
        "probs_superb": (
            {k: round(v, 3) for k, v in p_anchor.items()} if p_anchor else None
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


def _scores_depuis_etat_dominant(etat_dominant, niveau=2):
    scores = {
        "boredom": 0.0,
        "confusion": 0.0,
        "engagement": 0.0,
        "frustration": 0.0,
    }
    cle = {
        "Boredom": "boredom",
        "Confusion": "confusion",
        "Engagement": "engagement",
        "Frustration": "frustration",
    }.get(str(etat_dominant), "engagement")
    scores[cle] = float(niveau)
    return scores


def selectionner_utterance_iemocap(
    df,
    scenario_id=None,
    etat_dominant=None,
    scores_daisee=None,
    prefer_wav=False,
):
    sid = 0 if scenario_id is None else int(scenario_id)

    if scores_daisee:
        b = scores_daisee.get("boredom", 0)
        c = scores_daisee.get("confusion", 0)
        e = scores_daisee.get("engagement", 0)
        f = scores_daisee.get("frustration", 0)
    elif etat_dominant:
        s = _scores_depuis_etat_dominant(etat_dominant)
        b, c, e, f = s["boredom"], s["confusion"], s["engagement"], s["frustration"]
    else:
        s = _scores_depuis_etat_dominant("Engagement")
        b, c, e, f = s["boredom"], s["confusion"], s["engagement"], s["frustration"]

    return selectionner_par_similarite(
        df,
        scenario_id=sid,
        boredom=b,
        confusion=c,
        engagement=e,
        frustration=f,
        prefer_wav=prefer_wav,
    )


def _onehot_etats(etat, score=1.0):
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

    if methode == "learned":
        pred = _predire_emotion_finetuned(y, sr)
        if pred is not None:
            etat, score, probs_etats = pred
            niveau = max(1, min(3, int(round(float(score) * 3))))
            ser_meta = getattr(_predire_emotion_finetuned, "_last_meta", {}) or {}
            hybrid = bool(ser_meta.get("w_anchor", ser_meta.get("w_superb", 0)))
            return {
                "modalite": "voix_iemocap",
                "emotion": etat,
                "score": round(float(score), 3),
                "etat": etat,
                "niveau": niveau,
                "probs_etats": probs_etats,
                "emotion_iemocap": row.get("emotion"),
                "annotation_etat": row.get("etat_daisee"),
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
                    "IEMOCAP audio (SER hybride WavLM DAiSEE+IEMOCAP ER)"
                    if hybrid
                    else "IEMOCAP audio (SER WavLM fine-tuné DAiSEE)"
                ),
                "ser_methode": "learned",
                "ser_details": ser_meta,
                "appariement": appariement,
            }

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
    scores_daisee=None,
    methode=None,
):
    df = charger_iemocap()
    row, appariement = selectionner_utterance_iemocap(
        df,
        scenario_id=scenario_id,
        etat_dominant=etat_dominant,
        scores_daisee=scores_daisee,
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


def apercu_audio_scenario(scenario_id, etat_dominant=None, scores_daisee=None):
    df = charger_iemocap()
    row, appariement = selectionner_utterance_iemocap(
        df,
        scenario_id=scenario_id,
        etat_dominant=etat_dominant,
        scores_daisee=scores_daisee,
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
