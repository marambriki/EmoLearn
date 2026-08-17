#!/usr/bin/env python3
# ============================================
# scripts/build_iemocap_av_corpus.py
#
# Option A — Sous-corpus IEMOCAP AUDIOVISUEL ALIGNÉ
# (visage + voix du MÊME locuteur, synchronisés) pour VALIDER la fusion
# multimodale dans un cadre réellement aligné.
#
# Compare, sur le même corpus aligné, plusieurs améliorations :
#   • VISAGE  : FER 1 frame (milieu)  vs  FER multi-frames (vote, moyenne des probas)
#   • VOIX    : SER heuristique (librosa)  vs  SER appris (wav2vec2 IEMOCAP)
#   • FUSION  : fusion historique (priorité pédagogique)  vs  fusion par VOTE pondéré
#   • QUALITÉ : nettoyage optionnel (segments courts, visage non détecté)
#
# Vérité terrain = annotation IEMOCAP mappée vers état DAiSEE (etat_daisee).
#
# Sorties : results/iemocap_av_fusion.csv + results/iemocap_av_metrics.json
# Contrainte données : seule Session1 contient les vidéos avi/DivX.
# ============================================

from __future__ import annotations

import argparse
import json
import os
import sys

import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
BACKEND = os.path.join(ROOT, "backend")
RESULTS = os.path.join(ROOT, "results")
sys.path.insert(0, BACKEND)

import iemocap_video as iv
from iemocap_audio import resolve_wav_path, wav_exists
import fer_offline
import ser_module
from fer_classifier import mapping_fer_daisee
from fusion_module import (
    fusionner_modalites,
    fusionner_par_vote,
    agreger_probs_etats,
)

ETATS = ["Boredom", "Confusion", "Engagement", "Frustration"]
ETATS_URGENTS = ["Confusion", "Frustration"]


# ---------------------------------------------------------------------------
# VOIX — SER aligné (même locuteur/segment), méthode au choix
# ---------------------------------------------------------------------------
def _enriched_row(row):
    uid = row["utterance_id"]
    return {
        "utterance_id": uid,
        "start_time": float(row["start_time"]),
        "end_time": float(row["end_time"]),
        "duree": float(row.get("duree", float(row["end_time"]) - float(row["start_time"]))),
        "emotion": row["emotion"],
        "valence": float(row["valence"]),
        "activation": float(row["activation"]),
        "dominance": float(row["dominance"]),
        "etat_daisee": row["etat_daisee"],
        "wav_path": resolve_wav_path(uid),
    }


def _ser_aligne(row, methode):
    """SER sur le même locuteur/segment. Repli annotation si .wav indisponible."""
    enriched = _enriched_row(row)
    wav_path = enriched["wav_path"]
    if wav_path and os.path.isfile(wav_path):
        res = ser_module._analyser_voix_audio(enriched, methode=methode)
        if res is not None:
            return res
    return ser_module._resultat_depuis_row(
        enriched, source="IEMOCAP annotation (repli, .wav indisponible)"
    )


# ---------------------------------------------------------------------------
# VISAGE — FER multi-frames (vote par moyenne des probabilités)
# ---------------------------------------------------------------------------
def _fer_multi(images):
    """Moyenne des probas d'émotion sur N frames → résultat FER agrégé + probs_etats."""
    cumul = {}
    n = 0
    for img in images:
        r = fer_offline.analyser_visage_image(img)
        if r.get("erreur"):
            continue
        for emo, p in (r.get("tous_scores") or {}).items():
            cumul[emo] = cumul.get(emo, 0.0) + float(p)
        n += 1
    if n == 0:
        return None
    avg = {emo: v / n for emo, v in cumul.items()}
    probs_etats = agreger_probs_etats(avg, mapping_fer_daisee)
    emo_top = max(avg, key=avg.get)
    etat = mapping_fer_daisee.get(emo_top, {"etat": "Boredom", "niveau": 1})
    return {
        "modalite": "visage_fer_iemocap_multi",
        "emotion": emo_top,
        "score": round(float(avg[emo_top]), 3),
        "etat": etat["etat"],
        "niveau": etat["niveau"],
        "tous_scores": avg,
        "probs_etats": probs_etats,
        "erreur": None,
    }


# ---------------------------------------------------------------------------
# Construction du corpus
# ---------------------------------------------------------------------------
def construire_corpus(args):
    df = pd.read_csv(os.path.join(ROOT, "data", "iemocap_parsed.csv"))
    df = df[df["utterance_id"].astype(str).str.startswith("Ses01")].copy()
    if not args.inclure_xxx:
        df = df[df["emotion"] != "xxx"].copy()

    # Vidéo + audio disponibles
    df = df[df["utterance_id"].apply(iv.avi_exists)].copy()
    df = df[df["utterance_id"].apply(lambda u: bool(resolve_wav_path(u)) and wav_exists(u))].copy()

    # Nettoyage qualité : segments trop courts
    df["_duree"] = df["end_time"].astype(float) - df["start_time"].astype(float)
    if args.duree_min > 0:
        df = df[df["_duree"] >= args.duree_min].copy()

    # Échantillon équilibré par état
    parts = []
    for etat in ETATS:
        sub = df[df["etat_daisee"] == etat]
        if len(sub) > 0:
            parts.append(sub.sample(min(args.n_per_state, len(sub)), random_state=args.seed))
    echantillon = pd.concat(parts).reset_index(drop=True)
    print(f"Échantillon : {len(echantillon)} utterances (durée_min={args.duree_min}s)")
    print(echantillon["etat_daisee"].value_counts().to_dict())

    lignes = []
    for i, row in echantillon.iterrows():
        uid = row["utterance_id"]
        images, vmeta = iv.extract_face_frames(
            uid, float(row["start_time"]), float(row["end_time"]), n_frames=args.fer_frames
        )
        if not images:
            lignes.append({"utterance_id": uid, "gt_etat": row["etat_daisee"],
                           "skip": True, "skip_raison": vmeta.get("erreur")})
            continue
        if args.exclure_visage_non_detecte and vmeta["n_visages_detectes"] == 0:
            lignes.append({"utterance_id": uid, "gt_etat": row["etat_daisee"],
                           "skip": True, "skip_raison": "aucun_visage"})
            continue

        # FER 1 frame (milieu) vs multi-frames
        mid = len(images) // 2
        fer_1f = fer_offline.analyser_visage_image(images[mid])
        fer_1f["probs_etats"] = agreger_probs_etats(fer_1f.get("tous_scores"), mapping_fer_daisee)
        fer_multi = _fer_multi(images) or fer_1f

        # SER heuristique vs fine-tuné DAiSEE (même segment)
        ser_heur = _ser_aligne(row, methode="heuristique")
        ser_learned = _ser_aligne(row, methode="learned")

        # Même fusion (incertitude) : perception faible vs perception améliorée
        fusion_old = fusionner_modalites(fer_1f, ser_heur)
        fusion_vote = fusionner_par_vote(fer_multi, ser_learned, poids_visage=args.poids_visage)

        lignes.append({
            "utterance_id": uid,
            "emotion_iemocap": row["emotion"],
            "gt_etat": row["etat_daisee"],
            "skip": False,
            "n_frames": vmeta["n_frames"],
            "n_visages": vmeta["n_visages_detectes"],
            "duree": round(float(row["_duree"]), 2),
            # Visage
            "fer_1f_etat": fer_1f["etat"],
            "fer_multi_etat": fer_multi["etat"],
            # Voix
            "ser_heur_etat": ser_heur["etat"],
            "ser_modele_etat": ser_learned["etat"],
            "ser_modele_source": ser_learned.get("source"),
            # Fusion
            "fusion_old_etat": fusion_old["etat"],
            "fusion_vote_etat": fusion_vote["etat"],
        })
        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{len(echantillon)} traités")

    return pd.DataFrame(lignes)


# ---------------------------------------------------------------------------
# Métriques
# ---------------------------------------------------------------------------
def _accuracy(df, col):
    v = df[(~df["skip"]) & df[col].notna()]
    return round(float((v[col] == v["gt_etat"]).mean()), 3) if len(v) else 0.0


def _rappel_urgents(df, col):
    v = df[(~df["skip"]) & df[col].notna()]
    urgents = v[v["gt_etat"].isin(ETATS_URGENTS)]
    if len(urgents) == 0:
        return 0.0
    return round(float(urgents[col].isin(ETATS_URGENTS).mean()), 3)


def _confusion(df, col):
    v = df[(~df["skip"]) & df[col].notna()]
    if len(v) == 0:
        return {}
    return v.groupby(["gt_etat", col]).size().unstack(fill_value=0).to_dict(orient="index")


def calculer_metriques(df):
    valides = df[~df["skip"]]
    cols = {
        "visage_fer_1frame": "fer_1f_etat",
        "visage_fer_multiframes": "fer_multi_etat",
        "voix_ser_heuristique": "ser_heur_etat",
        "voix_ser_finetuned_daisee": "ser_modele_etat",
        "fusion_historique": "fusion_old_etat",
        "fusion_vote_pondere": "fusion_vote_etat",
    }
    accuracy = {k: _accuracy(df, c) for k, c in cols.items()}
    rappel = {k: _rappel_urgents(df, c) for k, c in cols.items()}

    return {
        "n_total": len(df),
        "n_valide": int(len(valides)),
        "n_skip": int(df["skip"].sum()),
        "taux_visage_detecte": round(float((valides["n_visages"] > 0).mean()), 3) if len(valides) else 0.0,
        "accuracy_vs_annotation": accuracy,
        "rappel_etats_urgents": rappel,
        "gains": {
            "ser_appris_vs_heuristique": round(
                accuracy["voix_ser_appris_wav2vec2"] - accuracy["voix_ser_heuristique"], 3),
            "fer_multi_vs_1frame": round(
                accuracy["visage_fer_multiframes"] - accuracy["visage_fer_1frame"], 3),
            "fusion_vote_vs_historique": round(
                accuracy["fusion_vote_pondere"] - accuracy["fusion_historique"], 3),
            "fusion_vote_vs_meilleure_unimodale": round(
                accuracy["fusion_vote_pondere"]
                - max(accuracy["visage_fer_multiframes"], accuracy["voix_ser_appris_wav2vec2"]), 3),
        },
        "distribution_gt": valides["gt_etat"].value_counts().to_dict() if len(valides) else {},
        "confusion_fusion_vote": _confusion(df, "fusion_vote_etat"),
    }


def main():
    parser = argparse.ArgumentParser(description="Sous-corpus IEMOCAP audiovisuel aligné")
    parser.add_argument("--n-per-state", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fer-frames", type=int, default=5, help="frames FER par utterance (vote)")
    parser.add_argument("--poids-visage", type=float, default=0.60)
    parser.add_argument("--duree-min", type=float, default=1.0, help="durée min du segment (s)")
    parser.add_argument("--exclure-visage-non-detecte", action="store_true")
    parser.add_argument("--inclure-xxx", action="store_true")
    parser.add_argument("--out", default=os.path.join(RESULTS, "iemocap_av_fusion.csv"))
    args = parser.parse_args()

    os.makedirs(RESULTS, exist_ok=True)
    df = construire_corpus(args)
    df.to_csv(args.out, index=False)
    print(f"\n✅ CSV écrit : {args.out}")

    metriques = calculer_metriques(df)
    metrics_path = os.path.join(RESULTS, "iemocap_av_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metriques, f, indent=2, ensure_ascii=False)
    print(f"✅ Métriques écrites : {metrics_path}\n")
    print("Accuracy vs annotation :")
    print(json.dumps(metriques["accuracy_vs_annotation"], indent=2, ensure_ascii=False))
    print("\nGains :")
    print(json.dumps(metriques["gains"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
