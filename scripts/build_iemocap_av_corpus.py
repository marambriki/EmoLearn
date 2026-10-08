#!/usr/bin/env python3
# ============================================
# scripts/build_iemocap_av_corpus.py
#
# Option A — Sous-corpus IEMOCAP AUDIOVISUEL ALIGNÉ
# (visage + voix du MÊME locuteur, synchronisés) pour VALIDER la fusion
# multimodale dans un cadre réellement aligné.
#
# Compare, sur le même corpus aligné, le stack PRODUIT :
#   • VISAGE  : FER+AU 1 frame (milieu)  vs  FER+AU multi-frames
#   • VOIX    : SER heuristique (librosa)  vs  WavLM hybride (DAiSEE + ancre IEMOCAP)
#   • FUSION  : perception faible (1f + heuristique)  vs  vote à incertitude (multi + WavLM)
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
from au_module import extraire_au, moyenner_au
from fer_classifier import appliquer_mapping, model_au_disponible
from fer_offline import analyser_visage_image_pretrained
import ser_module
from fusion_module import fusionner_modalites, fusionner_par_vote

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
# VISAGE — FER+AU (même juge que le runtime produit)
# ---------------------------------------------------------------------------
def _fer_au(images):
    """Scores FER pretrained 7 + AU MediaPipe → classifieur FER+AU."""
    if not images:
        return None
    if not model_au_disponible():
        raise RuntimeError("Classifieur FER+AU introuvable (models/fer_daisee_clf_au.joblib).")
    sums = {}
    n = 0
    aus = []
    for img in images:
        r = analyser_visage_image_pretrained(img)
        if r.get("erreur") or not r.get("tous_scores"):
            continue
        n += 1
        for lab, p in r["tous_scores"].items():
            sums[lab] = sums.get(lab, 0.0) + float(p)
        aus.append(extraire_au(img))
    if n == 0:
        return None
    fer7 = {lab: total / n for lab, total in sums.items()}
    au_scores = moyenner_au(aus) if aus else moyenner_au([])
    emo7 = max(fer7, key=fer7.get)
    etat, niveau, mapping_source, probs_etats = appliquer_mapping(
        emo7, fer7, au_scores=au_scores
    )
    return {
        "modalite": "visage_fer_au",
        "emotion": emo7,
        "score": round(float(fer7[emo7]), 3),
        "etat": etat,
        "niveau": niveau,
        "tous_scores": fer7,
        "probs_etats": probs_etats,
        "mapping_source": mapping_source,
        "au_actif": True,
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

        # FER+AU : 1 frame (milieu) vs multi-frames
        mid = len(images) // 2
        fer_1f = _fer_au([images[mid]])
        fer_multi = _fer_au(images) or fer_1f
        if fer_1f is None:
            lignes.append({"utterance_id": uid, "gt_etat": row["etat_daisee"],
                           "skip": True, "skip_raison": "fer_au_echec"})
            continue

        # SER heuristique vs WavLM hybride (même segment)
        ser_heur = _ser_aligne(row, methode="heuristique")
        ser_learned = _ser_aligne(row, methode="learned")

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
            "fer_1f_etat": fer_1f["etat"],
            "fer_1f_mapping": fer_1f.get("mapping_source"),
            "fer_multi_etat": fer_multi["etat"],
            "fer_multi_mapping": fer_multi.get("mapping_source"),
            "ser_heur_etat": ser_heur["etat"],
            "ser_modele_etat": ser_learned["etat"],
            "ser_modele_source": ser_learned.get("source"),
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
        "visage_fer_au_1frame": "fer_1f_etat",
        "visage_fer_au_multiframes": "fer_multi_etat",
        "voix_ser_heuristique": "ser_heur_etat",
        "voix_ser_wavlm": "ser_modele_etat",
        "fusion_perception_faible": "fusion_old_etat",
        "fusion_vote_incertitude": "fusion_vote_etat",
    }
    accuracy = {k: _accuracy(df, c) for k, c in cols.items()}
    rappel = {k: _rappel_urgents(df, c) for k, c in cols.items()}
    meilleur_uni = max(
        accuracy["visage_fer_au_multiframes"],
        accuracy["voix_ser_wavlm"],
    )

    return {
        "stack": {
            "visage": "FER+AU (pretrained 7 + MediaPipe AU + RF)",
            "voix": "WavLM hybride DAiSEE + ancre IEMOCAP ER",
            "fusion": "vote à pondération continue par incertitude",
        },
        "n_total": len(df),
        "n_valide": int(len(valides)),
        "n_skip": int(df["skip"].sum()),
        "taux_visage_detecte": round(float((valides["n_visages"] > 0).mean()), 3) if len(valides) else 0.0,
        "accuracy_vs_annotation": accuracy,
        "rappel_etats_urgents": rappel,
        "gains": {
            "ser_wavlm_vs_heuristique": round(
                accuracy["voix_ser_wavlm"] - accuracy["voix_ser_heuristique"], 3),
            "fer_au_multi_vs_1frame": round(
                accuracy["visage_fer_au_multiframes"] - accuracy["visage_fer_au_1frame"], 3),
            "fusion_vote_vs_perception_faible": round(
                accuracy["fusion_vote_incertitude"] - accuracy["fusion_perception_faible"], 3),
            "fusion_vote_vs_meilleure_unimodale": round(
                accuracy["fusion_vote_incertitude"] - meilleur_uni, 3),
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
    parser.add_argument("--metrics", default=None)
    args = parser.parse_args()

    os.makedirs(RESULTS, exist_ok=True)
    metrics_path = args.metrics or os.path.join(RESULTS, "iemocap_av_metrics.json")
    if os.path.abspath(metrics_path) != os.path.abspath(
        os.path.join(RESULTS, "iemocap_av_metrics.json")
    ):
        pass
    else:
        hist_metrics = os.path.join(RESULTS, "iemocap_av_metrics_historique_wav2vec2.json")
        hist_csv = os.path.join(RESULTS, "iemocap_av_fusion_historique.csv")
        live_csv = os.path.join(RESULTS, "iemocap_av_fusion.csv")
        if os.path.isfile(metrics_path) and not os.path.isfile(hist_metrics):
            import shutil
            shutil.copy2(metrics_path, hist_metrics)
            if os.path.isfile(live_csv) and not os.path.isfile(hist_csv):
                shutil.copy2(live_csv, hist_csv)
            print(f"Archivé le banc wav2vec2 → {hist_metrics}")

    print(
        "Stack : FER+AU + WavLM hybride + fusion incertitude "
        f"(n_per_state={args.n_per_state}, frames={args.fer_frames})"
    )
    df = construire_corpus(args)
    df.to_csv(args.out, index=False)
    print(f"\n✅ CSV écrit : {args.out}")

    metriques = calculer_metriques(df)
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metriques, f, indent=2, ensure_ascii=False)
    print(f"✅ Métriques écrites : {metrics_path}\n")
    print("Accuracy vs annotation :")
    print(json.dumps(metriques["accuracy_vs_annotation"], indent=2, ensure_ascii=False))
    print("\nGains :")
    print(json.dumps(metriques["gains"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
