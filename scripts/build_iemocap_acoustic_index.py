#!/usr/bin/env python3
# ============================================
# Index acoustique IEMOCAP (features librosa)
# pour appariement par similarité scénario ↔ voix
# ============================================
"""
Usage :
  python scripts/build_iemocap_acoustic_index.py
  python scripts/build_iemocap_acoustic_index.py --max-samples 2000

Produit :
  data/iemocap_acoustic_index.csv
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
BACKEND = os.path.join(ROOT, "backend")
sys.path.insert(0, BACKEND)

from iemocap_audio import charger_segment_wav

OUT = os.path.join(ROOT, "data", "iemocap_acoustic_index.csv")
CSV_IN = os.path.join(ROOT, "data", "iemocap_parsed.csv")


def extraire_dims_acoustiques(y, sr):
    """4 dims alignées sur l'index d'appariement (librosa, yin rapide)."""
    import librosa

    if y is None or len(y) < max(int(sr * 0.05), 1):
        return None

    rms = float(np.mean(librosa.feature.rms(y=y)))
    zcr = float(np.mean(librosa.feature.zero_crossing_rate(y=y)))
    centroid = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))

    f0 = librosa.yin(y, fmin=75, fmax=400, sr=sr)
    f0_valid = f0[~np.isnan(f0)] if f0 is not None else np.array([])
    pitch_mean = float(np.mean(f0_valid)) if len(f0_valid) > 0 else 150.0

    energy = min(1.0, rms * 15)
    brightness = min(1.0, centroid / 4000)
    pitch_norm = min(1.0, max(0.0, (pitch_mean - 80) / 250))

    return {
        "energy_norm": energy,
        "pitch_norm": pitch_norm,
        "zcr_norm": min(1.0, zcr * 10),
        "brightness_norm": brightness,
    }


def charger_corpus():
    if not os.path.isfile(CSV_IN):
        raise FileNotFoundError(f"CSV introuvable : {CSV_IN}")
    df = pd.read_csv(CSV_IN)
    if "wav_exists" in df.columns:
        df = df[df["wav_exists"]]
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="0 = tout le corpus avec wav",
    )
    args = parser.parse_args()

    df = charger_corpus()
    if args.max_samples and args.max_samples < len(df):
        df = df.sample(n=args.max_samples, random_state=42)

    total = len(df)
    print(f"Index acoustique IEMOCAP — {total} segments", flush=True)

    rows = []
    skipped = 0
    for n, (_, row) in enumerate(df.iterrows(), start=1):
        y, sr = charger_segment_wav(
            row["utterance_id"],
            float(row["start_time"]),
            float(row["end_time"]),
            wav_path=row.get("wav_path"),
        )
        if y is None:
            skipped += 1
            continue
        feats = extraire_dims_acoustiques(y, sr)
        if feats is None:
            skipped += 1
            continue
        rows.append({"utterance_id": row["utterance_id"], **feats})
        if n % 100 == 0 or n == total:
            print(
                f"  {len(rows)} profils / {n} traites ({skipped} skip)",
                flush=True,
            )

    out = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    out.to_csv(OUT, index=False)
    print(f"OK {len(out)} profils -> {OUT} (skip={skipped})", flush=True)


if __name__ == "__main__":
    main()
