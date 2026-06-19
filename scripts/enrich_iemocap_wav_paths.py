#!/usr/bin/env python3
"""
Enrichit data/iemocap_parsed.csv avec les chemins .wav IEMOCAP.
Usage : python scripts/enrich_iemocap_wav_paths.py
"""

import os
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "backend"))

from iemocap_audio import enrich_row_wav_paths  # noqa: E402

CSV_IN = os.path.join(ROOT, "data", "iemocap_parsed.csv")
CSV_OUT = CSV_IN


def main():
    if not os.path.isfile(CSV_IN):
        print(f"Fichier introuvable : {CSV_IN}")
        sys.exit(1)

    df = pd.read_csv(CSV_IN)
    enriched = [enrich_row_wav_paths(row.to_dict()) for _, row in df.iterrows()]
    out = pd.DataFrame(enriched)

    out.to_csv(CSV_OUT, index=False)
    disponibles = int(out["wav_exists"].sum()) if "wav_exists" in out.columns else 0
    print(f"✅ CSV enrichi : {CSV_OUT}")
    print(f"   Utterances : {len(out)}")
    print(f"   .wav disponibles : {disponibles}/{len(out)}")


if __name__ == "__main__":
    main()
