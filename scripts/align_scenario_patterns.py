#!/usr/bin/env python3
"""Aligne pattern sur l'état dominant dans data/scenarios_50.csv."""

import os
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "backend"))

from daisee_utils import etat_dominant_depuis_scores, pattern_depuis_scores

CSV_PATH = os.path.join(ROOT, "data", "scenarios_50.csv")


def main():
    df = pd.read_csv(CSV_PATH)
    df.columns = [c.strip() for c in df.columns]
    changes = 0

    for idx, row in df.iterrows():
        old = str(row.get("pattern", ""))
        new = pattern_depuis_scores(
            row["Boredom"], row["Confusion"], row["Engagement"], row["Frustration"]
        )
        etat, niveau, _ = etat_dominant_depuis_scores(
            row["Boredom"], row["Confusion"], row["Engagement"], row["Frustration"]
        )
        if old != new:
            changes += 1
            print(f"  #{idx}: {old} -> {new}  ({etat} {niveau}/3)")
        df.at[idx, "pattern"] = new

    df.to_csv(CSV_PATH, index=False)
    print(f"\n✅ {CSV_PATH} mis à jour ({changes} patterns corrigés)")

    presets = {}
    for pattern in ["stable", "ennui_eleve", "frustration", "confusion", "engagement_faible", "critique"]:
        subset = df[df["pattern"] == pattern]
        if len(subset) > 0:
            presets[pattern] = int(subset.index[0])

    print("\nPresets suggérés pour theme.js :")
    labels = {
        "stable": "Stable",
        "ennui_eleve": "Ennui élevé",
        "frustration": "Frustration",
        "confusion": "Confusion",
        "engagement_faible": "Engagement faible",
        "critique": "Critique",
    }
    for pattern, sid in presets.items():
        print(f"  {{ id: {sid}, label: \"{labels[pattern]}\", pattern: \"{pattern}\" }},")


if __name__ == "__main__":
    main()
