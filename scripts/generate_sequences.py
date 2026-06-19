#!/usr/bin/env python3
"""Génère data/scenarios_sequences.csv à partir de scenarios_50.csv."""

import os

import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SCENARIOS = os.path.join(ROOT, "data", "scenarios_50.csv")
OUTPUT = os.path.join(ROOT, "data", "scenarios_sequences.csv")

SEQUENCE_SPECS = [
    ("Engagement stable", "stable", [0, 1, 2, 3, 4, 5]),
    ("Transition légère", "stable", [5, 6, 7, 8, 9]),
    ("Ennui progressif", "ennui_eleve", [15, 16, 17, 18, 19, 20, 21]),
    ("Ennui et désengagement", "ennui_eleve", [18, 19, 20, 21, 22, 23]),
    ("Frustration montante", "frustration", [25, 26, 27, 28, 29, 30]),
    ("Frustration critique", "frustration", [30, 31, 32, 33]),
    ("Confusion persistante", "confusion", [34, 35, 36, 37, 38, 39]),
    ("Confusion + blocage", "confusion", [38, 39, 40, 41]),
    ("Engagement faible prolongé", "engagement_faible", [41, 42, 43, 44, 45, 46]),
    ("État critique multi-états", "critique", [47, 48, 49, 46, 45]),
]


def main():
    df = pd.read_csv(SCENARIOS)
    rows = []
    for seq_id, (title, pattern, ids) in enumerate(SEQUENCE_SPECS):
        clip_ids = [str(df.iloc[i]["ClipID"]) for i in ids]
        rows.append(
            {
                "sequence_id": seq_id,
                "title": title,
                "pattern": pattern,
                "length": len(ids),
                "scenario_ids": ";".join(str(i) for i in ids),
                "clip_ids": ";".join(clip_ids),
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(OUTPUT, index=False)
    print(f"OK {len(out)} sequences -> {OUTPUT}")


if __name__ == "__main__":
    main()
