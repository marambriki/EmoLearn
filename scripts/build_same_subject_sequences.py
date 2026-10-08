#!/usr/bin/env python3
"""Construit des parcours séquence = clips DAiSEE du MÊME apprenant (dossier sujet)."""

from __future__ import annotations

import os
import sys

import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "backend"))

from daisee_utils import etat_dominant_depuis_scores, pattern_depuis_scores
from daisee_video import get_index, resolve_daisee_video_path

N_CLIPS = 5

# Sujets DAiSEE (présents dans scenarios_50). Le pattern reste interne
# (règles temporelles) — jamais affiché comme étiquette d'état détecté.
THEMES = [
    ("110007", "ennui_eleve"),
    ("210057", "ennui_eleve"),
    ("411021", "stable"),
    ("310077", "stable"),
    ("210055", "frustration"),
    ("400018", "critique"),
    ("210060", "confusion"),
    ("334463", "confusion"),
    ("310082", "engagement_faible"),
    ("110006", "frustration"),
]


def _sujet_depuis_chemin(path: str) -> str:
    parts = os.path.normpath(path).split(os.sep)
    # .../DataSet/{split}/{sujet}/{clip_dir}/{file}
    if len(parts) >= 4:
        return parts[-3]
    return ""


def index_sujets() -> dict[str, str]:
    out = {}
    for fname, path in get_index().items():
        out[str(fname)] = _sujet_depuis_chemin(path)
    return out


def charger_labels() -> pd.DataFrame:
    path = os.path.join(ROOT, "data", "DAiSEE", "Labels", "AllLabels.csv")
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    return df


def main():
    sujet_of = index_sujets()
    labels = charger_labels()
    labels["ClipID"] = labels["ClipID"].astype(str).str.strip()
    labels["sujet"] = labels["ClipID"].map(sujet_of)

    sc50 = pd.read_csv(os.path.join(ROOT, "data", "scenarios_50.csv"))
    sc50.columns = [c.strip() for c in sc50.columns]
    clip_to_sid = {str(row["ClipID"]): i for i, row in sc50.iterrows()}

    extra_rows = []
    extra_seen = set()
    sequences = []

    for seq_id, (sujet, pattern) in enumerate(THEMES):
        pool = labels[labels["sujet"] == sujet].copy()
        if pool.empty:
            print(f"⚠ sujet {sujet} absent des labels, skip")
            continue
        pool = pool.drop_duplicates(subset=["ClipID"])
        # clips déjà dans les 50 d'abord, puis autres du même sujet (ordre ClipID)
        in50 = [c for c in pool["ClipID"] if c in clip_to_sid]
        others = sorted(c for c in pool["ClipID"] if c not in clip_to_sid)
        ordered = in50 + others
        chosen = []
        for clip in ordered:
            if resolve_daisee_video_path(clip):
                chosen.append(clip)
            if len(chosen) >= N_CLIPS:
                break
        if len(chosen) < 2:
            print(f"⚠ sujet {sujet}: seulement {len(chosen)} clips vidéo, skip")
            continue

        sids = []
        for clip in chosen:
            if clip in clip_to_sid:
                sids.append(clip_to_sid[clip])
                continue
            if clip in extra_seen:
                continue
            extra_seen.add(clip)
            lab = pool[pool["ClipID"] == clip].iloc[0]
            extra_rows.append(
                {
                    "ClipID": clip,
                    "Boredom": int(lab["Boredom"]),
                    "Engagement": int(lab["Engagement"]),
                    "Confusion": int(lab["Confusion"]),
                    "Frustration": int(lab["Frustration"]),
                    "pattern": pattern_depuis_scores(
                        lab["Boredom"],
                        lab["Confusion"],
                        lab["Engagement"],
                        lab["Frustration"],
                    ),
                }
            )
            clip_to_sid[clip] = 50 + len(extra_rows) - 1
            sids.append(clip_to_sid[clip])

        etat, _, _ = etat_dominant_depuis_scores(
            int(pool[pool["ClipID"] == chosen[0]]["Boredom"].iloc[0]),
            int(pool[pool["ClipID"] == chosen[0]]["Confusion"].iloc[0]),
            int(pool[pool["ClipID"] == chosen[0]]["Engagement"].iloc[0]),
            int(pool[pool["ClipID"] == chosen[0]]["Frustration"].iloc[0]),
        )
        sequences.append(
            {
                "sequence_id": seq_id,
                "title": f"Apprenant {sujet}",
                "pattern": pattern,
                "length": len(sids),
                "sujet": sujet,
                "scenario_ids": ";".join(str(s) for s in sids),
                "clip_ids": ";".join(chosen),
            }
        )
        print(f"seq {seq_id} sujet={sujet} n={len(sids)} ids={sids}")

    extra_path = os.path.join(ROOT, "data", "scenarios_sequence_extra.csv")
    pd.DataFrame(extra_rows).to_csv(extra_path, index=False)
    seq_path = os.path.join(ROOT, "data", "scenarios_sequences.csv")
    pd.DataFrame(sequences).to_csv(seq_path, index=False)
    print(f"extra clips: {len(extra_rows)} -> {extra_path}")
    print(f"sequences: {len(sequences)} -> {seq_path}")


if __name__ == "__main__":
    main()
