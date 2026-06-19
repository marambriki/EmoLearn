"""Tests cohérence pattern / état dominant DAiSEE."""

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from daisee_utils import (
    etat_dominant_depuis_scores,
    pattern_depuis_scores,
    PATTERN_PAR_ETAT,
)


def test_scenario_17_coherent():
    """Ancien cas : ennui_eleve + Engagement dominant -> stable."""
    b, c, e, f = 2, 0, 3, 0
    etat, _, _ = etat_dominant_depuis_scores(b, c, e, f)
    pattern = pattern_depuis_scores(b, c, e, f)
    assert etat == "Engagement"
    assert pattern == "stable"


def test_tous_scenarios_csv_coherents():
    csv_path = os.path.join(
        os.path.dirname(__file__), "..", "data", "scenarios_50.csv"
    )
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]

    for _, row in df.iterrows():
        etat, _, _ = etat_dominant_depuis_scores(
            row["Boredom"], row["Confusion"], row["Engagement"], row["Frustration"]
        )
        pattern = pattern_depuis_scores(
            row["Boredom"], row["Confusion"], row["Engagement"], row["Frustration"]
        )

        if pattern == "critique":
            continue
        if pattern == "engagement_faible":
            assert row["Engagement"] <= 1
            continue

        assert pattern == PATTERN_PAR_ETAT[etat], (
            f"{row['ClipID']}: etat={etat}, pattern={pattern}"
        )
