#!/usr/bin/env python3
"""Debug SER sur un scénario (annotation vs prédiction)."""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from ser_module import analyser_voix_depuis_scenario, _SER_FINETUNED_DIR, _SER_ANCHOR_DIR


def main():
    sid = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    scores = {"boredom": 0, "confusion": 0, "engagement": 3, "frustration": 0}
    r = analyser_voix_depuis_scenario(scenario_id=sid, scores_daisee=scores)
    print("FT model dir:", _SER_FINETUNED_DIR)
    print("Anchor dir:", _SER_ANCHOR_DIR)
    print("source:", r.get("source"))
    print("annot_iemocap:", r.get("emotion_iemocap"))
    print("annot_etat:", r.get("annotation_etat"))
    print("pred_etat:", r.get("etat"), "score:", r.get("score"))
    print("ser_details:", json.dumps(r.get("ser_details") or {}, indent=2))


if __name__ == "__main__":
    main()
