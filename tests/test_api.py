"""Tests API FastAPI — prototype de simulation (clip + séquence)."""

import os
import sys

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from main import app

client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"
    assert r.json()["version"].startswith("2.5")


def test_accueil_simulation_clip():
    r = client.get("/")
    assert r.status_code == 200
    data = r.json()
    assert data["prototype"] == "simulation"
    assert data["unite"] == "clip"
    assert "simulation" in data["modes"]
    assert "sequence" in data["modes"]
    assert "evaluation" in data["modes"]
    assert data.get("temporel") is True


def test_scenarios_list():
    r = client.get("/scenarios")
    assert r.status_code == 200
    data = r.json()
    assert data["total"] == 50
    assert len(data["scenarios"]) == 50
    assert "video_disponible" in data["scenarios"][0]
    assert "clip_id" in data["scenarios"][0]


def test_scenario_video_meta():
    r = client.get("/scenarios/0/video")
    assert r.status_code == 200
    data = r.json()
    assert data["scenario_id"] == 0
    assert data["clip_id"]
    assert "disponible" in data


def test_scenario_video_not_found():
    r = client.get("/scenarios/9999/video")
    assert r.status_code == 404


def test_scenario_audio_meta():
    r = client.get("/scenarios/0/audio")
    assert r.status_code == 200
    data = r.json()
    assert data["scenario_id"] == 0
    assert data.get("utterance_id")
    assert "audio_disponible" in data
    assert "emotion_iemocap" in data


def test_scenario_ser_preview():
    r = client.get("/scenarios/0/ser-preview")
    assert r.status_code == 200
    data = r.json()
    assert data.get("preview") is True
    assert data.get("etat") in (
        "Boredom",
        "Confusion",
        "Engagement",
        "Frustration",
        None,
    )
    if not data.get("erreur"):
        assert data.get("score") is not None


def test_sequences_list():
    r = client.get("/sequences")
    assert r.status_code == 200
    data = r.json()
    assert data["total"] >= 1
    assert "scenario_ids" in data["sequences"][0]
    seq0 = data["sequences"][0]
    assert seq0.get("meme_apprenant") is True
    assert seq0.get("sujet")
    prefixes = {str(c).replace(".avi", "").replace(".mp4", "")[:6] for c in seq0["clip_ids"]}
    assert len(prefixes) == 1


def test_analyser_fer_removed():
    r = client.post("/analyser/fer")
    assert r.status_code == 404


def test_session_start_removed():
    assert client.post("/session/start").status_code == 404


def test_templates():
    r = client.get("/templates")
    assert r.status_code == 200
    assert r.json()["total"] == 20


@pytest.mark.skipif(
    not os.getenv("GROQ_API_KEY"),
    reason="GROQ_API_KEY requis pour génération LLM",
)
def test_chat_message_vide():
    r = client.post("/chat", json={"message": "   "})
    assert r.status_code == 200
    assert r.json().get("error") == "Message vide"


@pytest.mark.skipif(
    not os.getenv("GROQ_API_KEY"),
    reason="GROQ_API_KEY requis pour le tuteur conversationnel",
)
def test_chat_tuteur():
    r = client.post(
        "/chat",
        json={
            "message": "Qu'est-ce qu'un algorithme de tri ?",
            "etat": "Confusion",
            "niveau": 2,
            "template_id": "A1",
            "concept": "algorithmes de tri",
            "module": "Algorithmique",
            "utiliser_rag": True,
        },
    )
    assert r.status_code == 200
    data = r.json()
    assert data.get("error") is None
    assert "reponse" in data
    assert len(data["reponse"]) > 10
    assert data["etat"] == "Confusion"


def test_analyser_clip():
    r = client.post(
        "/analyser",
        json={
            "scenario_id": 33,
            "concept": "tri",
            "module": "Algo",
            "niveau": "Intermédiaire",
            "utiliser_temporel": True,
        },
    )
    assert r.status_code == 200
    data = r.json()
    assert "template_id" in data
    assert data.get("error") is None
    assert data.get("mode") == "simulation"
    assert "temporal" in data
    assert "declencher" in data.get("decision", {})


def test_analyser_sequence_sans_llm():
    r = client.post(
        "/analyser/sequence",
        json={
            "sequence_id": 0,
            "concept": "tri",
            "module": "Algo",
            "niveau": "Intermédiaire",
            "generer_contenu": False,
        },
    )
    assert r.status_code == 200
    data = r.json()
    assert data.get("error") is None
    assert data.get("mode") == "sequence"
    assert data.get("n_clips", 0) >= 2
    assert len(data.get("etapes", [])) == data["n_clips"]
    assert data["etapes"][0].get("temporal") is not None


def test_evaluation_pages():
    assert client.get("/evaluation").status_code == 200
    assert "L1" in client.get("/evaluation").text
    assert client.get("/evaluation/l1").status_code == 200
    assert client.get("/evaluation/l2").status_code == 200
    assert client.get("/evaluation/l4").status_code == 200


def test_evaluation_l1_post(tmp_path, monkeypatch):
    import shutil
    from pathlib import Path

    src = Path(__file__).resolve().parents[1] / "results" / "forms"
    dst = tmp_path / "forms"
    dst.mkdir()
    shutil.copy(src / "l1_mapping_reponses.csv", dst / "l1_mapping_reponses.csv")
    monkeypatch.setenv("EMOLEARN_FORMS_DIR", str(dst))

    item = {
        "fiche_id": 1,
        "C1_coherence": 4,
        "C2_pertinence": 5,
        "C3_template": 3,
        "C4_urgence": 4,
        "commentaire_expert": "ok",
    }
    r = client.post("/evaluation/l1", json={"expert_id": "Etest", "reponses": [item]})
    assert r.status_code == 200, r.text
    assert r.json()["n"] == 1
    assert r.json()["expert_id"] == "Etest"


def test_evaluation_l4_post(tmp_path, monkeypatch):
    import shutil
    from pathlib import Path

    src = Path(__file__).resolve().parents[1] / "results" / "forms"
    dst = tmp_path / "forms"
    dst.mkdir()
    shutil.copy(src / "l4_pilote_reponses.csv", dst / "l4_pilote_reponses.csv")
    monkeypatch.setenv("EMOLEARN_FORMS_DIR", str(dst))

    payload = {
        "participant_id": 1,
        "niveau": "L3",
        "familiarite_mooc_1_5": 3,
        "condition": "C2",
        "utilite_1_7": 5,
        "adequation_emotionnelle_1_7": 4,
        "satisfaction_contenu_1_7": 5,
        **{f"sus_{i}": 3 for i in range(1, 11)},
        "q_ouvert_aide": "clair",
    }
    r = client.post("/evaluation/l4", json=payload)
    assert r.status_code == 200, r.text
    assert r.json()["sus_score"] == 50.0
