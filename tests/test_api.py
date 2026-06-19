"""Tests API FastAPI (sans appel LLM si GROQ absent)."""

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


def test_scenarios_list():
    r = client.get("/scenarios")
    assert r.status_code == 200
    data = r.json()
    assert data["total"] == 50
    assert len(data["scenarios"]) == 50


def test_sequences_list():
    r = client.get("/sequences")
    assert r.status_code == 200
    data = r.json()
    assert data["total"] >= 1


def test_templates():
    r = client.get("/templates")
    assert r.status_code == 200
    assert r.json()["total"] == 20


def test_session_start():
    r = client.post("/session/start?sequence_id=0")
    assert r.status_code == 200
    data = r.json()
    assert "session_id" in data


@pytest.mark.skipif(
    not os.getenv("GROQ_API_KEY"),
    reason="GROQ_API_KEY requis pour génération LLM",
)
def test_analyser_clip():
    r = client.post(
        "/analyser",
        json={"scenario_id": 33, "concept": "tri", "module": "Algo", "niveau": "Intermédiaire"},
    )
    assert r.status_code == 200
    data = r.json()
    assert "template_id" in data
    assert data.get("error") is None
