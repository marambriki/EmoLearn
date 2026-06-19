"""Tests module RAG (recherche ChromaDB, sans LLM)."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))


def test_rag_recherche_tri():
    from rag_module import rechercher_documents, construire_contexte_rag

    docs = rechercher_documents(
        query="Expliquer les algorithmes de tri",
        concept="algorithmes de tri",
        etat="Confusion",
        n_results=3,
    )
    assert len(docs) >= 1
    assert all("score" in d and d["score"] >= 0.3 for d in docs)
    ctx = construire_contexte_rag(docs, concept="algorithmes de tri", etat="Confusion")
    assert "CONTEXTE DE RÉFÉRENCE" in ctx or "CONTEXTE PÉDAGOGIQUE" in ctx


def test_rag_lazy_import():
    from rag_module import get_embedding_model

    model = get_embedding_model()
    assert model is not None
