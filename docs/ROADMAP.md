# EmoLearn — Feuille de route (qualité scientifique & technique)

Document de suivi aligné sur le plan de mémoire et l'état du dépôt.

## Légende

| Statut | Signification |
|--------|----------------|
| ✅ | Fait |
| 🔄 | En cours |
| ⏳ | À faire (rédaction / terrain) |

---

## Phase 1 — Cohérence du pipeline

| # | Tâche | Fichiers | Statut |
|---|--------|----------|--------|
| 1.1 | Moteur de décision unifié | `backend/decision_engine.py` | ✅ |
| 1.2 | 20 templates en production | `backend/prompts/templates.py` | ✅ |
| 1.3 | LLM branché sur `template_id` | `backend/llm_module.py` | ✅ |
| 1.4 | API `/analyser` cohérente | `backend/main.py` | ✅ |
| 1.5 | Route `GET /templates` | `backend/main.py` | ✅ |
| 1.6 | Interface professionnelle | `frontend/src/App.js` | ✅ |
| 1.7 | Journal de décision JSON | `pipeline.enregistrer_decision` | ✅ |

---

## Phase 2 — Évaluation FER

| # | Tâche | Fichiers | Statut |
|---|--------|----------|--------|
| 2.1 | Script évaluation DAiSEE | `scripts/eval_fer_daisee.py` | ✅ |
| 2.2 | Mode FER image API | retiré du produit (simulation clip) | ❌ produit |
| 2.3 | Métriques F1 / confusion | `results/fer_metrics.json` | ✅ (si frames dispo) |
| 2.4 | Section mémoire Mode A vs B | Chapitre 4–5 | ⏳ |
| 2.5 | Notebook DeepFace reproductible | `notebooks/` | ⏳ |

---

## Phase 3 — Scénarios temporels (OS1)

| # | Tâche | Fichiers | Statut |
|---|--------|----------|--------|
| 3.1 | Analyse exploratoire DAiSEE | `notebooks/OS1*.ipynb` | ✅ |
| 3.2 | Séquences 5–10 clips | `data/scenarios_sequences.csv` | ✅ |
| 3.3 | Règles multi-clips | `backend/temporal_rules.py` | ✅ |
| 3.4 | Branché dans pipeline | `pipeline.py` (`utiliser_temporel`) | ✅ |
| 3.5 | API parcours séquence | `GET /sequences`, `POST /analyser/sequence` | ✅ |

---

## Phase 4 — IEMOCAP multimodal

| # | Tâche | Fichiers | Statut |
|---|--------|----------|--------|
| 4.1 | Parser IEMOCAP | `backend/iemocap_parser.py` | ✅ |
| 4.2 | Fusion + conflits | `backend/fusion_module.py` | ✅ |
| 4.3 | Documentation appariement | `docs/IEMOCAP.md` | ✅ |
| 4.4 | Limites mémoire mises à jour | `docs/LIMITATIONS.md` | ✅ |

---

## Phase 5 — Corpus & qualité

| # | Tâche | Fichiers | Statut |
|---|--------|----------|--------|
| 5.1 | Corpus généré | `data/corpus_200.csv` | ✅ |
| 5.2 | Audit cohérence | `scripts/audit_corpus.py` | ✅ |
| 5.3 | Régénération incohérences | `scripts/regenerate_corpus.py` | ✅ |
| 5.4 | Échantillon 60 experts | `data/corpus_evaluation_60.csv` | ✅ |
| 5.5 | Export fiches experts | `scripts/export_validation_experts.py` | ✅ |

---

## Phase 6 — Validation (terrain)

| Niveau | Objectif | Statut |
|--------|----------|--------|
| L1 | Delphi experts mapping | ⏳ formulaires `results/forms/l1_*.csv` + agrégation |
| L2 | 3 enseignants × 60 | ⏳ formulaires `results/forms/l2_*.csv` + agrégation |
| L3 | Métriques auto | ✅ `metrics_contenu.py` + export MD |
| L4 | Pilote n=30 | ⏳ |
| Proxy fusion produit | `eval_fusion_produit_proxy.py` | ✅ |

---

## Phase 7 — Qualité logicielle

| # | Tâche | Statut |
|---|--------|--------|
| 7.1 | Tests unitaires temporels | ✅ `tests/test_temporal.py` |
| 7.2 | Tests API | ✅ `tests/test_api.py` |
| 7.3 | UI séquences + FER + PDF | ✅ |
| 7.4 | Rédaction mémoire | ⏳ |

---

## Phase 8 — Entraînement léger (complément pré-entraînés)

| # | Tâche | Statut |
|---|--------|--------|
| 8.1 | Classifieur sklearn FER→DAiSEE | ✅ `scripts/train_fer_daisee_classifier.py` |
| 8.2 | Fine-tune tête SER wav2vec2 (Confusion) | ✅ `scripts/finetune_ser_wav2vec2.py` |
| 8.3 | Doc usage | ✅ `docs/TRAINING.md` |

---

## API v2.4 — Endpoints (prototype simulation — clip)

| Route | Description |
|-------|-------------|
| `GET /health` | Santé API |
| `GET /scenarios` | 50 clips DAiSEE |
| `POST /analyser` | Clip de simulation |
| `POST /chat` | Tuteur conversationnel |
| `GET /sessions` | Historique SQLite |
| `GET /templates` | 20 templates A1–D5 |
