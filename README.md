# EmoLearn — Personnalisation pédagogique adaptative multimodale

**EmoLearn** est un **prototype de recherche** (mémoire de master) qui détecte l'état
émotionnel d'un apprenant à partir de signaux **visage + voix**, décide s'il faut intervenir,
et **génère un contenu pédagogique adaptatif** via un LLM.

> ⚠️ **Périmètre assumé** : EmoLearn est un **prototype de simulation** (OS5), pas un système
> déployé sur de vrais MOOC en temps réel. L'engagement provient de corpus annotés (DAiSEE),
> pas d'une captation live. Voir [`docs/LIMITES_ET_TENSIONS.md`](docs/LIMITES_ET_TENSIONS.md).

---

## 1. Ce que fait le projet

```
Scénario / image + contexte pédagogique
        │
        ▼
  Perception émotionnelle      Visage (DAiSEE) + Voix (IEMOCAP)
        │                      → fusion 60 % / 40 % (gestion des conflits)
        ▼
  Décision                     déclencher ? + type de contenu (A/B/C/D) + template (A1–D5)
        │                      + règles temporelles (urgence sur séquence)
        ▼
  Génération                   LLM Groq (GPT-OSS 20B) + 20 templates + RAG optionnel
        │
        ▼
  Sortie                       contenu adaptatif → interface React + export PDF + journal
```

---

## 2. Deux pipelines distincts (à ne pas confondre)

| | **Pipeline 1 — Produit** | **Pipeline 2 — Banc d'essai (validation)** |
|---|---|---|
| But | Générer du contenu pédagogique | **Valider** la fusion multimodale |
| Visage | DAiSEE (engagement apprenant) | Vidéo IEMOCAP |
| Voix | IEMOCAP (appariée par état) | Audio IEMOCAP (**même locuteur**) |
| Lancement | API `uvicorn` + UI `npm start` | `python scripts/build_iemocap_av_corpus.py` |
| Branché dans l'app ? | ✅ Oui | ❌ Non (script hors ligne) |

Les deux **partagent le même moteur de fusion** (`backend/fusion_module.py`) : ce qui est
validé au banc d'essai est exactement le moteur utilisé en production.

### Rôle des corpus

- **DAiSEE** = **socle du projet** : taxonomie d'engagement (4 états × niveaux 0–3), scénarios,
  déclenchement des adaptations. Tout le reste est traduit dans cet espace.
- **IEMOCAP** = **modalité vocale** (annotations VAD + SER) + **corpus audiovisuel aligné** pour
  la validation de la fusion (Option A, voir [`docs/IEMOCAP.md`](docs/IEMOCAP.md) §9).

---

## 3. Installation

```bash
# Backend (Python 3.11)
pip install -r requirements.txt

# Frontend
cd frontend && npm install
```

Créer un fichier `.env` à la racine :

```
GROQ_API_KEY=...   # requis pour la génération LLM
GROQ_MODEL=openai/gpt-oss-20b   # optionnel (défaut) — llama-3.1-8b-instant retiré le 16/08/2026
```

> Données externes (accord académique) : placer DAiSEE et IEMOCAP sous `data/`.
> Voir [`docs/IEMOCAP.md`](docs/IEMOCAP.md) §8 pour la structure attendue.

---

## 4. Lancer le projet

```bash
# Backend — API FastAPI
cd backend
uvicorn main:app --reload          # http://localhost:8000

# Frontend — interface React (autre terminal)
cd frontend
npm start                          # http://localhost:3000
```

> Sous Windows PowerShell, pour les scripts : `$env:PYTHONIOENCODING="utf-8"`.

### Endpoints principaux (prototype de simulation — clip + séquence)

| Route | Description |
|-------|-------------|
| `GET /health` | Santé API |
| `GET /scenarios` · `GET /sequences` · `GET /templates` | Référentiels |
| `POST /analyser` | Analyse d’un clip (règles temporelles optionnelles) |
| `POST /analyser/sequence` | Parcours multi-clips avec historique temporel |
| `POST /chat` | Tuteur conversationnel |
| `GET /sessions` | Historique des analyses (SQLite) |

---

## 5. Banc d'essai fusion (validation Option A)

```bash
python scripts/build_iemocap_av_corpus.py --n-per-state 20 --fer-frames 5
```

Sorties : `results/iemocap_av_fusion.csv` (détail) + `results/iemocap_av_metrics.json` (métriques).

**Résultats actuels (n = 80, seed = 42, durée ≥ 1 s, 5 frames/utterance)** — présentés honnêtement :

| Modalité / approche | Accuracy | Rappel états urgents |
|---------------------|:---:|:---:|
| Visage — FER 1 frame | 0,375 | 0,250 |
| Visage — FER **multi-frames** (vote) | 0,412 | 0,275 |
| Voix — SER heuristique | 0,263 | 0,225 |
| Voix — **SER appris (wav2vec2)** | **0,487** | 0,200 |
| Fusion — priorité pédagogique | 0,338 | **0,400** |
| Fusion — **vote pondéré** | **0,500** | 0,125 |

> **Trois leviers mesurés** : le **SER appris** double presque l'accuracy de la voix (0,263 → 0,487,
> **+0,224**) ; le **vote multi-frames** stabilise le visage (+0,037, 100 % visages détectés) ; la
> **fusion par vote pondéré** dépasse enfin chaque modalité en accuracy (**0,500**).
> **Arbitrage mesuré** : la fusion *vote* maximise l'accuracy, la fusion *priorité pédagogique*
> maximise le **rappel des états urgents** (0,400) — wav2vec2 n'ayant pas de classe Confusion.
> Configurable via `EMOLEARN_SER_METHODE`. Le prototype retient le vote pondéré par incertitude
> (la priorité pédagogique n'est plus une stratégie de conflit, seulement un repli si les deux
> confiances sont nulles).
> Voir `docs/LIMITES_ET_TENSIONS.md` et `docs/IEMOCAP.md` §9.

---

## 6. Évaluations complémentaires (pipeline produit)

```bash
# Proxy fusion DAiSEE+IEMOCAP (pas une accuracy gold)
python scripts/eval_fusion_produit_proxy.py

# Formulaires experts L1/L2 + agrégation
python scripts/agreger_validation_experts.py --prepare-forms
python scripts/agreger_validation_experts.py

# Métriques de forme du corpus (+ extrait markdown mémoire)
python scripts/metrics_contenu.py

# Similarité LLM sur le corpus réel (BLEU / ROUGE ; + BERTScore si installé)
python scripts/metrics_llm_similarity.py --skip-bertscore
# Avec références experts (si disponibles) :
# python scripts/metrics_llm_similarity.py --mode gold --references data/references_contenu.csv
```

Sorties clés : `results/fusion_produit_proxy_metrics.json`, `results/forms/`, `results/corpus_metrics_memoire.md`, `results/llm_similarity_metrics.json`.

---

## 7. Tests

```bash
python -m pytest -q          # backend (API, temporel, SER, fusion, RAG)
cd frontend && npm test      # frontend
```

---

## 8. Structure

```
backend/        API + modules (FER, SER, fusion, décision, temporel, LLM, RAG)
frontend/       interface React (clip + séquence)
scripts/        évaluation, corpus, banc d'essai, validation experts
data/           DAiSEE, IEMOCAP, corpus, scénarios, séquences
docs/           positionnement scientifique, limites, FAQ jury
results/        métriques et sorties d'évaluation
tests/          tests unitaires
notebooks/      analyses exploratoires (OS1–OS4)
```

---

## 9. Documentation scientifique

- [`docs/IEMOCAP.md`](docs/IEMOCAP.md) — positionnement IEMOCAP + sous-corpus audiovisuel aligné
- [`docs/LIMITES_ET_TENSIONS.md`](docs/LIMITES_ET_TENSIONS.md) — limites et tensions assumées
- [`docs/LIMITES_ET_PERSPECTIVES.md`](docs/LIMITES_ET_PERSPECTIVES.md) — limites opérationnelles + perspectives
- [`docs/FAQ_JURY.md`](docs/FAQ_JURY.md) — réponses types pour la soutenance
- [`docs/LIMITATIONS.md`](docs/LIMITATIONS.md) — limites du prototype
- [`docs/PROTOCOLE_EVALUATION.md`](docs/PROTOCOLE_EVALUATION.md) — protocole multi-niveaux
- [`docs/ROADMAP.md`](docs/ROADMAP.md) — feuille de route et statut
