# Guide notebooks EmoLearn — Soutenance

Document de révision pour expliquer le **code** et les **résultats** de chaque notebook à l'encadrante.

**Dernière mise à jour :** notebooks `OS1Daisee.ipynb` et `OS1IEMOCAP.ipynb` (anciennement OS1 / OS1multi).

---

## 1. Carte des notebooks

| Notebook | Rôle | Fichier produit principal | Module backend |
|----------|------|---------------------------|----------------|
| **OS1Daisee** | Analyse exploratoire DAiSEE | `data/scenarios_50.csv` | `daisee_utils.py` |
| **OS1IEMOCAP** | Voix IEMOCAP + fusion multimodale | `data/iemocap_parsed.csv` | `ser_module.py`, `fusion_module.py` |
| **OS2fini** | Engagement → stratégie A/B/C/D | `data/scenarios_avec_strategies.csv` | `adaptation.py`, `decision_engine.py` |
| **OS3fini** | 20 templates Groq + corpus | `data/corpus_200.csv` | `prompts/templates.py` |
| **OS4** | Module LLM intégré | — | `llm_module.py` |
| **Rag** | RAG ChromaDB (sans vs avec) | base vectorielle | `rag_module.py` |
| **TesterRAG** | Test recherche sémantique seule | — | `rag_module.py` |

**Ne pas présenter :** `os3.ipynb` (brouillon), `OS1_analyse.ipynb` (doublon avec erreurs).

**Ordre oral recommandé (15–20 min) :** OS1Daisee → OS1IEMOCAP → OS2fini → OS3fini → OS4 + Rag → démo UI.

---

## 2. OS1Daisee.ipynb — Analyse DAiSEE

### Phrase d'ouverture

> « OS1 analyse le dataset **DAiSEE** : 4 dimensions d'engagement (Boredom, Engagement, Confusion, Frustration) notées de 0 à 3. On identifie des **patterns**, des **transitions temporelles**, et on construit **50 scénarios** pour tester le prototype. »

### Cellule 1 — Chargement et statistiques

```python
import pandas as pd
train = pd.read_csv('../data/DAiSEE/Labels/TrainLabels.csv')
val   = pd.read_csv('../data/DAiSEE/Labels/ValidationLabels.csv')
test  = pd.read_csv('../data/DAiSEE/Labels/TestLabels.csv')
train.describe()
```

| Élément | Explication |
|---------|-------------|
| `pd.read_csv` | Lit les **labels** officiels (pas les vidéos) |
| Colonnes | `ClipID`, `Boredom`, `Engagement`, `Confusion`, `Frustration` |
| `describe()` | Moyennes, écarts-types, min/max par dimension |

**Résultats :**

| Split | Clips |
|-------|-------|
| Train | **5 358** |
| Validation | **1 429** |
| Test | **1 784** |
| **Total** | **8 571** |

**À dire :** « On travaille sur les **annotations ground truth**, pas sur les pixels — c'est le mode simulation du prototype. »

### Cellule 2 — Distribution (`data/distribution_etats.png`)

```python
for etat in ['Boredom', 'Engagement', 'Confusion', 'Frustration']:
    ax.hist(train[etat], bins=4, range=(-0.5, 3.5))
```

- 4 barres par dimension (niveaux 0–3).
- **Engagement** concentré sur 2–3 ; **Confusion** rare à 3.

**À dire :** « Dataset **déséquilibré** → échantillonnage stratifié pour les 50 scénarios. »

### Cellule 3 — Matrice de corrélation (`data/correlation_etats.png`)

```python
corr = train[etats].corr()
sns.heatmap(corr, annot=True)
```

- Corrélation négative : Engagement ↔ Boredom.
- Corrélation positive possible : Confusion ↔ Frustration.

### Cellule 4 — Boxplots (`data/boxplots_etats.png`)

Compare médiane et dispersion des 4 états sur le train.

### Cellule 5 — Six patterns critiques

```python
p1 = train[(train['Confusion'] >= 2) & (train['Engagement'] <= 1)]
p2 = train[train['Frustration'] >= 2]
p3 = train[train['Boredom'] >= 2]
p4 = train[train['Engagement'] == 0]
p5 = train[(train['Confusion'] >= 2) & (train['Frustration'] >= 2) & (train['Engagement'] <= 1)]
p6 = train[train['Engagement'] >= 2]
```

| Pattern | Signification | Clips (train) | % |
|---------|---------------|---------------|---|
| P1 | Confusion haute + engagement faible | 49 | 0,9 % |
| P2 | Frustration élevée | 234 | 4,4 % |
| P3 | Ennui élevé | 1 229 | 22,9 % |
| P4 | Engagement nul | 34 | 0,6 % |
| P5 | Situation critique (multi-signaux) | 20 | 0,4 % |
| P6 | Bon engagement | 5 111 | **95,4 %** |

**À dire :** « 95 % des clips sont stables ; l'adaptation cible les cas rares mais pédagogiquement urgents. »

### Cellule 6 — Transitions temporelles

```python
train_sorted = train.sort_values('ClipID')
for i in range(1, len(train_sorted)):
    prev, curr = train_sorted.iloc[i-1], train_sorted.iloc[i]
    # Engagement → Boredom : eng. était ≥2, devient ≤1, et Boredom ≥2
    # Engagement → Confusion : idem avec Confusion ≥2
```

**Résultat : 189 transitions critiques** (dégradation engagement dans le temps).

**Lien backend :** `temporal_rules.py` — règles multi-clips.

### Cellule 7 — Seuils d'alerte

```python
# Boredom ≥ 2 pendant 3 clips consécutifs → alerte
# Confusion ≥ 2 pendant 2 clips → alerte
# Frustration ≥ 2 pendant 2 clips → alerte
```

| Alerte | Règle | Cas (~) |
|--------|-------|---------|
| Ennui | 3 clips consécutifs | ~340 |
| Confusion | 2 clips | ~32 |
| Frustration | 2 clips | ~32 |
| **Total** | | **404** |

### Cellule 8 — Segmentation 70/15/15

Export : `conception_70.csv`, `validation_15.csv`, `test_15.csv` — protocole méthodologique mémoire.

### Cellule 9 — Cinquante scénarios (`data/scenarios_50.csv`)

```python
stable = train[train['Engagement'] >= 2].sample(15, random_state=42)
stable['pattern'] = 'stable'
# + ennui×10, frustration×8, confusion×7, engagement_faible×6, critique×4
df_scenarios.to_csv('../data/scenarios_50.csv')
```

| Pattern | Nb scénarios |
|---------|--------------|
| stable | 15 |
| ennui_eleve | 10 |
| frustration | 8 |
| confusion | 7 |
| engagement_faible | 6 |
| critique | 4 |

`random_state=42` → **reproductibilité**.

**Lien prototype :** `main.py` charge ce CSV ; `pattern_depuis_scores()` recalcule la cohérence pattern/état dominant.

### Cellule 10 — Trajectoires (`data/trajectoires_temporelles.png`)

Courbes d'évolution des 4 scores sur des séquences de clips.

### Résumé OS1Daisee

| Métrique | Valeur |
|----------|--------|
| Clips total | 8 571 |
| Patterns | 6 |
| Transitions | 189 |
| Alertes | 404 |
| Scénarios | 50 |

---

## 3. OS1IEMOCAP.ipynb — Voix + fusion

### Phrase d'ouverture

> « On enrichit le pipeline avec **IEMOCAP** (modalité voix) : parser les annotations, mapper vers DAiSEE, fusionner avec le visage sur 50 scénarios. »

### Cellule 1 — Parser IEMOCAP

```python
from iemocap_parser import charger_iemocap_complet, sauvegarder_iemocap_csv
df_iemocap = charger_iemocap_complet(sessions=[1, 2, 3, 4, 5])
```

**Format ligne brute :**

```text
[6.2901 - 8.2357] Ses01F_impro01_F000 neu [2.5000, 2.5000, 2.5000]
                                              V        A        D
```

**Parser =** lire `.txt` → extraire timing, id, émotion, VAD → CSV structuré.

**Résultats :**

| Métrique | Valeur |
|----------|--------|
| Utterances | **10 039** |
| Sessions | 5 |
| Durée moyenne | ~4,46 s |
| VAD moyen | V=2,78, A=3,09, D=3,19 |

**Émotions IEMOCAP brutes (top) :** `xxx` ~2 507 (25 %), `fru`, `neu`, `ang`, `sad`…

**Après mapping DAiSEE :**

| État | Count | % |
|------|-------|---|
| Boredom | 5 302 | 52,8 % |
| Frustration | 2 954 | 29,4 % |
| Engagement | 1 636 | 16,3 % |
| Confusion | 147 | 1,5 % |

**Sauvegarde :** `data/iemocap_parsed.csv`

### Cellule 2 — Visualisations (`data/iemocap_analyse.png`)

4 graphiques : émotions brutes, états mappés, VAD, durées.

### Cellule 3 — Test SER (voix)

```python
from ser_module import analyser_voix_depuis_scenario
result = analyser_voix_depuis_scenario(scenario_id=i)
```

1. Filtre utterances IEMOCAP par **état dominant** du scénario DAiSEE.
2. Sélection : `scenario_id % len(pool)`.
3. Retourne émotion, état DAiSEE, niveau, score.

### Cellule 4 — Pipeline multimodal

```python
from fer_module import analyser_visage_depuis_scenario
from ser_module import analyser_voix_depuis_scenario
from fusion_module import fusionner_modalites

result_visage = analyser_visage_depuis_scenario(scenario)  # scores CSV
result_voix   = analyser_voix_depuis_scenario(scenario_id)
fusion        = fusionner_modalites(result_visage, result_voix)
```

**Fusion (`fusion_module.py`) :**

- Poids : **60 % visage (DAiSEE)** / **40 % voix (IEMOCAP)**.
- Accord → moyenne pondérée des niveaux.
- Conflit → **priorité pédagogique** : Frustration > Confusion > Boredom > Engagement.

**Résultats 50 scénarios :**

| Métrique | Valeur |
|----------|--------|
| Accords | **10 (20 %)** |
| Conflits | **40 (80 %)** |

**À dire :** « 80 % de conflits car visage = apprenant DAiSEE, voix = acteur IEMOCAP — limite assumée (`docs/IEMOCAP.md`). »

### Cellule 5 — Comparaison (`data/comparaison_daisee_iemocap.png`)

DAiSEE : Engagement dominant (5 111 clips ≥2). IEMOCAP mappé : Boredom dominant.

---

## 4. OS2fini.ipynb — Décision pédagogique

### Phrase d'ouverture

> « OS2 applique des **règles expertes** : 4 scores DAiSEE → type de contenu A/B/C/D. Pas de machine learning end-to-end. »

### Mapping FER (extension optionnelle)

```python
def mapper_emotion_daisee(emotion_fer):
    # happy → Engagement 3, angry → Frustration 3, neutral → Boredom 1, ...
```

Modèle : `dima806/facial_emotions_image_detection` (7 émotions → 4 états DAiSEE).

### Fonction centrale `selectionner_strategie` (`backend/adaptation.py`)

| Priorité | Condition | Type | Stratégie |
|----------|-----------|------|-----------|
| 1 | `confusion >= 3` ET `frustration >= 2` | **A+D** | Déblocage immédiat |
| 2 | `engagement == 0` | **B** | Réengagement urgent |
| 3 | `frustration >= 2` | **D+A** | Encouragement + simplification |
| 4 | `confusion >= 2` | **A** | Explication alternative |
| 5 | `boredom >= 2` | **C** | Exercice avancé |
| 6 | `engagement <= 1` | **B** | Exemple concret |
| 7 | `boredom == 1` | **B** | Application pratique |
| 8 | `engagement >= 2` | **D** | Renforcement positif |
| 9 | sinon | **aucune** | Continuation normale |

### Évaluation sur 50 scénarios

```python
for _, row in scenarios.iterrows():
    emotion, niveau, strategie, type_c = selectionner_strategie(
        row['Boredom'], row['Confusion'], row['Engagement'], row['Frustration']
    )
df_resultats.to_csv('../data/scenarios_avec_strategies.csv')
```

**Résultats :**

| Type | Scénarios | % |
|------|-----------|---|
| D+A | 16 | 32 % |
| C | 11 | 22 % |
| A | 8 | 16 % |
| B | 7 | 14 % |
| D | 6 | 12 % |
| A+D | 2 | 4 % |

**Graphique :** `data/os2_resultats.png`

### Choix du template précis (`decision_engine.selectionner_template_id`)

Exemples : confusion modérée → **A1** ; frustration forte → **D2** ; ennui élevé → **C1**.

---

## 5. OS3fini.ipynb — Vingt templates + corpus Groq

### Phrase d'ouverture

> « OS3 définit **20 prompts structurés** (A1–D5) et génère un corpus avec **Groq Llama 3.1**. »

### Structure d'un template (exemple A1)

```python
"A1": f"""L'apprenant montre une confusion de niveau {niveau} sur {concept}...
Génère une explication alternative en 200 mots qui :
1. Reformule avec des mots simples
2. Utilise une analogie concrète
3. Décompose en 3 étapes
4. Termine par une question de vérification
Ton encourageant et rassurant."""
```

### Les 20 templates

| Famille | IDs | Objectif |
|---------|-----|----------|
| **A** | A1–A5 | Confusion → explication |
| **B** | B1–B5 | Ennui / engagement faible → exemples |
| **C** | C1–C5 | Ennui élevé → exercices |
| **D** | D1–D5 | Frustration → soutien émotionnel |

### Génération corpus

```python
pattern_templates_complet = {
    'stable':             ['D4', 'B2', 'D5'],
    'ennui_eleve':        ['C1', 'C2', 'B3', 'C3', 'B1'],
    'frustration':        ['D1', 'D2', 'D3', 'D4'],
    'confusion':          ['A1', 'A2', 'A5', 'A4', 'A3'],
    'engagement_faible':  ['B1', 'B2', 'B3'],
    'critique':           ['A3', 'D2', 'B2', 'D1', 'A1']
}
```

**Résultats (`results/corpus_metrics.json`, `corpus_audit_summary.json`) :**

| Métrique | Valeur |
|----------|--------|
| Total contenus | **220** |
| Mots moyens | **225** (115–434) |
| Templates utilisés | **20/20** |
| Audit qualité | **97,7 % OK** |

**Mots moyens par famille :** D ~130 (court) ; C ~300 (exercices longs).

**Sauvegarde :** `data/corpus_200.csv`

### Structure universelle

`generer_contenu_universel(...)` → reprise dans `backend/prompts/templates.py`.

---

## 6. OS4.ipynb — Module LLM backend

### Phrase d'ouverture

> « OS4 teste `generer_contenu_adaptatif()` — même fonction que l'API `POST /analyser`. »

Le notebook enchaîne **4 appels Groq** (un par type A/B/C/D). Sur le plan gratuit Groq (`llama-3.1-8b-instant`, ~6 000 tokens/min), le **4ᵉ appel** peut échouer avec une erreur **429 (rate limit)** si les requêtes sont trop rapprochées — ce n'est pas un bug de logique.

**Mitigations en place :**
- **`llm_module.py`** : retry automatique sur `RateLimitError` (jusqu'à 4 tentatives, attentes 2 / 5 / 8 s).
- **`OS4.ipynb`** : pause de **5 s** entre chaque test pour limiter le débit.

```python
import time
from llm_module import generer_contenu_adaptatif

# Boucle sur 4 cas ; time.sleep(5) entre chaque appel
r1 = generer_contenu_adaptatif(etat="Confusion", niveau=2, concept="...", template_id="A1")
```

| Test | État | Niveau | Type | Mots |
|------|------|--------|------|------|
| 1 | Confusion | 2 | A | ~195 |
| 2 | Boredom | 3 | C | ~257 |
| 3 | Frustration | 2 | D | ~154 |
| 4 | Engagement faible (niv. 1) | 1 | B | ~234 |

**À dire si l'encadrante voit un 429 :** « Limite API Groq sur le tier gratuit ; le module réessaie automatiquement. En soutenance, l'UI ne fait qu'un appel à la fois. »

**OS3 = conception des prompts ; OS4 = industrialisation backend.**

---

## 7. Rag.ipynb et TesterRAG.ipynb

### Rag.ipynb — Comparaison sans / avec RAG

1. **Sans RAG :** génération LLM seul → `Sources RAG : Aucune`
2. **Avec RAG :** ChromaDB **9 documents** → **2 docs** récupérés par requête → injectés dans le prompt
3. **4 types** testés (Confusion, Ennui, Frustration, Engagement)

**Pipeline RAG :**

1. Requête (concept + état)
2. Embedding `sentence-transformers`
3. Recherche similarité ChromaDB
4. `construire_contexte_rag()` → prompt enrichi
5. Groq génère un contenu ancré sur le corpus

**Prototype actuel :** RAG **systématique** (`llm_module.py`).

### TesterRAG.ipynb — Test unitaire

```python
from rag_module import rechercher_documents, construire_contexte_rag
documents = rechercher_documents(query="...", etat="Confusion", n_results=3)
```

Validation technique de la recherche, sans génération LLM.

---

## 8. Glossaire

### Parser

Programme qui lit un fichier **texte brut** (annotations IEMOCAP `.txt`) et produit un **CSV structuré** (`iemocap_parsed.csv`, 10 039 lignes). Fichier : `backend/iemocap_parser.py`.

### `xxx` (IEMOCAP)

Code émotion = **non classifié / indéterminé** (~25 % des utterances). Mappé vers **Boredom niveau 1** par prudence.

### VAD

**V**alence, **A**ctivation, **D**ominance — scores dimensionnels 1–5 par utterance IEMOCAP. Complète les labels discrets (`hap`, `fru`…).

### Fusion multimodale

- **60 %** visage DAiSEE (engagement e-learning)
- **40 %** voix IEMOCAP
- Conflit → priorité pédagogique (Frustration > Confusion > Boredom > Engagement)

---

## 9. Chaîne complète — schéma

```
OS1Daisee     → scenarios_50.csv
OS1IEMOCAP    → iemocap_parsed.csv + fusion
OS2fini       → Type A/B/C/D (adaptation.py)
OS3fini       → 20 templates → corpus_200.csv
OS4 + Rag     → llm_module.py → POST /analyser (UI)
```

**Exemple bout en bout :**

```
Scores : Frustration=3, Engagement=1
  → OS2 : Type D+A
  → decision_engine : template D2
  → OS3/OS4 + RAG : contenu soutien ~170 mots
```

---

## 10. Chiffres à mémoriser

| # | Chiffre | Contexte |
|---|---------|----------|
| 1 | 8 571 | clips DAiSEE |
| 2 | 50 | scénarios benchmark |
| 3 | 10 039 | utterances IEMOCAP |
| 4 | 6 | patterns OS1 |
| 5 | 189 | transitions critiques |
| 6 | 404 | alertes multi-clips |
| 7 | 60/40 | poids fusion |
| 8 | 20 | templates A1–D5 |
| 9 | 220 | contenus corpus |
| 10 | 97,7 % | audit corpus OK |
| 11 | 32 % | scénarios Type D+A |
| 12 | 80 % | conflits fusion (limite) |

---

## 11. Questions fréquentes — réponses courtes

| Question | Réponse |
|----------|---------|
| Pourquoi pas ML pour la décision ? | Règles expertes interprétables, calibrées sur OS1. |
| `random_state=42` ? | Reproductibilité scientifique. |
| Pourquoi 80 % conflits fusion ? | Visage DAiSEE ≠ voix IEMOCAP ; priorité pédagogique. |
| Vidéo DAiSEE dans le flux ? | Non — labels CSV (simulation) ; vidéo = éval FER séparée. |
| OS3 vs OS4 ? | OS3 conçoit prompts ; OS4 teste le module backend. |
| Erreur 429 Groq dans OS4 ? | Rate limit API (4 appels d'affilée) ; retry auto dans `llm_module.py` + pause 5 s dans le notebook. |
| Corpus 200 ou 220 ? | Fichier `corpus_200.csv`, **220 lignes** générées. |
| IEMOCAP remplace DAiSEE ? | Non — IEMOCAP = voix seule ; DAiSEE = engagement e-learning. |

---

## 12. Démo soutenance — scénarios suggérés

| # | Pattern | Attendu |
|---|---------|---------|
| 1 | stable | Type D, renforcement |
| 7 | ennui_eleve | Type C, exercice |
| — | frustration | Type D+A, soutien |
| — | conflit fusion | Montrer explication fusion dans l'UI |

Commandes utiles :

```powershell
cd backend
C:\Python311\python.exe -m uvicorn main:app --reload --port 8000

cd frontend
npm start
```

---

*Document généré pour la soutenance EmoLearn — à lire avec les notebooks dans `notebooks/`.*
