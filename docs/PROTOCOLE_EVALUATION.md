# Protocole d’évaluation multi-niveaux — EmoLearn

Méthodologie alignée sur l’OS6 et le chapitre 5 du mémoire.

---

## Niveau 1 — Validation du modèle engagement → adaptation

### Objectif

Vérifier la cohérence théorique du mapping (4 états DAiSEE → 4 types de contenu + 20 templates).

### Participants

- 3 experts : sciences de l’éducation / ingénierie pédagogique / e-learning

### Matériel

- Document : matrice engagement-adaptation (extrait du mémoire OS2)
- 20 fiches : une par template (A1–D5) avec exemple de déclenchement
- 15 cas tests tirés de `data/scenarios_50.csv` (stratifiés par pattern)

### Procédure (Delphi simplifié — 2 tours)

**Tour 1 — Individuel (30 min/expert)**  
Pour chaque cas test, l’expert note sur Likert 1–5 :

| Critère | Description |
|---------|-------------|
| C1 Cohérence théorique | L’état DAiSEE justifie-t-il l’intervention ? |
| C2 Pertinence stratégie | Le type A/B/C/D est-il adapté ? |
| C3 Adéquation template | Le template_id (ex. A3) est-il le bon sous-type ? |
| C4 Urgence | Le déclenchement est-il opportun ? |

**Tour 2 — Synthèse (1h en visio)**  
Consensus sur les cas < 3/5 ; ajustement des règles dans `adaptation.py` / `decision_engine.py`.

### Livrable

- `results/validation_experts_mapping.csv`
- Section 5.2 du mémoire

---

## Niveau 2 — Qualité du contenu généré

### Objectif

Mesurer pertinence pédagogique, clarté et adéquation à l’état émotionnel.

### Échantillon

- **60 contenus** stratifiés : 15 × types A, B, C, D (via `template_id`)
- Source : régénération API figée OU sous-ensemble audité de `corpus_200.csv`
- Même `concept` pour limiter la variance : « les algorithmes de tri »

### Évaluateurs

- 3 enseignants (informatique / pédagogie)

### Grille (Likert 1–5 par critère)

| Code | Critère |
|------|---------|
| Q1 | Pertinence par rapport au concept |
| Q2 | Clarté et structure |
| Q3 | Adéquation à l’état d’engagement annoncé |
| Q4 | Qualité pédagogique globale |
| Q5 | Utilisabilité en contexte MOOC |
| Q6 | Absence d’erreurs factuelles (avec RAG) |

### Métriques automatiques (complément)

À calculer via `scripts/metrics_contenu.py` :

- Nombre de mots (cible : 150–250 selon template)
- Indice de lisibilité (adaptation française si possible)
- Taux de répétition de phrases
- Présence des 4 points des instructions template (heuristique mots-clés)

### Livrable

- `results/evaluation_experts_contenu.csv`
- Boxplots par `template_id`
- Section 5.3 du mémoire

---

## Niveau 3 — Validation technique FER (QR1)

### Protocole

1. Extraire N frames (ex. 500) depuis clips DAiSEE avec labels.
2. Inférer avec `fer_module.analyser_visage(path)`.
3. Mapper émotions FER → états DAiSEE (`mapping_fer_daisee`).
4. Comparer à l’état dominant du label clip.

### Métriques

- Accuracy, Precision, Recall, F1 (macro)
- Matrice de confusion
- Temps d’inférence moyen par frame

### Livrable

- `results/fer_metrics.json` + figures
- Section 4.2 / 5.1 du mémoire

---

## Niveau 4 — Étude pilote utilisateurs (n = 30)

### Design

- **Within-subject** : 2 conditions sur séquences vidéo courtes (DAiSEE)
  - **C1** : contenu statique (contrôle)
  - **C2** : contenu généré par EmoLearn (adaptatif)

### Participants

- Étudiants L2/L3 informatique ou équivalent
- Consentement éclairé + anonymisation

### Déroulement (≈ 45 min)

1. Questionnaire pré (profil, familiarité MOOC)
2. Visionnage 3–5 clips avec états différents
3. Après chaque clip : affichage adaptation C1 ou C2 (contrebalancer l’ordre)
4. Questionnaires post-scénario

### Instruments

| Outil | Mesure |
|-------|--------|
| Likert 1–7 | Utilité perçue de l’adaptation |
| Likert 1–7 | Adéquation émotionnelle perçue |
| Likert 1–7 | Satisfaction contenu |
| SUS (10 items) | Utilisabilité du prototype |
| 3 questions ouvertes | Qualitatif |

### Hypothèses exploratoires

- H1 : C2 > C1 sur utilité et adéquation émotionnelle
- H2 : SUS ≥ 68 (seuil acceptabilité)

### Analyse

- Statistiques descriptives
- Wilcoxon apparié (C1 vs C2) si normalité non vérifiée
- Analyse thématique des verbatims

### Livrable

- `results/pilote_utilisateurs.csv`
- Section 5.4 du mémoire

---

## Traçabilité pour la soutenance

Pour chaque session API, conserver :

```json
{
  "scenario_id": 0,
  "clip_id": "...",
  "fusion": { "etat": "...", "niveau": 3 },
  "decision": { "template_id": "A3", "strategie": "..." },
  "prompt_hash": "...",
  "model": "llama-3.1-8b-instant",
  "rag_sources": []
}
```

*(Implémentation future : `backend/decision_log.py`)*

---

## Calendrier suggéré

| Semaine | Activité |
|---------|----------|
| S1 | Tests API + audit corpus |
| S2 | Évaluation FER + rédaction 4.2 |
| S3 | Delphi experts (L1) |
| S4 | Évaluation contenu 60 ex. (L2) |
| S5–S6 | Pilote n=30 (L4) |
| S7 | Rédaction chapitre 5 |
