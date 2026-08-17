# FAQ jury — EmoLearn

Réponses courtes pour la soutenance (master recherche).

---

## 1. Quelles sont vos métriques de fusion DAiSEE + IEMOCAP ?

Dans le **pipeline produit**, visage (DAiSEE) et voix (IEMOCAP) ne sont pas la même personne → **pas d’accuracy gold multimodale**.

On évalue :
- chaque modalité séparément (FER, SER) ;
- une **proxy** de cohérence : état fusionné vs état dominant DAiSEE (`results/fusion_produit_proxy_metrics.json`) ;
- le **moteur de fusion** sur un banc IEMOCAP **aligné** (même locuteur) : accuracy vote **0,50**, rappel urgents priorité pédagogique **0,40**.

---

## 2. Pourquoi un appariement hétérogène ?

DAiSEE n’a pas d’audio d’apprentissage. IEMOCAP apporte la voix. C’est une **limite assumée** du prototype, mitigée par le banc aligné (Option A) pour valider le moteur de fusion.

---

## 3. Les templates pédagogiques sont-ils validés ?

Ils sont une **proposition argumentée**. Le protocole Delphi L1 et les formulaires sont prêts (`results/forms/`). Les scores experts restent à collecter — travail de terrain / perspectives.

---

## 4. Les règles temporelles sont-elles utilisées ?

Oui (v2.5) : branchées dans `pipeline.py`, exposées via `GET /sequences` et `POST /analyser/sequence`, visibles dans l’UI (mode séquence + urgence / règle).

---

## 5. Est-ce un système déployé en live ?

Non. C’est un **prototype de simulation** (OS5) : clips annotés, pas de captation temps réel d’apprenants MOOC.

---

## 6. Quelle est la contribution centrale ?

Une **architecture multimodale de bout en bout** (perception → décision → génération) avec **évaluation honnête** des limites (mapping, appariement, validation humaine partielle).

---

## 7. Que manque-t-il encore ?

Validation experte L1/L2 (scores), pilote utilisateurs L4, et un éventuel corpus audiovisuel d’apprentissage vraiment joint. Voir `docs/LIMITES_ET_PERSPECTIVES.md`.
