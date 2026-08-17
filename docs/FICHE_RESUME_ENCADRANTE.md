# EmoLearn — Fiche de synthèse du projet

**Type** : mémoire de master — prototype de recherche (simulation, OS5)
**API** : v2.5 — modes simulation / séquence / tuteur
**Date** : août 2026 (mise à jour 17/08)

---

## 1. Ce que j'ai développé

J'ai construit un pipeline complet de **personnalisation pédagogique adaptative multimodale**.
Je détecte l'état émotionnel/d'engagement d'un apprenant à partir de deux signaux — **visage** et
**voix** —, je décide s'il faut intervenir pédagogiquement, puis je **génère un contenu
adaptatif** via un LLM. J'ai également ajouté un **tuteur conversationnel**, un **parcours
multi-clips** avec règles temporelles, une **fusion à pondération continue par incertitude**,
et plusieurs **évaluations instrumentées** (fusion, entraînement, forme, similarité, fiabilité).

```
Scénario (clip vidéo + contexte pédagogique)
        │
        ▼
Perception émotionnelle   Visage (FER, DAiSEE)  +  Voix (SER, IEMOCAP)
        │                 → fusion 60/40 × confiance (pondération par incertitude)
        ▼
Décision pédagogique      déclenchement ? + type (A/B/C/D) + template (A1–D5)
        │                 + règles temporelles (urgence sur séquence de clips)
        ▼
Génération de contenu     LLM (Groq, Llama 3.1) + RAG (ChromaDB) + 20 templates
        │                 + tuteur conversationnel affect-aware (/chat)
        ▼
Restitution                interface React + export PDF + journal SQLite
```

### Deux pipelines (à ne pas confondre)

| | Pipeline produit | Banc d'essai (validation) |
| --- | --- | --- |
| But | Générer le contenu pédagogique | Valider le moteur de fusion |
| Visage | Vidéo DAiSEE réelle + FER | Vidéo IEMOCAP |
| Voix | Audio IEMOCAP **apparié par état** | Audio IEMOCAP **même locuteur**, synchronisé |
| Branché dans l'app | Oui (API + UI) | Non (script hors ligne) |

Les deux **partagent le même moteur de fusion** : ce que je valide au banc d'essai est exactement
ce que j'utilise en production.

**Contribution que je revendique** : une architecture multimodale de bout en bout (perception →
décision → génération), avec une **évaluation honnête** des limites (mapping, appariement,
validation humaine partielle).

### API principale (v2.5)

| Route | Rôle |
| --- | --- |
| `POST /analyser` | Analyse d'un clip (vidéo DAiSEE + audio IEMOCAP) |
| `GET /sequences` · `POST /analyser/sequence` | Parcours multi-clips + règles temporelles |
| `POST /chat` | Tuteur conversationnel affect-aware |
| `GET /scenarios` · `GET /templates` · `GET /sessions` | Référentiels et historique |
| `GET /rag/status` · `GET /health` | Santé RAG / API |

---

## 2. Techniques et algorithmes que j'ai mis en œuvre

### 2.1 Perception visage — FER

| Étape | Ce que j'ai fait |
| --- | --- |
| Détection | MediaPipe (BlazeFace ; courte/longue portée + miroir) |
| Extraction | 5 frames réparties sur la vidéo DAiSEE |
| Classification | HuggingFace **`dima806/facial_emotions_image_detection`** (ViT, 7 émotions) |
| Agrégation | **Vote multi-frames** (moyenne des probabilités) |
| Mapping sklearn | **Random Forest** (GridSearchCV) 7 scores → 4 états, avec **échantillon stratifié** + **oversampling** |
| Fine-tune ViT partiel | Tête 4 classes + couches dégelées (2 ou 4) |
| Fine-tune ViT **complet** | Tout le backbone (`--unfreeze-all`, LR 1e-5) — **meilleur F1**, modèle runtime `models/fer_vit_daisee/` |

Runtime configurable : `EMOLEARN_FER_MAPPING` (learned/auto/heuristique) et
`EMOLEARN_FER_MODE` (finetuned/pretrained/auto).

### 2.2 Perception voix — SER

| Étape | Ce que j'ai fait |
| --- | --- |
| Signal | Audio réel IEMOCAP (.wav), couverture 100 % |
| Heuristique | Features **librosa** (RMS, ZCR, centroïde spectral, pitch `pyin`) + règles → VAD |
| Fine-tune tête | **`superb/wav2vec2-base-superb-er`** : backbone gelé, tête → 4 états DAiSEE (Confusion via `sur`/`fea`) |
| Fine-tune **complet** | Tout wav2vec2 dégelé (`--unfreeze-all`, 1 200 samples, 5 époques, loss **class-weighted**) → `models/ser_wav2vec2_daisee_full/` |

Runtime : `EMOLEARN_SER_METHODE=learned` (défaut) ou `heuristique`.

### 2.3 Fusion multimodale (+ fiabilité produit)

| Composant | Ce que j'ai fait |
| --- | --- |
| Prior visage/voix | **60 % / 40 %** (balayage de sensibilité réalisé) |
| Pondération continue | Poids effectifs = `prior × confiance`, puis renormalisation (late fusion) |
| Repli | Priorité pédagogique **seulement** si les deux confiances sont nulles |
| Filtre frames FER | Rejet des frames hésitantes avant moyenne (`EMOLEARN_SEUIL_FRAME_FER=0.35`) |

Plus de seuil dur « gate » : une modalité peu confiante est **sous-pondérée**, pas forcément
ignorée. Éval : `scripts/eval_fusion_reliability.py`.

### 2.4 Règles temporelles + parcours séquence

Moteur de règles sur l'historique : déclenchement **immédiat**, **rapide** (2–3 clips) ou
**différé** (4–5 clips). Branché dans le pipeline et exposé via `/sequences` et
`/analyser/sequence` (mode séquence dans l'UI).

### 2.5 Moteur de décision pédagogique

À partir des 4 scores DAiSEE, je choisis un **type** puis **1 template parmi 20** :

| Type | Rôle | Templates |
| --- | --- | --- |
| A | Explication alternative | A1–A5 |
| B | Exemple concret | B1–B5 |
| C | Exercice progressif | C1–C5 |
| D | Encouragement / motivation | D1–D5 |

Sur 50 scénarios de benchmark, les templates les plus fréquents sont D1, C1, D2/D4, A1, B1 — la
couverture A–D est effective.

### 2.6 Génération — LLM + RAG + tuteur

| Composant | Ce que j'ai fait |
| --- | --- |
| LLM | **Groq — GPT-OSS 20B** (`openai/gpt-oss-20b`, temp. 0,3), retry/backoff |
| RAG | **ChromaDB** (HNSW, cosinus) + embeddings **`paraphrase-multilingual-MiniLM-L12-v2`** |
| Tuteur `/chat` | Ton, consigne et température adaptés à l'état détecté |
| Qualité contenu | Métriques de **forme** + similarité **BLEU / ROUGE** (cohérence interne par template) |

### 2.7 Stack technique

- **Backend** : Python 3.11, FastAPI, SQLite
- **Frontend** : React (interface, export PDF, mode séquence)
- **ML/DL** : scikit-learn, PyTorch, HuggingFace `transformers`, OpenCV, **MediaPipe** (BlazeFace), librosa
- **LLM** : Groq — `openai/gpt-oss-20b` (remplace Llama 3.1 8B retiré)
- **Tests** : pytest (API, fusion, SER, temporel, RAG, entraînement)
- **Notebooks** : analyses exploratoires OS1–OS4 (DAiSEE, IEMOCAP, FER, décision, RAG)

---

## 3. Données et mapping

| Corpus | Rôle |
| --- | --- |
| **DAiSEE** | Socle : taxonomie d'engagement (4 états × niveaux 0–3), scénarios, déclenchement |
| **IEMOCAP** | Voix + banc audiovisuel aligné (même locuteur) pour valider la fusion |
| **Corpus pédagogique** | ~200 contenus générés + échantillon d'évaluation experts (~60) |

### Mapping IEMOCAP → DAiSEE (hypothèse documentée)

| Émotion IEMOCAP | État DAiSEE |
| --- | --- |
| hap, exc | Engagement |
| neu, sad | Boredom |
| ang, fru, dis | Frustration |
| sur, fea | Confusion |

Hypothèse de construit, pas une équivalence démontrée. Sensibilité mesurée (§4.7) : jusqu'à
**17 %** des états changent selon la ré-interprétation.

**Audit du corpus** : 215/220 contenus OK (**97,7 %**) ; 5 cas à revoir (templates/ton suspects).

---

## 4. Résultats d'évaluation

> **Origine des métriques (à ne pas confondre)**  
> - §§4.1–4.3 : entraînement / test des briques (prototype, pas le banc AV aligné)  
> - §4.4 et §4.6 : **banc d'essai** IEMOCAP audiovisuel aligné (même locuteur)  
> - §4.5 : **proxy produit** (pas d'accuracy gold multimodale — visage ≠ voix)  
> - §§4.8–4.9 : qualité du contenu LLM  
> - §4.10 : fiabilité fusion (pondération par incertitude), cas synthétiques

### 4.1 Classifieur FER → DAiSEE (Random Forest)

DAiSEE déséquilibré (~75 % Engagement). Variantes sur 1 200 échantillons :

| Variante | Accuracy | F1 macro | Rappel Frustration | Note |
| --- | :---: | :---: | :---: | --- |
| Baseline déséquilibrée | 0,72 | 0,26 | 0 % | Collapse Engagement |
| Oversampling seul | 0,70 | 0,30 | 0 % | Gain F1 léger |
| **Stratifié 300/état (retenu)** | 0,28* | 0,27 | **20 %** | *test équilibré ; Confusion **41 %** |

### 4.2 Fine-tune ViT FER → DAiSEE

| Variante | Samples | Époques | Accuracy | F1 macro | Note |
| --- | :---: | :---: | :---: | :---: | --- |
| Partiel (2 couches) | 800 | 3 | 0,36 | 0,361 | Rappel Frustration 38 % |
| Large (4 couches) | ~1 566 | 5 | 0,378 | 0,370 | N'a pas battu le full |
| **Complet (backbone entier, retenu)** | 800 | 5 | **0,39** | **0,388** | Rappel Frustration **48 %** |

Le fine-tune **complet** reste le meilleur compromis ; le runtime garde `models/fer_vit_daisee/`.

### 4.3 Fine-tune SER wav2vec2 → DAiSEE

| Variante | Samples | Époques | Meilleure Acc. | Meilleur F1 macro |
| --- | :---: | :---: | :---: | :---: |
| Tête seule (backbone gelé) | 600 | 2 | 0,300 | 0,309 |
| **Complet (`--unfreeze-all`, class-weighted)** | 1 200 | 5 | **0,557** | **0,537** |

Le fine-tune **complet** (époque 4) est nettement supérieur à la tête seule. En banc d'essai
aligné (§4.4), même la version apprise (tête) passe déjà de 0,263 → **0,487** vs heuristique
(+0,224).

### 4.4 Banc d'essai fusion (n = 80, aligné)

Échantillon = **20 × 4 états** (Session 1 IEMOCAP, durée ≥ 1 s, 5 frames, seed 42), même
locuteur. Visages détectés : **100 %**.

| Modalité / approche | Accuracy | Rappel urgents |
| --- | :---: | :---: |
| Visage — FER 1 frame | 0,375 | 0,250 |
| Visage — FER multi-frames | 0,412 | 0,275 |
| Voix — SER heuristique | 0,263 | 0,225 |
| Voix — **SER appris (wav2vec2)** | **0,487** | 0,200 |
| Fusion — priorité pédagogique | 0,338 | **0,400** |
| Fusion — **vote pondéré** | **0,500** | 0,125 |

| Gain mesuré | Δ accuracy |
| --- | :---: |
| SER appris vs heuristique | +0,224 |
| FER multi-frames vs 1 frame | +0,037 |
| Fusion vote vs priorité pédagogique | +0,162 |
| Fusion vote vs meilleure unimodale | +0,013 |

**Arbitrage** : vote → accuracy ; priorité pédagogique → rappel des états critiques.

### 4.5 Proxy pipeline produit (n = 52)

Vérité terrain = état dominant DAiSEE (**pas** accuracy multimodale gold) :

| Métrique | Fusion produit |
| --- | :---: |
| Accuracy vs état dominant | 0,50 |
| Rappel états urgents | 1,00 |

### 4.6 Sensibilité au poids visage/voix (n = 80)

| Poids visage/voix | Accuracy | Rappel urgents |
| --- | :---: | :---: |
| 0,5 / 0,5 | 0,287 | 0,375 |
| **0,6 / 0,4 (retenu)** | 0,300 | **0,425** |
| 0,7 / 0,3 | **0,338** | 0,400 |
| 0,8 / 0,2 | 0,325 | 0,350 |
| 0,9 / 0,1 | 0,325 | 0,350 |

### 4.7 Sensibilité du mapping (10 039 utterances IEMOCAP)

| Variante | % d'états changés |
| --- | :---: |
| neutre → Engagement | 17,0 % |
| tristesse → Frustration | 10,8 % |
| surprise → Engagement | 1,1 % |
| dégoût → Boredom | 0,0 % |

### 4.8 Qualité de forme du corpus LLM (N = 208)

| Métrique | Valeur |
| --- | --- |
| Longueur moyenne | 222 mots (min 113, max 410) |
| Lisibilité (Kandel-Moles) | 95,0 |
| Taux de répétition moyen | 0,479 |
| Part dans la cible de longueur | 0,832 |

### 4.9 Similarité LLM — BLEU / ROUGE (N = 189)

Référence **auto** = contenu de longueur médiane par template (cohérence interne, **pas** gold
expert) :

| Métrique | Moyenne |
| --- | :---: |
| BLEU-4 | 0,139 |
| ROUGE-1 F1 | 0,449 |
| ROUGE-2 F1 | 0,176 |
| ROUGE-L F1 | 0,272 |

Ces scores mesurent la **proximité à une référence interne**, pas la pertinence pédagogique
(réservée à L2).

### 4.10 Fiabilité fusion — pondération par incertitude (cas synthétiques, n = 6)

Sans prétendre à 100 % de bonnes prédictions en conditions réelles, j'évalue la fusion produit
sur 6 cas critiques (modalité bruitée, incertitude totale, etc.) :

| Méthode | Accuracy (cas synthétiques) |
| --- | :---: |
| **Pondération continue** (`w = prior × confiance`) | **1,000** |

Exemple : une voix bruitée est **sous-pondérée** (au lieu de forcer un état urgent erroné).
Repli priorité pédagogique uniquement si les deux confiances sont nulles.

---

## 5. Validation experte et perspectives

| Niveau | Objectif | Statut |
| --- | --- | --- |
| L1 | Delphi mapping engagement → stratégie | Formulaires prêts (`results/forms/`) — scores à collecter |
| L2 | Notation enseignante des contenus (≈60) | Formulaires prêts — scores à collecter |
| L3 | Métriques automatiques (forme, fusion, similarité, fiabilité) | Fait |
| L4 | Pilote utilisateurs (n ≈ 30) | À faire |

**Perspectives prioritaires** :
1. Collecter les scores L1/L2 (terrain)
2. Rejouer le **banc d'essai aligné** avec SER/FER full fine-tune + fusion par incertitude
3. Pilote utilisateurs L4
4. Corpus audiovisuel d'apprentissage vraiment joint (si données)

---

## 6. Limites que j'assume

| # | Tension | Statut |
| --- | --- | --- |
| 1 | Fusion vs unimodal | Compromis mesuré (vote vs priorité pédagogique) |
| 2 | Mapping émotions → engagement | Hypothèse + analyse de sensibilité |
| 3 | Adaptatif vs simulation | Prototype (pas de captation live MOOC) |
| 4 | Appariement visage/voix hétérogène (produit) | Documenté + mitigé via banc aligné |
| 5 | Templates non validés par experts | Proposition + protocole prêt |
| 6 | Qualité LLM | Forme + similarité mesurées ; fond → L2 |
| 7 | Déséquilibre DAiSEE | Mitigé (stratifié + fine-tune ViT) |
| 8 | Pas d'accuracy gold multimodale produit | Proxy + banc aligné (visage ≠ voix en produit) |
| 9 | Confusion peu couverte par la voix (modèle 4 classes d'origine) | Améliorée via fine-tune DAiSEE (sur/fea) |
| 10 | Plafond FER (~0,39 F1) | Mitigé par pondération continue selon la confiance |
| 11 | Largeur vs profondeur | Contribution = chaîne complète + évaluation critique |

---

*Message clé : la force du projet n'est pas un score unique élevé, mais une chaîne complète,
instrumentée, avec des limites explicites plutôt que dissimulées.*
