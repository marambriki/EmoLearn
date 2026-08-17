# Entraînement léger EmoLearn — FER sklearn + SER wav2vec2 (tête)

Deux contributions **complémentaires** aux modèles pré-entraînés :
le backbone reste figé ; on apprend uniquement le mapping vers DAiSEE.

## 1. Classifieur FER → DAiSEE (sklearn)

```bash
# Random Forest + GridSearchCV (défaut) — choisit n_estimators / max_depth
python scripts/train_fer_daisee_classifier.py --from-cache --classifier rf

# Étape 1 — rééquilibrage fort (oversampling TRAIN) sur le cache existant
python scripts/train_fer_daisee_classifier.py --from-cache --oversample --classifier rf

# Étape 1b — nouvelles features stratifiées (quota ~égal / état) + oversampling
python scripts/train_fer_daisee_classifier.py --sample 1200 --stratified --oversample --classifier rf

# FER + Action Units (MediaPipe blendshapes → AU1–26 proxy)
python scripts/train_fer_daisee_classifier.py --sample 800 --stratified --oversample --with-au --classifier rf

# Première extraction depuis les vidéos DAiSEE
python scripts/train_fer_daisee_classifier.py --sample 400 --classifier rf

# Sans Grid Search (params fixes)
python scripts/train_fer_daisee_classifier.py --from-cache --no-grid-search
```

| Entrée | Sortie |
|--------|--------|
| Scores FER 7 classes (modèle pré-entraîné) | Boredom / Confusion / Frustration / Engagement |
| + AU (si `--with-au`) | `models/fer_daisee_clf_au.joblib` |
| Vidéos `data/DAiSEE/DataSet/` | `models/fer_daisee_clf.joblib` |

**AU (complément)** : MediaPipe Face Landmarker blendshapes → vecteur AU pédagogique
(AU1_2, AU4, AU5, AU6_7, AU9, AU12, AU15, AU17, AU20, AU25_26).  
Runtime : si `models/fer_daisee_clf_au.joblib` existe et `EMOLEARN_FER_AU=1` (défaut),
le pipeline utilise **FER 7 + AU** pour le mapping DAiSEE.

```bash
$env:EMOLEARN_FER_AU="1"   # défaut : FER+AU si modèle présent
$env:EMOLEARN_FER_AU="0"   # désactive AU
```

**Algo** = Random Forest (sklearn). **GridSearchCV** sélectionne les hyperparamètres  
(`n_estimators`, `max_depth`, `min_samples_leaf`) par validation croisée stratifiée.

**Rééquilibrage** :
- `--stratified` : échantillonnage labels avec quota par état (réduit ~75 % Engagement)
- `--oversample` : duplique les classes rares sur le **train** uniquement (+ `class_weight=balanced_subsample`)

**Résultats mesurés (août 2026)** :

| Variante | Acc. | F1 macro | Frustration recall | Note |
|----------|------|----------|--------------------|------|
| Baseline (déséquilibré) | 0,72 | 0,26 | 0 % | Collapse Engagement |
| Oversample seul | 0,70 | 0,30 | 0 % | Gain F1 léger |
| **Stratifié 300/état** | 0,28* | 0,27 | **20 %** | *test équilibré ; Confusion recall 41 % |

Le modèle **stratifié** est celui déployé (`models/fer_daisee_clf.joblib`) : priorise les états à risque pédagogiques.

Métriques : `results/fer_classifier_metrics.json`  
Cache features : `results/fer_features_cache.csv` (stratifié)

Si `gain_accuracy ≤ 0`, relancer avec plus de données :

```bash
python scripts/train_fer_daisee_classifier.py --sample 800 --classifier rf
```

**Usage runtime** (défaut = `learned` — classifieur entraîné) :

```bash
$env:EMOLEARN_FER_MAPPING="learned"      # défaut : classifieur sklearn
$env:EMOLEARN_FER_MAPPING="auto"         # learned seulement si gain > 0
$env:EMOLEARN_FER_MAPPING="heuristique"  # mapping fixe (baseline)
```

## 2. Fine-tune ViT FER → DAiSEE (backbone partiel + tête)

```bash
# Fine-tune partiel : tête + 2 dernières couches ViT
python scripts/finetune_fer_vit_daisee.py --sample 800 --epochs 3 --unfreeze-layers 2

# Fine-tune COMPLET (tout le backbone) — plus long, LR auto 1e-5
.\venv\Scripts\python.exe scripts/finetune_fer_vit_daisee.py --from-faces-cache --epochs 5 --unfreeze-all

# Rejouer sans ré-extraire les visages (partiel plus profond)
python scripts/finetune_fer_vit_daisee.py --from-faces-cache --epochs 5 --unfreeze-layers 4
```

| Entrée | Sortie |
|--------|--------|
| Visages DAiSEE (stratifiés) | 4 états DAiSEE |
| `dima806/facial_emotions_image_detection` | `models/fer_vit_daisee/` |

**Ce n’est pas la tête sklearn** : on remplace le classifieur ViT (7→4) et on
dégèle les dernières couches (`--unfreeze-layers`).  
Runtime : si le dossier existe, `fer_offline.py` l’utilise automatiquement  
(`EMOLEARN_FER_MODE=finetuned|pretrained|auto`).

Métriques : `results/fer_finetune_metrics.json`  
Cache images : `results/fer_faces_cache/`

**Résultats mesurés** :
- Partiel 800, unfreeze=2 : F1 macro **0,36** (vs ~0,27 sklearn)
- **Full backbone 800, 5 epochs** : F1 macro **0,39** (`results/fer_finetune_full_metrics.json`)
- Large 1566, unfreeze=4 : F1 macro **0,37** (`results/fer_finetune_large_metrics.json`) — pas meilleur que le full

## 3. Fine-tune tête SER wav2vec2 (incl. Confusion)

```bash
python scripts/finetune_ser_wav2vec2.py --max-samples 800 --epochs 5 --augment
# dernière couche encodeur
python scripts/finetune_ser_wav2vec2.py --max-samples 1200 --epochs 5 --unfreeze-last --augment
# COMPLET (tout wav2vec2) — LR auto 1e-5
.\venv\Scripts\python.exe scripts/finetune_ser_wav2vec2.py --max-samples 1200 --epochs 5 --unfreeze-all --augment
```

| Entrée | Sortie |
|--------|--------|
| Segments IEMOCAP `.wav` | 4 états DAiSEE |
| Labels via `emotion → etat_daisee` (**sur/fea → Confusion**) | `models/ser_wav2vec2_daisee/` |

Options utiles : `--augment` (gain/bruit/shift), `--label-smoothing 0.05`, cosine LR.

Métriques : `results/ser_finetune_metrics.json`

**Runtime produit (hybride, défaut)** — corrige le biais fréquent `hap → Confusion` du fine-tune :

1. Fine-tuné DAiSEE (4 états, Confusion incluse)
2. SUPERB ER (`neu/hap/ang/sad`) mappé vers DAiSEE (`hap→Engagement`)
3. Fusion adaptative : si SUPERB est confiant sur Engagement/Frustration et le FT dit Confusion → prioriser SUPERB

```bash
$env:EMOLEARN_SER_METHODE="learned"       # hybride FT+SUPERB (défaut)
$env:EMOLEARN_SER_HYBRID="0"              # FT seul (A/B)
$env:EMOLEARN_SER_METHODE="heuristique"   # baseline librosa
```
## 4. Fiabilité produit (pondération par incertitude)

Sans promettre 100 % de bonnes prédictions :

| Levier | Env / code | Effet |
|--------|------------|--------|
| Fusion | pondération continue | `w = prior 60/40 × confiance`, renormalisé |
| Repli | confiances nulles | priorité pédagogique |
| Filtre frames FER | `EMOLEARN_SEUIL_FRAME_FER=0.35` | Rejette frames hésitantes avant moyenne |

Éval synthétique : `python scripts/eval_fusion_reliability.py`  
→ `results/fusion_reliability_metrics.json`

Cache FER élargi (1566) : `results/fer_faces_cache_large/`  
Fine-tune partiel 4 couches : `results/fer_finetune_large_metrics.json` (best F1≈0,37)  
→ n’a **pas** battu le full backbone sur 800 (F1≈0,39) : le runtime garde `models/fer_vit_daisee/` issu du full.

## Positionnement scientifique

- **Pas** un re-training massif FER/LLM.
- Contribution : apprendre le **lien** features pré-entraînées → taxonomie DAiSEE + pondération par incertitude.
- Runtime SER = **hybride** fine-tuné DAiSEE + SUPERB ER (`hap/ang/…`) + heuristique librosa en repli.
- Plafond FER mesuré (~0,39 F1 sur 800) : la fiabilité produit repose aussi sur la confiance continue, pas seulement sur l’accuracy clip.

## Prérequis

- `scikit-learn`, `torch`, `transformers`, `librosa`, `opencv-python` (déjà dans `requirements.txt`)
- Données : `data/DAiSEE/DataSet/` + `data/IEMOCAP_full_release/` + `data/iemocap_parsed.csv`
