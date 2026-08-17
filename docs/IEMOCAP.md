# IEMOCAP dans EmoLearn — positionnement scientifique

## 1. Rôle dans le prototype

IEMOCAP (Interactive Emotional Dyadic Motion Capture) apporte la **modalité vocale** du pipeline multimodal EmoLearn. Il ne remplace pas DAiSEE : les deux jeux de données couvrent des contextes différents.

| Aspect | DAiSEE | IEMOCAP |
|--------|--------|---------|
| Contexte | Apprenants regardant des cours (e-learning) | Dialogues actés (sessions 1–5) |
| Modalité | Visage / engagement | Voix + annotations émotionnelles |
| Étiquettes | 4 états × niveau 0–3 | Émotions discrètes + VAD |
| Usage EmoLearn | Ground truth scénarios (`scenarios_50.csv`) | Enrichissement vocal en fusion |

## 2. Architecture technique

```
data/IEMOCAP_full_release/
  Session{N}/dialog/
    EmoEvaluation/*.txt          → annotations VAD
    wav/{dialogue_id}.wav        → audio dialogue (segments par utterance)
        ↓
backend/iemocap_parser.py  →  data/iemocap_parsed.csv (+ wav_path)
backend/iemocap_audio.py   →  extraction segment [start, end]
        ↓
backend/ser_module.py      →  analyser_voix_depuis_scenario()
                               • signal audio réel OBLIGATOIRE (couverture .wav = 100%)
                               • EMOLEARN_SER_METHODE=learned (défaut) : SER fine-tuné DAiSEE
                               • EMOLEARN_SER_METHODE=heuristique : SER librosa (features acoustiques)
        ↓
backend/fusion_module.py   →  fusionner_modalites()    [60 % visage / 40 % voix]
                               fusionner_par_vote()     [vote pondéré sur l'état]
```

### Fichiers clés

- `backend/iemocap_parser.py` — parse les lignes `[t1 - t2] utterance_id emotion [V,A,D]`
- `backend/iemocap_audio.py` — résout les chemins `.wav` et extrait les segments
- `backend/ser_module.py` — charge le CSV parsé, SER audio ou annotations, mappe vers DAiSEE
- `scripts/enrich_iemocap_wav_paths.py` — ajoute `wav_path` / `wav_exists` au CSV
- `data/mapping_iemocap_daisee.csv` — table de correspondance documentée
- `backend/fusion_module.py` — détection et résolution de conflits

### Signal audio réel (SER sur .wav) — obligatoire

Le pipeline analyse **toujours** le signal audio réel de l'utterance IEMOCAP
appariée (aucun mode « annotations CSV seules » : la couverture `.wav` du
corpus est de 100 %, 10 039/10 039 utterances).

Le module extrait le segment `[start_time, end_time]` du fichier dialogue `.wav`
(ex. `Ses01F_impro01.wav`) puis applique le SER selon `EMOLEARN_SER_METHODE` :
- `learned` (défaut) : modèle wav2vec2 fine-tuné DAiSEE (`models/ser_wav2vec2_daisee/`)
- `heuristique` : **features acoustiques** (RMS, pitch, centroid spectral)

La réponse API inclut `voix.source`, `voix.wav_path` et éventuellement
`voix.features_acoustiques`.

## 3. Mapping IEMOCAP → états DAiSEE

Les émotions IEMOCAP (codes 2–4 lettres) sont projetées sur la taxonomie pédagogique DAiSEE :

| Code IEMOCAP | État DAiSEE | Niveau typique | Justification pédagogique |
|--------------|-------------|----------------|---------------------------|
| `hap`, `exc` | Engagement | 3 | Valence positive, activation forte |
| `neu` | Boredom | 1 | État neutre prolongé ≈ faible stimulation |
| `sad` | Boredom | 2 | Faible engagement |
| `ang`, `fru` | Frustration | 3 | Blocage, tension |
| `dis` | Frustration | 2 | Rejet, gêne modérée |
| `sur`, `fea` | Confusion | 2–3 | Incertitude, surprise cognitive |
| `xxx` | Boredom | 1 | Non classifié → défaut prudent |

Ce mapping est une **interprétation de constructe** : IEMOCAP mesure des émotions conversationnelles, DAiSEE mesure l’engagement en apprentissage. Il doit être présenté comme une **hypothèse de travail à valider** (cohérence interne du pipeline + validation experte Delphi L1, et analyse de sensibilité `scripts/sensibilite_mapping.py`), **pas** comme une vérité psychologique établie.

## 4. Appariement avec les scénarios DAiSEE

**Implémentation actuelle (v2.3) :**

```python
etat_dominant, _, _ = etat_depuis_ligne(scenario_daisee)
subset = df_iemocap[df_iemocap["etat_daisee"] == etat_dominant]
idx = scenario_id % len(subset)
row = subset.iloc[idx]
```

1. Calcul de l'**état dominant** du scénario DAiSEE (Boredom, Confusion, Engagement, Frustration).
2. Filtrage des utterances IEMOCAP mappées sur le **même état**.
3. Sélection reproductible via `scenario_id % len(subset)` (même scénario → même voix).
4. En mode audio : préférence aux utterances avec `.wav` disponible.

La réponse API inclut `voix.appariement` (`methode`, `etat_cible`, `pool_size`, etc.).

- **Limite conservée :** pas de synchronisation temporelle réelle entre clip DAiSEE et locuteur IEMOCAP.
- **Amélioration vs modulo :** la voix est **émotionnellement cohérente** avec le scénario facial.

**Pistes d’amélioration (perspectives) :**

1. ✅ **Réalisé (Option A)** — sous-corpus IEMOCAP **audiovisuel aligné** (visage + voix du *même* locuteur, synchronisés) pour valider la fusion sans appariement hétérogène. Voir §9.
2. Fine-tuning SER sur clips DAiSEE si audio disponible

## 5. Fusion multimodale

Poids : **60 % visage (DAiSEE)** / **40 % voix (IEMOCAP)**.

### Types de conflit

| Type | Exemple | Stratégie |
|------|---------|-----------|
| Accord parfait | Confusion + Confusion | Moyenne pondérée |
| Conflit mineur | Confusion vs Frustration proche | Confiance maximale |
| Conflit majeur | Engagement vs Frustration | Priorité pédagogique (Frustration > Confusion > Boredom > Engagement) |

Priorité pédagogique : en cas de désaccord fort, privilégier l’état le plus **urgent** pour l’intervention (frustration et confusion avant ennui).

## 6. Ce qu’il faut écrire (et ne pas écrire) dans le mémoire

### À inclure

- IEMOCAP comme **extension multimodale exploratoire**
- Table de mapping + discussion validité de constructe
- Statistiques du CSV parsé (`get_statistiques_iemocap()`)
- Exemples de conflits résolus (`expliquer_fusion()`)

### À éviter

- « Détection vocale en temps réel de l’apprenant MOOC »
- « IEMOCAP et DAiSEE sont synchronisés »
- Présenter la fusion comme validation clinique

### Formulation suggérée

> « La modalité vocale s’appuie sur des annotations IEMOCAP réelles, mappées vers la taxonomie DAiSEE, et fusionnées avec l’état d’engagement facial selon des règles à priorité pédagogique. L’appariement actuel par index de scénario constitue une simplification méthodologique pour le prototype. »

## 7. Statistiques (référence)

Après parsing complet : voir `data/iemocap_parsed.csv` (~10 000 utterances).

Commande :

```bash
cd backend
python -c "from ser_module import get_statistiques_iemocap; import json; print(json.dumps(get_statistiques_iemocap(), indent=2))"
```

## 8. Prérequis données

1. Télécharger **IEMOCAP Full Release** (accord académique)
2. Placer sous `data/IEMOCAP_full_release/`
3. Générer le CSV : `python -c "from iemocap_parser import charger_iemocap_complet, sauvegarder_iemocap_csv; sauvegarder_iemocap_csv(charger_iemocap_complet())"`

Sans le dossier complet, le module utilise `data/iemocap_parsed.csv` s’il existe déjà.

## 9. Sous-corpus audiovisuel aligné (Option A)

### 9.1 Motivation

Le pipeline principal apparie **visage DAiSEE** et **voix IEMOCAP**, qui ne proviennent
**pas du même apprenant** (limite documentée §4 et dans `docs/LIMITATIONS.md`).
Pour **valider la fusion multimodale** dans un cadre réellement aligné, on exploite le fait
qu’IEMOCAP est **audiovisuel** : pour une même *utterance*, on dispose du **visage** et de la
**voix du même locuteur, synchronisés** sur le segment `[start_time, end_time]`.

> « Le pipeline principal reste DAiSEE (visage) + IEMOCAP (voix). En complément, un sous-corpus
> IEMOCAP audiovisuel (visage + voix du même locuteur, synchronisés) sert à valider la fusion
> multimodale dans un cadre réellement aligné, levant la limite d’appariement hétérogène. »

### 9.2 Chaîne technique

```
data/IEMOCAP_full_release/Session1/dialog/avi/DivX/{dialogue}.avi   (720x480, 29.97 fps, dyadique)
        ↓  backend/iemocap_video.py
   • frame au MILIEU du segment [start, end]
   • crop de la moitié du LOCUTEUR (gauche/droite)
   • détection visage : MediaPipe (courte portée → longue portée → miroir)
        ↓  backend/fer_offline.analyser_visage_image()  →  état facial (visage_fer_iemocap)
data/IEMOCAP_full_release/Session1/dialog/wav/{dialogue}.wav
        ↓  backend/ser_module (SER acoustique, même segment)  →  état vocal
        ↓  backend/fusion_module.fusionner_modalites()        →  état fusionné
   Comparaison à l’annotation IEMOCAP mappée (etat_daisee = vérité terrain)
```

Script : `scripts/build_iemocap_av_corpus.py`
(`--n-per-state`, `--seed`, `--inclure-xxx`).
Sorties : `results/iemocap_av_fusion.csv` (détail) + `results/iemocap_av_metrics.json` (métriques).

### 9.3 Disposition vidéo dyadique (point méthodologique clé)

Chaque vidéo `dialog` montre **deux acteurs côte à côte**. La règle de sélection du visage :

- l’acteur **porteur des marqueurs mocap** (lettre du dialogue `SesNN[F/M]`) est à **gauche** ;
- le partenaire est à **droite** ;
- locuteur (suffixe `_F###` / `_M###`) : `locuteur == porteur ⇒ gauche`, sinon `droite`.

Cette règle est implémentée dans `iemocap_video.speaker_side()`.

### 9.4 Contraintes de données

- **Seule la Session 1** contient le dossier `avi/DivX` dans ce dépôt (28 dialogues) →
  le sous-corpus est restreint à Session 1.
- Bug corrigé au passage : `iemocap_audio.dialogue_id_from_utterance` ne gérait que `_F###`
  (les `.wav` des locuteurs **masculins** étaient introuvables) → corrigé en `_[FM]\d+$`.

### 9.5 Résultats (run n = 80, 20 par état, seed = 42, durée min = 1 s, 5 frames/utterance)

| Indicateur | Valeur |
|------------|--------|
| Visages détectés | **100 %** (MediaPipe, vote multi-frames) |
| Segments retenus | 80/80 (nettoyage : durée ≥ 1 s, visage présent) |

**Comparaison directe des améliorations sur le même corpus aligné :**

| Modalité / approche | Accuracy ↑ | Rappel états urgents ↑ |
|---------------------|:---:|:---:|
| Visage — FER **1 frame** (milieu) | 0,375 | 0,250 |
| Visage — FER **multi-frames** (vote, 5 images) | **0,412** | 0,275 |
| Voix — SER **heuristique** (librosa) | 0,263 | 0,225 |
| Voix — SER **appris** (wav2vec2 IEMOCAP) | **0,487** | 0,200 |
| Fusion — **historique / priorité pédagogique** (FER 1f + SER heuristique) | 0,338 | **0,400** |
| Fusion — **vote pondéré** (FER multi + SER appris) | **0,500** | 0,125 |

**Gains mesurés :** SER appris vs heuristique **+0,224** ; FER multi vs 1 frame **+0,037** ;
fusion vote vs historique **+0,162** ; fusion vote vs **meilleure** modalité unimodale **+0,013**.

### 9.6 Lecture honnête (à écrire dans le mémoire)

- **Apport méthodologique** : la fusion est évaluée sur des modalités **réellement synchronisées**
  (même locuteur, même instant) → **lève la limite d’appariement hétérogène**.
- **SER appris = plus gros gain** : remplacer les seuils heuristiques par un **wav2vec2 pré-entraîné
  sur IEMOCAP** fait passer la voix de **0,263 → 0,487** (+0,224, quasi-doublement). La comparaison
  *heuristique vs appris* sur le **même** corpus est une **contribution méthodologique** du mémoire.
- **La fusion par vote dépasse enfin chaque modalité en accuracy** (0,500 > 0,487 voix et 0,412
  visage) : le **vote pondéré sur les probabilités d’état** (et non plus seulement sur le niveau)
  permet au multimodal d’apporter une vraie valeur ajoutée sur l’exactitude globale.
- **Arbitrage accuracy ↔ sécurité pédagogique, désormais mesuré** : la fusion **vote** maximise
  l’accuracy (0,500) mais **chute sur le rappel urgent** (0,125), car le wav2vec2 4-classes
  (ang/hap/neu/sad) **n’a pas de classe Confusion** et tire vers Boredom/Engagement. À l’inverse, la
  fusion **priorité pédagogique** garde le **meilleur rappel urgent** (0,400). Ces deux familles
  ont été comparées sur n=80 ; le prototype retient désormais **une seule** méthode :
  pondération continue par incertitude, avec priorité pédagogique seulement si les deux
  confiances sont nulles.
- **SER audio** : heuristique librosa (baseline) + fine-tuné DAiSEE (4 états, Confusion incluse).
  Voir matrice de confusion dans `results/iemocap_av_metrics.json`.
- Reproductible : `scripts/build_iemocap_av_corpus.py --n-per-state 20 --fer-frames 5` (seed fixé).
  Variables d’environnement : `EMOLEARN_SER_METHODE` (learned/heuristique).

### 9.7 Perspectives ouvertes par ce sous-corpus

1. ~~SER appris~~ **fait** : wav2vec2 vs heuristique mesuré (+0,224).
2. ~~Multi-frames~~ **fait** : vote sur 5 images (+0,037, 100 % visages détectés).
3. ~~Fusion sur l’état~~ **fait** : vote pondéré sur les probabilités d’état.
4. **À faire** : SER appris **multi-classes incluant Confusion/surprise** (le modèle actuel manque
   Confusion) ; **ré-étalonnage du mapping** émotions→DAiSEE et **étude de sensibilité**
   (`scripts/sensibilite_mapping.py`) ; **pondération de fusion adaptative** par confiance.
