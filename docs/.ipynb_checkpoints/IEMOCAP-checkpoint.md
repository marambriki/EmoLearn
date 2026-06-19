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
                               • mode CSV (défaut) : annotations
                               • mode audio : SER acoustique (librosa)
        ↓
backend/fusion_module.py   →  fusionner_modalites()  [60 % visage / 40 % voix]
```

### Fichiers clés

- `backend/iemocap_parser.py` — parse les lignes `[t1 - t2] utterance_id emotion [V,A,D]`
- `backend/iemocap_audio.py` — résout les chemins `.wav` et extrait les segments
- `backend/ser_module.py` — charge le CSV parsé, SER audio ou annotations, mappe vers DAiSEE
- `scripts/enrich_iemocap_wav_paths.py` — ajoute `wav_path` / `wav_exists` au CSV
- `data/mapping_iemocap_daisee.csv` — table de correspondance documentée
- `backend/fusion_module.py` — détection et résolution de conflits

### Mode audio (SER sur .wav)

Par défaut, le pipeline utilise les **annotations CSV** (rapide, reproductible).

Pour activer l'analyse du signal vocal :

```json
POST /analyser
{ "scenario_id": 0, "utiliser_audio": true }
```

Ou variable d'environnement : `EMOLearn_SER_AUDIO=1`

Le module extrait le segment `[start_time, end_time]` du fichier dialogue `.wav`
(ex. `Ses01F_impro01.wav`) et estime l'émotion via des **features acoustiques**
(RMS, pitch, centroid spectral). Si le `.wav` est absent, repli automatique sur le CSV.

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

Ce mapping est une **interprétation de constructe** : IEMOCAP mesure des émotions conversationnelles, DAiSEE mesure l’engagement en apprentissage. Il doit être présenté comme hypothèse de travail validée par la cohérence du pipeline, pas comme vérité psychologique universelle.

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

1. Couplage sur séquences filmées maison (visage + voix alignés)
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
