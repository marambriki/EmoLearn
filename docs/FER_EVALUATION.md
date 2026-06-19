# Guide d'évaluation FER (OS2 / QR1)

## Modes disponibles

### 1. Analyse des labels DAiSEE (sans vidéos)

```bash
python scripts/eval_fer_daisee.py labels --split train --sample 5000
```

Produit `results/fer_labels_analysis.json` : distribution des états dominants, co-occurrences, table de mapping FER→DAiSEE.

### 2. Benchmark des 50 scénarios (pipeline complet)

```bash
python scripts/eval_fer_daisee.py scenarios
```

Produit `results/scenarios_decisions.csv` et `results/scenarios_benchmark.json` (répartition des templates A1–D5).

### 3. Évaluation FER sur frames (nécessite dataset extrait)

1. Extraire les frames avec `data/DAiSEE/extractFrames.py` (ffmpeg) depuis le dataset DAiSEE officiel.
2. Lancer :

```bash
python scripts/eval_fer_daisee.py fer --frames-root "CHEMIN/vers/Data" --sample 100
```

Produit `results/fer_metrics.json` (accuracy, F1 macro, matrice de confusion) et `results/fer_confusion.png` si matplotlib est installé.

## Interprétation pour le mémoire

| Mode | Ce que vous pouvez affirmer |
|------|----------------------------|
| **labels** | Statistiques ground truth DAiSEE ; justification des seuils d'adaptation |
| **scenarios** | Couverture du mapping engagement → templates sur 50 cas représentatifs |
| **fer** | Performance du modèle `dima806/facial_emotions_image_detection` mappé vers les 4 états DAiSEE |

**Important :** sans frames locales, ne pas prétendre avoir mesuré l'accuracy FER en conditions réelles — utiliser le mode `fer` dès que les vidéos/frames sont disponibles.

## Résultats actuels (extrait train, n=5000)

- Engagement dominant : ~76 %
- Co-occurrence confusion+frustration ≥2 : 89 cas
- Boredom élevé (≥2) : 1147 cas

Voir `results/fer_labels_analysis.json` pour le détail complet.
