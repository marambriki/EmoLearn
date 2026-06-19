# Régénération du corpus — guide

## Contexte

L’audit initial (`scripts/audit_corpus.py`) a signalé **46 entrées** incohérentes sur 220 (~79 % OK).

Une première passe **`--metadata-only`** a corrigé les métadonnées (template, émotion, niveau) via `decision_engine` :

- Résultat : **91,4 % OK** (201/220)
- Backup : `data/corpus_200_backup.csv`

Il reste ~**19 entrées** dont le **texte** ne correspond pas au template (marqueurs manquants, ton inadapté).

## Commandes

### 1. Audit

```bash
python scripts/audit_corpus.py
```

### 2. Aperçu des corrections (sans modification)

```bash
python scripts/regenerate_corpus.py --dry-run
```

### 3. Corriger métadonnées uniquement (sans API LLM)

```bash
python scripts/regenerate_corpus.py --metadata-only
```

### 4. Régénération complète du contenu (nécessite `GROQ_API_KEY` dans `.env`)

```bash
pip install groq python-dotenv
python scripts/regenerate_corpus.py
```

Pour un test limité :

```bash
python scripts/regenerate_corpus.py --limit 5 --delay 2
```

### 5. Ré-auditer

```bash
python scripts/audit_corpus.py
python scripts/metrics_contenu.py
```

## Prérequis LLM

- Fichier `.env` à la racine avec `GROQ_API_KEY=...`
- Paquets : `groq`, `python-dotenv` (voir `requirements.txt`)

## Fichiers produits

| Fichier | Description |
|---------|-------------|
| `data/corpus_200_backup.csv` | Sauvegarde avant modification |
| `results/corpus_audit_report.csv` | Détail des problèmes par ligne |
| `data/corpus_evaluation_60.csv` | Échantillon pour évaluation experte (régénéré à chaque audit) |

## Interprétation pour le mémoire

- Indiquer que le corpus a été **audité automatiquement** puis **corrigé** selon le pipeline unifié (fusion + templates A1–D5).
- Les exemples de `corpus_evaluation_60.csv` servent à l’évaluation experte (chapitre 5).
- Mentionner le taux de cohérence post-correction (~91 %+).
