# EmoLearn — Livrable OS1 / OS2 / OS3


Ce dossier contient les notebooks des **objectifs spécifiques OS1, OS2 et OS3** du projet EmoLearn (adaptation pédagogique basée sur l'engagement émotionnel).

## Contenu

| Notebook | Description | Sortie principale |
|----------|-------------|-------------------|
| **OS1.ipynb** | Analyse exploratoire DAiSEE (visage) | `data/scenarios_50.csv` |
| **OS1IEMOCAP.ipynb** | Analyse IEMOCAP + SER + fusion multimodale | `data/iemocap_parsed.csv` |
| **OS2.ipynb** | FER (émotions faciales) + règles d'adaptation | `data/scenarios_avec_strategies.csv` |
| **OS3.ipynb** | Génération LLM de contenus adaptatifs | `data/corpus_200.csv` (optionnel) |

## Structure attendue du projet

```
EmoLearn/
├── livrable_encadrante/
│   ├── OS1.ipynb
│   ├── OS1IEMOCAP.ipynb
│   ├── OS2.ipynb
│   ├── OS3.ipynb
│   └── README.md
├── data/
│   ├── DAiSEE/              ← requis pour OS1
│   ├── IEMOCAP_full_release/ ← requis pour OS1IEMOCAP
│   ├── scenarios_50.csv
│   └── sample_face.jpg      ← requis pour OS2 (photo de visage)
├── backend/
│   ├── adaptation.py        ← requis pour OS2
│   ├── iemocap_parser.py    ← requis pour OS1IEMOCAP
│   ├── ser_module.py
│   ├── fer_module.py
│   └── fusion_module.py
└── .env                     ← requis pour OS3 (GROQ_API_KEY)
```

## Installation

```bash
pip install pandas matplotlib seaborn numpy pillow transformers torch groq python-dotenv librosa soundfile
```


### OS3 — clé API Groq

Créer un fichier `.env` à la racine `EmoLearn/` :

```
GROQ_API_KEY=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
```


