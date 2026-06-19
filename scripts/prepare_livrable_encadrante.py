"""Prépare OS1, OS1IEMOCAP, OS2, OS3 pour envoi à l'encadrante."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "notebooks"
DEST = ROOT / "livrable_encadrante"


def md_cell(text: str) -> dict:
    lines = [line + "\n" for line in text.strip().splitlines()]
    if lines:
        lines[-1] = lines[-1].rstrip("\n")
    return {"cell_type": "markdown", "metadata": {}, "source": lines}


def code_cell(text: str) -> dict:
    lines = [line + "\n" for line in text.strip().splitlines()]
    if lines:
        lines[-1] = lines[-1].rstrip("\n")
    return {"cell_type": "code", "metadata": {}, "outputs": [], "execution_count": None, "source": lines}


def save_nb(path: Path, nb: dict) -> None:
    for cell in nb["cells"]:
        if cell.get("cell_type") == "code":
            cell["outputs"] = []
            cell["execution_count"] = None
    while nb["cells"] and not "".join(nb["cells"][-1].get("source", [])).strip():
        nb["cells"].pop()
    with open(path, "w", encoding="utf-8") as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)
        f.write("\n")


def replace_in_nb(nb: dict, old: str, new: str) -> None:
    for cell in nb["cells"]:
        src = "".join(cell.get("source", []))
        if old in src:
            src = src.replace(old, new)
            cell["source"] = [line + "\n" for line in src.splitlines()]
            if cell["source"]:
                cell["source"][-1] = cell["source"][-1].rstrip("\n")


INTRO_OS1 = """# OS1 — Analyse exploratoire DAiSEE

**Objectif :** analyser les 4 états émotionnels (Boredom, Engagement, Confusion, Frustration), identifier les patterns critiques et produire 50 scénarios représentatifs.

**Prérequis :** dataset DAiSEE dans `data/DAiSEE/`

**Fichier produit :** `data/scenarios_50.csv`"""

INTRO_OS1IEMOCAP = """# OS1 — Analyse multimodale IEMOCAP

**Objectif :** parser le dataset IEMOCAP (voix + émotions), tester SER (Speech Emotion Recognition) et fusionner visage simulé (DAiSEE) + voix (IEMOCAP).

**Prérequis :**
- dataset IEMOCAP dans `data/IEMOCAP_full_release/`
- `data/scenarios_50.csv` (produit par OS1 DAiSEE)
- modules `backend/` : `iemocap_parser`, `ser_module`, `fer_module`, `fusion_module`

**Fichier produit :** `data/iemocap_parsed.csv`

**Ordre :** exécuter **OS1.ipynb** (DAiSEE) avant ce notebook."""

INTRO_OS2 = """# OS2 — Reconnaissance faciale et stratégies d'adaptation

**Objectif :** détecter l'émotion faciale (FER), la mapper vers DAiSEE, puis appliquer les règles de stratégie pédagogique (types A, B, C, D).

**Prérequis :**
- `data/scenarios_50.csv` (OS1)
- `data/sample_face.jpg` — photo de visage pour la démo
- `backend/adaptation.py`

**Fichier produit :** `data/scenarios_avec_strategies.csv`"""

INTRO_OS3 = """# OS3 — Génération de contenu adaptatif (LLM)

**Objectif :** générer du contenu pédagogique personnalisé selon l'état émotionnel (Groq / Llama 3.1).

**Prérequis :**
- fichier `.env` à la racine du projet : `GROQ_API_KEY=votre_cle`
- `data/scenarios_avec_strategies.csv` (OS2)

**Ordre d'exécution :** Connexion Groq → 20 templates → tests → corpus (optionnel)"""

IMAGE_BOOTSTRAP = """
from pathlib import Path

def trouver_image_test():
    candidats = (
        "sample_face.jpg",
        "sample_face.jfif",
        "sample_face.png",
        "images.jfif",
    )
    data_dir = (Path("..") / "data").resolve()
    for name in candidats:
        p = data_dir / name
        if p.is_file():
            return p
    raise FileNotFoundError(
        f"Aucune image trouvée dans {data_dir}. "
        "Placez une photo de visage (ex. sample_face.jpg) dans data/."
    )

IMAGE_TEST = trouver_image_test()
print(f"Image de test : {IMAGE_TEST}")
"""

README = """# EmoLearn — Livrable OS1 / OS2 / OS3

Bonjour,

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

## Exécution

1. Ouvrir Jupyter / VS Code dans le dossier `livrable_encadrante/`
2. Exécuter **OS1.ipynb** (DAiSEE) en premier
3. Exécuter **OS1IEMOCAP.ipynb** (IEMOCAP + fusion)
4. Placer une photo de visage dans `data/sample_face.jpg` (ou `.jfif`)
5. Exécuter **OS2.ipynb**, puis **OS3.ipynb**

### OS3 — clé API Groq

Créer un fichier `.env` à la racine `EmoLearn/` :

```
GROQ_API_KEY=votre_cle_ici
```

La cellule « Génération corpus 200+ » (OS3) peut prendre **plusieurs minutes** ; les tests des 4 types suffisent pour valider le fonctionnement.

## Prochaines étapes (non incluses ici)

- Intégration RAG (base de connaissances)
- Pipeline unifié et API
- Interface utilisateur

---

*Projet EmoLearn — mémoire / stage*
"""


def main() -> None:
    DEST.mkdir(exist_ok=True)

    with open(SRC / "OS1Daisee.ipynb", encoding="utf-8") as f:
        os1 = json.load(f)
    os1["cells"].insert(0, md_cell(INTRO_OS1))
    save_nb(DEST / "OS1.ipynb", os1)

    with open(SRC / "OS1IEMOCAP.ipynb", encoding="utf-8") as f:
        os1i = json.load(f)
    os1i["cells"].insert(0, md_cell(INTRO_OS1IEMOCAP))
    replace_in_nb(os1i, "fusionner_modalites, expliquer_fusion", "fusionner_modalites")
    replace_in_nb(
        os1i,
        "result_voix = analyser_voix_depuis_scenario(scenario_id=i)",
        "result_voix = analyser_voix_depuis_scenario(scenario_id=i, etat_dominant=result_visage['etat'])",
    )
    replace_in_nb(
        os1i,
        "result_voix = analyser_voix_depuis_scenario(scenario_id)",
        "result_voix = analyser_voix_depuis_scenario(scenario_id, etat_dominant=result_visage['etat'])",
    )
    replace_in_nb(
        os1i,
        "res_voix   = analyser_voix_depuis_scenario(i)",
        "res_voix   = analyser_voix_depuis_scenario(i, etat_dominant=res_visage['etat'])",
    )
    save_nb(DEST / "OS1IEMOCAP.ipynb", os1i)

    with open(SRC / "OS2fini.ipynb", encoding="utf-8") as f:
        os2 = json.load(f)
    os2["cells"].insert(0, md_cell(INTRO_OS2))
    os2["cells"].insert(1, code_cell(IMAGE_BOOTSTRAP))
    replace_in_nb(os2, r"r'C:\Users\Lenovo\Downloads\images.jfif'", "IMAGE_TEST")
    replace_in_nb(os2, "Image.open(r'IMAGE_TEST')", "Image.open(IMAGE_TEST)")
    replace_in_nb(os2, "analyser_image_complete(r'IMAGE_TEST')", "analyser_image_complete(IMAGE_TEST)")
    replace_in_nb(os2, "4 niveaux (immédiat, rapide, différé, pas)", "6 types de stratégie (A, B, C, D, A+D, D+A)")
    save_nb(DEST / "OS2.ipynb", os2)

    with open(SRC / "OS3fini.ipynb", encoding="utf-8") as f:
        os3 = json.load(f)
    os3["cells"].insert(0, md_cell(INTRO_OS3))
    save_nb(DEST / "OS3.ipynb", os3)

    (DEST / "README.md").write_text(README, encoding="utf-8")
    print(f"Livrable prêt : {DEST}")


if __name__ == "__main__":
    main()
