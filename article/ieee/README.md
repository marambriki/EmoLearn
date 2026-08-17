# Article IEEE — EmoLearn

IEEE **conference** paper (`IEEEtran`), in **English**, targeting **6--10 pages**.

## Fichiers

| Fichier | Rôle |
|---------|------|
| `emolearn_ieee.tex` | Article principal |
| `references.bib` | Bibliographie (style `IEEEtran`) |
| `README.md` | Ce guide |

## Personnaliser avant soumission

Dans `emolearn_ieee.tex`, remplacer :

- `Prénom Nom` / `email@example.com`
- `Université / Établissement`, ville, pays
- les infos de l'encadrante

## Compilation locale (MiKTeX / TeX Live)

```bash
cd article/ieee
pdflatex emolearn_ieee
bibtex emolearn_ieee
pdflatex emolearn_ieee
pdflatex emolearn_ieee
```

Ou dans TeXstudio : ouvrir `emolearn_ieee.tex`, bibliographie = **BibTeX** (pas Biber), puis Build & View.

À la première compilation, accepter l'installation du paquet `IEEEtran` si proposé.

## Overleaf

1. Nouveau projet → Upload du dossier `article/ieee/`
2. Compiler avec **pdfLaTeX**
3. Menu → Main document : `emolearn_ieee.tex`

## Variantes IEEE

- **Conférence** (actuel) : `\documentclass[conference]{IEEEtran}`
- **Journal** : `\documentclass[journal]{IEEEtran}`
- **Anglais** : remplacer `[french]{babel}` par `[english]{babel}` et traduire le corps

## Template officiel

Le modèle IEEE est aussi disponible sur Overleaf :
https://www.overleaf.com/latex/templates/ieee-conference-template/grfzhhncsfqn
