# Mémoire EmoLearn — Compilation LaTeX

## Prérequis
1. **MiKTeX** (distribution LaTeX) — https://miktex.org
2. **TeXstudio** (éditeur) — https://www.texstudio.org

À la première compilation, MiKTeX proposera d'installer automatiquement les
paquets manquants : accepter (« Install »).

## Comment compiler (dans TeXstudio)
1. Ouvrir `rapport/main.tex`.
2. Régler le backend bibliographie sur **Biber** :
   `Options > Configurer TeXstudio > Compilation > Bibliographie : biber`.
3. Construire dans l'ordre (bouton « Compiler », touche F5) :
   - **PDFLaTeX** → **Biber** → **PDFLaTeX** → **PDFLaTeX**
   - (Le bouton « Build & View » / F5 enchaîne généralement tout seul.)

## Arborescence
```
rapport/
├── main.tex                     ← fichier principal (préambule + includes)
├── references.bib               ← bibliographie (BibTeX)
├── pages/
│   ├── garde.tex                ← page de garde (À PERSONNALISER)
│   ├── remerciements.tex        ← À PERSONNALISER
│   └── resume.tex               ← résumé FR + abstract EN
└── chapters/
    ├── chapitre1_introduction.tex   ← RÉDIGÉ (Introduction générale)
    ├── chapitre2_etat_art.tex       ← squelette
    ├── chapitre3_conception.tex     ← squelette
    ├── chapitre4_implementation.tex ← squelette
    ├── chapitre5_validation.tex     ← squelette
    ├── chapitre6_conclusion.tex     ← squelette
    └── annexes.tex                  ← squelette
```

## À personnaliser en priorité
- `pages/garde.tex` : université, intitulé exact du diplôme, votre nom,
  nom de l'encadrante, membres du jury, année.
- `pages/remerciements.tex`.

## Notes
- Les citations utilisent biblatex/biber. Pour citer : `\cite{cle}`.
- Le Chapitre 1 cite déjà ~17 références présentes dans `references.bib`.
