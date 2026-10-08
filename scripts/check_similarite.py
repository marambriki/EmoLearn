# -*- coding: utf-8 -*-
"""
Détecteur de recouvrement textuel entre les chapitres du mémoire et les
sources extraites (_pdf_extract/). Signale les séquences de N mots communes,
qui sont exactement ce que repèrent les logiciels anti-plagiat.

Usage : python scripts/check_similarite.py [N]
"""
import os
import re
import sys
import unicodedata

RACINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOSSIER_TEX = os.path.join(RACINE, "rapport", "chapters")
DOSSIER_SRC = os.path.join(RACINE, "_pdf_extract")
TAILLE_NGRAM = int(sys.argv[1]) if len(sys.argv) > 1 else 8

# Suites de mots trop banales pour constituer un indice de plagiat
STOP_SEQUENCES = ("et al", "dans le cadre de", "il est important de")


def normaliser(texte):
    texte = unicodedata.normalize("NFKD", texte)
    texte = "".join(c for c in texte if not unicodedata.combining(c))
    texte = texte.lower()
    texte = re.sub(r"[^a-z0-9 ]+", " ", texte)
    return re.sub(r"\s+", " ", texte).strip()


def nettoyer_latex(source):
    source = re.sub(r"%.*", "", source)
    source = re.sub(r"\\begin\{tikzpicture\}.*?\\end\{tikzpicture\}", " ", source, flags=re.S)
    source = re.sub(r"\\(cite|ref|label|includegraphics)\{[^}]*\}", " ", source)
    source = re.sub(r"\\[a-zA-Z@]+\*?", " ", source)
    source = re.sub(r"[{}$&~^_\\]", " ", source)
    return source


def ngrams(mots, n):
    return {" ".join(mots[i:i + n]): i for i in range(len(mots) - n + 1)}


def lire(chemin):
    with open(chemin, encoding="utf-8", errors="ignore") as f:
        return f.read()


def fusionner(positions, n):
    """Regroupe les n-grammes consécutifs en un seul passage."""
    positions = sorted(positions)
    blocs, debut, fin = [], positions[0], positions[0]
    for p in positions[1:]:
        if p <= fin + 1:
            fin = p
        else:
            blocs.append((debut, fin + n))
            debut = fin = p
    blocs.append((debut, fin + n))
    return blocs


def main():
    sources = {}
    for nom in sorted(os.listdir(DOSSIER_SRC)):
        if nom.endswith(".txt"):
            sources[nom] = set(ngrams(normaliser(lire(os.path.join(DOSSIER_SRC, nom))).split(), TAILLE_NGRAM))

    print(f"Sources analysees : {len(sources)}")
    print(f"Longueur de sequence detectee : {TAILLE_NGRAM} mots\n")

    total = 0
    for nom_tex in sorted(os.listdir(DOSSIER_TEX)):
        if not nom_tex.endswith(".tex"):
            continue
        mots = normaliser(nettoyer_latex(lire(os.path.join(DOSSIER_TEX, nom_tex)))).split()
        grammes = ngrams(mots, TAILLE_NGRAM)

        trouves = {}
        for gramme, pos in grammes.items():
            if any(s in gramme for s in STOP_SEQUENCES):
                continue
            for nom_src, ensemble in sources.items():
                if gramme in ensemble:
                    trouves.setdefault(nom_src, []).append(pos)

        if not trouves:
            print(f"[OK]      {nom_tex} — aucun recouvrement")
            continue

        print(f"[ALERTE]  {nom_tex}")
        for nom_src, positions in trouves.items():
            for debut, fin in fusionner(positions, TAILLE_NGRAM):
                extrait = " ".join(mots[debut:fin])
                total += 1
                print(f"    source : {nom_src}")
                print(f"    {fin - debut} mots : « {extrait} »\n")

    print(f"\nTotal passages a reformuler : {total}")


if __name__ == "__main__":
    main()
