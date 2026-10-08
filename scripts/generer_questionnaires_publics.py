#!/usr/bin/env python3
"""Génère des pages HTML autonomes (liens partageables, hors prototype)."""

from __future__ import annotations

import html
import os

import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT = os.path.join(ROOT, "questionnaires")
EMAIL = os.environ.get("EMOLEARN_FORM_EMAIL", "brikimaram308@gmail.com")
ACTION = f"https://formsubmit.co/{EMAIL}"
# URL publique du dossier questionnaires, sans slash final.
# Ex. https://marambriki.github.io/EmoLearn/questionnaires
FORM_BASE = os.environ.get("EMOLEARN_FORM_BASE", "").rstrip("/")

CSS = """
:root { --bg:#f4f1ea; --card:#fff; --ink:#1c1917; --muted:#57534e; --accent:#0f766e; --line:#e7e5e4; }
* { box-sizing:border-box; }
body { margin:0; font-family:Georgia,serif; background:var(--bg); color:var(--ink); line-height:1.45; }
main { max-width:720px; margin:0 auto; padding:24px 16px 72px; }
a { color:var(--accent); }
h1 { font-size:1.5rem; margin:0 0 8px; }
.muted { color:var(--muted); font-size:.95rem; }
.card { background:var(--card); border:1px solid var(--line); border-radius:10px; padding:16px; margin:16px 0; }
label, .lab { display:block; margin:10px 0 4px; font-size:.92rem; }
input[type=text], input[type=number], select, textarea {
  width:100%; padding:8px; font:inherit; border:1px solid var(--line); border-radius:6px;
}
textarea { min-height:70px; }
.likert { display:flex; flex-wrap:wrap; gap:10px 14px; }
.likert label { margin:0; display:flex; gap:4px; align-items:center; }
button { background:var(--accent); color:#fff; border:0; padding:12px 18px; border-radius:8px; font:inherit; cursor:pointer; }
.extrait { white-space:pre-wrap; font-size:.88rem; background:#fafaf9; padding:10px; border-radius:6px; max-height:420px; overflow:auto; }
nav a { margin-right:12px; }
.ancres { display:flex; justify-content:space-between; font-size:.8rem; color:var(--muted); margin-top:2px; }
.check { display:flex; gap:8px; align-items:flex-start; }
.check input { width:auto; margin-top:4px; }
.pied { font-size:.85rem; color:var(--muted); margin-top:28px; }
.sommaire { font-size:.88rem; line-height:1.8; }
.sommaire a { margin-right:10px; }
"""


ETAT_FR = {
    "Boredom": "Ennui",
    "Confusion": "Confusion",
    "Engagement": "Engagement",
    "Frustration": "Frustration",
}
PATTERN_FR = {
    "stable": "parcours stable",
    "ennui_eleve": "ennui élevé",
    "frustration": "frustration",
    "confusion": "confusion",
    "engagement_faible": "engagement faible",
    "critique": "situation critique (plusieurs états élevés)",
}
PATRON_FR = {
    "A1": "A1 — autre explication (confusion)",
    "A2": "A2 — micro-explication",
    "A3": "A3 — déblocage (confusion + frustration)",
    "A4": "A4 — explication de fond",
    "A5": "A5 — explication narrative",
    "B1": "B1 — exemple concret",
    "B2": "B2 — cas pratique",
    "B3": "B3 — mise en pratique",
    "B4": "B4 — application réelle",
    "B5": "B5 — exemple interdisciplinaire",
    "C1": "C1 — exercice (maîtrise / ennui)",
    "C2": "C2 — exercice de consolidation",
    "C3": "C3 — exercice créatif",
    "C4": "C4 — mini-projet",
    "C5": "C5 — exercice comparatif",
    "D1": "D1 — encouragement (frustration)",
    "D2": "D2 — soutien (frustration élevée)",
    "D3": "D3 — remotivation",
    "D4": "D4 — renforcement (bon engagement)",
    "D5": "D5 — félicitation",
}

PRESENTATION = f"""
<div class="card">
<h2>De quoi s’agit-il ?</h2>
<p><b>EmoLearn</b> est un <b>prototype de recherche</b> (mémoire de master), pas une appli déjà utilisée dans un vrai MOOC.</p>
<p>Dans un cours en ligne, l’enseignant ne voit pas si l’étudiant s’ennuie, est perdu ou frustré. EmoLearn simule cette boucle :</p>
<ol>
  <li><b>Perception</b> — à partir d’une courte vidéo de visage et d’un extrait de voix, le système estime un état d’engagement :
    <b>Engagement</b> (impliqué), <b>Ennui</b>, <b>Confusion</b> ou <b>Frustration</b>.</li>
  <li><b>Décision</b> — selon cet état, il choisit une stratégie pédagogique :
    <b>A</b> = réexpliquer, <b>B</b> = donner un exemple, <b>C</b> = proposer un exercice, <b>D</b> = encourager.</li>
  <li><b>Génération</b> — un modèle de langue rédige alors un petit texte de cours (ici : les algorithmes de tri), adapté à cet état.</li>
</ol>
<p>Réponses utilisées uniquement pour le mémoire, sans nom public. Conservation : durée du mémoire, puis suppression.</p>
</div>
"""

GLOSSAIRE = """
<div class="card">
<h2>Petit lexique</h2>
<ul>
  <li><b>Type A / B / C / D</b> : famille d’intervention (expliquer / exemple / exercice / encourager).</li>
  <li><b>Patron</b> (A1, D1…) : sous-type plus précis dans cette famille.</li>
  <li><b>État annoté</b> : étiquette du corpus vidéo (référence).</li>
  <li><b>État détecté</b> : ce que le prototype a estimé (visage + voix).</li>
</ul>
</div>
"""

CONSENT = """
<div class="card">
<label class="check"><input type="checkbox" name="consentement" value="oui" required/>
<span>J’ai lu la présentation. J’accepte que mes réponses (sans nom) servent à un mémoire de master. Cette page ne filme pas et n’enregistre ni webcam ni micro.</span></label>
</div>
"""


def likert(
    name: str,
    hi: int = 5,
    gauche: str = "Pas d’accord",
    droite: str = "Tout à fait d’accord",
) -> str:
    radios = "".join(
        f'<label><input type="radio" name="{html.escape(name)}" value="{i}" required/> {i}</label>'
        for i in range(1, hi + 1)
    )
    return (
        f'<div class="likert">{radios}</div>'
        f'<div class="ancres"><span>{html.escape(gauche)}</span><span>{html.escape(droite)}</span></div>'
    )


def strip_md(text: str) -> str:
    t = str(text or "").replace("**", "").replace("*", "")
    t = t.replace("tandis qu'les", "tandis que les")
    t = t.replace("tandis qu’les", "tandis que les")
    return t.strip()


def page(title: str, body: str) -> str:
    return f"""<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>{html.escape(title)}</title>
<meta name="referrer" content="origin"/>
<style>{CSS}</style>
</head>
<body><main>
<nav><a href="index.html">Accueil des questionnaires</a></nav>
{body}
</main></body></html>
"""


def hidden_formsubmit(subject: str) -> str:
    nxt = ""
    if FORM_BASE.startswith("https://"):
        nxt = (
            f'<input type="hidden" name="_next" '
            f'value="{html.escape(FORM_BASE + "/merci.html")}"/>\n'
        )
    return f"""
<input type="hidden" name="_subject" value="{html.escape(subject)}"/>
<input type="hidden" name="_template" value="table"/>
{nxt}<input type="text" name="_honey" style="display:none" tabindex="-1" autocomplete="off"/>
"""


def write(name: str, content: str) -> None:
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, name)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(content)
    print("Wrote", path)


def index() -> None:
    body = f"""
<h1>Questionnaires EmoLearn</h1>
<p class="muted">Enquête pour un mémoire de master.</p>
{PRESENTATION}
{GLOSSAIRE}
<div class="card">
  <h2>Étudiants</h2>
  <p><a href="etudiants.html">Ouvrir le questionnaire étudiants</a></p>
</div>
<div class="card">
  <h2>Enseignants</h2>
  <p><a href="enseignants.html">Ouvrir le questionnaire enseignants</a></p>
</div>
<div class="card">
  <h2>Experts pédagogie</h2>
  <p><a href="experts.html">Ouvrir le questionnaire experts</a></p>
</div>
"""
    write("index.html", page("Questionnaires EmoLearn", body))


def merci() -> None:
    write(
        "merci.html",
        page(
            "Merci",
            "<h1>Merci</h1><p>Votre réponse a bien été envoyée. Elle servira uniquement au mémoire. Vous pouvez fermer cette page.</p>"
            '<p><a href="index.html">Retour</a></p>',
        ),
    )


def etudiants() -> None:
    sus = [
        "Je voudrais utiliser ce type de système fréquemment.",
        "Je le trouverais inutilement complexe.",
        "Je le trouverais facile à utiliser.",
        "J’aurais besoin d’un support technique pour l’utiliser.",
        "Les fonctions me sembleraient bien intégrées.",
        "Il y aurait trop d’incohérences.",
        "La plupart des gens l’apprendraient très rapidement.",
        "Je le trouverais lourd à utiliser.",
        "Je me sentirais en confiance en l’utilisant.",
        "Il me faudrait apprendre beaucoup de choses avant de m’en servir.",
    ]
    sus_html = "".join(
        f'<div class="lab">SUS {i} — {html.escape(t)} (1–5)</div>{likert(f"sus_{i}")}'
        for i, t in enumerate(sus, 1)
    )
    body = f"""
<h1>Questionnaire étudiants — EmoLearn</h1>
{PRESENTATION}
<form action="{ACTION}" method="POST" referrerpolicy="origin">
{hidden_formsubmit("EmoLearn — questionnaire étudiants")}
{CONSENT}
<div class="card">
<p>Vous <b>n’utilisez pas</b> le prototype. Répondez seulement d’après la description, comme si cette idée existait dans un cours en ligne.</p>
<label>Niveau d’études</label>
<input type="text" name="niveau" placeholder="L2 / L3 informatique" required/>
<div class="lab">Familiarité avec les cours en ligne / MOOC</div>
{likert("familiarite_mooc", gauche="Pas du tout", droite="Beaucoup")}
<div class="lab">D’après la description (sans avoir testé le logiciel) : un contenu qui s’adapte à l’engagement me semblerait utile</div>
{likert("utilite", 7)}
<div class="lab">D’après la description : un tel contenu m’aiderait à rester dans le cours</div>
{likert("adequation", 7)}
<div class="lab">D’après la description : je serais satisfait(e) de recevoir ce type de texte pendant un module</div>
{likert("satisfaction", 7)}
<div class="lab">D’après la description : j’accepterais qu’une webcam / un micro estime mon engagement pendant un cours (données non stockées à des fins commerciales)</div>
{likert("acceptation_captation", 7)}
</div>
<div class="card"><h2>Facilité d’usage imaginée (SUS)</h2>
<p>Ces 10 questions ne portent pas sur un essai réel. Imaginez seulement le système décrit plus haut.</p>
{sus_html}
</div>
<div class="card">
<label>Qu’est-ce qui aiderait, selon vous (idée, pas essai du logiciel) ?</label><textarea name="aide"></textarea>
<label>Qu’est-ce qui gênerait, selon vous ?</label><textarea name="gene"></textarea>
<label>Une suggestion</label><textarea name="suggestion"></textarea>
</div>
<button type="submit">Envoyer</button>
</form>
"""
    write("etudiants.html", page("Questionnaire étudiants", body))


def enseignants() -> None:
    src = os.path.join(ROOT, "data", "corpus_evaluation_60.csv")
    df = pd.read_csv(src)
    sample = (
        df.groupby(df["template_id"].astype(str).str[0], group_keys=False)
        .head(2)
        .head(8)
    )
    cards = []
    for i, (_, d) in enumerate(sample.iterrows(), start=1):
        extra = html.escape(strip_md(str(d.get("contenu", ""))))
        tid = str(d.get("template_id", ""))
        etat_raw = str(d.get("emotion", "")).strip().title()
        etat = ETAT_FR.get(etat_raw, etat_raw)
        patron = PATRON_FR.get(tid, tid)
        cards.append(
            f"""<div class="card" id="texte-{i}">
<p><b>Texte {i}/8</b> — {html.escape(patron)} — état visé : <b>{html.escape(etat)}</b></p>
<div class="extrait">{extra}</div>
<div class="lab">Q1 D’après ce texte (sans avoir utilisé le logiciel) : pertinence par rapport aux algorithmes de tri</div>
{likert(f"t{i}_pertinence")}
<div class="lab">Q2 D’après ce texte : clarté et structure</div>
{likert(f"t{i}_clarte")}
<div class="lab">Q3 D’après ce texte : adéquation au ton / à l’état annoncé ({html.escape(etat)})</div>
{likert(f"t{i}_adequation")}
<div class="lab">Q4 D’après ce texte : qualité pédagogique globale</div>
{likert(f"t{i}_qualite")}
<div class="lab">Q5 D’après ce texte : utilisable tel quel dans un cours en ligne</div>
{likert(f"t{i}_utilisabilite")}
<div class="lab">Q6 D’après ce texte : pas d’erreur factuelle / chiffre inventé</div>
{likert(f"t{i}_factualite")}
<label>Commentaire texte {i}</label>
<textarea name="t{i}_commentaire"></textarea>
</div>"""
        )
    body = f"""
<h1>Questionnaire enseignants — EmoLearn</h1>
{PRESENTATION}
{GLOSSAIRE}
<form action="{ACTION}" method="POST" referrerpolicy="origin">
{hidden_formsubmit("EmoLearn — questionnaire enseignants")}
{CONSENT}
<div class="card">
<p>Vous <b>n’utilisez pas</b> le prototype. Vous jugez uniquement les extraits affichés ci-dessous (textes générés), d’après la description du projet.</p>
<label>Votre profil (ex. enseignant informatique)</label>
<input type="text" name="profil" required/>
<p class="muted sommaire">Aller au texte :
{" ".join(f'<a href="#texte-{i}">{i}</a>' for i in range(1, 9))}
</p>
</div>
{''.join(cards)}
<button type="submit">Envoyer</button>
</form>
"""
    write("enseignants.html", page("Questionnaire enseignants", body))


def experts() -> None:
    src = os.path.join(ROOT, "results", "forms", "l1_mapping_reponses.csv")
    df = pd.read_csv(src).drop_duplicates("fiche_id")
    n = len(df)
    cards = []
    for _, r in df.iterrows():
        fid = int(r["fiche_id"])
        etat_gt = ETAT_FR.get(str(r.get("etat_gt", "")), str(r.get("etat_gt", "")))
        etat_fus = ETAT_FR.get(str(r.get("etat_fusion", "")), str(r.get("etat_fusion", "")))
        pat = PATTERN_FR.get(str(r.get("pattern", "")), str(r.get("pattern", "")))
        t_gt = PATRON_FR.get(str(r.get("template_attendu_gt", "")), str(r.get("template_attendu_gt", "")))
        t_obs = PATRON_FR.get(str(r.get("template_observe", "")), str(r.get("template_observe", "")))
        meta = (
            f"Cas {fid}/{n} — {html.escape(pat)}<br/>"
            f"Référence corpus : <b>{html.escape(etat_gt)}</b> "
            f"→ {html.escape(t_gt)}<br/>"
            f"<b>À juger</b> — état + patron proposés (à noter, sans essai du logiciel) : <b>{html.escape(etat_fus)}</b> "
            f"→ {html.escape(t_obs)}"
        )
        cards.append(
            f"""<div class="card" id="cas-{fid}"><p>{meta}</p>
<div class="lab">C1 D’après ce cas (sans avoir utilisé le logiciel) : l’état détecté justifierait une intervention</div>{likert(f"c{fid}_C1")}
<div class="lab">C2 D’après ce cas : le type A/B/C/D de la ligne « À juger » serait adapté</div>{likert(f"c{fid}_C2")}
<div class="lab">C3 D’après ce cas : le patron (A1, D1…) de la ligne « À juger » serait le bon sous-type</div>{likert(f"c{fid}_C3")}
<div class="lab">C4 D’après ce cas : un déclenchement à ce moment serait opportun</div>{likert(f"c{fid}_C4")}
<label>Commentaire cas {fid}</label>
<textarea name="c{fid}_commentaire"></textarea>
</div>"""
        )
    body = f"""
<h1>Questionnaire experts — mapping pédagogique</h1>
{PRESENTATION}
{GLOSSAIRE}
<form action="{ACTION}" method="POST" referrerpolicy="origin">
{hidden_formsubmit("EmoLearn — questionnaire experts")}
{CONSENT}
<div class="card">
<p>Vous <b>n’utilisez pas</b> le prototype. Vous jugez uniquement les cas écrits ci-dessous (mapping état → stratégie), pas une session réelle dans le logiciel.</p>
<p>Pour chaque cas, deux lignes : <b>Référence corpus</b> (étiquette annotée) et <b>À juger</b> (état + patron proposés sur le papier). Vos notes C1–C4 portent sur la ligne <b>À juger</b>. La référence sert seulement de contexte.</p>
</div>
<div class="card">
<label>Votre fonction (pédagogie, e-learning…)</label>
<input type="text" name="expert" required/>
<p class="muted sommaire">Aller au cas :
{" ".join(f'<a href="#cas-{int(r["fiche_id"])}">{int(r["fiche_id"])}</a>' for _, r in df.iterrows())}
</p>
</div>
{''.join(cards)}
<button type="submit">Envoyer</button>
</form>
"""
    write("experts.html", page("Questionnaire experts", body))


def main():
    index()
    merci()
    etudiants()
    enseignants()
    experts()


if __name__ == "__main__":
    main()
