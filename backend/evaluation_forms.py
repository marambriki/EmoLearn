# ============================================
# Formulaires web L1 / L2 / L4 → CSV results/forms/
# ============================================

from __future__ import annotations

import html
import os
from datetime import datetime, timezone

import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FORMS_DIR = os.environ.get(
    "EMOLEARN_FORMS_DIR", os.path.join(ROOT, "results", "forms")
)

L1_FILE = "l1_mapping_reponses.csv"
L2_FILE = "l2_contenu_reponses.csv"
L4_FILE = "l4_pilote_reponses.csv"

L1_SCORES = ["C1_coherence", "C2_pertinence", "C3_template", "C4_urgence"]
L2_SCORES = [
    "Q1_pertinence",
    "Q2_clarte",
    "Q3_adequation",
    "Q4_qualite",
    "Q5_utilisabilite",
    "Q6_factualite",
]
L1_LABELS = {
    "C1_coherence": "C1 — L'état justifie-t-il une intervention ?",
    "C2_pertinence": "C2 — Le type A/B/C/D est-il adapté ?",
    "C3_template": "C3 — Le patron (A1, D1…) est-il le bon sous-type ?",
    "C4_urgence": "C4 — Le déclenchement est-il opportun ?",
}
L2_LABELS = {
    "Q1_pertinence": "Q1 — Pertinence par rapport au concept",
    "Q2_clarte": "Q2 — Clarté et structure",
    "Q3_adequation": "Q3 — Adéquation à l'état d'engagement",
    "Q4_qualite": "Q4 — Qualité pédagogique globale",
    "Q5_utilisabilite": "Q5 — Utilisable dans un MOOC",
    "Q6_factualite": "Q6 — Pas d'erreur factuelle / chiffre inventé",
}


def forms_dir() -> str:
    return os.environ.get("EMOLEARN_FORMS_DIR", FORMS_DIR)


def _path(name: str) -> str:
    return os.path.join(forms_dir(), name)


def _read(name: str) -> pd.DataFrame:
    path = _path(name)
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def _write(name: str, df: pd.DataFrame) -> None:
    os.makedirs(forms_dir(), exist_ok=True)
    df.to_csv(_path(name), index=False, encoding="utf-8-sig")


def _clamp(value, lo: int, hi: int) -> int:
    n = int(value)
    if n < lo or n > hi:
        raise ValueError(f"Note {n} hors échelle {lo}–{hi}")
    return n


def l1_cas() -> list[dict]:
    df = _read(L1_FILE)
    df = df.drop_duplicates(subset=["fiche_id"], keep="first")
    return df.fillna("").to_dict(orient="records")


def l2_fiches() -> list[dict]:
    df = _read(L2_FILE)
    df = df.drop_duplicates(subset=["fiche_id"], keep="first")
    rows = df.fillna("").to_dict(orient="records")
    for r in rows:
        r["contenu_extrait"] = str(r.get("contenu_extrait", ""))[:800]
    return rows


def enregistrer_l1(expert_id: str, reponses: list[dict]) -> dict:
    expert_id = (expert_id or "").strip()
    if not expert_id:
        raise ValueError("Indiquez votre identifiant (ex. E1).")
    df = _read(L1_FILE)
    if "expert_id" not in df.columns:
        df["expert_id"] = ""
    mask_keep = df["expert_id"].fillna("").astype(str).str.strip() != expert_id
    df = df.loc[mask_keep]
    cas = {int(r["fiche_id"]): r for r in l1_cas()}
    nouvelles = []
    for item in reponses:
        fid = int(item["fiche_id"])
        base = dict(cas[fid])
        for col in L1_SCORES:
            base[col] = _clamp(item[col], 1, 5)
        base["commentaire_expert"] = str(item.get("commentaire_expert") or "")
        base["expert_id"] = expert_id
        nouvelles.append(base)
    out = pd.concat([df, pd.DataFrame(nouvelles)], ignore_index=True)
    _write(L1_FILE, out)
    return {
        "ok": True,
        "niveau": "L1",
        "expert_id": expert_id,
        "n": len(nouvelles),
        "fichier": _path(L1_FILE),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def enregistrer_l2(enseignant_id: str, reponses: list[dict]) -> dict:
    enseignant_id = (enseignant_id or "").strip()
    if not enseignant_id:
        raise ValueError("Indiquez votre identifiant (ex. ENS1).")
    df = _read(L2_FILE)
    if "enseignant_id" not in df.columns:
        df["enseignant_id"] = ""
    ids_new = {int(item["fiche_id"]) for item in reponses}
    eid = df["enseignant_id"].fillna("").astype(str).str.strip()
    fid = pd.to_numeric(df["fiche_id"], errors="coerce")
    mask_keep = ~((eid == enseignant_id) & fid.isin(ids_new))
    df = df.loc[mask_keep]
    fiches = {int(r["fiche_id"]): r for r in l2_fiches()}
    nouvelles = []
    for item in reponses:
        fid = int(item["fiche_id"])
        base = dict(fiches[fid])
        for col in L2_SCORES:
            base[col] = _clamp(item[col], 1, 5)
        base["commentaire"] = str(item.get("commentaire") or "")
        base["enseignant_id"] = enseignant_id
        nouvelles.append(base)
    out = pd.concat([df, pd.DataFrame(nouvelles)], ignore_index=True)
    _write(L2_FILE, out)
    return {
        "ok": True,
        "niveau": "L2",
        "enseignant_id": enseignant_id,
        "n": len(nouvelles),
        "fichier": _path(L2_FILE),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def _sus_score(items: dict) -> float:
    total = 0
    for i in range(1, 11):
        v = _clamp(items[f"sus_{i}"], 1, 5)
        total += (v - 1) if i % 2 == 1 else (5 - v)
    return round(total * 2.5, 1)


def enregistrer_l4(payload: dict) -> dict:
    pid = int(payload["participant_id"])
    if pid < 1 or pid > 30:
        raise ValueError("participant_id entre 1 et 30")
    df = _read(L4_FILE)
    df = df.astype(object)
    idx = df.index[df["participant_id"].astype(int) == pid]
    if len(idx) == 0:
        raise ValueError(f"Participant {pid} absent du formulaire L4")
    i = idx[0]
    df.at[i, "niveau"] = str(payload.get("niveau") or "")
    df.at[i, "familiarite_mooc_1_5"] = _clamp(payload["familiarite_mooc_1_5"], 1, 5)
    df.at[i, "clip_id"] = str(payload.get("clip_id") or "")
    df.at[i, "condition"] = str(payload.get("condition") or "")
    for col in ("utilite_1_7", "adequation_emotionnelle_1_7", "satisfaction_contenu_1_7"):
        df.at[i, col] = _clamp(payload[col], 1, 7)
    sus_items = {f"sus_{k}": payload[f"sus_{k}"] for k in range(1, 11)}
    for k, v in sus_items.items():
        df.at[i, k] = _clamp(v, 1, 5)
    df.at[i, "sus_score"] = _sus_score(sus_items)
    for col in ("q_ouvert_aide", "q_ouvert_gene", "q_ouvert_suggestion"):
        df.at[i, col] = str(payload.get(col) or "")
    _write(L4_FILE, df)
    return {
        "ok": True,
        "niveau": "L4",
        "participant_id": pid,
        "sus_score": float(df.at[i, "sus_score"]),
        "fichier": _path(L4_FILE),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def page_html(title: str, body: str) -> str:
    return f"""<!DOCTYPE html>
<html lang="fr">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>{html.escape(title)} — EmoLearn</title>
  <style>
    :root {{ --bg:#f4f1ea; --card:#fff; --ink:#1c1917; --muted:#78716c; --accent:#0f766e; --line:#e7e5e4; }}
    * {{ box-sizing:border-box; }}
    body {{ margin:0; font-family:Georgia, serif; background:var(--bg); color:var(--ink); line-height:1.45; }}
    main {{ max-width:760px; margin:0 auto; padding:24px 16px 64px; }}
    a {{ color:var(--accent); }}
    h1 {{ font-size:1.6rem; margin:0 0 8px; }}
    .muted {{ color:var(--muted); font-size:.95rem; }}
    .card {{ background:var(--card); border:1px solid var(--line); border-radius:10px; padding:16px; margin:16px 0; }}
    label, .lab {{ display:block; font-size:.9rem; margin:10px 0 4px; }}
    input, select, textarea {{ width:100%; padding:8px; font:inherit; border:1px solid var(--line); border-radius:6px; }}
    textarea {{ min-height:64px; }}
    .likert {{ display:flex; gap:8px; flex-wrap:wrap; }}
    .likert label {{ margin:0; display:flex; gap:4px; align-items:center; font-size:.85rem; }}
    button {{ background:var(--accent); color:#fff; border:0; padding:10px 18px; border-radius:8px; font:inherit; cursor:pointer; }}
    button:disabled {{ opacity:.5; }}
    .ok {{ background:#ecfdf5; border:1px solid #a7f3d0; padding:12px; border-radius:8px; }}
    .err {{ background:#fef2f2; border:1px solid #fecaca; padding:12px; border-radius:8px; }}
    nav a {{ margin-right:12px; }}
    .extrait {{ white-space:pre-wrap; font-size:.9rem; background:#fafaf9; padding:10px; border-radius:6px; max-height:220px; overflow:auto; }}
  </style>
</head>
<body><main>
  <nav><a href="/evaluation">Questionnaires</a></nav>
  {body}
</main></body></html>"""


PRESENTATION_HTML = """
<div class="card">
<h2>De quoi s'agit-il ?</h2>
<p><b>EmoLearn</b> est un <b>prototype de recherche</b> (mémoire de master), pas une application déjà déployée sur un MOOC.</p>
<p>Dans un cours en ligne, l'enseignant ne voit pas si l'étudiant s'ennuie, est perdu ou frustré. EmoLearn simule cette boucle :</p>
<ol>
  <li><b>Perception</b> — visage + voix → un état : Engagement, Ennui, Confusion ou Frustration.</li>
  <li><b>Décision</b> — A = réexpliquer, B = exemple, C = exercice, D = encouragement.</li>
  <li><b>Génération</b> — un texte de cours (algorithmes de tri) adapté à cet état.</li>
</ol>
<p>Ce questionnaire est autonome : pas besoin d'ouvrir le prototype.</p>
</div>
"""


def html_accueil() -> str:
    body = f"""
    <h1>Questionnaires EmoLearn</h1>
    <p class="muted">Enquête de mémoire. Les notes vont dans results/forms/.</p>
    {PRESENTATION_HTML}
    <div class="card"><h2>L1 — Experts (mapping)</h2>
      <p>15 cas : l'état justifie-t-il le patron proposé ? ~20–30 min.</p>
      <p><a href="/evaluation/l1">Ouvrir le formulaire L1</a></p></div>
    <div class="card"><h2>L2 — Enseignants (contenu)</h2>
      <p>Extraits générés : pertinence, clarté, adéquation émotionnelle, qualité.</p>
      <p><a href="/evaluation/l2">Ouvrir le formulaire L2</a></p></div>
    <div class="card"><h2>L4 — Étudiants (pilote)</h2>
      <p>Utilité, adéquation, satisfaction, SUS (10 items), 3 questions ouvertes.</p>
      <p><a href="/evaluation/l4">Ouvrir le formulaire L4</a></p></div>
    """
    return page_html("Questionnaires", body)


def _likert(name: str, hi: int = 5) -> str:
    radios = "".join(
        f'<label><input type="radio" name="{html.escape(name)}" value="{i}" required/> {i}</label>'
        for i in range(1, hi + 1)
    )
    return f'<div class="likert">{radios}</div>'


def html_l1() -> str:
    cards = []
    for cas in l1_cas():
        fid = int(cas["fiche_id"])
        meta = (
            f"Cas {fid} · clip {html.escape(str(cas.get('clip_id','')))} · "
            f"pattern {html.escape(str(cas.get('pattern','')))}<br/>"
            f"État annoté : <b>{html.escape(str(cas.get('etat_gt','')))}</b> "
            f"→ patron attendu {html.escape(str(cas.get('template_attendu_gt','')))} "
            f"({html.escape(str(cas.get('famille_attendue_gt','')))})<br/>"
            f"État fusionné : <b>{html.escape(str(cas.get('etat_fusion','')))}</b> "
            f"→ patron observé {html.escape(str(cas.get('template_observe','')))}"
        )
        champs = "".join(
            f"<div class='lab'>{html.escape(L1_LABELS[c])}</div>{_likert(f'f{fid}_{c}')}"
            for c in L1_SCORES
        )
        cards.append(
            f"<div class='card'><p>{meta}</p>{champs}"
            f"<label>Commentaire</label>"
            f"<textarea name='f{fid}_commentaire'></textarea></div>"
        )
    body = f"""
    <h1>L1 — Mapping engagement → adaptation</h1>
    {PRESENTATION_HTML}
    <p class="muted">1 = pas d'accord · 5 = tout à fait d'accord. Identifiant libre (E1, votre nom…).</p>
    <form id="form">
      <div class="card"><label>Votre identifiant</label>
        <input name="expert_id" required placeholder="E1"/></div>
      {''.join(cards)}
      <button type="submit">Enregistrer mes notes</button>
    </form>
    <p id="msg"></p>
    <script>
    document.getElementById('form').onsubmit = async (e) => {{
      e.preventDefault();
      const fd = new FormData(e.target);
      const reponses = [];
      for (const card of { [int(c['fiche_id']) for c in l1_cas()] }) {{
        const item = {{fiche_id: card, commentaire_expert: fd.get('f'+card+'_commentaire') || ''}};
        let ok = true;
        {repr(L1_SCORES)}.forEach(col => {{
          const v = fd.get('f'+card+'_'+col);
          if (!v) ok = false;
          item[col] = Number(v);
        }});
        if (!ok) {{ document.getElementById('msg').innerHTML = '<p class="err">Complétez toutes les notes du cas '+card+'.</p>'; return; }}
        reponses.push(item);
      }}
      const r = await fetch('/evaluation/l1', {{
        method:'POST', headers:{{'Content-Type':'application/json'}},
        body: JSON.stringify({{expert_id: fd.get('expert_id'), reponses}})
      }});
      const data = await r.json();
      document.getElementById('msg').innerHTML = data.ok
        ? '<p class="ok">Merci. '+data.n+' cas enregistrés pour '+data.expert_id+'.</p>'
        : '<p class="err">'+(data.detail || data.error || 'Erreur')+'</p>';
    }};
    </script>
    """
    return page_html("L1 mapping", body)


def html_l2(offset: int = 0, limit: int = 5) -> str:
    fiches = l2_fiches()
    total = len(fiches)
    offset = max(0, min(offset, max(0, total - 1)))
    chunk = fiches[offset : offset + limit]
    cards = []
    ids = []
    for f in chunk:
        fid = int(f["fiche_id"])
        ids.append(fid)
        extra = html.escape(str(f.get("contenu_extrait", "")))
        meta = (
            f"Fiche {fid}/{total} · patron {html.escape(str(f.get('template_id','')))} · "
            f"état annoncé : <b>{html.escape(str(f.get('etat','')))}</b> · "
            f"concept : {html.escape(str(f.get('concept','')))}"
        )
        champs = "".join(
            f"<div class='lab'>{html.escape(L2_LABELS[c])}</div>{_likert(f'f{fid}_{c}')}"
            for c in L2_SCORES
        )
        cards.append(
            f"<div class='card'><p>{meta}</p><div class='extrait'>{extra}</div>{champs}"
            f"<label>Commentaire</label><textarea name='f{fid}_commentaire'></textarea></div>"
        )
    prev_off = max(0, offset - limit)
    next_off = offset + limit
    nav = f'<p><a href="/evaluation/l2?offset={prev_off}">Précédent</a>'
    if next_off < total:
        nav += f' · <a href="/evaluation/l2?offset={next_off}">Suivant ({next_off + 1}–{min(total, next_off + limit)})</a>'
    nav += "</p>"
    body = f"""
    <h1>L2 — Qualité du contenu généré</h1>
    {PRESENTATION_HTML}
    <p class="muted">Extraits {offset + 1}–{offset + len(chunk)} sur {total}. Même identifiant à chaque page (ENS1…).</p>
    {nav}
    <form id="form">
      <div class="card"><label>Votre identifiant enseignant</label>
        <input name="enseignant_id" required placeholder="ENS1"/></div>
      {''.join(cards)}
      <button type="submit">Enregistrer cette page</button>
    </form>
    <p id="msg"></p>
    {nav}
    <script>
    const IDS = {ids};
    const COLS = {repr(L2_SCORES)};
    document.getElementById('form').onsubmit = async (e) => {{
      e.preventDefault();
      const fd = new FormData(e.target);
      const reponses = [];
      for (const card of IDS) {{
        const item = {{fiche_id: card, commentaire: fd.get('f'+card+'_commentaire') || ''}};
        let ok = true;
        COLS.forEach(col => {{
          const v = fd.get('f'+card+'_'+col);
          if (!v) ok = false;
          item[col] = Number(v);
        }});
        if (!ok) {{ document.getElementById('msg').innerHTML = '<p class="err">Complétez la fiche '+card+'.</p>'; return; }}
        reponses.push(item);
      }}
      const r = await fetch('/evaluation/l2', {{
        method:'POST', headers:{{'Content-Type':'application/json'}},
        body: JSON.stringify({{enseignant_id: fd.get('enseignant_id'), reponses}})
      }});
      const data = await r.json();
      document.getElementById('msg').innerHTML = data.ok
        ? '<p class="ok">Page enregistrée ('+data.n+' fiches) pour '+data.enseignant_id+'.</p>'
        : '<p class="err">'+(data.detail || data.error || 'Erreur')+'</p>';
    }};
    </script>
    """
    return page_html("L2 contenu", body)


SUS_ITEMS = [
    "Je pense que je voudrais utiliser ce système fréquemment.",
    "Je trouve le système inutilement complexe.",
    "Je trouve le système facile à utiliser.",
    "J'aurais besoin d'un support technique pour utiliser ce système.",
    "Les différentes fonctions sont bien intégrées.",
    "Il y a trop d'incohérences dans ce système.",
    "La plupart des gens apprendraient à l'utiliser très rapidement.",
    "Je trouve le système très lourd à utiliser.",
    "Je me suis senti(e) en confiance en l'utilisant.",
    "J'ai dû apprendre beaucoup de choses avant de pouvoir l'utiliser.",
]


def html_l4() -> str:
    sus = "".join(
        f"<div class='lab'>SUS {i} — {html.escape(txt)}</div>{_likert(f'sus_{i}')}"
        for i, txt in enumerate(SUS_ITEMS, start=1)
    )
    body = f"""
    <h1>L4 — Pilote étudiants</h1>
    {PRESENTATION_HTML}
    <p class="muted">Pas besoin d'avoir vu le logiciel. Likert 1–7 pour l'utilité ; SUS 1–5 pour la facilité d'usage imaginée.</p>
    <form id="form">
      <div class="card">
        <label>Numéro participant (1–30)</label>
        <input name="participant_id" type="number" min="1" max="30" required/>
        <label>Niveau (L2, L3…)</label>
        <input name="niveau" placeholder="L3 informatique"/>
        <label>Familiarité MOOC (1–5)</label>
        {_likert('familiarite_mooc_1_5')}
        <label>Clip vu (optionnel)</label>
        <input name="clip_id"/>
        <label>Condition</label>
        <select name="condition">
          <option value="C2">C2 — contenu EmoLearn (adaptatif)</option>
          <option value="C1">C1 — contenu statique</option>
        </select>
        <div class="lab">Utilité perçue de l'adaptation (1–7)</div>
        {_likert('utilite_1_7', 7)}
        <div class="lab">Adéquation émotionnelle perçue (1–7)</div>
        {_likert('adequation_emotionnelle_1_7', 7)}
        <div class="lab">Satisfaction à l'égard du contenu (1–7)</div>
        {_likert('satisfaction_contenu_1_7', 7)}
      </div>
      <div class="card"><h2>SUS</h2>{sus}</div>
      <div class="card">
        <label>Qu'est-ce qui a aidé ?</label><textarea name="q_ouvert_aide"></textarea>
        <label>Qu'est-ce qui a gêné ?</label><textarea name="q_ouvert_gene"></textarea>
        <label>Une suggestion</label><textarea name="q_ouvert_suggestion"></textarea>
      </div>
      <button type="submit">Enregistrer</button>
    </form>
    <p id="msg"></p>
    <script>
    document.getElementById('form').onsubmit = async (e) => {{
      e.preventDefault();
      const fd = new FormData(e.target);
      const body = Object.fromEntries(fd.entries());
      ['participant_id','familiarite_mooc_1_5','utilite_1_7','adequation_emotionnelle_1_7','satisfaction_contenu_1_7'].forEach(k => body[k] = Number(body[k]));
      for (let i=1;i<=10;i++) body['sus_'+i] = Number(fd.get('sus_'+i));
      const r = await fetch('/evaluation/l4', {{
        method:'POST', headers:{{'Content-Type':'application/json'}},
        body: JSON.stringify(body)
      }});
      const data = await r.json();
      document.getElementById('msg').innerHTML = data.ok
        ? '<p class="ok">Merci. SUS = '+data.sus_score+'/100.</p>'
        : '<p class="err">'+(data.detail || data.error || 'Erreur')+'</p>';
    }};
    </script>
    """
    return page_html("L4 pilote", body)
