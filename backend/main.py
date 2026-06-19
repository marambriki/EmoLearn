# ============================================
# backend/main.py
# EmoLearn — API FastAPI multimodale v2.2
# ============================================

import os
import sys
import tempfile
import traceback

import pandas as pd
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

sys.path.append(os.path.dirname(__file__))

from fer_module import analyser_visage
from decision_log import enregistrer_decision
from daisee_utils import etat_dominant_depuis_scores, pattern_depuis_scores # recalcule la cohérence pattern/état dominant.
from database import initialiser_db, sauvegarder_session, recuperer_sessions
from session_state import creer_session, get_session, lister_sessions_actives
from pipeline import analyser_clip, formater_reponse_api, scores_depuis_scenario

app = FastAPI(title="EmoLearn Multimodal API", version="2.2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

initialiser_db()

DATA_DIR = os.path.join(os.path.dirname(__file__), "../data")
scenarios_path = os.path.join(DATA_DIR, "scenarios_50.csv")
sequences_path = os.path.join(DATA_DIR, "scenarios_sequences.csv")

df_scenarios = pd.read_csv(scenarios_path)
df_scenarios.columns = [col.strip() for col in df_scenarios.columns]
for idx, row in df_scenarios.iterrows():
    df_scenarios.at[idx, "pattern"] = pattern_depuis_scores(
        row["Boredom"], row["Confusion"], row["Engagement"], row["Frustration"]
    )

df_sequences = None
if os.path.exists(sequences_path):
    df_sequences = pd.read_csv(sequences_path)
    print(f"✅ {len(df_sequences)} séquences temporelles chargées")
else:
    print("⚠️ scenarios_sequences.csv absent — lancez scripts/generate_sequences.py")

print(f"✅ {len(df_scenarios)} scénarios chargés")


class AnalyseRequest(BaseModel):
    scenario_id: int
    concept: str = "les algorithmes de tri"
    module: str = "Algorithmique"
    niveau: str = "Intermédiaire"
    moment_cours: str = "Milieu du module"
    session_id: str | None = None
    mode: str = "simulation"  # simulation | fer
    utiliser_audio: bool = False


class SequenceRequest(BaseModel):
    sequence_id: int
    concept: str = "les algorithmes de tri"
    module: str = "Algorithmique"
    niveau: str = "Intermédiaire"
    moment_cours: str = "Milieu du module"
    generer_contenu: bool = True
    utiliser_audio: bool = False


class SessionClipRequest(BaseModel):
    session_id: str
    scenario_id: int
    concept: str = "les algorithmes de tri"
    module: str = "Algorithmique"
    niveau: str = "Intermédiaire"
    moment_cours: str = "Milieu du module"
    utiliser_audio: bool = False


def _get_scenario(scenario_id):
    if scenario_id < 0 or scenario_id >= len(df_scenarios):
        return None
    return df_scenarios.iloc[scenario_id]


def _executer_et_persister(
    scenario_id,
    concept,
    module,
    niveau,
    moment_cours="Milieu du module",
    mode="simulation",
    image_path=None,
    temporal=None,
    session_id=None,
    clip_index=None,
    utiliser_audio=False,
):
    scenario = _get_scenario(scenario_id)
    if scenario is None:
        return {"error": "Scénario non trouvé"}

    analyse = analyser_clip(
        scenario=scenario,
        scenario_id=scenario_id,
        concept=concept,
        module=module,
        niveau=niveau,
        moment_cours=moment_cours,
        mode=mode,
        image_path=image_path,
        temporal=temporal,
        utiliser_audio=utiliser_audio,
    )

    response = formater_reponse_api(
        analyse,
        scenario_id=scenario_id,
        session_id=session_id,
        clip_index=clip_index,
    )

    if analyse["decision"].get("declencher", True) and analyse["resultat_llm"]:
        enregistrer_decision(
            scenario_id=scenario_id,
            clip_id=str(scenario["ClipID"]),
            fusion=analyse["fusion"],
            decision=analyse["decision"],
            resultat_llm=analyse["resultat_llm"],
        )
        sauvegarder_session(
            scenario_id=scenario_id,
            emotion=analyse["decision"]["etat"],
            niveau=analyse["decision"]["niveau"],
            strategie=(
                f"{analyse['decision']['template_id']} — "
                f"{analyse['decision']['strategie']}"
            ),
            type_contenu=analyse["decision"]["type_contenu"],
            contenu=analyse["resultat_llm"].get("contenu", ""),
        )

    return response


@app.get("/")
def accueil():
    return {
        "message": "EmoLearn Multimodal API v2.2",
        "status": "Opérationnel",
        "modalites": ["visage (DAiSEE/FER)", "voix (IEMOCAP CSV ou .wav)"],
        "modes": ["simulation", "fer", "sequence"],
        "scenarios": len(df_scenarios),
        "sequences": len(df_sequences) if df_sequences is not None else 0,
        "templates": "A1–D5 (20 templates)",
    }


@app.get("/health")
def health():
    return {"status": "ok", "version": "2.2.0"}


@app.get("/rag/status")
def rag_status():
    """Vérifie si le RAG est prêt (dépendances + base ChromaDB)."""
    import importlib.util

    deps_ok = importlib.util.find_spec("sentence_transformers") is not None
    rag_db = os.path.join(DATA_DIR, "rag_db", "chroma.sqlite3")
    db_ok = os.path.isfile(rag_db)
    return {
        "disponible": deps_ok and db_ok,
        "deps_ok": deps_ok,
        "db_ok": db_ok,
        "chemin_db": rag_db if db_ok else None,
    }


@app.post("/analyser")
def analyser_scenario(request: AnalyseRequest):
    try:
        temporal = None
        session_id = request.session_id

        if session_id:
            session = get_session(session_id)
            if session:
                scenario = _get_scenario(request.scenario_id)
                if scenario is None:
                    return {"error": "Scénario non trouvé"}
                scores = scores_depuis_scenario(scenario)
                _clip, temporal = session.ajouter_clip(
                    request.scenario_id,
                    scenario["ClipID"],
                    scores,
                    str(scenario.get("pattern", "stable")),
                )
            else:
                return {"error": "Session introuvable"}

        return _executer_et_persister(
            scenario_id=request.scenario_id,
            concept=request.concept,
            module=request.module,
            niveau=request.niveau,
            moment_cours=request.moment_cours,
            mode=request.mode,
            temporal=temporal,
            session_id=session_id,
            utiliser_audio=request.utiliser_audio,
        )
    except Exception as e:
        traceback.print_exc()
        return {"error": str(e)}


@app.post("/analyser/fer")
async def analyser_fer(
    scenario_id: int = 0,
    concept: str = "les algorithmes de tri",
    module: str = "Algorithmique",
    niveau: str = "Intermédiaire",
    image: UploadFile = File(...),
):
    """Analyse FER réelle sur une image uploadée + scores DAiSEE du scénario associé."""
    try:
        scenario = _get_scenario(scenario_id)
        if scenario is None:
            return {"error": "Scénario non trouvé"}

        suffix = os.path.splitext(image.filename or "face.jpg")[1] or ".jpg"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            content = await image.read()
            tmp.write(content)
            tmp_path = tmp.name

        try:
            fer_result = analyser_visage(tmp_path)
            response = _executer_et_persister(
                scenario_id=scenario_id,
                concept=concept,
                module=module,
                niveau=niveau,
                mode="fer",
                image_path=tmp_path,
            )
            response["fer"] = {
                "emotion_detectee": fer_result.get("emotion"),
                "confiance": fer_result.get("score"),
                "etat_mappe": fer_result.get("etat"),
                "niveau_mappe": fer_result.get("niveau"),
                "erreur": fer_result.get("erreur"),
            }
            return response
        finally:
            os.unlink(tmp_path)
    except Exception as e:
        traceback.print_exc()
        return {"error": str(e)}


@app.post("/session/start")
def demarrer_session(sequence_id: int | None = None):
    pattern = None
    if sequence_id is not None and df_sequences is not None:
        if 0 <= sequence_id < len(df_sequences):
            pattern = str(df_sequences.iloc[sequence_id]["pattern"])
    session = creer_session(sequence_id=sequence_id, pattern=pattern)
    return session.to_dict()


@app.post("/session/clip")
def ajouter_clip_session(request: SessionClipRequest):
    try:
        session = get_session(request.session_id)
        if not session:
            return {"error": "Session introuvable"}

        scenario = _get_scenario(request.scenario_id)
        if scenario is None:
            return {"error": "Scénario non trouvé"}

        scores = scores_depuis_scenario(scenario)
        clip, temporal = session.ajouter_clip(
            request.scenario_id,
            scenario["ClipID"],
            scores,
            str(scenario.get("pattern", "stable")),
        )

        response = _executer_et_persister(
            scenario_id=request.scenario_id,
            concept=request.concept,
            module=request.module,
            niveau=request.niveau,
            moment_cours=request.moment_cours,
            temporal=temporal,
            session_id=request.session_id,
            clip_index=clip["index"],
            utiliser_audio=request.utiliser_audio,
        )

        if response.get("contenu") and not response.get("contenu_skipped"):
            session.enregistrer_adaptation(
                clip["index"], response["decision"], response.get("contenu")
            )

        response["session"] = session.to_dict()
        return response
    except Exception as e:
        traceback.print_exc()
        return {"error": str(e)}


@app.get("/session/{session_id}")
def get_session_info(session_id: str):
    session = get_session(session_id)
    if not session:
        return {"error": "Session introuvable"}
    return session.to_dict()


@app.get("/sessions/actives")
def sessions_actives():
    return {"total": len(lister_sessions_actives()), "sessions": lister_sessions_actives()}


@app.post("/analyser/sequence")
def analyser_sequence(request: SequenceRequest):
    """Parcourt une séquence multi-clips et déclenche les adaptations temporelles."""
    try:
        if df_sequences is None:
            return {"error": "Fichier scenarios_sequences.csv manquant"}

        if request.sequence_id < 0 or request.sequence_id >= len(df_sequences):
            return {"error": "Séquence non trouvée"}

        row = df_sequences.iloc[request.sequence_id]
        scenario_ids = [int(x) for x in str(row["scenario_ids"]).split(";")]
        session = creer_session(
            sequence_id=request.sequence_id,
            pattern=str(row["pattern"]),
        )

        parcours = []
        adaptations = []

        for idx, sid in enumerate(scenario_ids):
            scenario = _get_scenario(sid)
            if scenario is None:
                continue

            scores = scores_depuis_scenario(scenario)
            clip, temporal = session.ajouter_clip(
                sid, scenario["ClipID"], scores, str(scenario.get("pattern", "stable"))
            )

            analyse = analyser_clip(
                scenario=scenario,
                scenario_id=sid,
                concept=request.concept,
                module=request.module,
                niveau=request.niveau,
                moment_cours=request.moment_cours,
                temporal=temporal,
                generer_contenu=request.generer_contenu,
                utiliser_audio=request.utiliser_audio,
            )

            clip_result = formater_reponse_api(
                analyse,
                scenario_id=sid,
                session_id=session.session_id,
                clip_index=idx,
            )
            parcours.append(clip_result)

            if analyse["decision"].get("declencher") and analyse["resultat_llm"]:
                adaptations.append(clip_result)
                session.enregistrer_adaptation(
                    idx, analyse["decision"], analyse["resultat_llm"].get("contenu")
                )
                if request.generer_contenu:
                    enregistrer_decision(
                        scenario_id=sid,
                        clip_id=str(scenario["ClipID"]),
                        fusion=analyse["fusion"],
                        decision=analyse["decision"],
                        resultat_llm=analyse["resultat_llm"],
                    )

        dernier = parcours[-1] if parcours else None
        return {
            "sequence_id": int(request.sequence_id),
            "title": str(row["title"]),
            "pattern": str(row["pattern"]),
            "session_id": session.session_id,
            "nb_clips": len(parcours),
            "nb_adaptations": len(adaptations),
            "parcours": parcours,
            "adaptations": adaptations,
            "resultat_final": dernier,
            "session": session.to_dict(),
        }
    except Exception as e:
        traceback.print_exc()
        return {"error": str(e)}


@app.get("/scenarios")
def lister_scenarios():
    items = []
    for idx, row in df_scenarios.iterrows():
        b = int(row["Boredom"])
        c = int(row["Confusion"])
        e = int(row["Engagement"])
        f = int(row["Frustration"])
        etat, niveau, _ = etat_dominant_depuis_scores(b, c, e, f)
        items.append(
            {
                "id": int(idx),
                "clip_id": str(row["ClipID"]),
                "pattern": str(row.get("pattern", "stable")),
                "boredom": b,
                "confusion": c,
                "engagement": e,
                "frustration": f,
                "etat_dominant": etat,
                "niveau_dominant": niveau,
            }
        )
    return {"total": len(items), "scenarios": items}


@app.get("/sequences")
def lister_sequences():
    if df_sequences is None:
        return {"total": 0, "sequences": [], "error": "Fichier sequences absent"}
    items = []
    for idx, row in df_sequences.iterrows():
        ids = [int(x) for x in str(row["scenario_ids"]).split(";")]
        items.append(
            {
                "id": int(row["sequence_id"]),
                "title": str(row["title"]),
                "pattern": str(row["pattern"]),
                "length": int(row["length"]),
                "scenario_ids": ids,
                "clip_ids": str(row["clip_ids"]).split(";"),
            }
        )
    return {"total": len(items), "sequences": items}


@app.get("/sessions")
def get_sessions():
    sessions = recuperer_sessions()
    formatted = []
    for s in sessions:
        formatted.append(
            {
                "id": s[0],
                "scenario_id": s[1],
                "emotion": s[2],
                "niveau": s[3],
                "strategie": s[4],
                "type_contenu": s[5],
                "contenu": s[6],
                "timestamp": s[7] if len(s) > 7 else None,
            }
        )
    return {"total": len(formatted), "sessions": formatted}


@app.get("/statistiques")
def get_statistiques():
    sessions = recuperer_sessions()
    return {
        "total_analyses": len(sessions),
        "total_scenarios": len(df_scenarios),
        "total_sequences": len(df_sequences) if df_sequences is not None else 0,
        "patterns": df_scenarios["pattern"].value_counts().to_dict(),
        "version": "multimodal v2.2",
    }


@app.get("/templates")
def lister_templates_api():
    from prompts.templates import TEMPLATE_META, lister_templates

    return {
        "total": len(lister_templates()),
        "templates": [{"id": tid, **meta} for tid, meta in TEMPLATE_META.items()],
    }
