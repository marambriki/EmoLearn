# ============================================
# backend/main.py
# EmoLearn — API FastAPI — prototype de simulation (clip)
# ============================================

import os
import sys
import traceback

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel

sys.path.append(os.path.dirname(__file__))

from daisee_utils import etat_dominant_depuis_scores, pattern_depuis_scores
from daisee_video import (
    daisee_video_exists,
    extract_preview_jpeg,
    iter_mjpeg_stream,
    resolve_daisee_video_path,
)
from database import initialiser_db, sauvegarder_session, recuperer_sessions
from iemocap_audio import extract_segment_wav_bytes, resolve_wav_path
from pipeline import analyser_clip, enregistrer_decision, formater_reponse_api
from chat_module import repondre_tuteur
from ser_module import apercu_audio_scenario

app = FastAPI(title="EmoLearn Simulation API", version="2.5.0")

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

df_scenarios = pd.read_csv(scenarios_path)
df_scenarios.columns = [col.strip() for col in df_scenarios.columns]
# Un ClipID = une ligne ; index contigu 0..n-1 pour les ids API/UI
df_scenarios = df_scenarios.drop_duplicates(subset=["ClipID"], keep="first").reset_index(
    drop=True
)
for idx, row in df_scenarios.iterrows():
    df_scenarios.at[idx, "pattern"] = pattern_depuis_scores(
        row["Boredom"], row["Confusion"], row["Engagement"], row["Frustration"]
    )

sequences_path = os.path.join(DATA_DIR, "scenarios_sequences.csv")
df_sequences = (
    pd.read_csv(sequences_path)
    if os.path.isfile(sequences_path)
    else pd.DataFrame()
)

print(f"✅ {len(df_scenarios)} scénarios chargés (simulation clip)")
print(f"✅ {len(df_sequences)} séquences temporelles chargées")

# Précharge l'index DAiSEE pour éviter un 1er /scenarios très lent / timeouts UI
try:
    from daisee_video import get_index

    get_index()
except Exception as exc:
    print(f"⚠ Index DAiSEE non préchargé : {exc}")


class AnalyseRequest(BaseModel):
    scenario_id: int
    concept: str = "les algorithmes de tri"
    module: str = "Algorithmique"
    niveau: str = "Intermédiaire"
    moment_cours: str = "Milieu du module"
    utiliser_temporel: bool = True


class SequenceRequest(BaseModel):
    sequence_id: int
    concept: str = "les algorithmes de tri"
    module: str = "Algorithmique"
    niveau: str = "Intermédiaire"
    moment_cours: str = "Milieu du module"
    generer_contenu: bool = True
    generer_contenu_chaque_clip: bool = False


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    message: str
    historique: list[ChatMessage] = []
    etat: str = "Engagement"
    niveau: int = 2
    template_id: str = "A1"
    strategie: str = ""
    concept: str = "les algorithmes de tri"
    module: str = "Algorithmique"
    niveau_apprenant: str = "Intermédiaire"
    objectif_pedagogique: str = ""
    contenu_initial: str | None = None
    utiliser_rag: bool = True


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
    utiliser_temporel=True,
    historique_scores=None,
    generer_contenu=True,
    persister=True,
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
        generer_contenu=generer_contenu,
        historique_scores=historique_scores,
        utiliser_temporel=utiliser_temporel,
    )

    response = formater_reponse_api(analyse, scenario_id=scenario_id)

    if (
        persister
        and analyse["decision"].get("declencher", True)
        and analyse["resultat_llm"]
    ):
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


def _parse_sequence_ids(raw):
    return [int(x) for x in str(raw).split(";") if str(x).strip() != ""]


@app.get("/")
def accueil():
    return {
        "message": "EmoLearn Simulation API v2.5",
        "status": "Opérationnel",
        "prototype": "simulation",
        "unite": "clip",
        "modalites": ["visage (vidéo réelle DAiSEE + FER)", "voix (audio réel IEMOCAP + SER)"],
        "modes": ["simulation", "sequence", "chat_tuteur"],
        "scenarios": len(df_scenarios),
        "sequences": len(df_sequences),
        "templates": "A1–D5 (20 templates)",
        "temporel": True,
    }


@app.get("/health")
def health():
    from fer_offline import fer_finetuned_disponible, fer_runtime_mode, FINETUNED_DIR
    from ser_module import (
        SER_METHODE_DEFAUT,
        _SER_FINETUNED_DIR,
        ser_finetuned_disponible,
        ser_methode_active,
    )

    return {
        "status": "ok",
        "version": "2.5.0",
        "perception": {
            "fer_mode": fer_runtime_mode(),
            "fer_finetuned_disponible": fer_finetuned_disponible(),
            "fer_model_dir": FINETUNED_DIR if fer_finetuned_disponible() else None,
            "ser_methode": ser_methode_active(SER_METHODE_DEFAUT),
            "ser_finetuned_disponible": ser_finetuned_disponible(),
            "ser_model_dir": _SER_FINETUNED_DIR if ser_finetuned_disponible() else None,
        },
    }


@app.post("/chat")
def chat_tuteur(request: ChatRequest):
    """Tuteur conversationnel adapté à l'état émotionnel détecté."""
    try:
        if not request.message.strip():
            return {"error": "Message vide"}
        result = repondre_tuteur(
            message=request.message.strip(),
            historique=[m.model_dump() for m in request.historique],
            etat=request.etat,
            niveau=request.niveau,
            template_id=request.template_id,
            strategie=request.strategie,
            concept=request.concept,
            module=request.module,
            niveau_apprenant=request.niveau_apprenant,
            objectif_pedagogique=request.objectif_pedagogique,
            contenu_initial=request.contenu_initial,
            utiliser_rag=request.utiliser_rag,
        )
        return result
    except Exception as exc:
        traceback.print_exc()
        return {"error": str(exc)}


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
    """Analyse un clip de simulation (vidéo DAiSEE réelle + audio IEMOCAP réel)."""
    try:
        return _executer_et_persister(
            scenario_id=request.scenario_id,
            concept=request.concept,
            module=request.module,
            niveau=request.niveau,
            moment_cours=request.moment_cours,
            utiliser_temporel=request.utiliser_temporel,
        )
    except Exception as e:
        traceback.print_exc()
        return {"error": str(e)}


@app.get("/sequences")
def lister_sequences():
    """Liste les parcours multi-clips (règles temporelles)."""
    if df_sequences.empty:
        return {"total": 0, "sequences": []}
    items = []
    for idx, row in df_sequences.iterrows():
        sids = _parse_sequence_ids(row.get("scenario_ids", ""))
        items.append(
            {
                "id": int(row.get("sequence_id", idx)),
                "title": str(row.get("title", f"Séquence {idx}")),
                "pattern": str(row.get("pattern", "stable")),
                "length": int(row.get("length", len(sids))),
                "scenario_ids": sids,
                "clip_ids": [
                    c for c in str(row.get("clip_ids", "")).split(";") if c
                ],
            }
        )
    return {"total": len(items), "sequences": items}


@app.post("/analyser/sequence")
def analyser_sequence(request: SequenceRequest):
    """
    Analyse une séquence multi-clips avec historique temporel cumulé.
    Par défaut : LLM uniquement sur le dernier clip (économise l'API).
    """
    try:
        if df_sequences.empty:
            return {"error": "Aucune séquence disponible"}
        match = df_sequences[df_sequences["sequence_id"] == request.sequence_id]
        if match.empty:
            return {"error": "Séquence non trouvée"}
        row = match.iloc[0]
        scenario_ids = _parse_sequence_ids(row.get("scenario_ids", ""))
        if not scenario_ids:
            return {"error": "Séquence vide"}

        historique = []
        etapes = []
        derniere = None
        for i, sid in enumerate(scenario_ids):
            est_dernier = i == len(scenario_ids) - 1
            generer = request.generer_contenu and (
                request.generer_contenu_chaque_clip or est_dernier
            )
            step = _executer_et_persister(
                scenario_id=sid,
                concept=request.concept,
                module=request.module,
                niveau=request.niveau,
                moment_cours=request.moment_cours,
                utiliser_temporel=True,
                historique_scores=historique,
                generer_contenu=generer,
                persister=est_dernier and generer,
            )
            if step.get("error"):
                return step
            scores = step.get("decision", {}).get("scores_daisee") or {}
            historique.append(
                {
                    "boredom": int(scores.get("boredom", 0)),
                    "confusion": int(scores.get("confusion", 0)),
                    "engagement": int(scores.get("engagement", 0)),
                    "frustration": int(scores.get("frustration", 0)),
                }
            )
            etapes.append(
                {
                    "index": i,
                    "scenario_id": sid,
                    "clip_id": step.get("clip_id"),
                    "emotion": step.get("emotion"),
                    "niveau": step.get("niveau"),
                    "template_id": step.get("template_id"),
                    "declencher": step.get("decision", {}).get("declencher"),
                    "temporal": step.get("temporal"),
                    "contenu_genere": bool(
                        step.get("contenu") and not step.get("contenu_skipped")
                    ),
                }
            )
            derniere = step

        flat = dict(derniere or {})
        flat.update(
            {
                "mode": "sequence",
                "sequence_id": request.sequence_id,
                "title": str(row.get("title", "")),
                "pattern": str(row.get("pattern", "")),
                "n_clips": len(scenario_ids),
                "scenario_id": scenario_ids[-1],
                "etapes": etapes,
                "resultat_final": derniere,
                "sequence_meta": {
                    "sequence_id": request.sequence_id,
                    "title": str(row.get("title", "")),
                    "n_clips": len(scenario_ids),
                    "etapes": etapes,
                },
            }
        )
        return flat
    except Exception as e:
        traceback.print_exc()
        return {"error": str(e)}


@app.get("/scenarios")
def lister_scenarios():
    items = []
    for i, (_, row) in enumerate(df_scenarios.iterrows()):
        b = int(row["Boredom"])
        c = int(row["Confusion"])
        e = int(row["Engagement"])
        f = int(row["Frustration"])
        etat, niveau, _ = etat_dominant_depuis_scores(b, c, e, f)
        clip_id = str(row["ClipID"])
        items.append(
            {
                "id": i,
                "clip_id": clip_id,
                "pattern": str(row.get("pattern", "stable")),
                "boredom": b,
                "confusion": c,
                "engagement": e,
                "frustration": f,
                "etat_dominant": etat,
                "niveau_dominant": niveau,
                "video_disponible": daisee_video_exists(clip_id),
            }
        )
    return {"total": len(items), "scenarios": items}


@app.get("/scenarios/{scenario_id}/video")
def meta_video_scenario(scenario_id: int):
    """Métadonnées du clip DAiSEE associé au scénario (pour le lecteur UI)."""
    scenario = _get_scenario(scenario_id)
    if scenario is None:
        raise HTTPException(status_code=404, detail="Scénario non trouvé")
    clip_id = str(scenario["ClipID"])
    path = resolve_daisee_video_path(clip_id)
    disponible = bool(path) and os.path.isfile(path)
    return {
        "scenario_id": scenario_id,
        "clip_id": clip_id,
        "disponible": disponible,
        "frame_url": f"/scenarios/{scenario_id}/frame" if disponible else None,
        "stream_url": f"/scenarios/{scenario_id}/stream" if disponible else None,
    }


@app.get("/scenarios/{scenario_id}/frame")
def frame_video_scenario(scenario_id: int):
    """Frame JPEG de prévisualisation du clip DAiSEE."""
    scenario = _get_scenario(scenario_id)
    if scenario is None:
        raise HTTPException(status_code=404, detail="Scénario non trouvé")
    jpeg = extract_preview_jpeg(str(scenario["ClipID"]))
    if not jpeg:
        raise HTTPException(status_code=404, detail="Vidéo DAiSEE indisponible")
    return Response(content=jpeg, media_type="image/jpeg")


@app.get("/scenarios/{scenario_id}/stream")
def stream_video_scenario(scenario_id: int):
    """Flux MJPEG du clip DAiSEE (lisible dans le navigateur)."""
    scenario = _get_scenario(scenario_id)
    if scenario is None:
        raise HTTPException(status_code=404, detail="Scénario non trouvé")
    clip_id = str(scenario["ClipID"])
    if not daisee_video_exists(clip_id):
        raise HTTPException(status_code=404, detail="Vidéo DAiSEE indisponible")
    return StreamingResponse(
        iter_mjpeg_stream(clip_id),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.get("/scenarios/{scenario_id}/audio")
def meta_audio_scenario(scenario_id: int):
    """Utterance IEMOCAP appariée au scénario (pour le lecteur UI)."""
    scenario = _get_scenario(scenario_id)
    if scenario is None:
        raise HTTPException(status_code=404, detail="Scénario non trouvé")
    etat, _, _ = etat_dominant_depuis_scores(
        scenario["Boredom"],
        scenario["Confusion"],
        scenario["Engagement"],
        scenario["Frustration"],
    )
    meta = apercu_audio_scenario(scenario_id, etat_dominant=etat)
    if meta.get("audio_disponible"):
        meta["stream_url"] = f"/scenarios/{scenario_id}/audio/stream"
    else:
        meta["stream_url"] = None
    return meta


@app.get("/scenarios/{scenario_id}/audio/stream")
def stream_audio_scenario(scenario_id: int):
    """Segment WAV IEMOCAP apparié (lisible dans <audio>)."""
    scenario = _get_scenario(scenario_id)
    if scenario is None:
        raise HTTPException(status_code=404, detail="Scénario non trouvé")
    etat, _, _ = etat_dominant_depuis_scores(
        scenario["Boredom"],
        scenario["Confusion"],
        scenario["Engagement"],
        scenario["Frustration"],
    )
    meta = apercu_audio_scenario(scenario_id, etat_dominant=etat)
    wav_path = resolve_wav_path(meta["utterance_id"])
    audio = extract_segment_wav_bytes(
        meta["utterance_id"],
        meta["start_time"],
        meta["end_time"],
        wav_path=wav_path,
    )
    if not audio:
        raise HTTPException(status_code=404, detail="Audio IEMOCAP indisponible")
    return Response(
        content=audio,
        media_type="audio/wav",
        headers={
            "Content-Disposition": (
                f'inline; filename="{meta["utterance_id"]}.wav"'
            )
        },
    )


@app.get("/sessions")
def get_sessions():
    """Historique des analyses de clips (SQLite)."""
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
        "patterns": df_scenarios["pattern"].value_counts().to_dict(),
        "version": "simulation-clip v2.5",
        "sequences": len(df_sequences),
        "temporel": True,
    }


@app.get("/templates")
def lister_templates_api():
    from prompts.templates import TEMPLATE_META, lister_templates

    return {
        "total": len(lister_templates()),
        "templates": [{"id": tid, **meta} for tid, meta in TEMPLATE_META.items()],
    }
