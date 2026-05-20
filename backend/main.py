# ============================================
# backend/main.py
# ============================================

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import sys
import os

sys.path.append(os.path.dirname(__file__))
from adaptation import selectionner_strategie, get_description_strategie
from llm_module import generer_contenu_adaptatif
from database import initialiser_db, sauvegarder_session, recuperer_sessions

app = FastAPI(title="EmoLearn API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

initialiser_db()

scenarios_path = os.path.join(os.path.dirname(__file__), '../data/scenarios_50.csv')
df_scenarios = pd.read_csv(scenarios_path)
df_scenarios.columns = [col.strip() for col in df_scenarios.columns]
print(f"Colonnes : {df_scenarios.columns.tolist()}")
print(f"Scénarios chargés : {len(df_scenarios)}")

class AnalyseRequest(BaseModel):
    scenario_id : int
    concept     : str = "les algorithmes de tri"
    module      : str = "Algorithmique"
    niveau      : str = "Intermédiaire"

@app.get("/")
def accueil():
    return {"message": "EmoLearn API v1.0", "status": "✅ Opérationnel", "scenarios": len(df_scenarios)}

@app.get("/scenarios")
def get_scenarios():
    return {"total": len(df_scenarios), "scenarios": df_scenarios.to_dict(orient='records')}

@app.post("/analyser")
def analyser_scenario(request: AnalyseRequest):
    try:
        print(f"=== ANALYSE scenario_id={request.scenario_id} ===")

        if request.scenario_id < 0 or request.scenario_id >= len(df_scenarios):
            return {"error": "Scénario non trouvé"}

        scenario = df_scenarios.iloc[request.scenario_id]
        print(f"Scénario : {scenario['ClipID']} — pattern : {scenario['pattern']}")

        emotion, niveau, strategie, type_c = selectionner_strategie(
            int(scenario['Boredom']),
            int(scenario['Confusion']),
            int(scenario['Engagement']),
            int(scenario['Frustration'])
        )
        print(f"Stratégie : {strategie} — Type : {type_c}")

        description = get_description_strategie(type_c)

        mapping_type = {
            'A': 'Confusion', 'A+D': 'Confusion',
            'B': 'Engagement',
            'C': 'Boredom',
            'D': 'Frustration', 'D+A': 'Frustration'
        }
        etat_llm = mapping_type.get(type_c, 'Engagement')
        print(f"État LLM : {etat_llm} — Génération en cours...")

        resultat = generer_contenu_adaptatif(
            etat             = etat_llm,
            niveau           = niveau,
            concept          = request.concept,
            sujet_cours      = f"Informatique — {request.module}",
            module_actuel    = request.module,
            niveau_apprenant = request.niveau,
            objectif_pedagogique = f"Comprendre {request.concept}",
            duree_etat       = 2,
            pattern_detecte  = str(scenario['pattern']),
            moment_cours     = "Milieu du module"
        )
        print(f"Contenu généré : {resultat['nb_mots']} mots ✅")

        sauvegarder_session(
            scenario_id  = request.scenario_id,
            emotion      = emotion,
            niveau       = niveau,
            strategie    = strategie,
            type_contenu = type_c,
            contenu      = resultat['contenu']
        )

        return {
            "clip_id"     : str(scenario['ClipID']),
            "pattern"     : str(scenario['pattern']),
            "emotion"     : emotion,
            "niveau"      : niveau,
            "strategie"   : strategie,
            "type_contenu": type_c,
            "description" : description,
            "contenu"     : resultat['contenu'],
            "nb_mots"     : resultat['nb_mots']
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"error": str(e)}

@app.get("/sessions")
def get_sessions():
    sessions = recuperer_sessions()
    return {"total": len(sessions), "sessions": sessions}

@app.get("/statistiques")
def get_statistiques():
    sessions = recuperer_sessions()
    return {
        "total_analyses" : len(sessions),
        "total_scenarios": len(df_scenarios),
        "patterns"       : df_scenarios['pattern'].value_counts().to_dict()
    }