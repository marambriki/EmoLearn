# Limitations du prototype EmoLearn (v2.2)

Document aligné avec l'implémentation réelle — à intégrer au chapitre 6 du mémoire.

## Ce que le prototype fait

- **Simulation DAiSEE** : scores d'engagement depuis `scenarios_50.csv` (Mode B — défaut).
- **FER réel** : upload d'image via `POST /analyser/fer` (Mode A — optionnel).
- **Voix IEMOCAP** : annotations réelles, appariement heuristique `scenario_id % N`.
- **Fusion multimodale** : pondération 60 % visage / 40 % voix.
- **Temporalité** : séquences 5–10 clips, règles multi-clips (`temporal_rules.py`).
- **Génération** : Groq Llama 3.1 + 20 templates + **RAG systématique** (ChromaDB).

## Limitations assumées

| Limite | Impact | Mitigation mémoire |
|--------|--------|-------------------|
| Pas de collecte MOOC live | Pas de validation in situ | Prototype de simulation + pilote n=30 |
| Appariement IEMOCAP | Visage/voix pas du même apprenant | Documenter dans `docs/IEMOCAP.md` |
| FER ≠ labels DAiSEE | Mapping 7→4 classes heuristique | QR1 + `eval_fer_daisee.py` |
| LLM Groq vs GPT-4 plan initial | Légère différence de style | Justifier coût/reproductibilité |
| Validation experts non exécutée | Pas de scores Delphi | Fiches exportées, protocole prêt |
| Pas de fine-tuning | Dépendance modèles pré-entraînés | Périmètre master respecté |

## Mise à jour par rapport au sujet initial

Le sujet mentionnait « pas de traitement multimodal » — **obsolète** : IEMOCAP et fusion sont implémentés depuis l'extension du projet.

Le sujet prévoit un **prototype de simulation** (OS5) — cohérent avec l'architecture actuelle.
