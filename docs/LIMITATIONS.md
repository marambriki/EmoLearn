# Limitations du prototype EmoLearn (v2.2)

Document aligné avec l'implémentation réelle — à intégrer au chapitre 6 du mémoire.

## Ce que le prototype fait

- **Simulation DAiSEE (clip)** : scores d'engagement depuis `scenarios_50.csv` — mode unique de l’interface.
- **Voix IEMOCAP** : annotations réelles, appariement heuristique au scénario.
- **Fusion multimodale** : pondération 60 % visage / 40 % voix.
- **Génération** : Groq Llama 3.1 + 20 templates + **RAG** (ChromaDB).
- **Hors UI produit** : pas de FER webcam/upload, pas de séquence multi-clips.

## Limitations assumées

| Limite | Impact | Mitigation mémoire |
|--------|--------|-------------------|
| Pas de collecte MOOC live | Pas de validation in situ | Prototype de simulation + pilote n=30 |
| Appariement IEMOCAP | Visage/voix pas du même apprenant | Documenté ; **sous-corpus audiovisuel aligné** (Option A) validant la fusion sur le même locuteur — `docs/IEMOCAP.md` §9, `scripts/build_iemocap_av_corpus.py` |
| FER ≠ labels DAiSEE (plafond ~F1 0,39) | Prédictions visage encore bruitées | Fine-tune ViT + filtre frames + cache élargi (`docs/TRAINING.md` §2–4) |
| Confiance multimodal faible | État « sûr » affiché à tort | Pondération continue par incertitude (`fusion_module.py`) |
| LLM Groq vs GPT-4 plan initial | Légère différence de style | Justifier coût/reproductibilité |
| Validation experts non exécutée | Pas de scores Delphi | Fiches exportées, protocole prêt |
| Fine-tune limité (données / CPU) | Pas SOTA engagement | FER/SER fine-tunés mesurés + filet produit (`docs/TRAINING.md`) |

## Mise à jour par rapport au sujet initial

Le sujet mentionnait « pas de traitement multimodal » — **obsolète** : IEMOCAP et fusion sont implémentés depuis l'extension du projet.

Le sujet prévoit un **prototype de simulation** (OS5) — cohérent avec l'architecture actuelle.
