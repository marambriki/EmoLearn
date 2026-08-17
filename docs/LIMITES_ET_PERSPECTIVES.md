# Limites et perspectives — EmoLearn

Complément opérationnel à `LIMITES_ET_TENSIONS.md` pour le mémoire et la soutenance.

---

## 1. Ce qui est en place (prototype)

| Brique | Statut |
|--------|--------|
| Pipeline clip DAiSEE + IEMOCAP | ✅ API `/analyser` |
| Règles temporelles | ✅ branchées (`temporal_rules` → `pipeline`) |
| Parcours multi-clips | ✅ `/sequences`, `/analyser/sequence` + UI |
| Banc fusion aligné (Option A) | ✅ `iemocap_av_metrics.json` |
| Proxy fusion produit | ✅ `scripts/eval_fusion_produit_proxy.py` |
| Formulaires L1 / L2 | ✅ `results/forms/` + agrégation |
| Métriques forme contenu | ✅ `metrics_contenu.py` (+ export MD) |

---

## 2. Limites assumées

1. **Appariement hétérogène** — visage ≠ voix (personnes / contextes différents).
2. **Pas d’accuracy gold** sur le pipeline produit (pas de GT joint).
3. **Mapping IEMOCAP→DAiSEE** — hypothèse de construit (sensibilité mesurée).
4. **Validation humaine** — matériel prêt, scores L1/L2/L4 non encore collectés.
5. **Qualité LLM** — forme mesurée ; fond pédagogique = L2.
6. **Classes rares** — Confusion / Frustration sous-représentées (F1 macro FER faible).
7. **Simulation** — pas de captation live MOOC.

---

## 3. Perspectives (priorisées)

| Priorité | Action | Effort |
|----------|--------|--------|
| P1 | Collecter L1 (2–3 experts, Likert) | Terrain |
| P2 | Noter L2 sur 20–60 contenus | Terrain |
| P3 | Réentraîner SER (epochs↑, class-weighted) / FER sample↑ | Calcul |
| P4 | Pilote utilisateurs n≈30 (L4) | Terrain |
| P5 | Corpus audiovisuel apprentissage aligné (si données) | Données |

---

## 4. Commandes utiles

```bash
# Proxy fusion produit
python scripts/eval_fusion_produit_proxy.py

# Formulaires experts + agrégation
python scripts/agreger_validation_experts.py --prepare-forms
python scripts/agreger_validation_experts.py

# Métriques contenu + extrait mémoire
python scripts/metrics_contenu.py

# Tests
python -m pytest -q
```

---

## 5. Message de clôture (soutenance)

> La force du projet n’est pas un score unique élevé, mais une chaîne complète,
> instrumentée, et des limites **explicites** plutôt que dissimulées.
> La validation experte et le pilote utilisateurs constituent les prochaines étapes
> empiriques naturelles.
