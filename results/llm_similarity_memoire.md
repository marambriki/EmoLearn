## Métriques de similarité LLM (BLEU / ROUGE / BERTScore)

- N = 189 paires candidat / référence
- Type de référence : `auto_median_template`
- Mode auto : pour chaque template, la référence est le contenu de longueur médiane du corpus (cohérence interne, pas gold expert).
- BLEU-4 moyen : 0.1388 (σ=0.0643)
- ROUGE-1 F1 moyen : 0.4486 (σ=0.071)
- ROUGE-2 F1 moyen : 0.1759 (σ=0.0724)
- ROUGE-L F1 moyen : 0.2717 (σ=0.0679)

> Ces scores mesurent la **proximité à une référence**, pas la pertinence pédagogique (validation experte L2). BERTScore tolère mieux les paraphrases que BLEU/ROUGE.
