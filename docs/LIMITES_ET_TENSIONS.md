# Limites et tensions du projet EmoLearn

Document de **recul critique** destiné au chapitre Discussion du mémoire. Il assume
explicitement les tensions conceptuelles du prototype. L'objectif n'est pas de les masquer
mais de montrer qu'elles sont **identifiées, cadrées et, quand c'est possible, testées**.

---

## 1. Multimodal : un compromis précision / sécurité pédagogique (mesuré)

**Tension initiale.** Avec un SER heuristique, la fusion ne dépassait pas le visage seul → la
valeur ajoutée du multimodal semblait remise en cause.

**Résolution (testée, pas seulement cadrée).** Trois améliorations ont été implémentées **et
mesurées** sur le même corpus aligné (n = 80, durée ≥ 1 s, 5 frames/utterance) :

1. **SER appris** (wav2vec2 pré-entraîné IEMOCAP) en remplacement des seuils heuristiques ;
2. **Vote multi-frames** pour le FER (5 images votées au lieu d'une) ;
3. **Fusion par vote pondéré** sur les probabilités d'**état** (et non plus seulement le niveau).

| Modalité / approche | Accuracy | Rappel états urgents |
|---------------------|:---:|:---:|
| Visage — FER 1 frame | 0,375 | 0,250 |
| Visage — FER **multi-frames** | 0,412 | 0,275 |
| Voix — SER heuristique | 0,263 | 0,225 |
| Voix — **SER appris (wav2vec2)** | **0,487** | 0,200 |
| Fusion — **priorité pédagogique** | 0,338 | **0,400** |
| Fusion — **vote pondéré** | **0,500** | 0,125 |

**Lecture assumée.**
- **Le SER appris double presque l'accuracy de la voix** (0,263 → 0,487, **+0,224**) : c'est le
  plus gros levier. La comparaison *heuristique vs appris* est une contribution méthodologique.
- **La fusion par vote dépasse enfin chaque modalité en accuracy** (0,500 > 0,487 / 0,412) : le
  multimodal apporte donc une réelle valeur ajoutée sur l'exactitude globale.
- **Arbitrage mesuré, pas un défaut** : la fusion *vote* maximise l'accuracy mais s'effondre sur le
  rappel urgent (0,125) car wav2vec2 (4 classes ang/hap/neu/sad) **n'a pas de classe Confusion** ;
  la fusion *priorité pédagogique* garde le **meilleur rappel urgent** (0,400). Comparaison
  historique sur n=80 ; le prototype retient le vote pondéré par incertitude
  (`EMOLEARN_SER_METHODE` pour le SER).

**Limite résiduelle assumée.** La Confusion n'est pas couverte par la voix (modèle 4 classes) →
elle repose sur le visage. Prochain levier : un SER multi-classes incluant Confusion/surprise.

---

## 2. Mapping IEMOCAP → DAiSEE : validité de construit

**Tension.** IEMOCAP annote des **émotions conversationnelles** (joie, colère, neutre…) ;
DAiSEE mesure l'**engagement d'apprentissage** (ennui, confusion…). Poser `colère → Frustration`
ou `surprise → Confusion` est une **hypothèse théorique**, pas une équivalence démontrée.

**Lecture assumée.** Le mapping est présenté comme une **interprétation de construit**, hypothèse
de travail à valider (niveau Delphi L1 du protocole). Il est documenté et traçable
(`data/mapping_iemocap_daisee.csv`, `docs/IEMOCAP.md` §3).

**Analyse de sensibilité (testée).** `scripts/sensibilite_mapping.py` mesure l'impact de
ré-interprétations plausibles : modifier `neu → Engagement` change **17 %** des états,
`sad → Frustration` **11 %**, `sur → Engagement` **1 %**. Le mapping a donc une **réelle
influence** sur certaines correspondances (neutre, tristesse) et une influence marginale sur
d'autres — ce qui **objective** où la validation experte est prioritaire. Sortie :
`results/sensibilite_mapping.json`.

---

## 3. Système adaptatif vs prototype de simulation

**Tension.** Le projet se présente comme un système d'apprentissage adaptatif émotionnel, mais
l'engagement provient de **scores DAiSEE pré-annotés**, pas d'une détection live d'un apprenant.

**Lecture assumée.** EmoLearn est explicitement un **prototype de simulation (OS5)**. Aucune
revendication de captation temps réel d'apprenants MOOC. Le mode FER image permet néanmoins
une détection réelle ponctuelle.

---

## 4. « Multimodal » par appariement hétérogène

**Tension.** DAiSEE n'a pas d'audio d'apprentissage ; la voix est donc importée d'IEMOCAP
(autre population, autre contexte), via un appariement par état — visage et voix **ne sont pas
la même personne** dans le pipeline produit.

**Lecture assumée + mitigation.** C'est documenté. Le **sous-corpus audiovisuel aligné**
(Option A) lève cette limite **pour la validation** : visage et voix y proviennent du **même
locuteur, synchronisés**. C'est le seul endroit où le multimodal est *réellement* multimodal.

---

## 5. Cœur pédagogique non encore validé par des experts

**Tension.** Le mapping **engagement → type de contenu → template (A1–D5)** est le cœur
scientifique, mais il est **proposé** par conception, pas encore validé empiriquement
(niveaux L1/L2 du protocole = à exécuter).

**Lecture assumée.** Les 20 templates sont présentés comme une **proposition argumentée**, avec
un protocole de validation experte prêt (`docs/PROTOCOLE_EVALUATION.md`, fiches dans `results/`).

---

## 6. Qualité du contenu LLM

**Tension.** Le contenu généré (Groq) n'est pas comparé à une référence pédagogique « gold » ;
les métriques actuelles (`scripts/metrics_contenu.py`) portent sur la **forme** (longueur,
lisibilité, répétitions), pas sur la pertinence de fond.

**Lecture assumée.** Évaluation de fond renvoyée à la validation humaine (L2) et, en complément,
à des métriques de similarité (BLEU/ROUGE/BERTScore) **si** une référence est définie
(`scripts/metrics_llm_similarity.py` + `data/references_contenu.csv`).

---

## 7. Largeur vs profondeur

**Tension.** Le projet couvre beaucoup (FER, SER, fusion, temporel, RAG, LLM, séquences,
sessions, banc d'essai). Risque de perception « large mais peu profond ».

**Lecture assumée.** Contribution centrale revendiquée : **une architecture multimodale de bout
en bout + son évaluation honnête** (banc d'essai aligné). Les autres composants sont des
**briques de support**, pas des contributions revendiquées au même niveau.

---

## Synthèse

| # | Tension | Statut |
|---|---------|--------|
| 1 | Fusion vs unimodal | **Résolu** : compromis mesuré (rappel urgents 0,40 > unimodal) |
| 2 | Mapping émotions→engagement | Hypothèse + **analyse de sensibilité** (jusqu'à 17 %) |
| 3 | Adaptatif vs simulation | Cadré comme prototype (OS5) |
| 4 | Appariement hétérogène | Documenté + mitigé (Option A) |
| 5 | Templates non validés | Proposition + protocole + formulaires L1/L2 prêts |
| 6 | Qualité LLM | Forme mesurée ; fond → validation humaine |
| 7 | Largeur vs profondeur | Contribution centrale clarifiée |

**Message clé pour la soutenance.** La force d'EmoLearn n'est pas un score élevé, mais une
**chaîne complète, cohérente et critiquée**, où chaque limite est identifiée et, lorsque
possible, **testée** plutôt que dissimulée.
