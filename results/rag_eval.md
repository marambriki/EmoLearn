# Évaluation du RAG — qualité du retrieval

> Généré le 2026-09-08 10:33 UTC · mode = `realiste` · k = 3

## Métriques globales

| Métrique | Valeur | Interprétation |
|---|---|---|
| **Hit Rate @3** | **100%** | Au moins 1 bon document trouvé |
| **Precision @3** | 75% | Proportion de documents pertinents |
| **MRR** | 1.000 | Rang moyen du 1er bon document (1.0 = idéal) |
| **Score top-1 moyen** | 0.661 | Similarité cosinus moyenne |

## Par concept

| Concept | Questions | Hit Rate | Precision | MRR |
|---|---|---|---|---|
| algorithmes de tri | 4 | 100% | 92% | 1.000 |
| complexité algorithmique | 3 | 100% | 56% | 1.000 |
| récursion | 3 | 100% | 33% | 1.000 |
| réseaux de neurones | 3 | 100% | 100% | 1.000 |
| structures de données | 3 | 100% | 67% | 1.000 |
| encouragement | 3 | 100% | 100% | 1.000 |

## Détail par question

| Question | Concept attendu | Hit | Prec | RR | Score top-1 |
|---|---|---|---|---|---|
| Comment fonctionne le tri à bulles ? | algorithmes de tri | ✅ | 1.00 | 1.00 | 0.692 |
| Quelle est la différence entre QuickSort et MergeSort ? | algorithmes de tri | ✅ | 1.00 | 1.00 | 0.573 |
| Explique le tri par insertion | algorithmes de tri | ✅ | 0.67 | 1.00 | 0.727 |
| Quel algorithme de tri est le plus rapide ? | algorithmes de tri | ✅ | 1.00 | 1.00 | 0.749 |
| C'est quoi la notation Big-O ? | complexité algorithmique | ✅ | 0.67 | 1.00 | 0.781 |
| Comment mesurer la complexité d'un algorithme ? | complexité algorithmique | ✅ | 0.67 | 1.00 | 0.769 |
| Quelle est la complexité d'une recherche dichotomique ? | complexité algorithmique | ✅ | 0.33 | 1.00 | 0.702 |
| C'est quoi une fonction récursive ? | récursion | ✅ | 0.33 | 1.00 | 0.684 |
| Pourquoi une récursion a besoin d'un cas de base ? | récursion | ✅ | 0.33 | 1.00 | 0.561 |
| Donne un exemple de récursion | récursion | ✅ | 0.33 | 1.00 | 0.405 |
| Comment fonctionne un réseau de neurones ? | réseaux de neurones | ✅ | 1.00 | 1.00 | 0.791 |
| C'est quoi la fonction d'activation ReLU ? | réseaux de neurones | ✅ | 1.00 | 1.00 | 0.718 |
| Explique la rétropropagation dans un réseau de neurones | réseaux de neurones | ✅ | 1.00 | 1.00 | 0.814 |
| Quelles sont les principales structures de données ? | structures de données | ✅ | 0.67 | 1.00 | 0.691 |
| C'est quoi une pile et une file ? | structures de données | ✅ | 0.67 | 1.00 | 0.416 |
| Comment fonctionne un arbre binaire de recherche ? | structures de données | ✅ | 0.67 | 1.00 | 0.631 |
| Je suis frustré, je n'arrive pas à comprendre | encouragement | ✅ | 1.00 | 1.00 | 0.661 |
| Comment rester motivé quand c'est difficile ? | encouragement | ✅ | 1.00 | 1.00 | 0.684 |
| Je m'ennuie, ce cours est trop long | encouragement | ✅ | 1.00 | 1.00 | 0.519 |
