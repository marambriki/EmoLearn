# Analyse des correspondances (OS6)

n = 50 scénarios joints (labels DAiSEE × décisions runtime).

## Taux
- Famille GT (règles sur scores DAiSEE) vs famille runtime : **98.0%**
- Patron GT vs patron runtime : **82.0%**
- État dominant GT vs état fusionné : **46.0%**
- Famille canonique (état fusionné → A/C/D) : **44.0%**
- Famille admissible pour l'etat fusionne : **56.0%**
- Famille si cascade one-hot sur l'etat fusionne vs journal : **22.0%**

## Cramér V
- État fusionné × famille observée : 0.412
- Famille GT × famille observée : 0.965
- État GT × état fusionné : 0.437

## Contingence état fusionné × famille
famille_observee  A  B  C   D
etat_fusion                  
Boredom           5  2  2   3
Confusion         4  0  0   2
Engagement        0  3  0   6
Frustration       1  3  9  10

Patrons distincts observés : 11/20. Non déclenchés : A4, B3, B4, B5, C3, C4, C5, D3, D5.

Si la famille GT (cascade sur labels DAiSEE) coincide avec la famille journalisee, le referentiel de 50 scenarios documente le moteur de decision sur les scores annotes, pas la cascade appliquee a l'etat fusionne. La concordance admissible et le contre-factuel one-hot mesurent alors l'ecart entre etat detecte et type effectivement servi.

Généré : 2026-09-27T20:42:57.673814+00:00
