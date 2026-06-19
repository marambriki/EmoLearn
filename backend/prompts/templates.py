# ============================================
# EmoLearn — 20 templates de prompts (OS3 / C5)
# Alignés sur le plan de mémoire (A1–A5, B1–B5, C1–C5, D1–D5)
# ============================================

TEMPLATE_META = {
  "A1": {"type": "A", "longueur": "200 mots", "format": "Paragraphe avec étapes", "ton": "Encourageant et rassurant", "objectif": "Clarifier"},
  "A2": {"type": "A", "longueur": "150 mots", "format": "Micro-explication décomposée", "ton": "Patient et décomposé", "objectif": "Clarifier"},
  "A3": {"type": "A+D", "longueur": "200 mots", "format": "Aide de déblocage", "ton": "Empathique et déculpabilisant", "objectif": "Débloquer"},
  "A4": {"type": "A", "longueur": "250 mots", "format": "Explication de fond", "ton": "Constructif et orienté stratégie", "objectif": "Clarifier"},
  "A5": {"type": "A", "longueur": "200 mots", "format": "Explication narrative", "ton": "Pédagogue et créatif", "objectif": "Clarifier"},
  "B1": {"type": "B", "longueur": "200 mots", "format": "Paragraphe engageant", "ton": "Dynamique et engageant", "objectif": "Stimuler"},
  "B2": {"type": "B", "longueur": "250 mots", "format": "Cas pratique", "ton": "Stimulant et provocateur de curiosité", "objectif": "Réengager"},
  "B3": {"type": "B", "longueur": "200 mots", "format": "Mise en œuvre pratique", "ton": "Pratique et orienté action", "objectif": "Stimuler"},
  "B4": {"type": "B", "longueur": "250 mots", "format": "Application professionnelle", "ton": "Motivant et orienté carrière", "objectif": "Contextualiser"},
  "B5": {"type": "B", "longueur": "200 mots", "format": "Exemple interdisciplinaire", "ton": "Créatif et décloisonnant", "objectif": "Stimuler"},
  "C1": {"type": "C", "longueur": "250 mots", "format": "Exercice structuré", "ton": "Challengeant et dynamique", "objectif": "Approfondir"},
  "C2": {"type": "C", "longueur": "200 mots", "format": "Exercice progressif", "ton": "Encourageant et progressif", "objectif": "Consolider"},
  "C3": {"type": "C", "longueur": "200 mots", "format": "Exercice créatif", "ton": "Ouvert et valorisant", "objectif": "Stimuler"},
  "C4": {"type": "C", "longueur": "250 mots", "format": "Mini-projet", "ton": "Ambitieux et structurant", "objectif": "Approfondir"},
  "C5": {"type": "C", "longueur": "200 mots", "format": "Exercice comparatif", "ton": "Analytique et stimulant", "objectif": "Approfondir"},
  "D1": {"type": "D", "longueur": "150 mots", "format": "Message motivant", "ton": "Empathique et encourageant", "objectif": "Encourager"},
  "D2": {"type": "D", "longueur": "180 mots", "format": "Message de soutien", "ton": "Bienveillant et déstressant", "objectif": "Encourager"},
  "D3": {"type": "D", "longueur": "200 mots", "format": "Message remotivant", "ton": "Empowering et re-centrant", "objectif": "Réengager"},
  "D4": {"type": "D", "longueur": "150 mots", "format": "Renforcement positif", "ton": "Positif et valorisant", "objectif": "Renforcer"},
  "D5": {"type": "D", "longueur": "150 mots", "format": "Message de félicitation", "ton": "Fier et confiant", "objectif": "Renforcer"},
}


def lister_templates():
    return list(TEMPLATE_META.keys())


def get_meta_template(template_id):
    return TEMPLATE_META.get(template_id, TEMPLATE_META["A1"])


def construire_instructions_template(
    template_id,
    concept,
    niveau,
    duree=2,
    module="Informatique",
    frustration_niveau=None,
):
    """
    Retourne les instructions spécifiques du template (corps du prompt utilisateur).
    """
    fr = frustration_niveau if frustration_niveau is not None else niveau

    builders = {
        "A1": lambda: (
            f"L'apprenant montre une confusion de niveau {niveau} sur le concept de {concept} "
            f"depuis {duree} clips ({duree * 10} secondes). Génère une explication alternative en 200 mots qui :\n"
            "1. Reformule le concept avec des mots plus simples\n"
            "2. Utilise une analogie concrète de la vie quotidienne\n"
            "3. Décompose en 3 étapes claires\n"
            "4. Termine par une question de vérification de compréhension\n"
            "Ton encourageant et rassurant."
        ),
        "A2": lambda: (
            f"L'apprenant est en confusion intense (niveau {niveau}) sur {concept} depuis {duree} clips. "
            "Génère une micro-explication de 150 mots qui :\n"
            "1. Part d'un exemple concret très simple\n"
            "2. Construit progressivement vers le concept abstrait\n"
            "3. Utilise des métaphores visuelles\n"
            "4. Propose un schéma textuel simple (ASCII art acceptable)\n"
            "Ton très simple, patient et décomposé."
        ),
        "A3": lambda: (
            f"L'apprenant combine confusion (niveau {niveau}) et frustration (niveau {fr}) sur {concept}. "
            "Génère une aide de déblocage en 200 mots qui :\n"
            "1. Commence par une phrase rassurante (\"Ce concept est complexe, c'est normal\")\n"
            "2. Identifie LE point de blocage probable\n"
            "3. Propose une approche radicalement différente (visuelle/narrative)\n"
            "4. Termine par un mini-exercice guidé très simple\n"
            "Ton empathique et déculpabilisant."
        ),
        "A4": lambda: (
            f"L'apprenant a montré plusieurs épisodes de confusion sur des concepts similaires à {concept}. "
            "Génère une explication de fond en 250 mots qui :\n"
            "1. Adresse le concept fondamental sous-jacent\n"
            "2. Crée des ponts avec ce qui a déjà été compris\n"
            "3. Propose une représentation mentale unificatrice\n"
            "4. Suggère une stratégie d'apprentissage pour ce type de concept\n"
            "Ton constructif et orienté stratégie."
        ),
        "A5": lambda: (
            f"L'apprenant est confus (niveau {niveau}) sur un concept particulièrement abstrait : {concept}. "
            "Génère une explication en 200 mots qui :\n"
            "1. Utilise une analogie physique/tangible\n"
            "2. Crée un mini-scénario narratif illustratif\n"
            "3. Alterne entre concret et abstrait graduellement\n"
            "4. Termine par une reformulation simple du concept\n"
            "Ton pédagogue et créatif."
        ),
        "B1": lambda: (
            f"L'apprenant montre de l'ennui (niveau {niveau}) sur le module {module}. "
            f"Génère un exemple d'application concrète en 200 mots qui :\n"
            f"1. Part d'une situation réelle et actuelle\n"
            f"2. Montre comment {concept} résout un problème concret\n"
            "3. Inclut des données/chiffres réels si possible\n"
            "4. Termine par \"Et dans votre quotidien, où pourriez-vous utiliser cela ?\"\n"
            "Ton dynamique et engageant."
        ),
        "B2": lambda: (
            f"L'apprenant a un engagement très faible (niveau 0-1) depuis {duree} clips. "
            f"Génère un cas pratique captivant en 250 mots qui :\n"
            f"1. Commence par une question intrigante ou un fait surprenant\n"
            f"2. Présente un cas d'usage inattendu de {concept}\n"
            "3. Montre l'impact réel/les enjeux\n"
            "4. Propose une mini-investigation (\"Essayez de prédire...\")\n"
            "Ton stimulant et provocateur de curiosité."
        ),
        "B3": lambda: (
            f"L'apprenant s'ennuie (niveau {niveau}) après une section très théorique sur {concept}. "
            "Génère un exemple pratique de mise en œuvre en 200 mots qui :\n"
            "1. \"Passons à la pratique !\" - transition claire\n"
            "2. Décrit un mini-projet concret étape par étape\n"
            "3. Montre le résultat attendu\n"
            "4. Suggère une variation à explorer\n"
            "Ton pratique et orienté action."
        ),
        "B4": lambda: (
            f"L'apprenant a besoin de voir la pertinence de {concept}. "
            "Génère un exemple d'application réelle en 250 mots qui :\n"
            "1. Choisit un domaine d'application actuel (IA, santé, environnement...)\n"
            f"2. Explique comment {concept} est crucial dans ce contexte\n"
            "3. Cite une entreprise/projet réel qui l'utilise\n"
            "4. Montre les compétences professionnelles associées\n"
            "Ton motivant et orienté carrière."
        ),
        "B5": lambda: (
            f"L'apprenant pourrait bénéficier d'un exemple interdisciplinaire sur {concept}. "
            "Génère un exemple en 200 mots qui :\n"
            f"1. Connecte {concept} à un domaine différent (art, sport, cuisine, musique...)\n"
            "2. Montre la transversalité du concept\n"
            "3. Utilise une comparaison inattendue\n"
            "4. Ouvre sur d'autres applications possibles\n"
            "Ton créatif et décloisonnant."
        ),
        "C1": lambda: (
            f"L'apprenant s'ennuie (niveau {niveau}) car le contenu semble trop simple sur {concept}. "
            "Génère un exercice de niveau avancé en 250 mots qui :\n"
            f"1. Pose un problème complexe nécessitant {concept}\n"
            "2. Ajoute une contrainte supplémentaire (optimisation, edge case...)\n"
            "3. Demande de justifier la solution\n"
            "4. Propose une extension \"bonus\" encore plus difficile\n"
            "Ton challengeant et respectueux de leur niveau."
        ),
        "C2": lambda: (
            f"L'apprenant est bien engagé mais pourrait approfondir {concept}. "
            "Génère un exercice de consolidation en 200 mots qui :\n"
            f"1. Combine {concept} avec un concept vu précédemment\n"
            "2. Présente un problème en 3 étapes de difficulté croissante\n"
            "3. Fournit des indices progressifs\n"
            "4. Propose une auto-évaluation\n"
            "Ton encourageant et progressif."
        ),
        "C3": lambda: (
            f"L'apprenant a bien compris {concept} mais manque de stimulation. "
            "Génère un exercice créatif en 200 mots qui :\n"
            "1. \"Imaginez que vous devez...\" (mise en situation ouverte)\n"
            f"2. Demande d'appliquer {concept} à un contexte original\n"
            "3. Valorise plusieurs solutions possibles\n"
            "4. Encourage la créativité dans l'approche\n"
            "Ton ouvert et valorisant de la créativité."
        ),
        "C4": lambda: (
            f"L'apprenant pourrait bénéficier d'un défi plus conséquent sur {concept}. "
            "Génère un mini-projet en 250 mots qui :\n"
            "1. Présente un objectif concret à atteindre\n"
            "2. Liste les étapes de réalisation (4-5 étapes)\n"
            "3. Mentionne les concepts à mobiliser\n"
            "4. Décrit le livrable attendu\n"
            "Ton ambitieux et structurant."
        ),
        "C5": lambda: (
            f"L'apprenant est prêt pour un exercice avancé sur {concept}. "
            "Génère un exercice de comparaison en 200 mots qui :\n"
            "1. Présente 2-3 approches différentes pour résoudre un problème\n"
            "2. Demande d'identifier forces/faiblesses de chacune\n"
            "3. Demande de justifier un choix selon le contexte\n"
            "4. Encourage la pensée critique\n"
            "Ton analytique et stimulant intellectuellement."
        ),
        "D1": lambda: (
            f"L'apprenant montre de la frustration (niveau {niveau}) sur {concept}. "
            "Génère un message d'encouragement en 150 mots qui :\n"
            "1. Valide la difficulté (\"Ce concept challenge beaucoup d'apprenants\")\n"
            "2. Rappelle les progrès déjà réalisés dans le cours\n"
            "3. Propose une perspective (\"Dans 2 modules, vous verrez pourquoi...\")\n"
            "4. Termine par une phrase motivante\n"
            "Ton empathique et encourageant."
        ),
        "D2": lambda: (
            f"L'apprenant est très frustré (niveau {niveau}) sur {concept}. "
            "Génère un message de soutien en 180 mots qui :\n"
            "1. \"Prenez une pause si nécessaire\" - déculpabilisation\n"
            "2. Normalise la difficulté (\"beaucoup d'apprenants trouvent ce passage difficile\")\n"
            "3. Propose une stratégie concrète de déblocage\n"
            "4. Rappelle que la persévérance est plus importante que la vitesse\n"
            "Ton bienveillant et déstressant."
        ),
        "D3": lambda: (
            f"L'apprenant semble découragé (engagement faible depuis {duree} clips) sur {concept}. "
            "Génère un message remotivant en 200 mots qui :\n"
            "1. Pose la question \"Qu'est-ce qui vous a amené à ce cours initialement ?\"\n"
            "2. Reconnecte avec les objectifs personnels\n"
            "3. Montre le chemin parcouru (rappel des acquis)\n"
            "4. Propose de se fixer un micro-objectif immédiat\n"
            "Ton empowering et re-centrant."
        ),
        "D4": lambda: (
            f"L'apprenant est bien engagé (niveau {niveau}) sur {concept}. "
            "Génère un renforcement positif en 150 mots qui :\n"
            "1. Félicite l'engagement et la persévérance\n"
            "2. Souligne un acquis spécifique récent\n"
            "3. Anticipe positivement la suite\n"
            "4. Encourage à maintenir cette dynamique\n"
            "Ton positif et valorisant."
        ),
        "D5": lambda: (
            f"L'apprenant a surmonté un passage difficile sur {concept}. "
            "Génère un message de félicitation en 150 mots qui :\n"
            "1. Célèbre la réussite (\"Vous avez réussi à...\")\n"
            "2. Explicite la compétence développée\n"
            "3. Fait le lien avec les prochaines étapes\n"
            "4. Renforce la confiance en soi\n"
            "Ton fier et confiant."
        ),
    }

    builder = builders.get(template_id, builders["A1"])
    return builder()
