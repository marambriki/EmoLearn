# ============================================
# backend/adaptation.py
# ============================================

def selectionner_strategie(boredom, confusion, engagement, frustration):
    if confusion >= 3 and frustration >= 2:
        return ("confusion+frustration", 3, "Aide de déblocage immédiate", "A+D")
    if engagement == 0:
        return ("engagement", 0, "Réengagement urgent", "B")
    if frustration >= 2:
        return ("frustration", frustration, "Encouragement + simplification", "D+A")
    if confusion >= 2:
        return ("confusion", confusion, "Explication alternative", "A")
    if boredom >= 2:
        return ("boredom", boredom, "Exercice avancé", "C")
    if engagement <= 1:
        return ("engagement", engagement, "Exemple concret stimulant", "B")
    if boredom == 1:
        return ("boredom", boredom, "Application pratique", "B")
    if engagement >= 2:
        return ("engagement", engagement, "Renforcement positif", "D")
    return ("neutre", 0, "Continuation normale", "aucune")


def get_description_strategie(type_contenu):
    descriptions = {
        "A"     : "Explication alternative — Reformulation du concept",
        "B"     : "Exemple concret — Cas pratique et application réelle",
        "C"     : "Exercice progressif — Challenge adapté au niveau",
        "D"     : "Encouragement motivationnel — Support émotionnel",
        "A+D"   : "Aide de déblocage — Explication + Encouragement",
        "D+A"   : "Soutien guidé — Encouragement + Simplification",
        "aucune": "Pas d'adaptation nécessaire"
    }
    return descriptions.get(type_contenu, "Type inconnu")