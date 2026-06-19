# ============================================
# backend/rag_module.py
# Module RAG — Retrieval Augmented Generation (lazy load)
# ============================================

import os

_embedding_model = None


def get_embedding_model():
    """Charge le modèle d'embeddings uniquement à la première utilisation RAG."""
    global _embedding_model
    if _embedding_model is None:
        from sentence_transformers import SentenceTransformer

        print("Chargement modèle d'embeddings RAG...")
        _embedding_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
        print("✅ Modèle d'embeddings RAG chargé")
    return _embedding_model


def encode_texts(texts):
    return get_embedding_model().encode(texts).tolist()

# ===== BASE DE CONNAISSANCES PÉDAGOGIQUES =====
# Documents de référence pour chaque concept
base_connaissances = [

    # ===== ALGORITHMIQUE =====
    {
        "id"     : "algo_tri_1",
        "concept": "algorithmes de tri",
        "type"   : "definition",
        "contenu": """Un algorithme de tri est une procédure qui réorganise 
        les éléments d'une liste selon un ordre défini (croissant, décroissant).
        Les algorithmes de tri principaux sont : le tri à bulles (O(n²)),
        le tri par insertion (O(n²)), le tri rapide QuickSort (O(n log n)),
        le tri fusion MergeSort (O(n log n)).""",
        "source" : "Cours Algorithmique M1"
    },
    {
        "id"     : "algo_tri_2",
        "concept": "algorithmes de tri",
        "type"   : "exemple",
        "contenu": """Exemple tri à bulles en Python :
        def tri_bulles(liste):
            n = len(liste)
            for i in range(n):
                for j in range(0, n-i-1):
                    if liste[j] > liste[j+1]:
                        liste[j], liste[j+1] = liste[j+1], liste[j]
            return liste
        Exemple : [5,3,1,4,2] → [1,2,3,4,5]""",
        "source" : "TD Algorithmique"
    },
    {
        "id"     : "algo_tri_3",
        "concept": "algorithmes de tri",
        "type"   : "exercice",
        "contenu": """Exercice avancé : Implémenter QuickSort avec pivot médian.
        Données : tableau de 10000 éléments aléatoires.
        Contrainte : complexité O(n log n) garantie.
        Comparer les performances avec MergeSort sur grands tableaux.""",
        "source" : "TP Algorithmique"
    },

    # ===== RÉSEAUX DE NEURONES =====
    {
        "id"     : "nn_1",
        "concept": "réseaux de neurones",
        "type"   : "definition",
        "contenu": """Un réseau de neurones artificiel est un modèle 
        computationnel inspiré du cerveau humain. Il est composé de :
        - Couche d'entrée : reçoit les données
        - Couches cachées : transformations non-linéaires
        - Couche de sortie : produit le résultat
        Chaque neurone applique : y = f(Wx + b)
        où W=poids, x=entrée, b=biais, f=fonction d'activation""",
        "source" : "Cours Deep Learning"
    },
    {
        "id"     : "nn_2",
        "concept": "réseaux de neurones",
        "type"   : "analogie",
        "contenu": """Analogie : Un réseau de neurones ressemble à une chaîne 
        de décision humaine. Comme un médecin qui analyse des symptômes :
        1. Il observe les données (symptômes = entrées)
        2. Il applique son expérience (couches cachées = apprentissage)
        3. Il donne un diagnostic (sortie = prédiction)
        Plus il voit de cas, plus il devient précis (entraînement).""",
        "source" : "Cours IA pour débutants"
    },

    # ===== STRUCTURES DE DONNÉES =====
    {
        "id"     : "struct_1",
        "concept": "structures de données",
        "type"   : "definition",
        "contenu": """Les structures de données principales sont :
        - Liste chaînée : éléments liés par pointeurs, insertion O(1)
        - Tableau : accès direct O(1), insertion O(n)
        - Pile (Stack) : LIFO, push/pop O(1)
        - File (Queue) : FIFO, enqueue/dequeue O(1)
        - Arbre binaire : recherche O(log n)
        - Table de hachage : accès moyen O(1)""",
        "source" : "Cours Structures de Données"
    },
    {
        "id"     : "struct_2",
        "concept": "structures de données",
        "type"   : "exemple",
        "contenu": """Exemple d'utilisation pratique :
        - Pile → historique navigateur web (retour arrière)
        - File → file d'attente imprimante
        - Arbre → système de fichiers (dossiers/sous-dossiers)
        - Table de hachage → dictionnaire Python, annuaire téléphonique
        Ces structures sont utilisées dans tous les logiciels modernes.""",
        "source" : "TD Structures de Données"
    },

    # ===== ENCOURAGEMENTS PÉDAGOGIQUES =====
    {
        "id"     : "encouragement_1",
        "concept": "encouragement",
        "type"   : "motivation",
        "contenu": """Données sur l'apprentissage : 85% des étudiants trouvent 
        les algorithmes difficiles au premier abord. La persévérance est 
        plus importante que les aptitudes naturelles. Des études montrent 
        que 10h de pratique délibérée suffisent pour maîtriser un concept 
        algorithmique de base. Chaque erreur est une opportunité d'apprentissage.""",
        "source" : "Psychologie de l'apprentissage"
    },
    {
        "id"     : "encouragement_2",
        "concept": "encouragement",
        "type"   : "methode",
        "contenu": """Méthodes pour surmonter la frustration en apprentissage :
        1. Décomposer le problème en sous-problèmes plus simples
        2. Chercher des exemples concrets avant la théorie
        3. Pratiquer avec de petits exercices avant les grands
        4. Se reposer 10 minutes puis reprendre
        5. Expliquer le concept à quelqu'un d'autre (technique Feynman)""",
        "source" : "Guide apprentissage efficace"
    }
]

# ===== INITIALISATION CHROMADB =====
def initialiser_rag():
    """
    Initialise la base vectorielle ChromaDB
    avec les documents de la base de connaissances
    """
    print("Initialisation base RAG ChromaDB...")

    import chromadb

    client = chromadb.PersistentClient(
        path=os.path.join(os.path.dirname(__file__), '../data/rag_db')
    )

    # Créer ou récupérer la collection
    collection = client.get_or_create_collection(
        name     = "emolearn_knowledge",
        metadata = {"hnsw:space": "cosine"}
    )

    # Ajouter les documents si la collection est vide
    if collection.count() == 0:
        print("Ajout des documents dans la base RAG...")

        ids      = [doc['id']      for doc in base_connaissances]
        contents = [doc['contenu'] for doc in base_connaissances]
        metadatas = [{
            'concept': doc['concept'],
            'type'   : doc['type'],
            'source' : doc['source']
        } for doc in base_connaissances]

        # Générer les embeddings
        embeddings = encode_texts(contents)

        collection.add(
            ids        = ids,
            documents  = contents,
            embeddings = embeddings,
            metadatas  = metadatas
        )
        print(f"✅ {len(base_connaissances)} documents ajoutés dans la base RAG !")
    else:
        print(f"✅ Base RAG existante : {collection.count()} documents")

    return collection

# ===== RECHERCHE DE DOCUMENTS PERTINENTS =====
def rechercher_documents(query, concept, etat, n_results=3):
    """
    Recherche les documents pertinents dans la base RAG
    Si le concept n'existe pas → génère contexte dynamique
    """
    collection = initialiser_rag()

    # Query enrichie
    query_enrichie  = f"{query} {concept} apprenant {etat}"
    query_embedding = encode_texts([query_enrichie])

    # Recherche dans ChromaDB
    results = collection.query(
        query_embeddings = query_embedding,
        n_results        = min(n_results, collection.count())
    )

    documents = []
    for i in range(len(results['documents'][0])):
        score = 1 - results['distances'][0][i]
        # ⚠️ Seuil de pertinence — si score < 0.3 → document non pertinent
        if score >= 0.3:
            documents.append({
                'contenu' : results['documents'][0][i],
                'metadata': results['metadatas'][0][i],
                'score'   : score
            })

    return documents


def construire_contexte_rag(documents, concept=None, etat=None):
    """
    Formate les documents pour le prompt LLM
    Si pas de documents pertinents → contexte générique
    """
    if not documents:
        # Contexte générique basé sur le concept
        return f"""=== CONTEXTE PÉDAGOGIQUE ===

Le concept enseigné est : {concept}
État émotionnel de l'apprenant : {etat}

Note : Génère un contenu pédagogique adapté à ce concept spécifique.
Utilise des exemples concrets et précis liés à ce domaine.
Ne pas inventer de faits non vérifiables.

=== FIN DU CONTEXTE ===\n\n"""

    contexte = "=== CONTEXTE DE RÉFÉRENCE VÉRIFIÉ ===\n\n"
    for i, doc in enumerate(documents, 1):
        contexte += f"[Document {i}] "
        contexte += f"Type: {doc['metadata']['type']} | "
        contexte += f"Source: {doc['metadata']['source']}\n"
        contexte += f"{doc['contenu']}\n\n"
    contexte += "=== FIN DU CONTEXTE ===\n\n"

    return contexte


def ajouter_document(concept, type_doc, contenu, source):
    """
    Ajoute un nouveau document dans la base RAG
    Utile pour enrichir la base au fil du temps
    """
    collection = initialiser_rag()

    doc_id    = f"{concept}_{type_doc}_{collection.count() + 1}"
    embedding = encode_texts([contenu])

    collection.add(
        ids        = [doc_id],
        documents  = [contenu],
        embeddings = embedding,
        metadatas  = [{
            'concept': concept,
            'type'   : type_doc,
            'source' : source
        }]
    )
    print(f"✅ Document ajouté : {doc_id}")
    return doc_id

print("✅ Module RAG chargé !")
