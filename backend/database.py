import sqlite3
import os

# Chemin de la base de données
DB_PATH = os.path.join(os.path.dirname(__file__), 'emolearn.db')

def initialiser_db():
    """Créer les tables de la base de données"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Table des sessions d'apprentissage
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            scenario_id INTEGER,
            emotion TEXT,
            niveau INTEGER,
            strategie TEXT,
            type_contenu TEXT,
            contenu TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Table des scénarios DAiSEE
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS scenarios (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            clip_id TEXT,
            boredom INTEGER,
            confusion INTEGER,
            engagement INTEGER,
            frustration INTEGER,
            pattern TEXT
        )
    ''')
    
    conn.commit()
    conn.close()
    print("Base de données initialisée avec succès !")

def sauvegarder_session(scenario_id, emotion, niveau, strategie, type_contenu, contenu):
    """Sauvegarder une session d'adaptation"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO sessions 
        (scenario_id, emotion, niveau, strategie, type_contenu, contenu)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', (scenario_id, emotion, niveau, strategie, type_contenu, contenu))
    conn.commit()
    conn.close()

def recuperer_sessions():
    """Récupérer toutes les sessions"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM sessions')
    sessions = cursor.fetchall()
    conn.close()
    return sessions

# Initialiser la base de données au démarrage
if __name__ == "__main__":
    initialiser_db()