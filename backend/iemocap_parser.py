# ============================================
# backend/iemocap_parser.py
# Parser IEMOCAP réel — lit les fichiers .txt
# ============================================

import os
import re
import pandas as pd

from iemocap_audio import enrich_row_wav_paths

# ===== CHEMIN IEMOCAP =====
IEMOCAP_PATH = os.path.join(
    os.path.dirname(__file__),
    '../data/IEMOCAP_full_release'
)

# ===== MAPPING ÉMOTIONS IEMOCAP → DAISEE =====
mapping_iemocap_daisee = {
    'neu' : {'etat': 'Boredom',     'niveau': 1},
    'fru' : {'etat': 'Frustration', 'niveau': 3},
    'ang' : {'etat': 'Frustration', 'niveau': 3},
    'sad' : {'etat': 'Boredom',     'niveau': 2},
    'hap' : {'etat': 'Engagement',  'niveau': 3},
    'exc' : {'etat': 'Engagement',  'niveau': 3},
    'sur' : {'etat': 'Confusion',   'niveau': 2},
    'dis' : {'etat': 'Frustration', 'niveau': 2},
    'fea' : {'etat': 'Confusion',   'niveau': 3},
    'xxx' : {'etat': 'Boredom',     'niveau': 1}
}

def parser_fichier_emo(filepath):
    """
    Parse un fichier EmoEvaluation IEMOCAP
    Exemple ligne :
    [6.2901 - 8.2357] Ses01F_impro01_F000 neu [2.5000, 2.5000, 2.5000]
    """
    utterances = []

    with open(filepath, 'r', encoding='utf-8') as f:
        contenu = f.read()

    # Pattern pour extraire les utterances
    pattern = r'\[(\d+\.\d+)\s*-\s*(\d+\.\d+)\]\s+(\S+)\s+(\w+)\s+\[(\d+\.\d+),\s*(\d+\.\d+),\s*(\d+\.\d+)\]'

    matches = re.findall(pattern, contenu)

    for match in matches:
        start_time  = float(match[0])
        end_time    = float(match[1])
        utterance_id = match[2]
        emotion      = match[3]
        valence      = float(match[4])
        activation   = float(match[5])
        dominance    = float(match[6])

        # Mapper vers DAiSEE
        etat_daisee = mapping_iemocap_daisee.get(
            emotion,
            {'etat': 'Boredom', 'niveau': 1}
        )

        row = {
            'utterance_id': utterance_id,
            'start_time'  : start_time,
            'end_time'    : end_time,
            'duree'       : round(end_time - start_time, 2),
            'emotion'     : emotion,
            'valence'     : valence,
            'activation'  : activation,
            'dominance'   : dominance,
            'etat_daisee' : etat_daisee['etat'],
            'niveau_daisee': etat_daisee['niveau'],
            'fichier'     : os.path.basename(filepath)
        }
        utterances.append(enrich_row_wav_paths(row))

    return utterances


def charger_iemocap_complet(sessions=[1, 2, 3, 4, 5]):
    """
    Charge toutes les annotations IEMOCAP
    depuis les 5 sessions
    """
    print(f"Chargement IEMOCAP sessions {sessions}...")
    toutes_utterances = []

    for session_num in sessions:
        session_path = os.path.join(
            IEMOCAP_PATH,
            f'Session{session_num}',
            'dialog',
            'EmoEvaluation'
        )

        if not os.path.exists(session_path):
            print(f"  ⚠️ Session{session_num} non trouvée")
            continue

        # Lire tous les fichiers .txt
        fichiers = [f for f in os.listdir(session_path)
                    if f.endswith('.txt')]

        for fichier in fichiers:
            filepath = os.path.join(session_path, fichier)
            utterances = parser_fichier_emo(filepath)
            toutes_utterances.extend(utterances)

        print(f"  ✅ Session{session_num} : {len(fichiers)} fichiers")

    df = pd.DataFrame(toutes_utterances)
    print(f"\n✅ Total IEMOCAP chargé : {len(df)} utterances")
    return df


def analyser_distribution(df):
    """
    Analyse la distribution des émotions
    """
    print("\n=== DISTRIBUTION ÉMOTIONS IEMOCAP ===\n")
    print("Émotions originales :")
    print(df['emotion'].value_counts())

    print("\nÉtats DAiSEE mappés :")
    print(df['etat_daisee'].value_counts())

    print(f"\nStatistiques dimensionnelles :")
    print(f"  Valence moyenne    : {df['valence'].mean():.2f}")
    print(f"  Activation moyenne : {df['activation'].mean():.2f}")
    print(f"  Dominance moyenne  : {df['dominance'].mean():.2f}")

    return df


def sauvegarder_iemocap_csv(df, path=None):
    """
    Sauvegarde le dataset parsé en CSV
    """
    if path is None:
        path = os.path.join(
            os.path.dirname(__file__),
            '../data/iemocap_parsed.csv'
        )

    df.to_csv(path, index=False)
    print(f"✅ IEMOCAP sauvegardé : {path}")
    return path


def get_utterance_par_emotion(df, emotion_daisee, n=10):
    """
    Retourne n utterances d'une émotion DAiSEE spécifique
    """
    subset = df[df['etat_daisee'] == emotion_daisee]
    return subset.sample(min(n, len(subset)))


print("✅ Parser IEMOCAP chargé !")