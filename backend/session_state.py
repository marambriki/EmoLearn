# ============================================
# État de session — historique multi-clips (OS1)
# ============================================

import uuid
from datetime import datetime, timezone

from temporal_rules import evaluer_declenchement_temporel

_sessions = {}


class SessionState:
    """Mémorise les clips d'une session d'apprentissage simulée."""

    def __init__(self, sequence_id=None, pattern=None):
        self.session_id = str(uuid.uuid4())[:8]
        self.sequence_id = sequence_id
        self.pattern = pattern
        self.clips = []
        self.adaptations = []
        self.created_at = datetime.now(timezone.utc).isoformat()

    def ajouter_clip(self, scenario_id, clip_id, scores, pattern="stable"):
        clip = {
            "scenario_id": int(scenario_id),
            "clip_id": str(clip_id),
            "boredom": int(scores["boredom"]),
            "confusion": int(scores["confusion"]),
            "engagement": int(scores["engagement"]),
            "frustration": int(scores["frustration"]),
            "pattern": str(pattern),
            "index": len(self.clips),
        }
        self.clips.append(clip)
        temporal = evaluer_declenchement_temporel(self.clips)
        return clip, temporal

    def enregistrer_adaptation(self, clip_index, decision, contenu=None):
        self.adaptations.append(
            {
                "clip_index": clip_index,
                "template_id": decision.get("template_id"),
                "strategie": decision.get("strategie"),
                "declenchement": decision.get("temporal"),
                "contenu_preview": (contenu or "")[:120],
            }
        )

    def to_dict(self):
        temporal = evaluer_declenchement_temporel(self.clips)
        return {
            "session_id": self.session_id,
            "sequence_id": self.sequence_id,
            "pattern": self.pattern,
            "nb_clips": len(self.clips),
            "clips": self.clips,
            "adaptations": self.adaptations,
            "temporal": temporal,
            "created_at": self.created_at,
        }


def creer_session(sequence_id=None, pattern=None):
    session = SessionState(sequence_id=sequence_id, pattern=pattern)
    _sessions[session.session_id] = session
    return session


def get_session(session_id):
    return _sessions.get(session_id)


def lister_sessions_actives():
    return [s.to_dict() for s in _sessions.values()]
