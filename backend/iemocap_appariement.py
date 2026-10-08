# ============================================
# backend/iemocap_appariement.py
# Appariement scénario DAiSEE ↔ utterance IEMOCAP
# par similarité de profil (scores + VAD + acoustique optionnelle)
# ============================================

from __future__ import annotations

import os

import numpy as np
import pandas as pd

ETATS = ("Boredom", "Confusion", "Engagement", "Frustration")

# Dimensions profil : 4 scores DAiSEE + VAD normalisés + durée + 4 dims acoustiques (option)
N_BASE = 8
N_ACOUSTIC = 4
PROFILE_DIM = N_BASE + N_ACOUSTIC

_ACOUSTIC_CACHE_PATH = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "data", "iemocap_acoustic_index.csv")
)
_index_matrix: np.ndarray | None = None
_index_df: pd.DataFrame | None = None
_acoustic_lookup: dict[str, np.ndarray] | None = None


def _soft_etat_vector(etat: str, niveau: float) -> np.ndarray:
    """One-hot DAiSEE pondéré par niveau / 3."""
    w = max(0.0, min(1.0, float(niveau) / 3.0))
    out = np.zeros(4, dtype=np.float64)
    if etat in ETATS:
        out[ETATS.index(etat)] = w
    return out


# ---------------------------------------------------------------------------
# VAD scénario DAiSEE → espace [0, 1] (compatible VAD IEMOCAP : (annot - 1) / 4)
#
# DAiSEE n'a pas de labels VAD (Gupta et al., 2016). Chaîne de dérivation :
#
#   1. Baker et al. (2010), IJHCS — positions QUALITATIVES des états d'apprentissage
#      sur le circumplex de Russell (1989) : valence × activation (Fig. 1).
#   2. Estimation sur [-1, +1] (lecture figure + prose ; pas de table officielle Baker).
#   3. Passage en [0, 1] :  x_01 = (x_russell + 1) / 2
#
#   État          | V [-1,+1] | A [-1,+1] |  → V [0,1]      → A [0,1]
#   --------------|-----------|-----------|----------------|------------
#   Boredom       |   -0,50   |   -0,60   |  (-0,5+1)/2=0,25 | 0,20
#   Confusion     |   -0,30   |   +0,10   |  0,35              | 0,55
#   Engagement    |   +0,50   |    0,00   |  0,75              | 0,50
#   Frustration   |   -0,70   |   +0,60   |  0,15              | 0,80
#
#   Dominance (D) : Baker ne formalise que V/A ; IEMOCAP annote aussi D (PAD).
#   D scénario = 0,5 (neutre) pour tous les états — pas d'hypothèse par état ;
#   dim 7 alignée sur l'annotation IEMOCAP milieu (3 → (3−1)/4 = 0,5).
#
#   Agrégation multi-label : VAD = Σ (score_i/3 × centroïde_i) / Σ (score_i/3)
# ---------------------------------------------------------------------------
_RUSSELL_VA_ESTIMATES = np.array(
    [
        [-0.50, -0.60],  # Boredom
        [-0.30, 0.10],  # Confusion
        [0.50, 0.00],  # Engagement (arousal neutre, Baker 2010)
        [-0.70, 0.60],  # Frustration
    ],
    dtype=np.float64,
)
# D [0, 1] — neutre 0,5 pour B/C/E/F (choix explicite, facile à justifier).
_RUSSELL_D_ESTIMATES = np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float64)


def _centroides_vad_baker_01() -> np.ndarray:
    """Centroïdes [V, A, D] en [0, 1] dérivés de _RUSSELL_VA_ESTIMATES."""
    va_01 = (_RUSSELL_VA_ESTIMATES + 1.0) / 2.0
    return np.column_stack([va_01[:, 0], va_01[:, 1], _RUSSELL_D_ESTIMATES])


_BAKER_VAD_CENTROIDS = _centroides_vad_baker_01()


def _vad_baker_depuis_scores(b: float, c: float, e: float, f: float) -> tuple[float, float, float]:
    """
    VAD scénario = moyenne pondérée des centroïdes B/C/E/F (scores normalisés /3).
    Centroïdes : Russell [-1,1] → [0,1] via (x+1)/2, positions qualitatives Baker (2010).
    """
    weights = np.array([b, c, e, f], dtype=np.float64)
    wsum = float(weights.sum())
    if wsum < 1e-9:
        return 0.5, 0.5, 0.5
    vad = (weights[:, None] * _BAKER_VAD_CENTROIDS).sum(axis=0) / wsum
    return float(vad[0]), float(vad[1]), float(vad[2])


def vecteur_profil_scenario(
    boredom: float,
    confusion: float,
    engagement: float,
    frustration: float,
) -> np.ndarray:
    """
    Profil cible dérivé des scores DAiSEE du scénario (pas seulement l'état dominant).
    VAD = centroïdes Baker/Russell pondérés par B/C/E/F ; proxy acoustique aligné index IEMOCAP.
    """
    b = float(boredom) / 3.0
    c = float(confusion) / 3.0
    e = float(engagement) / 3.0
    f = float(frustration) / 3.0
    valence, activation, dominance = _vad_baker_depuis_scores(b, c, e, f)
    duree_norm = 0.5
    base = np.array(
        [b, c, e, f, valence, activation, dominance, duree_norm],
        dtype=np.float64,
    )
    ac_proxy = np.array(
        [
            activation,
            valence,
            np.clip(c * 0.4 + f * 0.3, 0.0, 1.0),
            np.clip(e * 0.5 + (1.0 - b) * 0.2, 0.0, 1.0),
        ],
        dtype=np.float64,
    )
    return np.concatenate([base, ac_proxy])


def vecteur_profil_iemocap(row, acoustic: np.ndarray | None = None) -> np.ndarray:
    """Profil candidat : soft état + VAD IEMOCAP + durée + acoustique."""
    etat = str(row.get("etat_daisee", "Boredom"))
    niveau = float(row.get("niveau_daisee", row.get("niveau", 2)) or 2)
    soft = _soft_etat_vector(etat, niveau)

    valence = (float(row.get("valence", 2.5)) - 1.0) / 4.0
    activation = (float(row.get("activation", 2.5)) - 1.0) / 4.0
    dominance = (float(row.get("dominance", 2.5)) - 1.0) / 4.0
    duree = float(row.get("duree", 2.0) or 2.0)
    duree_norm = np.clip(np.log1p(duree) / np.log1p(8.0), 0.0, 1.0)

    base = np.concatenate(
        [soft, [valence, activation, dominance, duree_norm]],
    ).astype(np.float64)

    if acoustic is not None and len(acoustic) >= N_ACOUSTIC:
        return np.concatenate([base, acoustic[:N_ACOUSTIC]])
    return np.concatenate([base, np.zeros(N_ACOUSTIC, dtype=np.float64)])


def _cosine_similarity(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Similarité cosinus classique : toutes les dimensions ont le même poids."""
    q = query.astype(np.float64)
    q_norm = np.linalg.norm(q)
    if q_norm < 1e-12:
        q = np.ones_like(q) / len(q)
    else:
        q = q / q_norm

    m = matrix.astype(np.float64)
    m_norms = np.linalg.norm(m, axis=1, keepdims=True)
    m_norms = np.maximum(m_norms, 1e-12)
    m = m / m_norms
    return m @ q


def _filtrer_pool_par_etat(pool: pd.DataFrame, matrix: np.ndarray, etat_cible: str):
    """Restreint aux utterances dont l'état DAiSEE annoté correspond à la cible."""
    if pool.empty or "etat_daisee" not in pool.columns:
        return matrix, pool, False
    mask = pool["etat_daisee"].astype(str) == str(etat_cible)
    if not mask.any():
        return matrix, pool, False
    return matrix[mask.to_numpy()], pool[mask].reset_index(drop=True), True


def _charger_acoustic_lookup() -> dict[str, np.ndarray]:
    global _acoustic_lookup
    if _acoustic_lookup is not None:
        return _acoustic_lookup
    _acoustic_lookup = {}
    if not os.path.isfile(_ACOUSTIC_CACHE_PATH):
        return _acoustic_lookup
    try:
        ac = pd.read_csv(_ACOUSTIC_CACHE_PATH)
        for _, r in ac.iterrows():
            uid = str(r["utterance_id"])
            _acoustic_lookup[uid] = np.array(
                [
                    float(r.get("energy_norm", 0)),
                    float(r.get("pitch_norm", 0)),
                    float(r.get("zcr_norm", 0)),
                    float(r.get("brightness_norm", 0)),
                ],
                dtype=np.float64,
            )
    except Exception as exc:
        print(f"⚠ Index acoustique IEMOCAP illisible ({exc}) — VAD seul.")
        _acoustic_lookup = {}
    return _acoustic_lookup


def construire_index(
    df: pd.DataFrame,
    prefer_wav: bool = True,
    use_acoustic: bool = True,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Matrice (N, PROFILE_DIM) + sous-ensemble du dataframe aligné."""
    pool = df
    if prefer_wav and "wav_exists" in df.columns:
        wav_pool = df[df["wav_exists"]]
        if len(wav_pool) > 0:
            pool = wav_pool

    lookup = _charger_acoustic_lookup() if use_acoustic else {}
    rows = []
    vecs = []
    for _, row in pool.iterrows():
        uid = str(row["utterance_id"])
        ac = lookup.get(uid)
        vecs.append(vecteur_profil_iemocap(row, ac))
        rows.append(row)

    if not vecs:
        return np.zeros((0, PROFILE_DIM)), pool.iloc[0:0]

    out_df = pd.DataFrame(rows).reset_index(drop=True)
    return np.vstack(vecs), out_df


_index_use_acoustic: bool | None = None


def _get_index(
    df: pd.DataFrame,
    prefer_wav: bool,
    use_acoustic: bool = True,
) -> tuple[np.ndarray, pd.DataFrame]:
    global _index_matrix, _index_df, _index_use_acoustic
    if (
        _index_matrix is not None
        and _index_df is not None
        and _index_use_acoustic == use_acoustic
    ):
        return _index_matrix, _index_df
    _index_matrix, _index_df = construire_index(
        df, prefer_wav=prefer_wav, use_acoustic=use_acoustic,
    )
    _index_use_acoustic = use_acoustic
    return _index_matrix, _index_df


def selectionner_par_similarite(
    df: pd.DataFrame,
    scenario_id: int,
    boredom: float,
    confusion: float,
    engagement: float,
    frustration: float,
    prefer_wav: bool = True,
    top_k_ties: int = 5,
    use_acoustic: bool = True,
) -> tuple[dict, dict]:
    """
    Choisit l'utterance IEMOCAP la plus proche du profil DAiSEE (cosinus).
    Tie-break reproductible : scenario_id parmi les ex-aequo du top-k.
    """
    sid = int(scenario_id)
    matrix, pool = _get_index(df, prefer_wav=prefer_wav, use_acoustic=use_acoustic)
    if matrix.size == 0:
        row = df.iloc[sid % len(df)]
        return row.to_dict(), {
            "methode": "similarite_fallback_vide",
            "scenario_id": sid,
            "pool_size": int(len(df)),
        }

    etat_dom, niveau_dom, _ = _etat_depuis_scores(
        boredom, confusion, engagement, frustration,
    )
    matrix_work, pool_work, filtre_etat = _filtrer_pool_par_etat(
        pool, matrix, etat_dom,
    )

    query = vecteur_profil_scenario(boredom, confusion, engagement, frustration)
    sims = _cosine_similarity(query, matrix_work)

    # Pénalité légère si repli sur pool complet (état annoté ≠ cible)
    if not filtre_etat and "etat_daisee" in pool_work.columns:
        mismatch = (pool_work["etat_daisee"].astype(str) != str(etat_dom)).to_numpy()
        sims = sims - np.where(mismatch, 0.18, 0.0)

    best = float(np.max(sims))
    eps = 1e-6
    candidates = np.where(sims >= best - eps)[0]
    if len(candidates) > top_k_ties:
        order = candidates[np.argsort(-sims[candidates])[:top_k_ties]]
    else:
        order = candidates[np.argsort(-sims[candidates])]

    pick = int(order[sid % len(order)])
    row = pool_work.iloc[pick]

    base_methode = (
        "similarite_profil_acoustique"
        if use_acoustic and _charger_acoustic_lookup()
        else "similarite_profil_vad"
    )
    methode = (
        f"etat_{etat_dom.lower()}_puis_{base_methode}"
        if filtre_etat
        else f"{base_methode}_repli_etat"
    )

    meta = {
        "methode": methode,
        "etat_cible": etat_dom,
        "niveau_cible": int(niveau_dom),
        "filtre_etat_applique": filtre_etat,
        "pool_filtre_size": int(len(pool_work)),
        "etat_iemocap": row.get("etat_daisee"),
        "similarite": round(float(sims[pick]), 4),
        "similarite_max": round(best, 4),
        "pool_size": int(len(pool)),
        "index_pool": int(pick),
        "scenario_id": sid,
        "scores_cible": {
            "boredom": float(boredom),
            "confusion": float(confusion),
            "engagement": float(engagement),
            "frustration": float(frustration),
        },
        "profil_query": [round(float(x), 3) for x in query[:N_BASE]],
    }
    return row.to_dict(), meta


def _etat_depuis_scores(b, c, e, f):
    from daisee_utils import etat_dominant_depuis_scores

    return etat_dominant_depuis_scores(b, c, e, f)


def invalider_cache_index():
    """Force rebuild de l'index (tests / après rebuild acoustique)."""
    global _index_matrix, _index_df, _acoustic_lookup, _index_use_acoustic
    _index_matrix = None
    _index_df = None
    _acoustic_lookup = None
    _index_use_acoustic = None
