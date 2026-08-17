"""Tests AU (Action Units) + mapping FER+AU."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from au_module import (
    AU_FEATURE_ORDER,
    au_to_vector,
    blendshapes_to_au,
    moyenner_au,
)
from fer_classifier import features_fer_au, scores_to_vector


class _FakeCat:
    def __init__(self, name, score):
        self.category_name = name
        self.score = score


def test_blendshapes_to_au_smile():
    cats = [
        _FakeCat("mouthSmileLeft", 0.8),
        _FakeCat("mouthSmileRight", 0.6),
        _FakeCat("browDownLeft", 0.1),
        _FakeCat("browDownRight", 0.2),
    ]
    au = blendshapes_to_au(cats)
    assert au["AU12"] == 0.7
    assert abs(au["AU4"] - 0.15) < 1e-9
    assert set(au.keys()) == set(AU_FEATURE_ORDER)


def test_au_to_vector_order():
    au = {k: 0.0 for k in AU_FEATURE_ORDER}
    au["AU12"] = 0.9
    vec = au_to_vector(au)
    assert len(vec) == len(AU_FEATURE_ORDER)
    assert vec[list(AU_FEATURE_ORDER).index("AU12")] == 0.9


def test_features_fer_au_dim():
    fer = {
        "angry": 0.1,
        "disgust": 0.0,
        "fear": 0.0,
        "happy": 0.7,
        "neutral": 0.1,
        "sad": 0.05,
        "surprise": 0.05,
    }
    au = {k: 0.1 for k in AU_FEATURE_ORDER}
    x = features_fer_au(fer, au)
    assert x.shape == (7 + len(AU_FEATURE_ORDER),)
    assert abs(x[:7].sum() - scores_to_vector(fer).sum()) < 1e-9


def test_moyenner_au():
    a = {k: 0.0 for k in AU_FEATURE_ORDER}
    b = {k: 1.0 for k in AU_FEATURE_ORDER}
    m = moyenner_au([a, b])
    assert all(abs(v - 0.5) < 1e-9 for v in m.values())
