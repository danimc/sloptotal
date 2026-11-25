"""The per-sentence route: a missing model is a 503, never a page of zeros."""
import pytest
from fastapi.testclient import TestClient

from app.engines import sentences_revisor_es as sres
from app.main import app


@pytest.fixture(scope="module")
def client():
    return TestClient(app)


def test_token_sentence_maps_offsets_to_sentences():
    text, spans = sres.join(["Uno dos.", "Tres."])
    assert text == "Uno dos. Tres." and spans == [(0, 8), (9, 14)]
    assert sres.token_sentence([(0, 0), (0, 3), (4, 8), (8, 8), (9, 14)], spans) == [-1, 0, 0, -1, 1]


def test_missing_model_is_503(client, monkeypatch):
    monkeypatch.setattr(sres, "MODEL_DIR", "/nonexistent")
    r = client.post("/api/sentence-score", json={"sentences": ["Una frase."]})
    assert r.status_code == 503
