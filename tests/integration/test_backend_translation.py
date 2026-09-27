"""REST translation endpoints (dictionary + optional BanglaT5)."""

from __future__ import annotations

import os

import pytest
from fastapi.testclient import TestClient

from backend.bangla_service import reset_translation_singletons_for_tests
from backend.inference import reset_classifier_for_tests
from backend.main import app


@pytest.fixture()
def client() -> TestClient:
    reset_classifier_for_tests()
    reset_translation_singletons_for_tests()
    with TestClient(app) as test_client:
        yield test_client


def test_instant_word_dictionary(client: TestClient) -> None:
    resp = client.post("/api/translate/word", json={"english": "hello"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["source"] == "dictionary"
    assert body["bangla"]


def test_instant_batch_words(client: TestClient) -> None:
    resp = client.post(
        "/api/translate/words",
        json={"words": ["hello", "water", "unknownword"]},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["bangla_glosses"][0]
    assert body["bangla_glosses"][1]
    assert body["bangla_glosses"][2] is None
    assert " " in body["joined_bangla"]


@pytest.mark.skipif(
    os.environ.get("RUN_BANGLAT5_INTEGRATION") != "1",
    reason="BanglaT5 load is heavy; set RUN_BANGLAT5_INTEGRATION=1 to run",
)
def test_sentence_translation(client: TestClient) -> None:
    resp = client.post(
        "/api/translate/sentence",
        json={"english": "Thank you very much."},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["bangla"]
    assert body["latency_sec"] < 3.0
