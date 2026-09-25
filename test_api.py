"""Tests rapides — surtout pour vérifier que l'API répond."""

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert "version" in body


def test_sentiment_english():
    r = client.post(
        "/api/v1/analyze/sentiment",
        json={"text": "I love building APIs with FastAPI"},
    )
    assert r.status_code == 200
    assert r.json()["label"] == "positive"


def test_summary_without_llm():
    text = "First sentence is here. Second one adds detail. Third wraps it up."
    r = client.post("/api/v1/analyze/summary", json={"text": text, "max_sentences": 2})
    assert r.status_code == 200
    assert len(r.json()["summary"]) > 0
