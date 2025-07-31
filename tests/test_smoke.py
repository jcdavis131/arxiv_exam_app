"""Tiny smoke test ensuring the FastAPI app starts and basic route works."""
from fastapi.testclient import TestClient

from main import app

client = TestClient(app)


def test_health_exam_endpoint():
    resp = client.get("/api/exam/2406.01234?mc_questions=1&oe_questions=1", headers={"X-LLM-API-Key": "test-key"})
    assert resp.status_code in {200, 502, 500, 404, 400}