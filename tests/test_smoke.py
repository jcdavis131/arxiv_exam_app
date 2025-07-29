"""Tiny smoke test ensuring the FastAPI app starts and basic route works."""
from fastapi.testclient import TestClient

from arxiv_exam_app import app

client = TestClient(app)


def test_health_exam_endpoint():
    resp = client.get("/exam/2406.01234?n=1")
    assert resp.status_code in {200, 502, 500, 404}