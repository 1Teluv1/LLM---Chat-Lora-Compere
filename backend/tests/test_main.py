from pathlib import Path
import sys

from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.main import app


client = TestClient(app)


def test_health() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_debug_runtime_has_python() -> None:
    response = client.get("/debug/runtime")
    assert response.status_code == 200
    body = response.json()
    assert "python" in body
    assert "runtime_default" in body
