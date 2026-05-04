import os
import pytest
from fastapi.testclient import TestClient

from api.app import app

client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200

    data = response.json()
    assert data["status"] == "ok"
    assert "device" in data
    assert "sounds" in data


def test_sounds():
    response = client.get("/sounds")
    assert response.status_code == 200

    data = response.json()
    assert "sounds" in data
    assert isinstance(data["sounds"], list)
    assert len(data["sounds"]) > 0


def test_predict():
    test_audio_path = "test_bad.wav"
    assert os.path.exists(test_audio_path)

    with open(test_audio_path, "rb") as f:
        response = client.post(
            "/predict",
            files={"file": ("test_bad.wav", f, "audio/wav")},
            data={
                "expected_sounds": ["р", "т", "с"]
            },
        )

    print(response.status_code)
    print(response.text)

    assert response.status_code == 200

    data = response.json()
    assert "p_bad" in data
    assert "status" in data
    assert "flagged_sounds" in data


def test_history():
    response = client.get("/history?limit=5")
    assert response.status_code == 200

    data = response.json()
    assert "items" in data
    assert isinstance(data["items"], list)