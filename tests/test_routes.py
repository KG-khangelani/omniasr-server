"""Integration tests for API routes."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app.server import app


@pytest.fixture
def client():
    """Create a test client with mocked ASR service."""
    with patch("app.service.asr_service") as mock_service:
        mock_service.load_model = MagicMock()
        with TestClient(app) as test_client:
            yield test_client


def test_get_health_check(client: TestClient):
    """Health check should return valid JSON."""
    response = client.get("/health-check")

    assert response.headers["content-type"] == "application/json"
    assert response.status_code == 200
    assert response.text == '"ok"'


def test_get_models(client: TestClient):
    """Models endpoint."""
    response = client.get("/v1/models")

    assert response.status_code == 200

    data = response.json()

    assert "data" in data
    assert isinstance(data["data"], list)
    assert len(data["data"]) == 1

    model = response.json()["data"][0]

    assert "id" in model
    assert "object" in model
    assert "created" in model
    assert "owned_by" in model
    assert model["object"] == "model"
    assert model["owned_by"] == "omnilingual-asr"
    assert isinstance(model["created"], int)


@patch("app.routes.MODEL_NAME", "custom_model_name")
def test_get_models_uses_configured_model_name(client: TestClient):
    """Model ID should reflect the configured MODEL_NAME."""
    response = client.get("/v1/models")
    model = response.json()["data"][0]

    assert model["id"] == "custom_model_name"


def test_transcribe_returns_json(client: TestClient):
    """The default response should match the documented JSON shape."""
    with patch(
        "app.routes.asr_service.transcribe",
        new=AsyncMock(return_value="molo"),
    ) as transcribe:
        response = client.post(
            "/v1/audio/transcriptions",
            files={"file": ("sample.wav", b"RIFF-test", "audio/wav")},
        )

    assert response.status_code == 200
    assert response.json() == {"text": "molo"}
    transcribe.assert_awaited_once_with(b"RIFF-test", language=None)


def test_transcribe_returns_plain_text(client: TestClient):
    """The text format should return an unwrapped transcription."""
    with patch(
        "app.routes.asr_service.transcribe",
        new=AsyncMock(return_value="sawubona"),
    ):
        response = client.post(
            "/v1/audio/transcriptions",
            data={"response_format": "text"},
            files={"file": ("sample.wav", b"RIFF-test", "audio/wav")},
        )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain")
    assert response.text == "sawubona"


def test_transcribe_rejects_empty_file(client: TestClient):
    response = client.post(
        "/v1/audio/transcriptions",
        files={"file": ("empty.wav", b"", "audio/wav")},
    )

    assert response.status_code == 400
    assert response.json()["error"]["param"] == "file"


def test_transcribe_rejects_unsupported_response_format(client: TestClient):
    response = client.post(
        "/v1/audio/transcriptions",
        data={"response_format": "srt"},
        files={"file": ("sample.wav", b"RIFF-test", "audio/wav")},
    )

    assert response.status_code == 400
    error = response.json()["error"]
    assert error["param"] == "response_format"
    assert error["code"] == "unsupported_response_format"


def test_transcribe_requires_file(client: TestClient):
    response = client.post("/v1/audio/transcriptions")

    assert response.status_code == 400
    assert response.json()["error"]["type"] == "invalid_request_error"
