"""Tests du démarrage de l'API FastAPI.

Ils vérifient /health, le CORS et l'emplacement unique de l'agent.
Ils ne construisent pas EduRAGAssistant et ne chargent aucun modèle.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from starlette.requests import Request

from src.api.app import create_app
from src.api.dependencies import get_assistant, register_assistant


def _request_for(app: object) -> Request:
    return Request(
        {
            "type": "http",
            "app": app,
            "method": "GET",
            "path": "/health",
            "headers": [],
            "query_string": b"",
        }
    )


def test_health_returns_ok() -> None:
    app = create_app()
    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_startup_keeps_the_assistant_slot_empty() -> None:
    app = create_app()
    with TestClient(app):
        assert app.state.assistant is None


def test_health_does_not_require_an_assistant() -> None:
    app = create_app()
    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert app.state.assistant is None


def test_only_one_assistant_can_be_registered() -> None:
    app = create_app()
    first = object()
    register_assistant(app, first)

    with pytest.raises(RuntimeError, match="déjà initialisé"):
        register_assistant(app, object())

    assert app.state.assistant is first
    assert get_assistant(_request_for(app)) is first


def test_get_assistant_rejects_a_missing_agent() -> None:
    app = create_app()

    with pytest.raises(RuntimeError, match="n'est pas initialisé"):
        get_assistant(_request_for(app))


def test_create_app_keeps_the_assistant_given_once() -> None:
    assistant = object()
    app = create_app(assistant)

    with TestClient(app):
        assert app.state.assistant is assistant
        assert get_assistant(_request_for(app)) is assistant

    with pytest.raises(RuntimeError, match="déjà initialisé"):
        register_assistant(app, object())


def test_module_application_starts_without_an_assistant() -> None:
    from src.api.app import app

    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
    assert app.state.assistant is None


def test_cors_allows_the_local_react_origin() -> None:
    app = create_app()
    with TestClient(app) as client:
        response = client.get("/health", headers={"Origin": "http://localhost:5173"})

    assert response.headers["access-control-allow-origin"] == "http://localhost:5173"


def test_cors_allows_the_loopback_react_origin() -> None:
    app = create_app()
    with TestClient(app) as client:
        response = client.get("/health", headers={"Origin": "http://127.0.0.1:5173"})

    assert response.headers["access-control-allow-origin"] == "http://127.0.0.1:5173"


def test_cors_rejects_an_unknown_origin() -> None:
    app = create_app()
    with TestClient(app) as client:
        response = client.get("/health", headers={"Origin": "http://evil.example"})

    assert "access-control-allow-origin" not in response.headers


def test_cors_accepts_an_origin_from_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EDURAG_CORS_ORIGINS", "https://colab.research.google.com")
    app = create_app()
    with TestClient(app) as client:
        response = client.get(
            "/health",
            headers={"Origin": "https://colab.research.google.com"},
        )

    assert response.headers["access-control-allow-origin"] == "https://colab.research.google.com"
