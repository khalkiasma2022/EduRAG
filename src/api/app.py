"""Application HTTP d'EduRAG.

Streamlit reste utilisable.
Cette application répond sans charger Qwen, MiniLM ni les index.
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.dependencies import register_assistant

# Origines du serveur Vite, futur frontend React.
_DEFAULT_CORS_ORIGINS = (
    "http://localhost:5173",
    "http://127.0.0.1:5173",
)


def cors_origins() -> list[str]:
    """Retourne les origines autorisées.

    EDURAG_CORS_ORIGINS peut ajouter l'adresse publique utilisée sur Colab,
    séparée par des virgules.
    """
    configured = os.environ.get("EDURAG_CORS_ORIGINS", "")
    extra = [item.strip() for item in configured.split(",") if item.strip()]
    origins: list[str] = []
    for origin in (*_DEFAULT_CORS_ORIGINS, *extra):
        if origin not in origins:
            origins.append(origin)
    return origins


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Démarre l'API sans construire EduRAGAssistant.

    Un seul agent peut être enregistré avec register_assistant.
    Cet emplacement n'est pas remplacé au redémarrage du cycle de vie.
    """
    if not hasattr(app.state, "assistant"):
        app.state.assistant = None
    yield


def create_app(assistant: object | None = None) -> FastAPI:
    """Crée l'application et, au besoin, enregistre un seul agent."""
    app = FastAPI(title="EduRAG", lifespan=_lifespan)
    app.state.assistant = None
    if assistant is not None:
        register_assistant(app, assistant)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins(),
        allow_credentials=False,
        allow_methods=["GET", "POST", "DELETE"],
        allow_headers=["Accept", "Content-Type"],
    )

    @app.get("/health")
    def health() -> dict[str, str]:
        """Indique que l'API répond, sans interroger l'agent."""
        return {"status": "ok"}

    return app


app = create_app()
