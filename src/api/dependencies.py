"""Accès à l'unique agent EduRAG.

Les futures routes recevront cet agent.
Il n'est pas construit ici : son constructeur charge Qwen et MiniLM.
"""

from fastapi import FastAPI, Request


def register_assistant(app: FastAPI, assistant: object) -> None:
    """Enregistre l'agent une seule fois pour cette application."""
    if app.state.assistant is not None:
        raise RuntimeError("L'agent EduRAG est déjà initialisé.")
    app.state.assistant = assistant


def get_assistant(request: Request) -> object:
    """Retourne l'agent déjà enregistré."""
    assistant = getattr(request.app.state, "assistant", None)
    if assistant is None:
        raise RuntimeError("L'agent EduRAG n'est pas initialisé.")
    return assistant
