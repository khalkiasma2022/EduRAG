"""Sérialise les appels qui touchent Qwen ou FAISS.

Les lectures restent servies pendant une génération.
Le verrou asyncio attend sans occuper le thread du serveur.
La génération elle-même reste dans un fil, car la route est synchrone.
"""

import asyncio


def agent_call_is_exclusive(method: str, path: str) -> bool:
    """Indique si la requête utilise le modèle ou reconstruit l'index."""
    normalized = path.rstrip("/") or "/"
    if method == "POST" and normalized == "/documents":
        return True
    if method == "DELETE" and normalized.startswith("/documents/"):
        return True
    parts = normalized.split("/")
    return (
        method == "POST"
        and len(parts) == 4
        and parts[1] == "conversations"
        and parts[2] != ""
        and parts[3] == "messages"
    )


class ExclusiveAgentMiddleware:
    """Fait attendre les requêtes qui ne peuvent pas partager l'agent."""

    def __init__(self, app: object, lock: asyncio.Lock) -> None:
        """Retient l'application suivante et le verrou unique."""
        self.app = app
        self.lock = lock

    async def __call__(self, scope: dict, receive: object, send: object) -> None:
        """Laisse passer les lectures. Met en file les appels exclusifs."""
        if scope.get("type") == "http" and agent_call_is_exclusive(
            str(scope.get("method", "")),
            str(scope.get("path", "")),
        ):
            async with self.lock:
                await self.app(scope, receive, send)
            return
        await self.app(scope, receive, send)
