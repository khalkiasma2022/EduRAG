"""Tests de l'API des conversations.

Ils utilisent un faux assistant.
Ils ne construisent pas EduRAGAssistant et ne chargent aucun modèle.
"""

from __future__ import annotations

import ast
import asyncio
import sys
import threading
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from httpx import ASGITransport, AsyncClient

_FORBIDDEN_MODULES = (
    "torch",
    "transformers",
    "sentence_transformers",
    "src.assistant",
    "src.generation",
    "src.retrieval",
)

from src.api.app import create_app


class _Assistant:
    """Imite les conversations sans créer EduRAGAssistant."""

    def __init__(self) -> None:
        self.rows: list[dict[str, str | int]] = []
        self.messages: dict[int, list[dict[str, object]]] = {}
        self.sent: list[tuple[int, str]] = []
        self.fail = False
        self._next = 1
        self._active = 0
        self.peak = 0
        self._gate = threading.Lock()
        self.hold_question = ""
        self.entered = threading.Event()
        self.release = threading.Event()

    def create_conversation(self, title: str) -> int:
        if not isinstance(title, str) or title.strip() == "":
            raise ValueError("Le titre de la conversation est invalide.")
        conversation_id = self._next
        self._next += 1
        self.rows.insert(
            0,
            {
                "id": conversation_id,
                "title": title.strip(),
                "created_at": f"2026-11-25T12:00:0{conversation_id}.000000+00:00",
            },
        )
        self.messages[conversation_id] = []
        return conversation_id

    def list_conversations(self) -> list[dict[str, str | int]]:
        return [dict(row) for row in self.rows]

    def get_conversation(self, conversation_id: object) -> dict[str, object]:
        self._require(conversation_id)
        row = next(row for row in self.rows if row["id"] == conversation_id)
        return {**row, "messages": [dict(message) for message in self.messages[int(conversation_id)]]}

    def delete_conversation(self, conversation_id: object) -> None:
        self._require(conversation_id)
        self.rows = [row for row in self.rows if row["id"] != conversation_id]
        self.messages.pop(int(conversation_id), None)

    def send_message(self, conversation_id: object, question: str, top_k: int = 3) -> dict[str, object]:
        self._require(conversation_id)
        if not isinstance(question, str) or question.strip() == "":
            raise ValueError("Le message est invalide.")
        self.sent.append((int(conversation_id), question))
        with self._gate:
            self._active += 1
            self.peak = max(self.peak, self._active)
        try:
            if question == self.hold_question:
                self.entered.set()
                self.release.wait(3)
            if self.fail:
                self._add_message(int(conversation_id), "user", question, [])
                raise RuntimeError("La génération de la réponse a échoué.")
            question_message = self._add_message(int(conversation_id), "user", question, [])
            answer_message = self._add_message(
                int(conversation_id),
                "assistant",
                "Dans la mitochondrie.",
                [
                    {
                        "text": "La mitochondrie produit de l'énergie.",
                        "filename": "cours.pdf",
                        "page_number": 2,
                        "chunk_id": 4,
                        "stored_path": r"C:\secret\cours.pdf",
                    }
                ],
            )
        finally:
            with self._gate:
                self._active -= 1
        return {
            "conversation_id": int(conversation_id),
            "question": question_message,
            "answer": answer_message,
        }

    def ask(self, question: str, top_k: int = 3) -> dict[str, object]:
        raise AssertionError("La route ne doit pas appeler ask.")

    def _require(self, conversation_id: object) -> None:
        if isinstance(conversation_id, bool) or not isinstance(conversation_id, int) or conversation_id <= 0:
            raise ValueError("L'identifiant de la conversation est invalide.")
        if not any(row["id"] == conversation_id for row in self.rows):
            raise ValueError("Conversation introuvable.")

    def _add_message(
        self,
        conversation_id: int,
        role: str,
        content: str,
        sources: list[dict[str, object]],
    ) -> dict[str, object]:
        message = {
            "id": len(self.messages[conversation_id]) + 1,
            "conversation_id": conversation_id,
            "role": role,
            "content": content,
            "sources": sources,
            "created_at": f"2026-11-25T12:01:0{len(self.messages[conversation_id])}.000000+00:00",
        }
        self.messages[conversation_id].append(message)
        return dict(message)


def _client(assistant: _Assistant | None = None) -> tuple[TestClient, _Assistant]:
    assistant = _Assistant() if assistant is None else assistant
    return TestClient(create_app(assistant)), assistant


def _source() -> dict[str, str | int]:
    return {
        "text": "La mitochondrie produit de l'énergie.",
        "filename": "cours.pdf",
        "page_number": 2,
        "chunk_id": 4,
    }


def test_create_conversation_returns_the_stored_title() -> None:
    client, assistant = _client()

    response = client.post("/conversations", json={"title": "  Mitochondrie  "})

    assert response.status_code == 201
    body = response.json()
    assert body["id"] == 1
    assert body["title"] == "Mitochondrie"
    assert body["created_at"] == assistant.rows[0]["created_at"]
    assert "messages" not in body
    assert assistant.sent == []


def test_create_conversation_rejects_an_empty_title() -> None:
    client, assistant = _client()

    response = client.post("/conversations", json={"title": "   "})

    assert response.status_code == 400
    assert response.json()["detail"] == "Le titre de la conversation est invalide."
    assert assistant.rows == []


def test_list_conversations_keeps_the_store_order() -> None:
    client, _assistant = _client()
    client.post("/conversations", json={"title": "Ancienne"})
    client.post("/conversations", json={"title": "Récente"})

    response = client.get("/conversations")

    assert response.status_code == 200
    assert [item["title"] for item in response.json()] == ["Récente", "Ancienne"]
    assert [item["id"] for item in response.json()] == [2, 1]


def test_get_conversation_returns_messages_and_sources() -> None:
    client, _assistant = _client()
    created = client.post("/conversations", json={"title": "Cours"}).json()
    sent = client.post(
        f"/conversations/{created['id']}/messages",
        json={"question": "Où est produite l'énergie ?"},
    )

    response = client.get(f"/conversations/{created['id']}")

    assert sent.status_code == 200
    assert response.status_code == 200
    body = response.json()
    assert [message["role"] for message in body["messages"]] == ["user", "assistant"]
    assert body["messages"][0]["sources"] == []
    assert body["messages"][1]["sources"] == [_source()]
    assert r"C:\secret" not in response.text
    assert sent.json()["answer"]["sources"] == [_source()]
    assert sent.json()["question"]["content"] == "Où est produite l'énergie ?"


def test_get_conversation_returns_web_sources() -> None:
    client, assistant = _client()
    conversation_id = assistant.create_conversation("Web")
    web_source = {
        "type": "web",
        "title": "Documentation officielle Python",
        "url": "https://www.python.org/",
        "snippet": "Welcome to Python.org",
    }
    assistant._add_message(conversation_id, "user", "Nouveautés Python ?", [])
    assistant._add_message(
        conversation_id,
        "assistant",
        "Voici une synthèse [1].",
        [web_source],
    )

    response = client.get(f"/conversations/{conversation_id}")

    assert response.status_code == 200
    body = response.json()
    assert body["messages"][1]["sources"] == [web_source]


def test_web_source_with_forbidden_protocol_is_rejected() -> None:
    client, assistant = _client()
    conversation_id = assistant.create_conversation("Web invalide")
    assistant._add_message(conversation_id, "user", "Question", [])
    assistant._add_message(
        conversation_id,
        "assistant",
        "Réponse",
        [
            {
                "type": "web",
                "title": "Bad",
                "url": "javascript:alert(1)",
                "snippet": "no",
            }
        ],
    )

    response = client.get(f"/conversations/{conversation_id}")

    assert response.status_code == 500
    assert response.json()["detail"] == "Les sources enregistrées sont illisibles."


def test_delete_conversation_removes_its_messages() -> None:
    client, assistant = _client()
    created = client.post("/conversations", json={"title": "À retirer"}).json()
    client.post(f"/conversations/{created['id']}/messages", json={"question": "Une question"})

    response = client.delete(f"/conversations/{created['id']}")

    assert response.status_code == 204
    assert response.content == b""
    assert assistant.messages == {}
    assert assistant.rows == []
    assert client.get(f"/conversations/{created['id']}").status_code == 404


def test_missing_conversation_returns_404() -> None:
    client, assistant = _client()

    reading = client.get("/conversations/9")
    deletion = client.delete("/conversations/9")
    message = client.post("/conversations/9/messages", json={"question": "Bonjour"})

    assert reading.status_code == 404
    assert deletion.status_code == 404
    assert message.status_code == 404
    assert message.json()["detail"] == "Conversation introuvable."
    assert assistant.sent == []


def test_send_message_is_called_once() -> None:
    client, assistant = _client()
    created = client.post("/conversations", json={"title": "Cours"}).json()

    response = client.post(
        f"/conversations/{created['id']}/messages",
        json={"question": "Que produit la mitochondrie ?"},
    )

    assert response.status_code == 200
    assert assistant.sent == [(created["id"], "Que produit la mitochondrie ?")]
    assert response.json()["conversation_id"] == created["id"]
    assert response.json()["answer"]["content"] == "Dans la mitochondrie."


def test_empty_question_is_rejected() -> None:
    client, assistant = _client()
    created = client.post("/conversations", json={"title": "Cours"}).json()

    response = client.post(f"/conversations/{created['id']}/messages", json={"question": "   "})

    assert response.status_code == 400
    assert response.json()["detail"] == "Le message est invalide."
    assert assistant.sent == []
    assert assistant.messages[created["id"]] == []


def test_invalid_question_is_rejected() -> None:
    client, assistant = _client()
    created = client.post("/conversations", json={"title": "Cours"}).json()

    response = client.post(f"/conversations/{created['id']}/messages", json={"question": 12})

    assert response.status_code == 422
    assert assistant.sent == []


def test_generation_error_returns_no_invented_answer() -> None:
    assistant = _Assistant()
    assistant.fail = True
    client, _assistant = _client(assistant)
    created = client.post("/conversations", json={"title": "Cours"}).json()

    response = client.post(f"/conversations/{created['id']}/messages", json={"question": "Explique"})

    assert response.status_code == 500
    assert response.json()["detail"] == "La génération de la réponse a échoué."
    assert assistant.sent == [(created["id"], "Explique")]
    assert [message["role"] for message in assistant.messages[created["id"]]] == ["user"]


def test_invalid_conversation_id_is_rejected() -> None:
    client, _assistant = _client()

    response = client.get("/conversations/0")

    assert response.status_code == 400
    assert response.json()["detail"] == "L'identifiant de la conversation est invalide."


@pytest.mark.parametrize(
    ("method", "path", "exclusive"),
    [
        ("POST", "/documents", True),
        ("DELETE", "/documents/abc", True),
        ("GET", "/documents", False),
        ("POST", "/conversations/1/messages", True),
        ("GET", "/conversations", False),
        ("GET", "/conversations/1", False),
        ("DELETE", "/conversations/1", False),
        ("POST", "/conversations", False),
        ("GET", "/health", False),
    ],
)
def test_only_model_and_index_calls_are_exclusive(method: str, path: str, exclusive: bool) -> None:
    from src.api.guard import agent_call_is_exclusive

    assert agent_call_is_exclusive(method, path) is exclusive


def test_generation_waits_its_turn_without_blocking_health() -> None:
    assistant = _Assistant()
    assistant.hold_question = "Première question longue"
    app = create_app(assistant)
    created_id = assistant.create_conversation("Cours")

    async def scenario() -> None:
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            first = asyncio.create_task(
                client.post(
                    f"/conversations/{created_id}/messages",
                    json={"question": "Première question longue"},
                )
            )
            assert await asyncio.to_thread(assistant.entered.wait, 3)
            health = await client.get("/health")
            second = asyncio.create_task(
                client.post(
                    f"/conversations/{created_id}/messages",
                    json={"question": "Deuxième question"},
                )
            )
            await asyncio.sleep(0.1)
            assert health.status_code == 200
            assert not second.done()
            assistant.release.set()
            first_response = await first
            second_response = await second

        assert first_response.status_code == 200
        assert second_response.status_code == 200

    asyncio.run(scenario())
    assert assistant.peak == 1
    assert assistant.sent == [
        (created_id, "Première question longue"),
        (created_id, "Deuxième question"),
    ]


def test_conversation_api_does_not_load_ai_models() -> None:
    import src.api.routes.conversations as conversations

    source = Path(conversations.__file__).read_text(encoding="utf-8")
    for node in ast.walk(ast.parse(source)):
        modules: list[str] = []
        if isinstance(node, ast.Import):
            modules = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            modules = [node.module]
        for module in modules:
            assert not module.startswith(_FORBIDDEN_MODULES), module

    before = {
        name
        for name in sys.modules
        if name == "src.assistant" or name.startswith(_FORBIDDEN_MODULES)
    }
    client, _assistant = _client()
    response = client.get("/conversations")

    assert response.status_code == 200
    loaded = {
        name
        for name in sys.modules
        if name == "src.assistant" or name.startswith(_FORBIDDEN_MODULES)
    }
    assert loaded == before
