"""Tests de l'API Smart Revision.

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
    "src.revision",
)

from src.api.app import create_app


class _Assistant:
    """Imite les fiches sans créer EduRAGAssistant."""

    def __init__(self) -> None:
        self.documents = {"doc-1": "cours.pdf"}
        self.rows: list[dict[str, str | int]] = []
        self.calls: list[tuple[object, ...]] = []
        self.exports: list[tuple[str, object]] = []
        self.messages = 0
        self._next = 1
        self._active = 0
        self.peak = 0
        self._gate = threading.Lock()
        self.hold = False
        self.entered = threading.Event()
        self.release = threading.Event()
        self.message_started = threading.Event()

    def create_revision(
        self,
        document_id: str,
        page_start: int | None = None,
        page_end: int | None = None,
        language: str = "fr",
        level: str = "beginner",
        sheet_type: str = "complete",
    ) -> dict[str, str | int]:
        self.calls.append((document_id, page_start, page_end, language, level, sheet_type))
        with self._gate:
            self._active += 1
            self.peak = max(self.peak, self._active)
        try:
            if self.hold:
                self.entered.set()
                self.release.wait(3)
            self._validate(document_id, page_start, page_end, language, level, sheet_type)
            sheet_id = self._next
            self._next += 1
            row = {
                "id": sheet_id,
                "document_id": document_id,
                "filename": self.documents[document_id],
                "page_start": 1 if page_start is None else page_start,
                "page_end": 3 if page_end is None else page_end,
                "language": language,
                "level": level,
                "sheet_type": sheet_type,
                "content": "La mitochondrie produit de l'énergie.",
                "created_at": f"2026-11-25T13:00:0{sheet_id}.000000+00:00",
                "stored_path": r"C:\secret\cours.pdf",
            }
            self.rows.insert(0, row)
            return dict(row)
        finally:
            with self._gate:
                self._active -= 1

    def list_revisions(self) -> list[dict[str, str | int]]:
        return [dict(row) for row in self.rows]

    def get_revision(self, sheet_id: object) -> dict[str, str | int]:
        return dict(self._find(sheet_id))

    def export_revision_markdown(self, sheet_id: object, output_path: str | Path) -> Path:
        self.exports.append(("markdown", sheet_id))
        self._find(sheet_id)
        path = Path(output_path)
        path.write_bytes("## Fiche\n\nBonjour\n".encode("utf-8"))
        return path

    def export_revision_pdf(self, sheet_id: object, output_path: str | Path) -> Path:
        self.exports.append(("pdf", sheet_id))
        self._find(sheet_id)
        path = Path(output_path)
        path.write_bytes(b"%PDF-1.4\nfiche arabe\n")
        return path

    def send_message(self, conversation_id: object, question: str, top_k: int = 3) -> dict[str, object]:
        self.message_started.set()
        self.messages += 1
        message = {
            "id": 1,
            "conversation_id": conversation_id,
            "role": "user",
            "content": question,
            "sources": [],
            "created_at": "2026-11-25T13:02:00.000000+00:00",
        }
        answer = {
            "id": 2,
            "conversation_id": conversation_id,
            "role": "assistant",
            "content": "Réponse",
            "sources": [],
            "created_at": "2026-11-25T13:02:01.000000+00:00",
        }
        return {"conversation_id": conversation_id, "question": message, "answer": answer}

    def _validate(
        self,
        document_id: object,
        page_start: object,
        page_end: object,
        language: object,
        level: object,
        sheet_type: object,
    ) -> None:
        if not isinstance(document_id, str) or document_id.strip() == "":
            raise ValueError("L'identifiant du document est invalide.")
        if document_id not in self.documents:
            raise ValueError("Document introuvable dans la bibliothèque.")
        if (page_start is None) != (page_end is None):
            raise ValueError("L'intervalle de pages est invalide.")
        if page_start is not None and (
            isinstance(page_start, bool)
            or isinstance(page_end, bool)
            or not isinstance(page_start, int)
            or not isinstance(page_end, int)
            or page_start < 1
            or page_end < page_start
        ):
            raise ValueError("L'intervalle de pages est invalide.")
        if language not in {"fr", "en", "ar"}:
            raise ValueError("La langue demandée n'est pas prise en charge.")
        if level not in {"beginner", "intermediate", "advanced"}:
            raise ValueError("Le niveau demandé n'est pas pris en charge.")
        if sheet_type not in {"short", "complete"}:
            raise ValueError("Le type de fiche demandé n'est pas pris en charge.")

    def _find(self, sheet_id: object) -> dict[str, str | int]:
        if isinstance(sheet_id, bool) or not isinstance(sheet_id, int) or sheet_id <= 0:
            raise ValueError("L'identifiant de la fiche est invalide.")
        for row in self.rows:
            if row["id"] == sheet_id:
                return row
        raise ValueError("Fiche introuvable.")


def _client(assistant: _Assistant | None = None) -> tuple[TestClient, _Assistant]:
    assistant = _Assistant() if assistant is None else assistant
    return TestClient(create_app(assistant)), assistant


def _public_sheet(row: dict[str, str | int]) -> dict[str, str | int]:
    return {key: row[key] for key in (
        "id",
        "document_id",
        "filename",
        "page_start",
        "page_end",
        "language",
        "level",
        "sheet_type",
        "content",
        "created_at",
    )}


def test_create_revision_is_called_once() -> None:
    client, assistant = _client()

    response = client.post(
        "/revisions",
        json={
            "document_id": "doc-1",
            "language": "ar",
            "level": "advanced",
            "sheet_type": "short",
            "page_start": 2,
            "page_end": 4,
        },
    )

    assert response.status_code == 201
    assert assistant.calls == [("doc-1", 2, 4, "ar", "advanced", "short")]
    assert response.json() == _public_sheet(assistant.rows[0])
    assert r"C:\secret" not in response.text


def test_create_revision_defaults_to_the_whole_document() -> None:
    client, assistant = _client()

    response = client.post("/revisions", json={"document_id": "doc-1"})

    assert response.status_code == 201
    assert assistant.calls == [("doc-1", None, None, "fr", "beginner", "complete")]


@pytest.mark.parametrize(
    ("field", "value", "detail"),
    [
        ("language", "de", "La langue demandée n'est pas prise en charge."),
        ("level", "expert", "Le niveau demandé n'est pas pris en charge."),
        ("sheet_type", "long", "Le type de fiche demandé n'est pas pris en charge."),
    ],
)
def test_create_revision_rejects_an_unsupported_choice(field: str, value: str, detail: str) -> None:
    client, assistant = _client()
    body = {"document_id": "doc-1", field: value}

    response = client.post("/revisions", json=body)

    assert response.status_code == 400
    assert response.json()["detail"] == detail
    assert assistant.rows == []
    assert len(assistant.calls) == 1


def test_create_revision_reports_a_missing_document() -> None:
    client, assistant = _client()

    response = client.post("/revisions", json={"document_id": "absent"})

    assert response.status_code == 404
    assert response.json()["detail"] == "Document introuvable dans la bibliothèque."
    assert assistant.rows == []


@pytest.mark.parametrize(
    "pages",
    [
        {"page_start": 4, "page_end": 2},
        {"page_start": 2},
    ],
)
def test_create_revision_rejects_an_invalid_page_range(pages: dict[str, int]) -> None:
    client, assistant = _client()

    response = client.post("/revisions", json={"document_id": "doc-1", **pages})

    assert response.status_code == 400
    assert response.json()["detail"] == "L'intervalle de pages est invalide."
    assert assistant.rows == []


def test_list_revisions_keeps_the_store_order_without_generation() -> None:
    client, assistant = _client()
    client.post("/revisions", json={"document_id": "doc-1", "sheet_type": "short"})
    client.post("/revisions", json={"document_id": "doc-1", "sheet_type": "complete"})
    assistant.calls.clear()

    response = client.get("/revisions")

    assert response.status_code == 200
    assert [item["id"] for item in response.json()] == [2, 1]
    assert [item["sheet_type"] for item in response.json()] == ["complete", "short"]
    assert assistant.calls == []
    assert r"C:\secret" not in response.text


def test_get_revision_reads_the_saved_sheet() -> None:
    client, assistant = _client()
    created = client.post("/revisions", json={"document_id": "doc-1"}).json()
    assistant.calls.clear()

    response = client.get(f"/revisions/{created['id']}")

    assert response.status_code == 200
    assert response.json()["content"] == "La mitochondrie produit de l'énergie."
    assert response.json()["id"] == created["id"]
    assert assistant.calls == []


def test_get_revision_reports_an_unknown_sheet() -> None:
    client, assistant = _client()

    response = client.get("/revisions/9")

    assert response.status_code == 404
    assert response.json()["detail"] == "Fiche introuvable."
    assert assistant.calls == []


def test_get_revision_rejects_an_invalid_id() -> None:
    client, _assistant = _client()

    response = client.get("/revisions/0")

    assert response.status_code == 400
    assert response.json()["detail"] == "L'identifiant de la fiche est invalide."


def test_export_markdown_reads_the_saved_sheet() -> None:
    client, assistant = _client()
    created = client.post("/revisions", json={"document_id": "doc-1"}).json()
    assistant.calls.clear()

    response = client.get(f"/revisions/{created['id']}/markdown")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/markdown")
    assert response.content == "## Fiche\n\nBonjour\n".encode("utf-8")
    assert 'filename="fiche-1.md"' in response.headers["content-disposition"]
    assert assistant.calls == []
    assert assistant.exports == [("markdown", 1)]


def test_export_pdf_reads_the_saved_sheet() -> None:
    client, assistant = _client()
    created = client.post("/revisions", json={"document_id": "doc-1", "language": "ar"}).json()
    assistant.calls.clear()

    response = client.get(f"/revisions/{created['id']}/pdf")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/pdf")
    assert response.content == b"%PDF-1.4\nfiche arabe\n"
    assert 'filename="fiche-1.pdf"' in response.headers["content-disposition"]
    assert assistant.calls == []
    assert assistant.exports == [("pdf", 1)]


def test_export_reports_an_unknown_sheet() -> None:
    client, assistant = _client()

    response = client.get("/revisions/4/pdf")

    assert response.status_code == 404
    assert response.json()["detail"] == "Fiche introuvable."
    assert assistant.calls == []
    assert assistant.exports == [("pdf", 4)]


@pytest.mark.parametrize(
    ("method", "path", "exclusive"),
    [
        ("POST", "/revisions", True),
        ("GET", "/revisions", False),
        ("GET", "/revisions/1", False),
        ("GET", "/revisions/1/markdown", False),
        ("GET", "/revisions/1/pdf", False),
    ],
)
def test_only_revision_generation_is_exclusive(method: str, path: str, exclusive: bool) -> None:
    from src.api.guard import agent_call_is_exclusive

    assert agent_call_is_exclusive(method, path) is exclusive


def test_revision_generation_uses_the_shared_lock() -> None:
    assistant = _Assistant()
    assistant.hold = True
    app = create_app(assistant)

    async def scenario() -> None:
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            first = asyncio.create_task(client.post("/revisions", json={"document_id": "doc-1"}))
            assert await asyncio.to_thread(assistant.entered.wait, 3)
            health = await client.get("/health")
            listed = await client.get("/revisions")
            message = asyncio.create_task(
                client.post("/conversations/1/messages", json={"question": "Pendant la fiche"})
            )
            await asyncio.sleep(0.1)
            assert health.status_code == 200
            assert listed.status_code == 200
            assert not message.done()
            assert not assistant.message_started.is_set()
            assistant.release.set()
            created = await first
            sent = await message

        assert created.status_code == 201
        assert sent.status_code == 200

    asyncio.run(scenario())
    assert assistant.peak == 1
    assert assistant.calls == [("doc-1", None, None, "fr", "beginner", "complete")]
    assert assistant.messages == 1


def test_revision_api_does_not_load_ai_models() -> None:
    import src.api.routes.revisions as revisions

    source = Path(revisions.__file__).read_text(encoding="utf-8")
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
        if name.startswith(_FORBIDDEN_MODULES)
    }
    client, _assistant = _client()
    response = client.get("/revisions")

    assert response.status_code == 200
    loaded = {
        name
        for name in sys.modules
        if name.startswith(_FORBIDDEN_MODULES)
    }
    assert loaded == before
