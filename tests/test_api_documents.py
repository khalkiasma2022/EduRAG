"""Tests de l'API de gestion des PDF.

Ils utilisent un faux assistant.
Ils ne construisent pas EduRAGAssistant et ne chargent aucun modèle.
"""

from __future__ import annotations

import ast
import io
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

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
    """Imite la bibliothèque sans créer EduRAGAssistant."""

    def __init__(self) -> None:
        self.rows: list[dict[str, str]] = []
        self.added: list[Path] = []
        self.removed: list[str] = []
        self.duplicate = False

    def list_documents(self) -> list[dict[str, str]]:
        return [dict(row) for row in self.rows]

    def add_pdf(self, file_path: str | Path) -> dict[str, str]:
        path = Path(file_path)
        if not path.is_file():
            raise AssertionError("add_pdf doit recevoir un fichier présent.")
        if "/" in path.name or "\\" in path.name:
            raise AssertionError("add_pdf doit recevoir un nom de fichier seul.")
        self.added.append(path)
        if self.duplicate:
            raise ValueError("Ce document est déjà dans la bibliothèque.")
        entry = {
            "document_id": f"doc-{len(self.rows) + 1}",
            "filename": path.name,
            "stored_path": r"C:\secret\bibliotheque\cours.pdf",
            "sha256": "a" * 64,
        }
        self.rows.append(entry)
        return dict(entry)

    def remove_document(self, document_id: str) -> None:
        self.removed.append(document_id)
        if not any(row["document_id"] == document_id for row in self.rows):
            raise ValueError("Document introuvable dans la bibliothèque.")
        self.rows = [row for row in self.rows if row["document_id"] != document_id]


def _text_pdf(*pages: str) -> bytes:
    """Crée un PDF dont le texte reste extractible."""
    writer = PdfWriter()
    for text in pages:
        page = writer.add_blank_page(width=612, height=792)
        page[NameObject("/Resources")] = DictionaryObject(
            {
                NameObject("/Font"): DictionaryObject(
                    {
                        NameObject("/F1"): DictionaryObject(
                            {
                                NameObject("/Type"): NameObject("/Font"),
                                NameObject("/Subtype"): NameObject("/Type1"),
                                NameObject("/BaseFont"): NameObject("/Helvetica"),
                            }
                        )
                    }
                )
            }
        )
        stream = DecodedStreamObject()
        stream.set_data(f"BT /F1 24 Tf 100 700 Td ({text}) Tj ET".encode("ascii"))
        page[NameObject("/Contents")] = stream
    buffer = io.BytesIO()
    writer.write(buffer)
    return buffer.getvalue()


def _blank_pdf() -> bytes:
    """Crée un PDF sans texte extractible."""
    writer = PdfWriter()
    writer.add_blank_page(width=612, height=792)
    buffer = io.BytesIO()
    writer.write(buffer)
    return buffer.getvalue()


def _client(assistant: _Assistant | None = None) -> tuple[TestClient, _Assistant]:
    assistant = _Assistant() if assistant is None else assistant
    app = create_app(assistant)
    return TestClient(app), assistant


def test_list_documents_hides_internal_fields() -> None:
    assistant = _Assistant()
    assistant.rows = [
        {
            "document_id": "ancien",
            "filename": "cours-1.pdf",
            "stored_path": r"C:\secret\ancien.pdf",
            "sha256": "b" * 64,
        },
        {
            "document_id": "recent",
            "filename": "cours-2.pdf",
            "stored_path": "/secret/recent.pdf",
            "sha256": "c" * 64,
        },
    ]
    client, _assistant = _client(assistant)

    response = client.get("/documents")

    assert response.status_code == 200
    assert response.json() == [
        {"document_id": "ancien", "filename": "cours-1.pdf"},
        {"document_id": "recent", "filename": "cours-2.pdf"},
    ]
    assert "stored_path" not in response.text
    assert "sha256" not in response.text
    assert assistant.added == []
    assert assistant.removed == []


def test_list_documents_can_be_empty() -> None:
    client, assistant = _client()

    response = client.get("/documents")

    assert response.status_code == 200
    assert response.json() == []
    assert assistant.added == []


def test_upload_accepts_a_real_pdf_and_hides_internal_fields() -> None:
    client, assistant = _client()

    response = client.post(
        "/documents",
        files={"file": ("cours.pdf", _text_pdf("Bonjour"), "application/pdf")},
    )

    assert response.status_code == 201
    assert response.json() == {"document_id": "doc-1", "filename": "cours.pdf"}
    assert "stored_path" not in response.text
    assert "sha256" not in response.text
    assert len(assistant.added) == 1
    saved = assistant.added[0]
    assert saved.name == "cours.pdf"
    assert not saved.exists()


@pytest.mark.parametrize(
    "filename",
    ["secret.pdf", r"..\secret.pdf", "./secret.pdf", r"C:\Users\eleve\secret.pdf"],
)
def test_upload_keeps_only_the_file_name(filename: str) -> None:
    client, assistant = _client()

    response = client.post(
        "/documents",
        files={"file": (filename, _text_pdf("Algebre"), "application/pdf")},
    )

    assert response.status_code == 201
    assert response.json()["filename"] == "secret.pdf"
    assert assistant.added[0].name == "secret.pdf"


def test_upload_rejects_an_empty_file() -> None:
    client, assistant = _client()

    response = client.post(
        "/documents",
        files={"file": ("cours.pdf", b"", "application/pdf")},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Le fichier PDF est vide."
    assert assistant.added == []


def test_upload_rejects_a_file_that_is_not_a_pdf() -> None:
    client, assistant = _client()

    response = client.post(
        "/documents",
        files={"file": ("notes.txt", b"Ceci n'est pas un PDF.", "text/plain")},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Le fichier sélectionné doit être un PDF."
    assert assistant.added == []


def test_upload_rejects_pdf_bytes_with_the_wrong_type() -> None:
    client, assistant = _client()

    response = client.post(
        "/documents",
        files={"file": ("cours.pdf", _text_pdf("Bonjour"), "image/png")},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Le fichier sélectionné doit être un PDF."
    assert assistant.added == []


def test_upload_rejects_invalid_pdf_content() -> None:
    client, assistant = _client()

    response = client.post(
        "/documents",
        files={"file": ("cours.pdf", b"%PDF-1.4\npas un vrai document", "application/pdf")},
    )

    assert response.status_code == 400
    assert "invalide" in response.json()["detail"].lower()
    assert "stored_path" not in response.text
    assert assistant.added == []


def test_upload_rejects_a_pdf_without_extractable_text() -> None:
    client, assistant = _client()

    response = client.post(
        "/documents",
        files={"file": ("cours.pdf", _blank_pdf(), "application/pdf")},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Le PDF ne contient aucun texte extractible : cours.pdf"
    assert assistant.added == []


def test_upload_rejects_a_file_over_the_configured_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EDURAG_MAX_UPLOAD_BYTES", "64")
    client, assistant = _client()
    payload = _text_pdf("Bonjour")
    assert len(payload) > 64

    response = client.post(
        "/documents",
        files={"file": ("cours.pdf", payload, "application/pdf")},
    )

    assert response.status_code == 413
    assert response.json()["detail"] == "Le PDF dépasse la taille autorisée."
    assert assistant.added == []


def test_upload_reports_a_duplicate_document() -> None:
    assistant = _Assistant()
    assistant.duplicate = True
    client, _assistant = _client(assistant)

    response = client.post(
        "/documents",
        files={"file": ("cours.pdf", _text_pdf("Bonjour"), "application/pdf")},
    )

    assert response.status_code == 409
    assert response.json()["detail"] == "Ce document est déjà dans la bibliothèque."
    assert "sha256" not in response.text
    assert len(assistant.added) == 1


def test_delete_removes_an_existing_document() -> None:
    assistant = _Assistant()
    assistant.rows = [
        {
            "document_id": "doc-1",
            "filename": "cours.pdf",
            "stored_path": "/secret/cours.pdf",
            "sha256": "d" * 64,
        }
    ]
    client, _assistant = _client(assistant)

    response = client.delete("/documents/doc-1")

    assert response.status_code == 204
    assert response.content == b""
    assert assistant.removed == ["doc-1"]
    assert assistant.rows == []


def test_delete_reports_an_unknown_document() -> None:
    client, assistant = _client()

    response = client.delete("/documents/absent")

    assert response.status_code == 404
    assert response.json()["detail"] == "Document introuvable dans la bibliothèque."
    assert assistant.removed == ["absent"]


def test_openapi_documents_document_errors() -> None:
    client, _assistant = _client()
    paths = client.app.openapi()["paths"]

    assert "200" in paths["/documents"]["get"]["responses"]
    post = paths["/documents"]["post"]["responses"]
    assert {"201", "400", "409", "413"} <= set(post)
    delete = paths["/documents/{document_id}"]["delete"]["responses"]
    assert {"204", "404"} <= set(delete)


def test_document_api_does_not_load_ai_models() -> None:
    import src.api.routes.documents as documents

    source = Path(documents.__file__).read_text(encoding="utf-8")
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
    response = client.get("/documents")

    assert response.status_code == 200
    loaded = {
        name
        for name in sys.modules
        if name == "src.assistant" or name.startswith(_FORBIDDEN_MODULES)
    }
    assert loaded == before
