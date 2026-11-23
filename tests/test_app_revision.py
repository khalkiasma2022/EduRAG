"""Tests des fonctions Streamlit de Smart Revision.

Ils utilisent un faux assistant et des fichiers temporaires.
Ils n'appellent pas get_assistant() et ne chargent aucun modèle.
"""

import tempfile
from pathlib import Path

import pytest

import app as app_module


_DOCUMENTS = Path(__file__).resolve().parents[1] / "data" / "documents"
_MISSING_COURSE = (
    "Le cours PDF d'origine n'est plus dans la bibliothèque, "
    "mais cette fiche reste consultable."
)


class _Upload:
    """Imite le fichier remis par la zone d'import."""

    def __init__(self, name: str, data: bytes) -> None:
        self.name = name
        self.data = data

    def getvalue(self) -> bytes:
        return self.data


class _Assistant:
    """Enregistre les appels sans créer EduRAGAssistant."""

    def __init__(self, sheets: list[dict] | None = None) -> None:
        self.paths: list[Path] = []
        self.created: list[dict[str, object]] = []
        self.opened: list[object] = []
        self.exports: list[tuple[str, object, Path]] = []
        self.sheets = list(sheets or [])

    def add_pdf(self, file_path: str | Path) -> dict[str, str]:
        path = Path(file_path)
        self.paths.append(path)
        assert path.is_file()
        assert "\\" not in path.name and "/" not in path.name
        temp_root = Path(tempfile.gettempdir()).resolve()
        assert path.resolve().parent.parent == temp_root
        assert _DOCUMENTS.resolve() not in path.resolve().parents
        if path.read_bytes() == b"bad":
            raise ValueError("Ce document est déjà dans la bibliothèque.")
        return {"filename": path.name, "document_id": "doc-1"}

    def create_revision(
        self,
        document_id: str,
        page_start: int | None = None,
        page_end: int | None = None,
        language: str = "fr",
        level: str = "beginner",
        sheet_type: str = "complete",
    ) -> dict[str, str | int]:
        self.created.append(
            {
                "document_id": document_id,
                "page_start": page_start,
                "page_end": page_end,
                "language": language,
                "level": level,
                "sheet_type": sheet_type,
            }
        )
        if document_id == "trop":
            raise ValueError("Le cours sélectionné est trop volumineux.")
        return _sheet(len(self.created), document_id, language, level, sheet_type, page_start, page_end)

    def get_revision(self, sheet_id: object) -> dict[str, str | int]:
        self.opened.append(sheet_id)
        for sheet in self.sheets:
            if sheet["id"] == sheet_id:
                return dict(sheet)
        raise ValueError("Fiche introuvable.")

    def export_revision_markdown(self, sheet_id: object, output_path: str | Path) -> Path:
        return self._write_export("markdown", sheet_id, output_path, f"# Fiche {sheet_id}\n".encode("utf-8"))

    def export_revision_pdf(self, sheet_id: object, output_path: str | Path) -> Path:
        return self._write_export("pdf", sheet_id, output_path, b"%PDF-1.4 fiche " + str(sheet_id).encode("ascii"))

    def _write_export(self, kind: str, sheet_id: object, output_path: str | Path, data: bytes) -> Path:
        path = Path(output_path)
        self.exports.append((kind, sheet_id, path))
        assert _DOCUMENTS.resolve() not in path.resolve().parents
        if sheet_id == 9:
            raise ValueError("Fiche introuvable.")
        path.write_bytes(data)
        return path


def _sheet(
    sheet_id: int,
    document_id: str = "doc-1",
    language: str = "fr",
    level: str = "beginner",
    sheet_type: str = "short",
    page_start: int | None = 1,
    page_end: int | None = 1,
    filename: str = "cours.pdf",
    created_at: str = "2026-11-23T12:00:00.000000+00:00",
) -> dict[str, str | int]:
    """Construit une fiche déjà enregistrée."""
    return {
        "id": sheet_id,
        "document_id": document_id,
        "filename": filename,
        "created_at": created_at,
        "language": language,
        "level": level,
        "sheet_type": sheet_type,
        "page_start": 1 if page_start is None else page_start,
        "page_end": 3 if page_end is None else page_end,
        "content": f"Fiche {sheet_id}",
    }


@pytest.fixture(autouse=True)
def _block_real_assistant(monkeypatch: pytest.MonkeyPatch) -> None:
    """Interdit le chargement du véritable assistant."""

    def forbidden() -> None:
        raise AssertionError("get_assistant ne doit pas être appelé.")

    monkeypatch.setattr(app_module, "get_assistant", forbidden)


def test_uploaded_pdf_is_passed_to_add_pdf_and_removed() -> None:
    assistant = _Assistant()

    added = app_module._store_uploaded_pdf(assistant, _Upload("cours.pdf", b"%PDF-1.4"))

    assert added["filename"] == "cours.pdf"
    assert assistant.paths[0].name == "cours.pdf"
    assert not assistant.paths[0].exists()
    assert not assistant.paths[0].parent.exists()


def test_uploaded_pdf_is_removed_when_add_pdf_fails() -> None:
    assistant = _Assistant()

    with pytest.raises(ValueError, match="déjà"):
        app_module._store_uploaded_pdf(assistant, _Upload(r"..\secret.pdf", b"bad"))

    assert assistant.paths[0].name == "secret.pdf"
    assert not assistant.paths[0].exists()
    assert not assistant.paths[0].parent.exists()


@pytest.mark.parametrize(
    "uploaded_name",
    [
        "secret.pdf",
        r"..\secret.pdf",
        r".\secret.pdf",
        "../secret.pdf",
        "./secret.pdf",
        "dossier/secret.pdf",
        r"dossier\secret.pdf",
        r"C:\Users\eleve\secret.pdf",
        "/tmp/dossier/secret.pdf",
    ],
)
def test_uploaded_pdf_name_is_portable_across_windows_and_linux(uploaded_name: str) -> None:
    assistant = _Assistant()

    added = app_module._store_uploaded_pdf(assistant, _Upload(uploaded_name, b"%PDF-1.4"))

    assert added["filename"] == "secret.pdf"
    assert assistant.paths[0].name == "secret.pdf"
    assert not assistant.paths[0].exists()
    assert not assistant.paths[0].parent.exists()


def test_non_pdf_upload_is_refused_before_add_pdf() -> None:
    assistant = _Assistant()

    with pytest.raises(ValueError, match="PDF"):
        app_module._store_uploaded_pdf(assistant, _Upload("notes.txt", b"texte"))

    assert assistant.paths == []


def test_whole_document_sends_no_page_bounds() -> None:
    assistant = _Assistant()

    saved = app_module._create_selected_revision(
        assistant, "doc-1", "short", "ar", "advanced", True, 9, 1
    )

    assert assistant.created == [
        {
            "document_id": "doc-1",
            "page_start": None,
            "page_end": None,
            "language": "ar",
            "level": "advanced",
            "sheet_type": "short",
        }
    ]
    assert saved["id"] == 1


def test_valid_page_interval_is_forwarded() -> None:
    assistant = _Assistant()

    app_module._create_selected_revision(
        assistant, "doc-1", "complete", "en", "intermediate", False, 2, 4
    )

    assert assistant.created[-1]["page_start"] == 2
    assert assistant.created[-1]["page_end"] == 4
    assert assistant.created[-1]["language"] == "en"
    assert assistant.created[-1]["level"] == "intermediate"
    assert assistant.created[-1]["sheet_type"] == "complete"


@pytest.mark.parametrize(("start", "end"), [(5, 2), (None, 3), (2, None)])
def test_invalid_page_interval_does_not_create_a_revision(start: object, end: object) -> None:
    assistant = _Assistant()

    with pytest.raises(ValueError):
        app_module._create_selected_revision(
            assistant, "doc-1", "short", "fr", "beginner", False, start, end
        )

    assert assistant.created == []


def test_revision_id_changes_only_after_a_successful_generation() -> None:
    assistant = _Assistant()
    state: dict[str, object] = {"revision_id": 3}

    saved = app_module._create_selected_revision(
        assistant, "doc-1", "short", "fr", "beginner", True, None, None
    )
    app_module._remember_revision(state, saved)

    assert len(assistant.created) == 1
    assert state["revision_id"] == saved["id"]


def test_failed_generation_keeps_the_previous_revision_id() -> None:
    assistant = _Assistant()
    state: dict[str, object] = {"revision_id": 3}

    with pytest.raises(ValueError, match="volumineux"):
        saved = app_module._create_selected_revision(
            assistant, "trop", "complete", "fr", "beginner", True, None, None
        )
        app_module._remember_revision(state, saved)

    assert len(assistant.created) == 1
    assert state["revision_id"] == 3


def test_saved_sheets_keep_their_order_and_visible_information() -> None:
    sheets = [
        _sheet(2, filename="recent.pdf", language="ar", level="advanced", sheet_type="complete", created_at="2026-11-23T12:02:00.000000+00:00"),
        _sheet(1, filename="ancien.pdf", language="fr", level="beginner", sheet_type="short", created_at="2026-11-23T12:01:00.000000+00:00"),
    ]

    summaries = [app_module._sheet_summary(sheet) for sheet in sheets]

    assert [sheet["id"] for sheet in sheets] == [2, 1]
    assert summaries == [
        "recent.pdf — 2026-11-23T12:02:00.000000+00:00 — Fiche complète — Arabe — Avancé",
        "ancien.pdf — 2026-11-23T12:01:00.000000+00:00 — Fiche courte — Français — Débutant",
    ]
    assert "stored_path" not in summaries[0]


def test_reopening_a_sheet_uses_get_revision_and_a_separate_id() -> None:
    older = _sheet(1, document_id="doc-absent", filename="ancien.pdf")
    assistant = _Assistant([older])
    state: dict[str, object] = {"revision_id": 2}

    opened = app_module._open_saved_revision(assistant, 1)
    app_module._remember_opened_revision(state, opened["id"])

    assert opened["content"] == "Fiche 1"
    assert assistant.opened == [1]
    assert assistant.created == []
    assert state["revision_id"] == 2
    assert state["opened_revision_id"] == 1


def test_missing_source_course_is_reported_while_the_sheet_stays_readable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sheet = _sheet(1, document_id="doc-absent")
    documents = [{"document_id": "doc-1", "filename": "cours.pdf"}]
    messages: list[str] = []
    monkeypatch.setattr(app_module.st, "subheader", lambda *args, **kwargs: None)
    monkeypatch.setattr(app_module.st, "write", lambda *args, **kwargs: None)
    monkeypatch.setattr(app_module.st, "markdown", lambda *args, **kwargs: None)
    monkeypatch.setattr(app_module.st, "info", lambda text: messages.append(text))
    assistant = _Assistant([sheet])

    assert app_module._course_is_missing(sheet, documents) is True
    app_module._show_revision(sheet, documents)
    reopened = app_module._open_saved_revision(assistant, sheet["id"])

    assert messages == [_MISSING_COURSE]
    assert reopened["content"] == "Fiche 1"
    assert assistant.created == []


def test_exports_use_each_sheet_id_then_the_cached_bytes() -> None:
    assistant = _Assistant()
    state: dict[str, object] = {"revision_id": 4, "opened_revision_id": 2}

    new_markdown = app_module._cached_export(state, assistant, state["revision_id"], "markdown")
    new_markdown_again = app_module._cached_export(state, assistant, state["revision_id"], "markdown")
    new_pdf = app_module._cached_export(state, assistant, state["revision_id"], "pdf")
    opened_markdown = app_module._cached_export(state, assistant, state["opened_revision_id"], "markdown")

    assert new_markdown == new_markdown_again == b"# Fiche 4\n"
    assert new_pdf.startswith(b"%PDF-1.4") and b"fiche 4" in new_pdf
    assert opened_markdown == b"# Fiche 2\n"
    assert assistant.exports[:3] == [
        ("markdown", 4, assistant.exports[0][2]),
        ("pdf", 4, assistant.exports[1][2]),
        ("markdown", 2, assistant.exports[2][2]),
    ]
    assert [call[0:2] for call in assistant.exports] == [
        ("markdown", 4),
        ("pdf", 4),
        ("markdown", 2),
    ]
    assert assistant.created == []
    assert state["revision_id"] == 4
    assert state["opened_revision_id"] == 2
    assert all(not path.exists() for _kind, _sheet_id, path in assistant.exports)
    assert all(not path.parent.exists() for _kind, _sheet_id, path in assistant.exports)


def test_missing_sheet_export_raises_and_leaves_no_temporary_file() -> None:
    assistant = _Assistant()

    with pytest.raises(ValueError, match="introuvable"):
        app_module._cached_export({}, assistant, 9, "markdown")

    assert assistant.exports[0][0:2] == ("markdown", 9)
    assert not assistant.exports[0][2].exists()
    assert assistant.created == []
