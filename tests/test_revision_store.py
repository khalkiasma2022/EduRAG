"""Tests de l'enregistrement SQLite des fiches de révision.

Les fiches sont des dictionnaires. Aucun modèle n'est chargé.
"""

import sqlite3
from pathlib import Path

import pytest

from src.revision.store import RevisionStore


def _sheet(**changes: object) -> dict[str, object]:
    """Reproduit le dictionnaire renvoyé par RevisionSheet.generate()."""
    sheet = {
        "content": "Une variable garde une valeur.",
        "document_id": "doc-python",
        "filename": "cours_python.pdf",
        "page_start": 2,
        "page_end": 3,
        "language": "fr",
        "level": "beginner",
        "sheet_type": "short",
    }
    sheet.update(changes)
    return sheet


def test_store_creates_the_database_and_the_table(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "revisions.sqlite"
    assert not path.exists()

    RevisionStore(path)

    assert path.is_file()
    connection = sqlite3.connect(path)
    try:
        table = connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'revision_sheets'"
        ).fetchone()
        columns = [row[1] for row in connection.execute("PRAGMA table_info(revision_sheets)")]
    finally:
        connection.close()

    assert table == ("revision_sheets",)
    assert columns == [
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
    ]


def test_save_and_get_keep_the_sheet(tmp_path: Path) -> None:
    store = RevisionStore(tmp_path / "revisions.sqlite")

    saved = store.save(_sheet())
    loaded = store.get(saved["id"])

    assert loaded == saved
    assert loaded["content"] == "Une variable garde une valeur."
    assert isinstance(loaded["id"], int)
    assert loaded["id"] > 0


def test_list_sheets_returns_the_newest_first(tmp_path: Path) -> None:
    store = RevisionStore(tmp_path / "revisions.sqlite")
    first = store.save(_sheet(content="Première fiche"))
    second = store.save(_sheet(content="Deuxième fiche", sheet_type="complete"))

    listed = store.list_sheets()

    assert [sheet["id"] for sheet in listed] == [second["id"], first["id"]]
    assert [sheet["content"] for sheet in listed] == ["Deuxième fiche", "Première fiche"]


def test_saved_sheet_keeps_all_metadata(tmp_path: Path) -> None:
    store = RevisionStore(tmp_path / "revisions.sqlite")
    source = _sheet(
        content="Définition d'une fonction.",
        document_id="doc-42",
        filename="cours_python.pdf",
        page_start=1,
        page_end=3,
        language="en",
        level="advanced",
        sheet_type="complete",
    )

    saved = store.save(source)

    assert saved["document_id"] == "doc-42"
    assert saved["filename"] == "cours_python.pdf"
    assert saved["page_start"] == 1
    assert saved["page_end"] == 3
    assert saved["language"] == "en"
    assert saved["level"] == "advanced"
    assert saved["sheet_type"] == "complete"
    assert saved["content"] == "Définition d'une fonction."
    assert isinstance(saved["created_at"], str)
    assert saved["created_at"] != ""


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"content": ""}, "contenu"),
        ({"content": "   "}, "contenu"),
        ({"content": None}, "contenu"),
        ({"document_id": ""}, "document"),
        ({"filename": "  "}, "fichier"),
        ({"page_start": 0, "page_end": 1}, "pages"),
        ({"page_start": True, "page_end": 2}, "pages"),
        ({"page_start": 3, "page_end": 2}, "pages"),
        ({"language": "de"}, "langue"),
        ({"level": "expert"}, "niveau"),
        ({"sheet_type": "medium"}, "fiche"),
    ],
)
def test_save_rejects_invalid_sheets(tmp_path: Path, changes: dict[str, object], message: str) -> None:
    store = RevisionStore(tmp_path / "revisions.sqlite")

    with pytest.raises(ValueError, match=message):
        store.save(_sheet(**changes))

    assert store.list_sheets() == []


def test_save_rejects_a_value_that_is_not_a_sheet(tmp_path: Path) -> None:
    store = RevisionStore(tmp_path / "revisions.sqlite")

    with pytest.raises(ValueError, match="invalide"):
        store.save("pas une fiche")

    assert store.list_sheets() == []


def test_sheet_remains_after_the_store_is_reopened(tmp_path: Path) -> None:
    path = tmp_path / "revisions.sqlite"
    saved = RevisionStore(path).save(_sheet(content="Fiche conservée"))

    reopened = RevisionStore(path)

    assert reopened.get(saved["id"]) == saved
    assert reopened.list_sheets() == [saved]
