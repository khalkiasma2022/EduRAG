"""Tests de l'export Markdown d'une fiche enregistrée.

Les fiches sont des dictionnaires. Aucun modèle n'est chargé.
"""

from pathlib import Path

import pytest

from src.revision.export import RevisionExport


def _sheet(**changes: object) -> dict[str, object]:
    """Reproduit le dictionnaire renvoyé par RevisionStore.get()."""
    sheet = {
        "id": 4,
        "document_id": "doc-python",
        "filename": "cours_python.pdf",
        "page_start": 2,
        "page_end": 3,
        "language": "fr",
        "level": "beginner",
        "sheet_type": "short",
        "content": "Une variable garde une valeur.",
        "created_at": "2026-11-23T09:15:00.000000+00:00",
    }
    sheet.update(changes)
    return sheet


def test_to_markdown_contains_the_sheet_and_its_metadata() -> None:
    markdown = RevisionExport().to_markdown(_sheet())

    assert markdown.startswith("# Fiche de révision\n")
    assert "- Cours : cours_python.pdf" in markdown
    assert "- Pages : 2 à 3" in markdown
    assert "- Langue : Français" in markdown
    assert "- Niveau : Débutant" in markdown
    assert "- Type : Fiche courte" in markdown
    assert "- Date de création : 2026-11-23T09:15:00.000000+00:00" in markdown
    assert "Une variable garde une valeur." in markdown


def test_to_markdown_keeps_french_content() -> None:
    content = "L'été, une fonction peut être créée à partir d'une définition."
    markdown = RevisionExport().to_markdown(_sheet(content=content, sheet_type="complete", level="intermediate"))

    assert content in markdown
    assert "- Type : Fiche complète" in markdown
    assert "- Niveau : Intermédiaire" in markdown


def test_to_markdown_keeps_arabic_content() -> None:
    content = "المتغير يحتفظ بقيمة ويمكن استخدامه لاحقاً."
    markdown = RevisionExport().to_markdown(_sheet(content=content, language="ar", level="advanced"))

    assert content in markdown
    assert "- Langue : Arabe" in markdown
    assert "- Niveau : Avancé" in markdown


def test_export_markdown_writes_a_utf8_file(tmp_path: Path) -> None:
    exporter = RevisionExport()
    sheet = _sheet(content="Révision : été, à, ç, et المتغير.")
    output = tmp_path / "fiche.md"

    written = exporter.export_markdown(sheet, output)

    assert written == output
    assert output.read_text(encoding="utf-8") == exporter.to_markdown(sheet)
    assert "المتغير" in output.read_text(encoding="utf-8")


def test_export_markdown_creates_the_parent_directory(tmp_path: Path) -> None:
    output = tmp_path / "exports" / "nested" / "fiche.md"
    assert not output.parent.exists()

    written = RevisionExport().export_markdown(_sheet(), output)

    assert written == output
    assert output.is_file()
    assert output.read_text(encoding="utf-8").startswith("# Fiche de révision\n")


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"content": ""}, "contenu"),
        ({"content": None}, "contenu"),
        ({"filename": "  "}, "fichier"),
        ({"page_start": True, "page_end": 2}, "pages"),
        ({"page_start": 4, "page_end": 2}, "pages"),
        ({"language": "de"}, "langue"),
        ({"level": "expert"}, "niveau"),
        ({"sheet_type": "medium"}, "fiche"),
        ({"created_at": ""}, "date"),
        ({"id": 0}, "identifiant"),
        ({"document_id": ""}, "document"),
    ],
)
def test_export_rejects_invalid_sheets(tmp_path: Path, changes: dict[str, object], message: str) -> None:
    output = tmp_path / "fiche.md"

    with pytest.raises(ValueError, match=message):
        RevisionExport().export_markdown(_sheet(**changes), output)

    assert not output.exists()


@pytest.mark.parametrize("output_path", ["", "   "])
def test_export_rejects_an_empty_output_path(output_path: str) -> None:
    with pytest.raises(ValueError, match="vide"):
        RevisionExport().export_markdown(_sheet(), output_path)
