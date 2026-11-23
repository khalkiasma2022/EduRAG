"""Tests de l'export PDF d'une fiche enregistrée.

Ils vérifient le fichier produit et le texte qu'on peut en relire.
Ils ne prouvent pas, à eux seuls, que les lettres arabes sont visuellement reliées.
Aucun modèle n'est chargé.
"""

from pathlib import Path

import pytest
from pypdf import PdfReader

from src.revision.export import RevisionExport, _content_blocks


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


def _pdf_text(path: Path) -> str:
    """Relit le texte enregistré dans le PDF, dans l'ordre logique disponible."""
    reader = PdfReader(path)
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def test_export_pdf_writes_a_french_sheet(tmp_path: Path) -> None:
    content = "\n".join(
        [
            "# Variables",
            "L'été, une fonction peut être créée.",
            "- garder une valeur",
            "",
            "| Notion | Définition |",
            "| --- | --- |",
            "| Variable | Valeur nommée |",
        ]
    )
    output = tmp_path / "fiche-fr.pdf"

    written = RevisionExport().export_pdf(
        _sheet(content=content, language="fr", level="intermediate", sheet_type="complete"),
        output,
    )

    text = _pdf_text(written)
    assert written == output
    assert written.read_bytes().startswith(b"%PDF")
    assert written.stat().st_size > 0
    assert "Fiche de révision" in text
    assert "cours_python.pdf" in text
    assert "2 à 3" in text
    assert "Français" in text
    assert "Intermédiaire" in text
    assert "Fiche complète" in text
    assert "2026-11-23T09:15:00.000000+00:00" in text
    assert "L'été, une fonction peut être créée." in text
    assert "Variables" in text
    assert "garder une valeur" in text
    assert "Valeur nommée" in text


def test_export_pdf_writes_an_english_sheet(tmp_path: Path) -> None:
    output = tmp_path / "fiche-en.pdf"

    written = RevisionExport().export_pdf(
        _sheet(
            content="A variable stores a value.",
            language="en",
            level="advanced",
            sheet_type="short",
            filename="python_course.pdf",
        ),
        output,
    )

    text = _pdf_text(written)
    assert "A variable stores a value." in text
    assert "Anglais" in text
    assert "Avancé" in text
    assert "Fiche courte" in text
    assert "python_course.pdf" in text


def test_export_pdf_writes_an_arabic_sheet_in_logical_order(tmp_path: Path) -> None:
    content = "اللغة العربية تستخدم الحروف المتصلة."
    output = tmp_path / "fiche-ar.pdf"

    written = RevisionExport().export_pdf(
        _sheet(content=content, language="ar", level="beginner", sheet_type="complete"),
        output,
    )

    text = _pdf_text(written)
    assert written.stat().st_size > 0
    assert written.read_bytes().startswith(b"%PDF")
    assert b"NotoNaskhArabic" in written.read_bytes()
    assert "Arabe" in text
    assert "Fiche complète" in text
    assert _content_blocks(content) == [("paragraph", content)]


def test_export_pdf_keeps_numbers_and_bullets(tmp_path: Path) -> None:
    content = "\n".join(
        [
            "- puce simple",
            "* autre puce",
            "1. première étape",
            "2. deuxième étape",
            "10. étape suivante",
            "3) troisième forme",
        ]
    )
    output = tmp_path / "listes.pdf"

    written = RevisionExport().export_pdf(_sheet(content=content, language="fr"), output)

    text = _pdf_text(written)
    assert "• puce simple" in text
    assert "• autre puce" in text
    assert "1. première étape" in text
    assert "2. deuxième étape" in text
    assert "10. étape suivante" in text
    assert "3) troisième forme" in text
    assert _content_blocks(content) == [
        ("item", "puce simple"),
        ("item", "autre puce"),
        ("numbered", "1.", "première étape"),
        ("numbered", "2.", "deuxième étape"),
        ("numbered", "10.", "étape suivante"),
        ("numbered", "3)", "troisième forme"),
    ]


def test_export_pdf_keeps_arabic_list_numbers_in_logical_order(tmp_path: Path) -> None:
    content = "\n".join(
        [
            "- عنصر أول",
            "1. الخطوة الأولى",
            "2. الخطوة الثانية",
        ]
    )
    output = tmp_path / "listes-ar.pdf"

    written = RevisionExport().export_pdf(
        _sheet(content=content, language="ar", sheet_type="complete"),
        output,
    )

    assert written.stat().st_size > 0
    assert b"NotoNaskhArabic" in written.read_bytes()
    assert _content_blocks(content) == [
        ("item", "عنصر أول"),
        ("numbered", "1.", "الخطوة الأولى"),
        ("numbered", "2.", "الخطوة الثانية"),
    ]
    markdown = RevisionExport().to_markdown(_sheet(content=content, language="ar"))
    assert "- عنصر أول" in markdown
    assert "1. الخطوة الأولى" in markdown


def test_export_pdf_keeps_the_end_of_a_long_sheet(tmp_path: Path) -> None:
    lines = ["Cette ligne explique une notion importante du cours de révision."] * 140
    lines.append("Phrase finale conservée.")
    output = tmp_path / "fiche-longue.pdf"

    written = RevisionExport().export_pdf(_sheet(content="\n".join(lines)), output)

    reader = PdfReader(written)
    assert len(reader.pages) >= 2
    assert "Phrase finale conservée." in _pdf_text(written)


def test_export_pdf_creates_the_parent_directory(tmp_path: Path) -> None:
    output = tmp_path / "exports" / "nested" / "fiche.pdf"
    assert not output.parent.exists()

    written = RevisionExport().export_pdf(_sheet(), output)

    assert written == output
    assert output.is_file()
    assert output.stat().st_size > 0


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"content": ""}, "contenu"),
        ({"content": None}, "contenu"),
        ({"page_start": 4, "page_end": 2}, "pages"),
        ({"language": "de"}, "langue"),
        ({"created_at": ""}, "date"),
        ({"id": True}, "identifiant"),
    ],
)
def test_export_pdf_rejects_invalid_sheets(tmp_path: Path, changes: dict[str, object], message: str) -> None:
    output = tmp_path / "fiche.pdf"

    with pytest.raises(ValueError, match=message):
        RevisionExport().export_pdf(_sheet(**changes), output)

    assert not output.exists()


@pytest.mark.parametrize("output_path", ["", "   "])
def test_export_pdf_rejects_an_empty_output_path(output_path: str) -> None:
    with pytest.raises(ValueError, match="vide"):
        RevisionExport().export_pdf(_sheet(), output_path)


@pytest.mark.parametrize("name", ["fiche.md", "fiche.txt", "fiche"])
def test_export_pdf_rejects_a_wrong_extension(tmp_path: Path, name: str) -> None:
    output = tmp_path / name

    with pytest.raises(ValueError, match=r"\.pdf"):
        RevisionExport().export_pdf(_sheet(), output)

    assert not output.exists()


def test_markdown_export_still_writes_the_same_text(tmp_path: Path) -> None:
    exporter = RevisionExport()
    sheet = _sheet(content="L'été, une variable garde une valeur.")

    markdown = exporter.to_markdown(sheet)
    written = exporter.export_markdown(sheet, tmp_path / "fiche.md")

    assert markdown.startswith("# Fiche de révision\n")
    assert "L'été, une variable garde une valeur." in markdown
    assert written.read_text(encoding="utf-8") == markdown
