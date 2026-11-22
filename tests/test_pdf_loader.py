"""Tests du chargement du texte des PDF.

Ils couvrent l'extraction et les erreurs de lecture.
Le découpage en chunks n'est pas testé ici.
"""

from pathlib import Path

import pytest
from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

from src.ingestion.pdf_loader import load_pdf


def _write_text_pdf(path: Path, pages: list[str]) -> None:
    """Crée un petit PDF dont chaque entrée de `pages` devient une page."""
    writer = PdfWriter()
    for text in pages:
        page = writer.add_blank_page(width=612, height=792)
        # Helvetica est une police standard : le texte reste extractible sans fichier de police.
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
    with path.open("wb") as handle:
        writer.write(handle)


def _write_blank_pdf(path: Path) -> None:
    """Crée un PDF valide, avec une page blanche et aucun texte."""
    writer = PdfWriter()
    writer.add_blank_page(width=612, height=792)
    with path.open("wb") as handle:
        writer.write(handle)


def test_load_pdf_extracts_text(tmp_path: Path) -> None:
    pdf_path = tmp_path / "cours.pdf"
    _write_text_pdf(pdf_path, ["Bonjour EduRAG"])

    pages = load_pdf(str(pdf_path))

    assert len(pages) == 1
    assert pages[0]["text"] == "Bonjour EduRAG"
    assert pages[0]["filename"] == "cours.pdf"


def test_load_pdf_extracts_each_page(tmp_path: Path) -> None:
    pdf_path = tmp_path / "cours.pdf"
    _write_text_pdf(pdf_path, ["Page un", "Page deux", "Page trois"])

    pages = load_pdf(pdf_path)

    assert [page["text"] for page in pages] == ["Page un", "Page deux", "Page trois"]


def test_load_pdf_keeps_page_numbers(tmp_path: Path) -> None:
    pdf_path = tmp_path / "lecon.pdf"
    _write_text_pdf(pdf_path, ["Introduction", "Developpement"])

    pages = load_pdf(pdf_path)

    assert [page["page_number"] for page in pages] == [1, 2]
    assert {page["filename"] for page in pages} == {"lecon.pdf"}


def test_load_pdf_missing_file(tmp_path: Path) -> None:
    missing = tmp_path / "absent.pdf"

    with pytest.raises(FileNotFoundError):
        load_pdf(missing)


def test_load_pdf_invalid_file(tmp_path: Path) -> None:
    invalid = tmp_path / "notes.pdf"
    invalid.write_text("ceci n'est pas un PDF", encoding="utf-8")

    with pytest.raises(ValueError, match="invalide"):
        load_pdf(invalid)


def test_load_pdf_empty_pdf(tmp_path: Path) -> None:
    blank = tmp_path / "vide.pdf"
    _write_blank_pdf(blank)

    with pytest.raises(ValueError, match="texte"):
        load_pdf(blank)
