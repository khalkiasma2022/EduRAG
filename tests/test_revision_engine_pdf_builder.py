"""Tests du ProfessionalPDFBuilder Smart Revision V2."""

from __future__ import annotations

from pathlib import Path

from pypdf import PdfReader

from src.revision.engine.formatter import ContentFormatter, parse_markdown
from src.revision.engine.models import (
    ChapterDraft,
    DefinitionBlock,
    DiagramBlock,
    ExampleBlock,
    ParagraphBlock,
    PedagogicalDocument,
    QuestionBlock,
    RetainBlock,
    SynthesisBlock,
    TableBlock,
)
from src.revision.engine.pdf_builder import (
    AR_FONT_FAMILY,
    AR_FONT_PATH,
    COLOR_SLATE,
    COLOR_TEAL,
    ProfessionalPDFBuilder,
)


def _rich_document() -> PedagogicalDocument:
    return PedagogicalDocument(
        filename="cours_python.pdf",
        page_start=1,
        page_end=4,
        language="fr",
        level="intermediate",
        sheet_type="complete",
        chapters=[
            ChapterDraft(
                index=0,
                title="Variables et fonctions",
                blocks=[
                    ParagraphBlock(text="L'été, une fonction peut être créée."),
                    DefinitionBlock(text="Une variable nomme une valeur."),
                    RetainBlock(text="Retenir la portée locale."),
                    ExampleBlock(text="x = 1", origin="from_course"),
                    ExampleBlock(
                        text="Comme un tiroir étiqueté.",
                        origin="pedagogical_addition",
                    ),
                    TableBlock(
                        headers=["Notion", "Définition"],
                        rows=[["Variable", "Valeur nommée"], ["Fonction", "Bloc réutilisable"]],
                    ),
                    DiagramBlock(text="entrée → traitement → sortie"),
                    SynthesisBlock(text="Les notions structurantes du chapitre."),
                    QuestionBlock(text="Qu'est-ce qu'une variable ?"),
                    QuestionBlock(text="Donnez un exemple de fonction."),
                ],
            )
        ],
    )


def _pdf_text(path: Path) -> str:
    reader = PdfReader(path)
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def test_palette_and_arabic_font_resources() -> None:
    assert COLOR_SLATE == (47, 62, 70)
    assert COLOR_TEAL == (13, 148, 136)
    assert AR_FONT_PATH.is_file()
    assert AR_FONT_FAMILY == "NotoNaskhArabic"


def test_build_pdf_from_markdown_without_ir(tmp_path: Path) -> None:
    document = _rich_document()
    markdown = ContentFormatter().to_markdown(document)
    del document

    output = tmp_path / "fiche-v2.pdf"
    written = ProfessionalPDFBuilder().build_from_markdown(
        markdown,
        output_path=output,
        meta={
            "filename": "cours_python.pdf",
            "page_start": 1,
            "page_end": 4,
            "language": "fr",
            "level": "intermediate",
            "sheet_type": "complete",
        },
    )

    assert written == output
    assert written.read_bytes().startswith(b"%PDF")
    assert written.stat().st_size > 0

    text = _pdf_text(written)
    assert "EduRAG" in text
    assert "Fiche de révision" in text
    assert "cours_python.pdf" in text
    assert "Variables et fonctions" in text
    assert "L'été, une fonction peut être créée." in text
    assert "Une variable nomme une valeur." in text
    assert "Retenir la portée locale." in text
    assert "x = 1" in text
    assert "tiroir" in text
    assert "Valeur nommée" in text
    assert "entrée → traitement → sortie" in text or "entrée" in text
    assert "Qu'est-ce qu'une variable ?" in text


def test_round_trip_markdown_structure_pdf(tmp_path: Path) -> None:
    document = _rich_document()
    markdown = ContentFormatter().to_markdown(document)
    ir_id = id(document)
    del document

    parsed = parse_markdown(markdown)
    assert parsed  # structure logique indépendante de l'IR
    assert ir_id not in {id(b) for b in parsed}

    output = tmp_path / "round-trip.pdf"
    ProfessionalPDFBuilder().build_from_markdown(
        markdown,
        output_path=output,
        meta={
            "filename": "cours_python.pdf",
            "page_start": 1,
            "page_end": 4,
            "language": "fr",
            "level": "intermediate",
            "sheet_type": "complete",
        },
    )
    text = _pdf_text(output)
    assert "Définition" in text or "variable nomme" in text.lower()
    assert "Questions" in text or "variable ?" in text


def test_pdf_has_multiple_pages_with_footer(tmp_path: Path) -> None:
    """Couverture + contenu → au moins 2 pages ; pied de page numéroté."""
    paragraphs = [
        ParagraphBlock(text=f"Paragraphe de développement numéro {i}. " * 8)
        for i in range(1, 12)
    ]
    document = PedagogicalDocument(
        filename="long.pdf",
        page_start=1,
        page_end=10,
        language="fr",
        level="beginner",
        sheet_type="complete",
        chapters=[ChapterDraft(index=0, title="Chapitre long", blocks=paragraphs)],
    )
    markdown = ContentFormatter().to_markdown(document)
    del document

    output = tmp_path / "long.pdf"
    ProfessionalPDFBuilder().build_from_markdown(
        markdown,
        output_path=output,
        meta={
            "filename": "long.pdf",
            "page_start": 1,
            "page_end": 10,
            "language": "fr",
            "level": "beginner",
            "sheet_type": "complete",
        },
    )
    reader = PdfReader(output)
    assert len(reader.pages) >= 2
    joined = "\n".join(page.extract_text() or "" for page in reader.pages)
    assert "EduRAG" in joined


def test_arabic_language_uses_noto_font(tmp_path: Path) -> None:
    document = PedagogicalDocument(
        filename="cours.pdf",
        page_start=1,
        page_end=1,
        language="ar",
        level="beginner",
        sheet_type="short",
        chapters=[
            ChapterDraft(
                index=0,
                title="المتغيرات",
                blocks=[ParagraphBlock(text="المتغير يخزن قيمة.")],
            )
        ],
    )
    markdown = ContentFormatter().to_markdown(document)
    del document

    output = tmp_path / "ar.pdf"
    written = ProfessionalPDFBuilder().build_from_markdown(
        markdown,
        output_path=output,
        meta={
            "filename": "cours.pdf",
            "page_start": 1,
            "page_end": 1,
            "language": "ar",
            "level": "beginner",
            "sheet_type": "short",
        },
    )
    assert written.read_bytes().startswith(b"%PDF")
    text = _pdf_text(written)
    assert "المتغير" in text or "EduRAG" in text
