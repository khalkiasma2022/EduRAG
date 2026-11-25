"""Tests du ContentFormatter Smart Revision V2 (Markdown canonique + parse)."""

from __future__ import annotations

from src.revision.engine.formatter import (
    ContentFormatter,
    MdCallout,
    MdDiagram,
    MdHeading,
    MdParagraph,
    MdQuestions,
    MdSynthesis,
    MdTable,
    parse_markdown,
)
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


def _document(*blocks, title: str = "Variables") -> PedagogicalDocument:
    return PedagogicalDocument(
        filename="cours_python.pdf",
        page_start=1,
        page_end=3,
        language="fr",
        level="beginner",
        sheet_type="complete",
        chapters=[ChapterDraft(index=0, title=title, blocks=list(blocks))],
    )


def test_formats_document_and_chapter_headings() -> None:
    markdown = ContentFormatter().to_markdown(_document(ParagraphBlock(text="Intro.")))
    assert markdown.startswith("# Fiche de révision — cours_python.pdf\n")
    assert "## Variables\n" in markdown


def test_formats_paragraph() -> None:
    markdown = ContentFormatter().to_markdown(
        _document(ParagraphBlock(text="Une variable stocke une valeur."))
    )
    assert "Une variable stocke une valeur." in markdown


def test_formats_definition_callout() -> None:
    markdown = ContentFormatter().to_markdown(
        _document(DefinitionBlock(text="Une variable nomme une valeur."))
    )
    assert "> **Définition.** Une variable nomme une valeur." in markdown


def test_formats_retain_callout() -> None:
    markdown = ContentFormatter().to_markdown(
        _document(RetainBlock(text="Retenir la portée locale."))
    )
    assert "> **À retenir.** Retenir la portée locale." in markdown


def test_formats_course_and_pedagogical_examples() -> None:
    markdown = ContentFormatter().to_markdown(
        _document(
            ExampleBlock(text="x = 1", origin="from_course"),
            ExampleBlock(text="Penser à un tiroir étiqueté.", origin="pedagogical_addition"),
        )
    )
    assert "> **Exemple (cours).** x = 1" in markdown
    assert "> **Exemple (ajout pédagogique).** Penser à un tiroir étiqueté." in markdown


def test_formats_table_gfm() -> None:
    markdown = ContentFormatter().to_markdown(
        _document(
            TableBlock(
                headers=["Notion", "Rôle"],
                rows=[["Variable", "Valeur nommée"], ["Fonction", "Bloc réutilisable"]],
            )
        )
    )
    assert "| Notion | Rôle |" in markdown
    assert "| --- | --- |" in markdown
    assert "| Variable | Valeur nommée |" in markdown
    assert "| Fonction | Bloc réutilisable |" in markdown


def test_formats_diagram() -> None:
    markdown = ContentFormatter().to_markdown(
        _document(DiagramBlock(text="entrée → traitement → sortie"))
    )
    assert "**Schéma :** entrée → traitement → sortie" in markdown


def test_formats_synthesis_section() -> None:
    markdown = ContentFormatter().to_markdown(
        _document(SynthesisBlock(text="Les variables nomment des valeurs."))
    )
    assert "### Synthèse" in markdown
    assert "Les variables nomment des valeurs." in markdown


def test_formats_revision_questions() -> None:
    markdown = ContentFormatter().to_markdown(
        _document(
            QuestionBlock(text="Qu'est-ce qu'une variable ?"),
            QuestionBlock(text="Donnez un exemple."),
        )
    )
    assert "## Questions de révision" in markdown
    assert "1. Qu'est-ce qu'une variable ?" in markdown
    assert "2. Donnez un exemple." in markdown


def test_formats_subsection_heading_via_nested_structure() -> None:
    """Un second chapitre avec ### via synthèse / questions couvre la hiérarchie."""
    doc = PedagogicalDocument(
        filename="cours.pdf",
        page_start=1,
        page_end=1,
        language="fr",
        level="beginner",
        sheet_type="short",
        chapters=[
            ChapterDraft(
                index=0,
                title="Chapitre un",
                blocks=[
                    ParagraphBlock(text="Corps."),
                    SynthesisBlock(text="Point clé."),
                ],
            )
        ],
    )
    markdown = ContentFormatter().to_markdown(doc)
    assert "## Chapitre un" in markdown
    assert "### Synthèse" in markdown


def test_round_trip_preserves_block_types_and_french_accents() -> None:
    doc = PedagogicalDocument(
        filename="été_cours.pdf",
        page_start=1,
        page_end=2,
        language="fr",
        level="intermediate",
        sheet_type="complete",
        chapters=[
            ChapterDraft(
                index=0,
                title="Été et fonctions",
                blocks=[
                    ParagraphBlock(text="L'été, une fonction peut être créée."),
                    DefinitionBlock(text="Une fonction encapsule un comportement."),
                    RetainBlock(text="Mémoriser l'en-tête et le corps."),
                    ExampleBlock(text="def f(): return 1", origin="from_course"),
                    ExampleBlock(
                        text="Comme une recette de cuisine.",
                        origin="pedagogical_addition",
                    ),
                    TableBlock(
                        headers=["Notion", "Définition"],
                        rows=[["Paramètre", "Valeur d'entrée"], ["Retour", "Résultat"]],
                    ),
                    DiagramBlock(text="appel → corps → retour"),
                    SynthesisBlock(text="Les fonctions structurent le programme."),
                    QuestionBlock(text="À quoi sert un paramètre ?"),
                    QuestionBlock(text="Qu'est-ce qu'une valeur de retour ?"),
                ],
            )
        ],
    )
    markdown = ContentFormatter().to_markdown(doc)
    del doc

    parsed = parse_markdown(markdown)

    assert isinstance(parsed[0], MdHeading) and parsed[0].level == 1
    assert "été_cours.pdf" in parsed[0].text
    assert any(
        isinstance(b, MdHeading) and b.level == 2 and b.text == "Été et fonctions"
        for b in parsed
    )
    assert any(
        isinstance(b, MdParagraph) and b.text == "L'été, une fonction peut être créée."
        for b in parsed
    )
    assert any(
        isinstance(b, MdCallout)
        and b.kind == "definition"
        and "encapsule" in b.text
        for b in parsed
    )
    assert any(
        isinstance(b, MdCallout) and b.kind == "retain" and "en-tête" in b.text
        for b in parsed
    )
    assert any(
        isinstance(b, MdCallout) and b.kind == "example_course" and "def f()" in b.text
        for b in parsed
    )
    assert any(
        isinstance(b, MdCallout)
        and b.kind == "example_added"
        and "recette" in b.text
        for b in parsed
    )
    tables = [b for b in parsed if isinstance(b, MdTable)]
    assert len(tables) == 1
    assert tables[0].headers == ["Notion", "Définition"]
    assert tables[0].rows == [
        ["Paramètre", "Valeur d'entrée"],
        ["Retour", "Résultat"],
    ]
    assert any(
        isinstance(b, MdDiagram) and "appel → corps → retour" in b.text for b in parsed
    )
    assert any(
        isinstance(b, MdSynthesis)
        and "structurent" in " ".join(b.items)
        for b in parsed
    )
    questions = [b for b in parsed if isinstance(b, MdQuestions)]
    assert len(questions) == 1
    assert questions[0].items == [
        "À quoi sert un paramètre ?",
        "Qu'est-ce qu'une valeur de retour ?",
    ]


def test_parse_does_not_require_ir_object() -> None:
    markdown = "\n".join(
        [
            "# Support",
            "## Chapitre",
            "> **Définition.** Texte.",
            "> **Exemple (cours).** x = 1",
            "> **Exemple (ajout pédagogique).** Analogie.",
            "**Schéma :** a → b",
            "### Synthèse",
            "- Point un",
            "## Questions de révision",
            "1. Question ?",
        ]
    )
    parsed = parse_markdown(markdown)
    kinds = [type(b).__name__ for b in parsed]
    assert "MdHeading" in kinds
    assert "MdCallout" in kinds
    assert "MdDiagram" in kinds
    assert "MdSynthesis" in kinds
    assert "MdQuestions" in kinds
