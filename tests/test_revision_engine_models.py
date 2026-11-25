"""Tests des modèles IR Pydantic du moteur Smart Revision V2."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.revision.engine.models import (
    ChapterDraft,
    ChapterPlan,
    Concept,
    ContentBlock,
    CourseModel,
    CoursePage,
    CourseRelation,
    CourseUnit,
    DefinitionBlock,
    ExampleBlock,
    LearningObjective,
    ParagraphBlock,
    PedagogicalDocument,
    PedagogicalPlan,
    QualityIssue,
    QualityReport,
    RetainBlock,
)


def test_course_page_requires_positive_page_number() -> None:
    page = CoursePage(page_number=1, text="Bonjour", filename="cours.pdf")
    assert page.page_number == 1

    with pytest.raises(ValidationError):
        CoursePage(page_number=0, text="x", filename="cours.pdf")


def test_concept_kind_is_restricted() -> None:
    concept = Concept(
        id="c1",
        label="Variable",
        kind="definition",
        pages=[1],
        essential=True,
    )
    assert concept.depends_on == []

    with pytest.raises(ValidationError):
        Concept(id="c2", label="X", kind="unknown", pages=[1], essential=False)


def test_content_block_discriminator_paragraph() -> None:
    block = ContentBlock.model_validate(
        {"type": "paragraph", "text": "Explication courte.", "source_pages": [1, 2]}
    )
    assert isinstance(block, ParagraphBlock)
    assert block.type == "paragraph"


def test_content_block_discriminator_definition_and_retain() -> None:
    definition = ContentBlock.model_validate(
        {"type": "definition", "text": "Une variable stocke une valeur.", "source_pages": [2]}
    )
    retain = ContentBlock.model_validate(
        {"type": "retain", "text": "Retenir la portée.", "source_pages": [2]}
    )
    assert isinstance(definition, DefinitionBlock)
    assert isinstance(retain, RetainBlock)


def test_example_block_requires_origin() -> None:
    course_example = ContentBlock.model_validate(
        {
            "type": "example",
            "text": "x = 1",
            "origin": "from_course",
            "source_pages": [3],
        }
    )
    assert isinstance(course_example, ExampleBlock)
    assert course_example.origin == "from_course"

    with pytest.raises(ValidationError):
        ContentBlock.model_validate({"type": "example", "text": "y = 2", "source_pages": [3]})


def test_pedagogical_plan_and_document_round_trip_fields() -> None:
    objective = LearningObjective(text="Comprendre les variables", concept_ids=["c1"])
    chapter = ChapterPlan(
        index=0,
        title="Variables",
        objectives=[objective],
        concept_ids=["c1"],
        page_refs=[1, 2],
        needs_example=True,
        needs_table=False,
        needs_questions=False,
        depth="short",
    )
    plan = PedagogicalPlan(
        language="fr",
        level="beginner",
        sheet_type="short",
        chapters=[chapter],
    )
    draft = ChapterDraft(
        index=0,
        title="Variables",
        blocks=[
            ParagraphBlock(text="Une variable nomme une valeur.", source_pages=[1]),
        ],
        concept_ids_covered=["c1"],
    )
    document = PedagogicalDocument(
        filename="cours.pdf",
        page_start=1,
        page_end=2,
        language="fr",
        level="beginner",
        sheet_type="short",
        chapters=[draft],
    )
    assert plan.chapters[0].title == "Variables"
    assert document.chapters[0].blocks[0].type == "paragraph"


def test_course_model_and_quality_report() -> None:
    model = CourseModel(
        units=[CourseUnit(text="Intro", page_start=1, page_end=1, title="Intro")],
        concepts=[
            Concept(id="c1", label="Fonction", kind="method", pages=[2], essential=True),
        ],
        relations=[
            CourseRelation(
                source_id="c1",
                target_id="c1",
                relation_type="self",
                pages=[2],
            )
        ],
        themes=["Programmation"],
        essential_concept_ids=["c1"],
    )
    report = QualityReport(
        ok=False,
        issues=[QualityIssue(code="coverage", severity="error", message="Manque c1")],
        rewrite_chapter_indexes=[0],
    )
    assert model.essential_concept_ids == ["c1"]
    assert report.issues[0].severity == "error"
