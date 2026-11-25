"""Modèles IR Pydantic du moteur Smart Revision V2."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, Field, TypeAdapter


class CoursePage(BaseModel):
    """Page extraite d'un PDF de cours."""

    page_number: int = Field(ge=1)
    text: str
    filename: str


class CourseUnit(BaseModel):
    """Unité de texte nettoyée, éventuellement titrée."""

    text: str
    page_start: int = Field(ge=1)
    page_end: int = Field(ge=1)
    title: str | None = None


ConceptKind = Literal["definition", "formula", "method", "mechanism", "relation", "other"]


class Concept(BaseModel):
    """Notion identifiée dans le cours."""

    id: str
    label: str
    kind: ConceptKind
    pages: list[int] = Field(default_factory=list)
    essential: bool = False
    depends_on: list[str] = Field(default_factory=list)
    notes: str = ""


class CourseRelation(BaseModel):
    """Lien entre deux concepts."""

    source_id: str
    target_id: str
    relation_type: str
    pages: list[int] = Field(default_factory=list)


class CourseModel(BaseModel):
    """Représentation structurée du cours."""

    units: list[CourseUnit] = Field(default_factory=list)
    concepts: list[Concept] = Field(default_factory=list)
    relations: list[CourseRelation] = Field(default_factory=list)
    themes: list[str] = Field(default_factory=list)
    essential_concept_ids: list[str] = Field(default_factory=list)


class LearningObjective(BaseModel):
    """Objectif d'apprentissage lié à des concepts."""

    text: str
    concept_ids: list[str] = Field(default_factory=list)


ChapterDepth = Literal["short", "complete"]


class ChapterPlan(BaseModel):
    """Chapitre prévu par le planificateur pédagogique."""

    index: int = Field(ge=0)
    title: str
    objectives: list[LearningObjective] = Field(default_factory=list)
    concept_ids: list[str] = Field(default_factory=list)
    page_refs: list[int] = Field(default_factory=list)
    needs_example: bool = False
    needs_table: bool = False
    needs_questions: bool = False
    depth: ChapterDepth = "short"


class PedagogicalPlan(BaseModel):
    """Plan pédagogique global."""

    language: str
    level: str
    sheet_type: Literal["short", "complete"]
    chapters: list[ChapterPlan] = Field(default_factory=list)


ExampleOrigin = Literal["from_course", "pedagogical_addition"]


class ParagraphBlock(BaseModel):
    """Paragraphe explicatif."""

    type: Literal["paragraph"] = "paragraph"
    text: str
    source_pages: list[int] = Field(default_factory=list)


class DefinitionBlock(BaseModel):
    """Définition encadrée."""

    type: Literal["definition"] = "definition"
    text: str
    source_pages: list[int] = Field(default_factory=list)


class ExampleBlock(BaseModel):
    """Exemple tiré du cours ou ajout pédagogique."""

    type: Literal["example"] = "example"
    text: str
    origin: ExampleOrigin
    source_pages: list[int] = Field(default_factory=list)


class RetainBlock(BaseModel):
    """Bloc « À retenir »."""

    type: Literal["retain"] = "retain"
    text: str
    source_pages: list[int] = Field(default_factory=list)


class TableBlock(BaseModel):
    """Tableau comparatif."""

    type: Literal["table"] = "table"
    headers: list[str] = Field(default_factory=list)
    rows: list[list[str]] = Field(default_factory=list)
    source_pages: list[int] = Field(default_factory=list)


class DiagramBlock(BaseModel):
    """Schéma décrit textuellement."""

    type: Literal["diagram"] = "diagram"
    text: str
    source_pages: list[int] = Field(default_factory=list)


class SynthesisBlock(BaseModel):
    """Synthèse de chapitre ou finale."""

    type: Literal["synthesis"] = "synthesis"
    text: str
    source_pages: list[int] = Field(default_factory=list)


class QuestionBlock(BaseModel):
    """Question de révision."""

    type: Literal["question"] = "question"
    text: str
    source_pages: list[int] = Field(default_factory=list)


ContentBlockUnion = Annotated[
    ParagraphBlock
    | DefinitionBlock
    | ExampleBlock
    | RetainBlock
    | TableBlock
    | DiagramBlock
    | SynthesisBlock
    | QuestionBlock,
    Field(discriminator="type"),
]

_CONTENT_BLOCK_ADAPTER: TypeAdapter[ContentBlockUnion] = TypeAdapter(ContentBlockUnion)


class ContentBlock:
    """Façade de validation pour l'union discriminée des blocs."""

    @staticmethod
    def model_validate(data: object) -> ContentBlockUnion:
        return _CONTENT_BLOCK_ADAPTER.validate_python(data)


class ChapterDraft(BaseModel):
    """Chapitre rédigé, prêt à formater."""

    index: int = Field(ge=0)
    title: str
    blocks: list[
        ParagraphBlock
        | DefinitionBlock
        | ExampleBlock
        | RetainBlock
        | TableBlock
        | DiagramBlock
        | SynthesisBlock
        | QuestionBlock
    ] = Field(default_factory=list)
    concept_ids_covered: list[str] = Field(default_factory=list)


class PedagogicalDocument(BaseModel):
    """Document pédagogique complet avant formatage Markdown."""

    filename: str
    page_start: int = Field(ge=1)
    page_end: int = Field(ge=1)
    language: str
    level: str
    sheet_type: Literal["short", "complete"]
    chapters: list[ChapterDraft] = Field(default_factory=list)


class QualityIssue(BaseModel):
    """Problème détecté par le contrôle qualité."""

    code: str
    severity: Literal["error", "warning"]
    message: str
    chapter_index: int | None = None


class QualityReport(BaseModel):
    """Résultat du contrôle qualité."""

    ok: bool
    issues: list[QualityIssue] = Field(default_factory=list)
    rewrite_chapter_indexes: list[int] = Field(default_factory=list)
