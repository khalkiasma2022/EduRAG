"""Tests Correction 4 — densité des chapitres et réécritures QC non appauvrissantes."""

from __future__ import annotations

import json

import pytest

from src.revision.engine.coverage import rewrite_impoverishment_reason
from src.revision.engine.generator import OriginalContentGenerator
from src.revision.engine.llm_guard import (
    MAX_CONCEPTS_PER_CHAPTER,
    MAX_COURSE_CONCEPTS,
    MAX_REWRITE_CHAPTERS,
    MIN_TOKENS_PER_CONCEPT,
    TOKENS_CHAPTER,
    TOKENS_REWRITE,
    LlmBudgetGuard,
    ensure_chapter_density,
    ensure_course_concept_capacity,
    max_attempts_per_step,
)
from src.revision.engine.models import (
    ChapterDraft,
    ChapterPlan,
    Concept,
    CourseModel,
    CoursePage,
    DefinitionBlock,
    ExampleBlock,
    LearningObjective,
    ParagraphBlock,
    PedagogicalDocument,
    PedagogicalPlan,
    SynthesisBlock,
    TableBlock,
)
from src.revision.engine.planner import PedagogicalPlanner, _heuristic_plan
from src.revision.engine.quality import QualityController


class _TokenIds:
    def __init__(self, count: int) -> None:
        self.shape = (-1, count)


class _FakeTokenizer:
    chat_template = None

    def __call__(self, prompt: str, return_tensors: str = "pt") -> dict[str, _TokenIds]:
        return {"input_ids": _TokenIds(max(1, len(prompt) // 8))}


class _FakeConfig:
    max_position_embeddings = 32768
    use_sliding_window = False
    rope_parameters = {"rope_type": "default"}


class _FakeModel:
    config = _FakeConfig()


class ScriptedLLM:
    def __init__(self, replies: list[str] | None = None) -> None:
        self.replies = list(replies or [])
        self.calls = 0
        self.prompts: list[str] = []
        self.tokenizer = _FakeTokenizer()
        self.model = _FakeModel()

    def is_loaded(self) -> bool:
        return True

    def generate(self, prompt: str, max_new_tokens: int = 512) -> str:
        self.calls += 1
        self.prompts.append(prompt)
        if not self.replies:
            raise AssertionError("Plus de réponses LLM scriptées.")
        return self.replies.pop(0)


def _concept(cid: str, *, essential: bool = True, page: int = 1) -> Concept:
    return Concept(
        id=cid,
        label=cid.replace("_", " "),
        kind="definition",
        pages=[page],
        essential=essential,
        notes=f"Définition utile de {cid} pour le cours.",
    )


def _course(n: int) -> CourseModel:
    concepts = [_concept(f"c_{i}", essential=(i % 3 == 0), page=(i % 5) + 1) for i in range(n)]
    return CourseModel(
        concepts=concepts,
        themes=["Test"],
        essential_concept_ids=[c.id for c in concepts if c.essential],
    )


def _plan_for_ids(groups: list[list[str]]) -> PedagogicalPlan:
    chapters = []
    for index, ids in enumerate(groups):
        chapters.append(
            ChapterPlan(
                index=index,
                title=f"Chapitre {index}",
                objectives=[LearningObjective(text="obj", concept_ids=list(ids))],
                concept_ids=list(ids),
                page_refs=[1],
                needs_example=True,
                depth="complete",
            )
        )
    return PedagogicalPlan(
        language="fr",
        level="intermediate",
        sheet_type="complete",
        chapters=chapters,
    )


def _pages() -> list[CoursePage]:
    return [
        CoursePage(page_number=i, text=f"Contenu page {i}.", filename="cours.pdf")
        for i in range(1, 6)
    ]


def _chapter_json(*, concept_ids: list[str], blocks: list[dict], index: int = 0) -> str:
    return json.dumps(
        {
            "index": index,
            "title": "Chapitre",
            "concept_ids_covered": concept_ids,
            "blocks": blocks,
        },
        ensure_ascii=False,
    )


def test_density_constants_match_token_budget() -> None:
    assert MAX_CONCEPTS_PER_CHAPTER == 4
    assert TOKENS_CHAPTER == 640
    assert MIN_TOKENS_PER_CONCEPT == TOKENS_CHAPTER // MAX_CONCEPTS_PER_CHAPTER
    assert MAX_COURSE_CONCEPTS == 40
    assert MAX_REWRITE_CHAPTERS == 2
    assert TOKENS_REWRITE == 384


def test_chapter_with_four_concepts_is_accepted() -> None:
    ensure_chapter_density(4)
    ids = [f"c_{i}" for i in range(4)]
    course = _course(4)
    plan = _heuristic_plan(course, "fr")
    assert all(len(ch.concept_ids) <= 4 for ch in plan.chapters)
    assert {cid for ch in plan.chapters for cid in ch.concept_ids} == set(ids)


def test_chapter_with_more_than_four_concepts_is_rejected() -> None:
    with pytest.raises(ValueError, match="Densité pédagogique excessive|concepts"):
        ensure_chapter_density(5)

    course = _course(5)
    # Plan artificiellement dense (1 chapitre / 5 notions).
    plan = _plan_for_ids([[c.id for c in course.concepts]])
    llm = ScriptedLLM(
        [
            _chapter_json(
                concept_ids=[c.id for c in course.concepts],
                blocks=[
                    {
                        "type": "paragraph",
                        "text": "Texte trop dense pour être généré correctement.",
                        "source_pages": [1],
                    }
                ],
            )
        ]
    )
    with pytest.raises(ValueError, match="Densité|dense|budget|concepts"):
        OriginalContentGenerator(llm, LlmBudgetGuard()).generate(
            course,
            plan,
            filename="cours.pdf",
            page_start=1,
            page_end=1,
            source_pages=_pages(),
        )


def test_hundreds_of_concepts_are_refused_without_dropping_any() -> None:
    course = _course(200)
    with pytest.raises(ValueError, match="capacité pédagogique|40|Découpez"):
        ensure_course_concept_capacity(len(course.concepts))
    with pytest.raises(ValueError, match="capacité pédagogique|40|Découpez"):
        PedagogicalPlanner(ScriptedLLM(["{}"]), LlmBudgetGuard()).plan(course, language="fr")
    # Aucun concept n'a été retiré du modèle source.
    assert len(course.concepts) == 200


def test_max_density_exceeded_on_heuristic_path() -> None:
    course = _course(MAX_COURSE_CONCEPTS + 1)
    with pytest.raises(ValueError, match="capacité pédagogique|Découpez"):
        _heuristic_plan(course, "fr")


def test_budget_incompatible_with_dense_chapter() -> None:
    # 5 notions × 160 tokens > 640.
    with pytest.raises(ValueError, match="Densité|Budget insuffisant|dense"):
        ensure_chapter_density(MAX_CONCEPTS_PER_CHAPTER + 1)


def test_explicit_refusal_preserves_all_source_concepts() -> None:
    course = _course(160)
    before = [c.id for c in course.concepts]
    with pytest.raises(ValueError, match="Découpez|capacité"):
        PedagogicalPlanner(ScriptedLLM(["not-json"]), LlmBudgetGuard()).plan(
            course, language="fr"
        )
    assert [c.id for c in course.concepts] == before


def test_rewrite_dropping_explanation_is_rejected() -> None:
    course = CourseModel(
        concepts=[
            Concept(
                id="c_var",
                label="variable",
                kind="definition",
                pages=[1],
                essential=True,
                notes="Une variable stocke une valeur.",
            ),
            Concept(
                id="c_fun",
                label="fonction",
                kind="method",
                pages=[2],
                essential=True,
                notes="Une fonction encapsule un comportement.",
            ),
        ],
        essential_concept_ids=["c_var", "c_fun"],
    )
    plan = PedagogicalPlan(
        language="fr",
        level="intermediate",
        sheet_type="complete",
        chapters=[
            ChapterPlan(
                index=0,
                title="Variables",
                concept_ids=["c_var"],
                page_refs=[1],
                needs_example=True,
                depth="complete",
            ),
            ChapterPlan(
                index=1,
                title="Fonctions",
                concept_ids=["c_fun"],
                page_refs=[2],
                needs_example=True,
                depth="complete",
            ),
        ],
    )
    document = PedagogicalDocument(
        filename="cours.pdf",
        page_start=1,
        page_end=2,
        language="fr",
        level="intermediate",
        sheet_type="complete",
        chapters=[
            ChapterDraft(
                index=0,
                title="Variables",
                concept_ids_covered=["c_var"],
                blocks=[
                    ParagraphBlock(
                        text="On introduit la variable pour nommer une information claire.",
                        source_pages=[1],
                    ),
                    DefinitionBlock(
                        text="Une variable stocke une valeur.",
                        source_pages=[1],
                    ),
                    ExampleBlock(
                        text="Comme une boîte étiquetée.",
                        origin="pedagogical_addition",
                        source_pages=[],
                    ),
                ],
            ),
            ChapterDraft(
                index=1,
                title="Fonctions",
                concept_ids_covered=["c_fun"],
                blocks=[
                    ParagraphBlock(
                        text="Voir https://exemple.invente/fun puis la fonction utile.",
                        source_pages=[2],
                    ),
                    DefinitionBlock(
                        text="Une fonction encapsule un comportement.",
                        source_pages=[2],
                    ),
                    ExampleBlock(
                        text="Analogie : une recette réutilisable.",
                        origin="pedagogical_addition",
                        source_pages=[],
                    ),
                ],
            ),
        ],
    )
    bad = _chapter_json(
        index=1,
        concept_ids=["c_fun"],
        blocks=[
            {
                "type": "paragraph",
                "text": "Texte vague sans expliquer la notion demandée ici.",
                "source_pages": [2],
            },
            {
                "type": "synthesis",
                "text": "Conclusion trop générale sans ancrage.",
                "source_pages": [2],
            },
        ],
    )
    with pytest.raises(ValueError, match="appauvrie|Couverture|expliqué|qualité"):
        QualityController(
            ScriptedLLM([bad] * max_attempts_per_step()),
            LlmBudgetGuard(),
        ).review(document, course, plan, source_pages=_pages())


def test_rewrite_preserving_full_coverage_is_accepted() -> None:
    course = CourseModel(
        concepts=[
            Concept(
                id="c_fun",
                label="fonction",
                kind="method",
                pages=[2],
                essential=True,
                notes="Une fonction encapsule un comportement.",
            )
        ],
        essential_concept_ids=["c_fun"],
    )
    plan = PedagogicalPlan(
        language="fr",
        level="intermediate",
        sheet_type="complete",
        chapters=[
            ChapterPlan(
                index=0,
                title="Fonctions",
                concept_ids=["c_fun"],
                page_refs=[2],
                needs_example=True,
                depth="complete",
            )
        ],
    )
    document = PedagogicalDocument(
        filename="cours.pdf",
        page_start=2,
        page_end=2,
        language="fr",
        level="intermediate",
        sheet_type="complete",
        chapters=[
            ChapterDraft(
                index=0,
                title="Fonctions",
                concept_ids_covered=["c_fun"],
                blocks=[
                    ParagraphBlock(
                        text="Voir https://exemple.invente/x avant d'expliquer la fonction.",
                        source_pages=[2],
                    ),
                    DefinitionBlock(
                        text="Une fonction encapsule un comportement.",
                        source_pages=[2],
                    ),
                    ExampleBlock(
                        text="Analogie : une recette réutilisable.",
                        origin="pedagogical_addition",
                        source_pages=[],
                    ),
                    SynthesisBlock(
                        text="Retenir le rôle d'une fonction.",
                        source_pages=[2],
                    ),
                ],
            )
        ],
    )
    good = _chapter_json(
        concept_ids=["c_fun"],
        blocks=[
            {
                "type": "paragraph",
                "text": "On organise un comportement réutilisable dans une fonction claire.",
                "source_pages": [2],
            },
            {
                "type": "definition",
                "text": "Une fonction encapsule un comportement.",
                "source_pages": [2],
            },
            {
                "type": "example",
                "text": "Analogie : un atelier qui transforme des pièces.",
                "origin": "pedagogical_addition",
                "source_pages": [],
            },
            {
                "type": "synthesis",
                "text": "Retenir le rôle d'une fonction dans le programme.",
                "source_pages": [2],
            },
        ],
    )
    reviewed, report = QualityController(ScriptedLLM([good]), LlmBudgetGuard()).review(
        document, course, plan, source_pages=_pages()
    )
    assert report.ok is True
    assert 0 in report.rewrite_chapter_indexes
    assert any(b.type == "example" for b in reviewed.chapters[0].blocks)
    assert any(getattr(b, "source_pages", None) for b in reviewed.chapters[0].blocks)


def test_pages_and_examples_preserved_by_impoverishment_guard() -> None:
    course = CourseModel(
        concepts=[
            Concept(
                id="c_var",
                label="variable",
                kind="definition",
                pages=[1],
                essential=True,
                notes="Une variable stocke une valeur.",
            )
        ],
        essential_concept_ids=["c_var"],
    )
    plan_chapter = ChapterPlan(
        index=0,
        title="Variables",
        concept_ids=["c_var"],
        page_refs=[1],
        needs_example=True,
        depth="complete",
    )
    original = ChapterDraft(
        index=0,
        title="Variables",
        concept_ids_covered=["c_var"],
        blocks=[
            DefinitionBlock(text="Une variable stocke une valeur.", source_pages=[1]),
            ExampleBlock(
                text="Comme une boîte étiquetée.",
                origin="pedagogical_addition",
                source_pages=[],
            ),
            TableBlock(headers=["Notion", "Rôle"], rows=[["variable", "nommer"]], source_pages=[1]),
        ],
    )
    without_example = ChapterDraft(
        index=0,
        title="Variables",
        concept_ids_covered=["c_var"],
        blocks=[
            DefinitionBlock(text="Une variable stocke une valeur.", source_pages=[1]),
            TableBlock(headers=["Notion", "Rôle"], rows=[["variable", "nommer"]], source_pages=[1]),
        ],
    )
    without_pages = ChapterDraft(
        index=0,
        title="Variables",
        concept_ids_covered=["c_var"],
        blocks=[
            DefinitionBlock(text="Une variable stocke une valeur.", source_pages=[]),
            ExampleBlock(
                text="Comme une boîte étiquetée.",
                origin="pedagogical_addition",
                source_pages=[],
            ),
            TableBlock(headers=["Notion", "Rôle"], rows=[["variable", "nommer"]], source_pages=[]),
        ],
    )
    assert "exemples" in (
        rewrite_impoverishment_reason(
            original, without_example, chapter_plan=plan_chapter, course=course
        ) or ""
    ).casefold()
    assert "pages" in (
        rewrite_impoverishment_reason(
            original, without_pages, chapter_plan=plan_chapter, course=course
        ) or ""
    ).casefold()


def test_normal_pipeline_without_false_reject() -> None:
    course = _course(4)
    plan = _heuristic_plan(course, "fr")
    assert len(plan.chapters) <= 10
    assert all(len(ch.concept_ids) <= MAX_CONCEPTS_PER_CHAPTER for ch in plan.chapters)
    covered = {cid for ch in plan.chapters for cid in ch.concept_ids}
    assert covered == {c.id for c in course.concepts}

    chapter = plan.chapters[0].model_copy(update={"index": 0})
    concepts = [c for c in course.concepts if c.id in set(chapter.concept_ids)]
    labels = ", ".join(c.label for c in concepts)
    notes_block = " ".join(c.notes for c in concepts)
    payload = _chapter_json(
        index=0,
        concept_ids=list(chapter.concept_ids),
        blocks=[
            {
                "type": "paragraph",
                "text": (
                    f"Ce chapitre explique clairement {labels} avec une rédaction "
                    "originale fidèle au cours."
                ),
                "source_pages": [1],
            },
            {
                "type": "definition",
                "text": notes_block,
                "source_pages": [1],
            },
            {
                "type": "example",
                "text": "Analogie pédagogique claire pour ancrer les notions.",
                "origin": "pedagogical_addition",
                "source_pages": [],
            },
            {
                "type": "synthesis",
                "text": f"Retenir l'essentiel concernant {labels}.",
                "source_pages": [1],
            },
        ],
    )
    single = plan.model_copy(update={"chapters": [chapter]})
    doc = OriginalContentGenerator(ScriptedLLM([payload]), LlmBudgetGuard()).generate(
        course,
        single,
        filename="cours.pdf",
        page_start=1,
        page_end=5,
        source_pages=_pages(),
    )
    reviewed, report = QualityController(ScriptedLLM(), LlmBudgetGuard()).review(
        doc, course, single, source_pages=_pages()
    )
    assert report.ok is True
    assert set(reviewed.chapters[0].concept_ids_covered) == set(chapter.concept_ids)
