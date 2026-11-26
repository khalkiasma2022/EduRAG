"""Tests de la façade RevisionEngine (orchestration Phase 6)."""

from __future__ import annotations

import pytest

from src.revision.engine.facade import RevisionEngine
from src.revision.engine.formatter import ContentFormatter
from src.revision.engine.llm_guard import (
    LEGACY_STORE_LEVEL,
    LEGACY_STORE_SHEET_TYPE,
    MAX_ANALYSIS_WINDOWS,
    LlmBudgetGuard,
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
    QualityIssue,
    QualityReport,
    SynthesisBlock,
)
from src.revision.engine.reader import CourseReadResult


class RecordingReader:
    def __init__(self, result: CourseReadResult) -> None:
        self.result = result
        self.calls: list[tuple] = []

    def read(
        self,
        document_id: str,
        page_start: int | None = None,
        page_end: int | None = None,
    ) -> CourseReadResult:
        self.calls.append((document_id, page_start, page_end))
        return self.result


class RecordingUnderstanding:
    def __init__(self, llm: object, guard: LlmBudgetGuard, course: CourseModel) -> None:
        self.llm = llm
        self.guard = guard
        self.course = course
        self.calls: list[object] = []

    def analyze(self, windows: object) -> CourseModel:
        self.calls.append(windows)
        self.guard.reserve(1)
        return self.course


class RecordingPlanner:
    def __init__(self, llm: object, guard: LlmBudgetGuard, fixed_plan: PedagogicalPlan) -> None:
        self.llm = llm
        self.guard = guard
        self.fixed_plan = fixed_plan
        self.calls: list[tuple] = []

    def plan(
        self,
        course: CourseModel,
        sheet_type: str,
        level: str,
        language: str,
    ) -> PedagogicalPlan:
        self.calls.append((course, sheet_type, level, language))
        self.guard.reserve(1)
        # Adapter le plan aux paramètres demandés pour vérifier la transmission.
        return self.fixed_plan.model_copy(
            update={
                "sheet_type": sheet_type,
                "level": level,
                "language": language,
            }
        )


class RecordingGenerator:
    def __init__(
        self,
        llm: object,
        guard: LlmBudgetGuard,
        document: PedagogicalDocument,
    ) -> None:
        self.llm = llm
        self.guard = guard
        self.document = document
        self.calls: list[dict] = []

    def generate(self, course, plan, *, filename, page_start, page_end, source_pages=None):
        self.calls.append(
            {
                "course": course,
                "plan": plan,
                "filename": filename,
                "page_start": page_start,
                "page_end": page_end,
                "source_pages": source_pages,
            }
        )
        self.guard.reserve(1)
        return self.document.model_copy(
            update={
                "filename": filename,
                "page_start": page_start,
                "page_end": page_end,
                "language": plan.language,
                "level": plan.level,
                "sheet_type": plan.sheet_type,
            }
        )


class RecordingQuality:
    def __init__(
        self,
        llm: object,
        guard: LlmBudgetGuard,
        *,
        fail: bool = False,
        mutate_title: str | None = None,
    ) -> None:
        self.llm = llm
        self.guard = guard
        self.fail = fail
        self.mutate_title = mutate_title
        self.calls: list[dict] = []

    def review(self, document, course, plan, *, source_pages=None):
        self.calls.append(
            {
                "document": document,
                "course": course,
                "plan": plan,
                "source_pages": source_pages,
            }
        )
        if self.fail:
            raise ValueError("Contrôle qualité en échec : défauts persistants.")
        reviewed = document.model_copy(deep=True)
        if self.mutate_title and reviewed.chapters:
            reviewed.chapters[0] = reviewed.chapters[0].model_copy(
                update={"title": self.mutate_title}
            )
        report = QualityReport(ok=True, issues=[], rewrite_chapter_indexes=[])
        return reviewed, report


class RecordingFormatter:
    def __init__(self) -> None:
        self.calls: list[PedagogicalDocument] = []

    def to_markdown(self, document: PedagogicalDocument) -> str:
        self.calls.append(document)
        return ContentFormatter().to_markdown(document)


class SharedLLM:
    """Une seule instance simulée — generate ne doit pas être requis si stages mockés."""

    def __init__(self) -> None:
        self.generate_calls = 0

    def generate(self, prompt: str, max_new_tokens: int = 512) -> str:
        self.generate_calls += 1
        raise AssertionError("Les stages mockés ne doivent pas appeler llm.generate.")


def _pages() -> list[CoursePage]:
    return [
        CoursePage(page_number=1, text="Une variable stocke une valeur.", filename="cours.pdf"),
        CoursePage(page_number=2, text="Une fonction encapsule un comportement.", filename="cours.pdf"),
    ]


def _read_result() -> CourseReadResult:
    pages = _pages()
    return CourseReadResult(
        pages=pages,
        filename="cours.pdf",
        page_start=1,
        page_end=2,
        windows=[pages[:1], pages[1:]],
    )


def _course() -> CourseModel:
    return CourseModel(
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
                depends_on=["c_var"],
                notes="Une fonction encapsule un comportement.",
            ),
        ],
        themes=["Programmation"],
        essential_concept_ids=["c_var", "c_fun"],
    )


def _plan(language: str = "fr") -> PedagogicalPlan:
    return PedagogicalPlan(
        language=language,
        level=LEGACY_STORE_LEVEL,
        sheet_type=LEGACY_STORE_SHEET_TYPE,
        chapters=[
            ChapterPlan(
                index=0,
                title="Variables",
                objectives=[LearningObjective(text="Comprendre", concept_ids=["c_var"])],
                concept_ids=["c_var"],
                page_refs=[1],
                needs_example=True,
                needs_questions=False,
                depth="complete",
            ),
            ChapterPlan(
                index=1,
                title="Fonctions",
                objectives=[LearningObjective(text="Utiliser", concept_ids=["c_fun"])],
                concept_ids=["c_fun"],
                page_refs=[2],
                needs_example=True,
                needs_questions=False,
                depth="complete",
            ),
        ],
    )


def _document() -> PedagogicalDocument:
    return PedagogicalDocument(
        filename="cours.pdf",
        page_start=1,
        page_end=2,
        language="fr",
        level=LEGACY_STORE_LEVEL,
        sheet_type=LEGACY_STORE_SHEET_TYPE,
        chapters=[
            ChapterDraft(
                index=0,
                title="Variables",
                concept_ids_covered=["c_var"],
                blocks=[
                    ParagraphBlock(
                        text="On nomme une information pour la réutiliser.",
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
                    SynthesisBlock(text="Retenir le nommage.", source_pages=[1]),
                ],
            ),
            ChapterDraft(
                index=1,
                title="Fonctions",
                concept_ids_covered=["c_fun"],
                blocks=[
                    ParagraphBlock(
                        text="On regroupe ensuite un comportement réutilisable.",
                        source_pages=[2],
                    ),
                    DefinitionBlock(
                        text="Une fonction encapsule un comportement.",
                        source_pages=[2],
                    ),
                    SynthesisBlock(text="La fonction s'appuie sur les variables.", source_pages=[2]),
                ],
            ),
        ],
    )


def _build_engine(
    *,
    language: str = "fr",
    quality_fail: bool = False,
    mutate_qc_title: str | None = "Variables QC",
    understanding_error: Exception | None = None,
    planner_error: Exception | None = None,
    generator_error: Exception | None = None,
):
    llm = SharedLLM()
    guard = LlmBudgetGuard()
    course = _course()
    plan = _plan(language=language)
    document = _document()
    read_result = _read_result()

    reader = RecordingReader(read_result)
    understanding = RecordingUnderstanding(llm, guard, course)
    planner = RecordingPlanner(llm, guard, plan)
    generator = RecordingGenerator(llm, guard, document)
    quality = RecordingQuality(
        llm, guard, fail=quality_fail, mutate_title=mutate_qc_title
    )
    formatter = RecordingFormatter()

    if understanding_error is not None:
        def boom_analyze(windows):
            understanding.calls.append(windows)
            raise understanding_error

        understanding.analyze = boom_analyze  # type: ignore[method-assign]
    if planner_error is not None:
        def boom_plan(course_arg, sheet_type, level, language):
            planner.calls.append((course_arg, sheet_type, level, language))
            raise planner_error

        planner.plan = boom_plan  # type: ignore[method-assign]
    if generator_error is not None:
        def boom_generate(*args, **kwargs):
            generator.calls.append({"args": args, "kwargs": kwargs})
            raise generator_error

        generator.generate = boom_generate  # type: ignore[method-assign]

    class _Lib:
        def list_documents(self):
            return []

    engine = RevisionEngine(
        library=_Lib(),
        llm=llm,
        guard=guard,
        reader=reader,  # type: ignore[arg-type]
        understanding=understanding,  # type: ignore[arg-type]
        planner=planner,  # type: ignore[arg-type]
        generator=generator,  # type: ignore[arg-type]
        quality=quality,  # type: ignore[arg-type]
        formatter=formatter,  # type: ignore[arg-type]
    )
    return engine, reader, understanding, planner, generator, quality, formatter, guard, llm


def test_pipeline_runs_in_order_and_returns_flat_dict() -> None:
    engine, reader, understanding, planner, generator, quality, formatter, guard, llm = (
        _build_engine()
    )

    result = engine.run(
        "doc-1",
        page_start=1,
        page_end=2,
        language="fr",
        level="beginner",
        sheet_type="complete",
    )

    assert reader.calls == [("doc-1", None, None)]
    assert understanding.calls == [reader.result.windows]
    assert planner.calls[0][0] is understanding.course
    assert planner.calls[0][1:] == (LEGACY_STORE_SHEET_TYPE, LEGACY_STORE_LEVEL, "fr")
    assert generator.calls[0]["course"] is understanding.course
    assert generator.calls[0]["plan"].sheet_type == LEGACY_STORE_SHEET_TYPE
    assert generator.calls[0]["filename"] == "cours.pdf"
    assert generator.calls[0]["source_pages"] == reader.result.pages
    assert quality.calls[0]["document"].filename == "cours.pdf"
    assert quality.calls[0]["course"] is understanding.course
    assert quality.calls[0]["plan"].language == "fr"
    assert len(formatter.calls) == 1
    assert formatter.calls[0].chapters[0].title == "Variables QC"

    assert result["document_id"] == "doc-1"
    assert result["filename"] == "cours.pdf"
    assert result["page_start"] == 1
    assert result["page_end"] == 2
    assert result["language"] == "fr"
    assert result["level"] == LEGACY_STORE_LEVEL
    assert result["sheet_type"] == LEGACY_STORE_SHEET_TYPE
    assert "## Variables QC" in result["content"]
    assert "store" not in result
    assert "id" not in result
    assert llm.generate_calls == 0
    assert guard.calls == 3  # understanding + planner + generator


def test_run_refuses_pdf_exceeding_window_capacity_without_calling_llm() -> None:
    engine, reader, understanding, planner, generator, quality, formatter, guard, llm = (
        _build_engine()
    )
    oversized = [
        [CoursePage(page_number=i, text=f"p{i}", filename="cours.pdf")]
        for i in range(1, MAX_ANALYSIS_WINDOWS + 2)
    ]
    reader.result = CourseReadResult(
        pages=[page for window in oversized for page in window],
        filename="cours.pdf",
        page_start=1,
        page_end=MAX_ANALYSIS_WINDOWS + 1,
        windows=oversized,
    )

    with pytest.raises(ValueError, match="capacité|fenêtres"):
        engine.run("doc-1", language="fr")

    assert understanding.calls == []
    assert planner.calls == []
    assert generator.calls == []
    assert quality.calls == []
    assert formatter.calls == []
    assert guard.calls == 0
    assert llm.generate_calls == 0


def test_legacy_params_are_ignored_but_language_is_kept() -> None:
    engine, reader, _u, planner, generator, _q, _f, _g, _llm = _build_engine(
        language="en"
    )
    result = engine.run(
        "doc-1",
        page_start=2,
        page_end=3,
        language="en",
        level="advanced",
        sheet_type="short",
    )
    assert reader.calls == [("doc-1", None, None)]
    assert planner.calls[0][1:] == (LEGACY_STORE_SHEET_TYPE, LEGACY_STORE_LEVEL, "en")
    assert generator.calls[0]["plan"].sheet_type == LEGACY_STORE_SHEET_TYPE
    assert generator.calls[0]["plan"].level == LEGACY_STORE_LEVEL
    assert generator.calls[0]["plan"].language == "en"
    assert result["sheet_type"] == LEGACY_STORE_SHEET_TYPE
    assert result["level"] == LEGACY_STORE_LEVEL
    assert result["language"] == "en"


def test_quality_is_mandatory_before_formatter() -> None:
    engine, _r, _u, _p, _g, quality, formatter, _guard, _llm = _build_engine(
        quality_fail=True
    )
    with pytest.raises(ValueError, match="qualité"):
        engine.run("doc-1", sheet_type="complete")
    assert len(quality.calls) == 1
    assert formatter.calls == []


def test_errors_from_each_stage_propagate_without_partial_success() -> None:
    cases = [
        {"understanding_error": ValueError("JSON de compréhension invalide.")},
        {"planner_error": ValueError("JSON de planification invalide.")},
        {"generator_error": ValueError("JSON de chapitre invalide.")},
        {"quality_fail": True},
    ]
    for kwargs in cases:
        engine, reader, understanding, planner, generator, quality, formatter, _g, _l = (
            _build_engine(**kwargs)
        )
        with pytest.raises(ValueError):
            engine.run("doc-1")
        assert formatter.calls == []


def test_shared_guard_and_single_llm_instance() -> None:
    llm = SharedLLM()
    guard = LlmBudgetGuard()

    class _Lib:
        def list_documents(self):
            return []

    understanding = RecordingUnderstanding(llm, guard, _course())
    planner = RecordingPlanner(llm, guard, _plan())
    generator = RecordingGenerator(llm, guard, _document())
    quality = RecordingQuality(llm, guard)
    engine = RevisionEngine(
        library=_Lib(),
        llm=llm,
        guard=guard,
        reader=RecordingReader(_read_result()),  # type: ignore[arg-type]
        understanding=understanding,  # type: ignore[arg-type]
        planner=planner,  # type: ignore[arg-type]
        generator=generator,  # type: ignore[arg-type]
        quality=quality,  # type: ignore[arg-type]
        formatter=RecordingFormatter(),  # type: ignore[arg-type]
    )
    assert engine.understanding.llm is llm
    assert engine.planner.llm is llm
    assert engine.generator.llm is llm
    assert engine.quality.llm is llm
    assert engine.understanding.guard is guard
    assert engine.planner.guard is guard
    assert engine.generator.guard is guard
    assert engine.quality.guard is guard

    other_guard = LlmBudgetGuard()
    with pytest.raises(ValueError, match="LlmBudgetGuard"):
        RevisionEngine(
            library=_Lib(),
            llm=llm,
            guard=guard,
            reader=RecordingReader(_read_result()),  # type: ignore[arg-type]
            understanding=RecordingUnderstanding(llm, other_guard, _course()),  # type: ignore[arg-type]
            planner=planner,  # type: ignore[arg-type]
            generator=generator,  # type: ignore[arg-type]
            quality=quality,  # type: ignore[arg-type]
            formatter=RecordingFormatter(),  # type: ignore[arg-type]
        )


def test_rejects_second_llm_instance() -> None:
    class _Lib:
        def list_documents(self):
            return []

    llm_a = SharedLLM()
    llm_b = SharedLLM()
    guard = LlmBudgetGuard()
    with pytest.raises(ValueError, match="Qwen|instance"):
        RevisionEngine(
            library=_Lib(),
            llm=llm_a,
            guard=guard,
            reader=RecordingReader(_read_result()),  # type: ignore[arg-type]
            understanding=RecordingUnderstanding(llm_a, guard, _course()),  # type: ignore[arg-type]
            planner=RecordingPlanner(llm_b, guard, _plan()),  # type: ignore[arg-type]
            generator=RecordingGenerator(llm_a, guard, _document()),  # type: ignore[arg-type]
            quality=RecordingQuality(llm_a, guard),  # type: ignore[arg-type]
            formatter=RecordingFormatter(),  # type: ignore[arg-type]
        )


def test_does_not_save_or_touch_store() -> None:
    engine, *_rest = _build_engine()
    result = engine.run("doc-1")
    assert set(result) == {
        "content",
        "document_id",
        "filename",
        "page_start",
        "page_end",
        "language",
        "level",
        "sheet_type",
    }
