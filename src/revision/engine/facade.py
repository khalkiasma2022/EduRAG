"""Façade RevisionEngine : orchestration du pipeline Smart Revision V2."""

from __future__ import annotations

from src.revision.engine.course_cache import CourseUnderstandingCache
from src.revision.engine.formatter import ContentFormatter
from src.revision.engine.generator import OriginalContentGenerator
from src.revision.engine.llm_guard import (
    LEGACY_STORE_LEVEL,
    LEGACY_STORE_SHEET_TYPE,
    LlmBudgetGuard,
    ensure_generation_capacity,
    ensure_pipeline_capacity,
)
from src.revision.engine.planner import PedagogicalPlanner
from src.revision.engine.progress import log_progress
from src.revision.engine.quality import QualityController
from src.revision.engine.reader import CourseReader
from src.revision.engine.understanding import CourseUnderstanding


class RevisionEngine:
    """Orchestre Reader → Understanding → Planner → Generator → QC → Formatter.

    Indépendant de FastAPI, SQLite, Streamlit et React.
    N'enregistre aucune fiche : retourne un dictionnaire plat prêt pour un save externe.
    """

    def __init__(
        self,
        library: object,
        llm: object,
        *,
        guard: LlmBudgetGuard | None = None,
        reader: CourseReader | None = None,
        understanding: CourseUnderstanding | None = None,
        planner: PedagogicalPlanner | None = None,
        generator: OriginalContentGenerator | None = None,
        quality: QualityController | None = None,
        formatter: ContentFormatter | None = None,
        course_cache: CourseUnderstandingCache | None = None,
    ) -> None:
        if not callable(getattr(library, "list_documents", None)):
            raise TypeError("library doit fournir list_documents.")
        if not callable(getattr(llm, "generate", None)):
            raise TypeError("llm doit fournir generate.")

        self.library = library
        self.llm = llm
        self.guard = guard if guard is not None else LlmBudgetGuard()
        self.course_cache = course_cache

        self.reader = reader if reader is not None else CourseReader(library)
        self.understanding = (
            understanding
            if understanding is not None
            else CourseUnderstanding(llm, self.guard)
        )
        self.planner = (
            planner if planner is not None else PedagogicalPlanner(llm, self.guard)
        )
        self.generator = (
            generator
            if generator is not None
            else OriginalContentGenerator(llm, self.guard)
        )
        self.quality = (
            quality if quality is not None else QualityController(llm, self.guard)
        )
        self.formatter = formatter if formatter is not None else ContentFormatter()

        self._assert_shared_guard()
        self._assert_shared_llm()

    def run(
        self,
        document_id: str,
        page_start: int | None = None,
        page_end: int | None = None,
        language: str = "fr",
        level: str = "beginner",
        sheet_type: str = "complete",
    ) -> dict[str, str | int]:
        """Exécute le pipeline complet et renvoie un dict plat (sans sauvegarde).

        ``page_start``, ``page_end``, ``level`` et ``sheet_type`` sont ignorés pour
        le contenu (compatibilité API). L'analyse porte toujours sur tout le PDF.
        Les valeurs legacy sont figées pour SQLite.
        """
        del page_start, page_end, level, sheet_type
        log_progress(f"Smart Revision démarrée pour {document_id}")
        read_result = self.reader.read(document_id, None, None)
        log_progress(
            f"Lecture PDF « {read_result.filename} » "
            f"({read_result.page_start}-{read_result.page_end}, "
            f"{len(read_result.windows)} fenêtres)"
        )

        cached = (
            self.course_cache.load(document_id) if self.course_cache is not None else None
        )
        if cached is not None:
            ensure_pipeline_capacity(
                windows=len(read_result.windows),
                skip_understanding=True,
            )
            course = cached
            log_progress(
                f"Compréhension en cache ({len(course.concepts)} concepts)"
            )
        else:
            ensure_pipeline_capacity(windows=len(read_result.windows))
            course = self.understanding.analyze(read_result.windows)
            log_progress(f"Compréhension terminée ({len(course.concepts)} concepts)")
            if self.course_cache is not None:
                self.course_cache.save(
                    document_id,
                    filename=read_result.filename,
                    page_start=read_result.page_start,
                    page_end=read_result.page_end,
                    course=course,
                )

        plan = self.planner.plan(
            course,
            sheet_type=LEGACY_STORE_SHEET_TYPE,
            level=LEGACY_STORE_LEVEL,
            language=language,
        )
        log_progress(f"Plan prêt ({len(plan.chapters)} chapitres)")
        ensure_generation_capacity(self.guard, chapters=len(plan.chapters))
        document = self.generator.generate(
            course,
            plan,
            filename=read_result.filename,
            page_start=read_result.page_start,
            page_end=read_result.page_end,
            source_pages=read_result.pages,
        )
        reviewed_document, _report = self.quality.review(
            document,
            course,
            plan,
            source_pages=read_result.pages,
        )
        log_progress("Mise en forme Markdown…")
        markdown = self.formatter.to_markdown(reviewed_document)
        if not isinstance(markdown, str) or markdown.strip() == "":
            raise ValueError("Le Markdown pédagogique généré est vide.")
        log_progress("Smart Revision terminée.")

        return {
            "content": markdown.strip(),
            "document_id": document_id,
            "filename": read_result.filename,
            "page_start": int(read_result.page_start),
            "page_end": int(read_result.page_end),
            "language": language,
            "level": LEGACY_STORE_LEVEL,
            "sheet_type": LEGACY_STORE_SHEET_TYPE,
        }

    def _assert_shared_guard(self) -> None:
        for name in ("understanding", "planner", "generator", "quality"):
            component = getattr(self, name)
            component_guard = getattr(component, "guard", None)
            if component_guard is not None and component_guard is not self.guard:
                raise ValueError(
                    "Tous les modules LLM doivent partager le même LlmBudgetGuard."
                )

    def _assert_shared_llm(self) -> None:
        for name in ("understanding", "planner", "generator", "quality"):
            component = getattr(self, name)
            component_llm = getattr(component, "llm", None)
            if component_llm is not None and component_llm is not self.llm:
                raise ValueError(
                    "Tous les modules LLM doivent réutiliser la même instance Qwen."
                )
