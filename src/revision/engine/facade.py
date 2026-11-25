"""Façade RevisionEngine : orchestration du pipeline Smart Revision V2."""

from __future__ import annotations

from src.revision.engine.formatter import ContentFormatter
from src.revision.engine.generator import OriginalContentGenerator
from src.revision.engine.llm_guard import LlmBudgetGuard
from src.revision.engine.planner import PedagogicalPlanner
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
    ) -> None:
        if not callable(getattr(library, "list_documents", None)):
            raise TypeError("library doit fournir list_documents.")
        if not callable(getattr(llm, "generate", None)):
            raise TypeError("llm doit fournir generate.")

        self.library = library
        self.llm = llm
        self.guard = guard if guard is not None else LlmBudgetGuard()

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
        """Exécute le pipeline complet et renvoie un dict plat (sans sauvegarde)."""
        read_result = self.reader.read(document_id, page_start, page_end)
        course = self.understanding.analyze(read_result.windows)
        plan = self.planner.plan(
            course,
            sheet_type=sheet_type,
            level=level,
            language=language,
        )
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
        markdown = self.formatter.to_markdown(reviewed_document)
        if not isinstance(markdown, str) or markdown.strip() == "":
            raise ValueError("Le Markdown pédagogique généré est vide.")

        return {
            "content": markdown.strip(),
            "document_id": document_id,
            "filename": read_result.filename,
            "page_start": int(read_result.page_start),
            "page_end": int(read_result.page_end),
            "language": language,
            "level": level,
            "sheet_type": sheet_type,
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
