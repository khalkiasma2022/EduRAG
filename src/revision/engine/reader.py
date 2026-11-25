"""Lecture et fenêtrage du cours pour le moteur Smart Revision V2."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from src.ingestion.pdf_loader import load_pdf
from src.revision.engine.llm_guard import MAX_ANALYSIS_WINDOWS, MAX_COURSE_TOKENS_PER_WINDOW
from src.revision.engine.models import CoursePage
from src.revision.sheet import _find_document, _select_pages


def _default_count_tokens(text: str) -> int:
    """Estimation locale sans tokenizer (~1 jeton pour 4 caractères)."""
    cleaned = text.strip()
    if cleaned == "":
        return 0
    return max(1, (len(cleaned) + 3) // 4)


def _clean_page_text(text: str) -> str:
    """Supprime les pages vides et les espaces excessifs."""
    return " ".join(text.split())


@dataclass(frozen=True)
class CourseReadResult:
    """Pages sélectionnées, métadonnées et fenêtres d'analyse."""

    pages: list[CoursePage]
    filename: str
    page_start: int
    page_end: int
    windows: list[list[CoursePage]]


class CourseReader:
    """Résout un document de la bibliothèque et le découpe en fenêtres LLM."""

    def __init__(
        self,
        library: object,
        count_tokens: Callable[[str], int] | None = None,
    ) -> None:
        if not callable(getattr(library, "list_documents", None)):
            raise TypeError("library doit fournir list_documents.")
        self.library = library
        self.count_tokens = count_tokens or _default_count_tokens

    def read(
        self,
        document_id: str,
        page_start: int | None = None,
        page_end: int | None = None,
    ) -> CourseReadResult:
        """Relit le PDF (ou un intervalle) et construit les fenêtres d'analyse."""
        entry = _find_document(self.library, document_id)
        raw_pages = load_pdf(entry["stored_path"])
        for page in raw_pages:
            page["filename"] = entry["filename"]
        selected = _select_pages(raw_pages, page_start, page_end)

        pages: list[CoursePage] = []
        for page in selected:
            text = _clean_page_text(str(page["text"]))
            if text == "":
                continue
            pages.append(
                CoursePage(
                    page_number=int(page["page_number"]),
                    text=text,
                    filename=str(page["filename"]),
                )
            )
        if not pages:
            raise ValueError("Les pages du cours sont invalides.")

        windows = self._split_windows(pages)
        return CourseReadResult(
            pages=pages,
            filename=pages[0].filename,
            page_start=pages[0].page_number,
            page_end=pages[-1].page_number,
            windows=windows,
        )

    def _split_windows(self, pages: list[CoursePage]) -> list[list[CoursePage]]:
        """Empile les pages jusqu'au budget tokens, sans coupe silencieuse."""
        windows: list[list[CoursePage]] = []
        current: list[CoursePage] = []
        current_tokens = 0

        for page in pages:
            page_tokens = self.count_tokens(page.text)
            if page_tokens > MAX_COURSE_TOKENS_PER_WINDOW:
                raise ValueError(
                    "Le cours sélectionné est trop volumineux. "
                    "Sélectionnez moins de pages."
                )
            if current and current_tokens + page_tokens > MAX_COURSE_TOKENS_PER_WINDOW:
                windows.append(current)
                current = []
                current_tokens = 0
            current.append(page)
            current_tokens += page_tokens

        if current:
            windows.append(current)

        if len(windows) > MAX_ANALYSIS_WINDOWS:
            raise ValueError(
                "Le cours sélectionné est trop volumineux. "
                "Sélectionnez moins de pages."
            )
        return windows
