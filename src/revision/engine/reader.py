"""Lecture et fenêtrage du cours pour le moteur Smart Revision V2."""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass

from src.ingestion.pdf_loader import load_pdf
from src.revision.engine.llm_guard import MAX_ANALYSIS_WINDOWS, MAX_COURSE_TOKENS_PER_WINDOW
from src.revision.engine.models import CoursePage
from src.revision.sheet import _find_document, _select_pages

# Espaces horizontaux uniquement (préserve les retours à la ligne).
_HORIZONTAL_SPACE = re.compile(r"[^\S\n]+")


def _default_count_tokens(text: str) -> int:
    """Estimation locale sans tokenizer (~1 jeton pour 4 caractères)."""
    cleaned = text.strip()
    if cleaned == "":
        return 0
    return max(1, (len(cleaned) + 3) // 4)


def _clean_page_text(text: str) -> str:
    """Normalise les espaces horizontaux sans écraser la structure du texte."""
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [_HORIZONTAL_SPACE.sub(" ", line).strip() for line in normalized.split("\n")]
    collapsed: list[str] = []
    blank_pending = False
    for line in lines:
        if line == "":
            if collapsed and not blank_pending:
                collapsed.append("")
            blank_pending = True
            continue
        collapsed.append(line)
        blank_pending = False
    while collapsed and collapsed[0] == "":
        collapsed.pop(0)
    while collapsed and collapsed[-1] == "":
        collapsed.pop()
    return "\n".join(collapsed)


def _validate_page_token_count(value: object) -> int:
    """Exige un entier strictement positif (booléens exclus)."""
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(
            "count_tokens doit retourner un entier strictement positif "
            f"pour une page non vide (reçu {value!r})."
        )
    return value


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
        if count_tokens is None:
            self.count_tokens = _default_count_tokens
        else:
            if not callable(count_tokens):
                raise TypeError("count_tokens doit être appelable.")
            self.count_tokens = count_tokens
        self.library = library

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
            page_tokens = _validate_page_token_count(self.count_tokens(page.text))
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

        if not windows:
            raise ValueError("Les pages du cours sont invalides.")
        if len(windows) > MAX_ANALYSIS_WINDOWS:
            raise ValueError(
                "Le cours sélectionné dépasse la capacité d'analyse "
                f"({MAX_ANALYSIS_WINDOWS} fenêtres)."
            )
        return windows
