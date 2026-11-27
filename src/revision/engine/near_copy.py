"""Détection de paraphrase trop proche du PDF source (générateur + QC)."""

from __future__ import annotations

import re

from src.revision.engine.models import CourseModel, CoursePage

_NUMBER_RE = re.compile(r"(?<![A-Za-z])\d+(?:[.,]\d+)?")
_FORMULA_RE = re.compile(r"\b[A-Za-z]\s*=\s*\d+\b")

# Fenêtre assez longue pour distinguer jargon inévitable et collage de phrases.
_WINDOW_WORDS = 10
_MIN_CHUNK_CHARS = 48
_MIN_FULL_CHARS = 40


def normalize_text(text: str) -> str:
    return " ".join(text.casefold().split())


def page_source_corpus(course: CourseModel, pages: list[CoursePage]) -> str:
    """Corpus anti-copie : pages et unités uniquement (pas les notes de concepts)."""
    parts = [page.text for page in pages]
    parts.extend(unit.text for unit in course.units)
    return "\n".join(parts)


def full_source_corpus(course: CourseModel, pages: list[CoursePage]) -> str:
    """Corpus large (nombres, contradictions) : pages + unités + notes + labels."""
    parts = [page.text for page in pages]
    parts.extend(unit.text for unit in course.units)
    parts.extend(concept.notes for concept in course.concepts if concept.notes)
    parts.extend(concept.label for concept in course.concepts)
    return "\n".join(parts)


def whitelist_terms(course: CourseModel) -> set[str]:
    """Jargon autorisé : labels, formules, fragments courts — pas les phrases PDF."""
    terms: set[str] = set()
    for concept in course.concepts:
        label = concept.label.strip()
        if label:
            terms.add(label)
        for line in concept.notes.splitlines():
            line = line.strip()
            if not line or "CONTRADICTION" in line.upper():
                continue
            for match in _NUMBER_RE.finditer(line):
                terms.add(match.group(0))
            for token in _FORMULA_RE.findall(line):
                terms.add(token)
            # Autoriser le jargon court hors phrase complète (pas de point final).
            if (
                len(line) <= 40
                and not line.endswith((".", "。", "!", "?", "…"))
                and len(line.split()) <= 6
            ):
                terms.add(line)
    return terms


def strip_whitelist(text: str, whitelist: set[str]) -> str:
    result = text
    for term in sorted(whitelist, key=len, reverse=True):
        if not term:
            continue
        pattern = re.compile(re.escape(term), re.IGNORECASE)
        result = pattern.sub(" ", result)
    return result


def is_near_copy(text: str, source_text: str, *, whitelist: set[str] | None = None) -> bool:
    """True si le paragraphe est une copie trop littérale du corpus pages/unités."""
    allowed = whitelist or set()
    cleaned = normalize_text(strip_whitelist(text, allowed))
    source_clean = normalize_text(strip_whitelist(source_text, allowed))
    if len(cleaned) < _MIN_FULL_CHARS:
        return False
    if cleaned in source_clean:
        return True
    words = cleaned.split()
    if len(words) < _WINDOW_WORDS:
        return False
    window = _WINDOW_WORDS
    for start in range(0, len(words) - window + 1):
        chunk = " ".join(words[start : start + window])
        if len(chunk) >= _MIN_CHUNK_CHARS and chunk in source_clean:
            return True
    return False
