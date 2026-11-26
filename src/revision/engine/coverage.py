"""Vérification réelle de la couverture pédagogique d'un chapitre.

Ne se fie pas aux seuls `concept_ids_covered` déclarés par le LLM : chaque
concept prévu (essentiel ou secondaire) doit laisser une trace explicative
dans le texte généré (label, synonyme, acronyme, paraphrase issue des notes,
ou contenu propre à son `kind`).
"""

from __future__ import annotations

import re
from typing import Iterable

from src.revision.engine.models import (
    ChapterDraft,
    ChapterPlan,
    Concept,
    CourseModel,
    DefinitionBlock,
    DiagramBlock,
    ExampleBlock,
    ParagraphBlock,
    RetainBlock,
    SynthesisBlock,
    TableBlock,
)

_MIN_EXPLANATORY_CHARS = 24
_MIN_NOTE_PHRASE = 16
_SHORT_TERM_MAX = 3

_ACRONYM_TOKEN = re.compile(r"\b([A-Z]{2,10})\b")
_PAREN_CHUNK = re.compile(r"\(([^)]{1,80})\)")
_WORD_SPLIT = re.compile(r"[^\w\u0600-\u06FF]+", re.UNICODE)
_FORMULA_HINT = re.compile(r"[=+\-*/^]|\\frac|\d")
_EXPLAIN_CUES = re.compile(
    r"(?:"
    r"\b(?:est|sont|permet(?:tent)?|consiste|désigne|définit|correspond|"
    r"means?|is|are|allows?|consists?|defines?|refers|"
    r"يعرف|يعني|هو|هي|يسمح|يتكون)\b"
    r"|:\s*\S"
    r")",
    re.IGNORECASE,
)
_ALIAS_SPLIT = re.compile(
    r"\s*(?:/|–|—|\bou\b|\bor\b|\baussi appelé(?:e)?\b|\balso known as\b|\baka\b)\s*",
    re.IGNORECASE,
)


def chapter_explanation_corpus(draft: ChapterDraft) -> str:
    """Assemble le texte pédagogique utile d'un chapitre (hors questions)."""
    parts: list[str] = []
    for block in draft.blocks:
        if getattr(block, "type", None) == "question":
            continue
        if isinstance(
            block,
            (
                ParagraphBlock,
                DefinitionBlock,
                ExampleBlock,
                RetainBlock,
                SynthesisBlock,
                DiagramBlock,
            ),
        ):
            text = getattr(block, "text", "") or ""
            if text.strip():
                parts.append(text.strip())
        elif isinstance(block, TableBlock):
            parts.extend(str(h).strip() for h in block.headers if str(h).strip())
            for row in block.rows:
                if isinstance(row, list):
                    parts.extend(str(cell).strip() for cell in row if str(cell).strip())
    return "\n".join(parts)


def has_explanatory_body(draft: ChapterDraft) -> bool:
    """Vrai si le chapitre contient au moins un bloc explicatif non vide."""
    corpus = chapter_explanation_corpus(draft)
    return len(_normalize(corpus)) >= _MIN_EXPLANATORY_CHARS


def concept_evidence_terms(concept: Concept) -> list[str]:
    """Termes / phrases servant de preuves d'ancrage du concept dans le texte."""
    terms: list[str] = []
    label = (concept.label or "").strip()
    if label:
        terms.append(label)
        for part in _ALIAS_SPLIT.split(label):
            part = part.strip(" .;,:")
            if part:
                terms.append(part)
        terms.extend(_paren_aliases(label))
        acronym = _initials_acronym(label)
        if acronym:
            terms.append(acronym)

    for line in _usable_note_lines(concept.notes):
        terms.append(line)
        terms.extend(_paren_aliases(line))
        terms.extend(_acronym_expansions(line))
        for acronym in _ACRONYM_TOKEN.findall(line):
            terms.append(acronym)
        # Première proposition courte utile (synonyme / définition partielle).
        head = re.split(r"[.;\n]", line, maxsplit=1)[0].strip()
        if head:
            terms.append(head)
            terms.extend(_acronym_expansions(head))

    return _unique_preserve(terms)


def concept_is_explained(concept: Concept, corpus: str) -> bool:
    """Décide si le concept est réellement expliqué dans `corpus`."""
    normalized_corpus = _normalize(corpus)
    if len(normalized_corpus) < _MIN_EXPLANATORY_CHARS:
        return False

    terms = concept_evidence_terms(concept)
    matched_term: str | None = None
    for term in sorted(terms, key=len, reverse=True):
        if _term_supported_in_corpus(term, corpus, normalized_corpus):
            matched_term = term
            break

    if matched_term is None:
        return False

    return _kind_evidence_ok(concept, corpus, normalized_corpus, matched_term)


def unexplained_planned_concepts(
    draft: ChapterDraft,
    *,
    chapter_plan: ChapterPlan,
    course: CourseModel,
) -> list[Concept]:
    """Concepts du plan absents d'une explication réelle dans le chapitre."""
    by_id = {c.id: c for c in course.concepts}
    corpus = chapter_explanation_corpus(draft)
    missing: list[Concept] = []
    for concept_id in chapter_plan.concept_ids:
        concept = by_id.get(concept_id)
        if concept is None:
            continue
        if not concept_is_explained(concept, corpus):
            missing.append(concept)
    return missing


def unexplained_declared_concepts(
    draft: ChapterDraft,
    *,
    course: CourseModel,
) -> list[Concept]:
    """Concepts déclarés dans concept_ids_covered sans explication pertinente."""
    by_id = {c.id: c for c in course.concepts}
    corpus = chapter_explanation_corpus(draft)
    missing: list[Concept] = []
    seen: set[str] = set()
    for concept_id in draft.concept_ids_covered:
        if concept_id in seen:
            continue
        seen.add(concept_id)
        concept = by_id.get(concept_id)
        if concept is None:
            continue
        if not concept_is_explained(concept, corpus):
            missing.append(concept)
    return missing


def coverage_failure_message(concepts: Iterable[Concept]) -> str:
    """Message d'erreur distinguant essentiels et secondaires."""
    concepts = list(concepts)
    if not concepts:
        return "Concepts non expliqués dans le chapitre."
    essentials = [c for c in concepts if c.essential]
    secondary = [c for c in concepts if not c.essential]
    parts: list[str] = []
    if essentials:
        parts.append(
            "concepts essentiels non expliqués : "
            + ", ".join(f"{c.id} ({c.label})" for c in essentials)
        )
    if secondary:
        parts.append(
            "concepts secondaires non expliqués : "
            + ", ".join(f"{c.id} ({c.label})" for c in secondary)
        )
    return "Couverture pédagogique insuffisante — " + " ; ".join(parts)


def validate_chapter_concept_coverage(
    draft: ChapterDraft,
    *,
    chapter_plan: ChapterPlan,
    course: CourseModel,
) -> list[Concept]:
    """Retourne la liste des concepts (plan ∪ déclarés) non réellement expliqués."""
    planned = unexplained_planned_concepts(
        draft, chapter_plan=chapter_plan, course=course
    )
    declared = unexplained_declared_concepts(draft, course=course)
    by_id: dict[str, Concept] = {}
    for concept in planned + declared:
        by_id[concept.id] = concept
    return list(by_id.values())


def explained_planned_ids(
    draft: ChapterDraft,
    *,
    chapter_plan: ChapterPlan,
    course: CourseModel,
) -> set[str]:
    """Identifiants du plan réellement expliqués dans le brouillon."""
    missing = {c.id for c in unexplained_planned_concepts(
        draft, chapter_plan=chapter_plan, course=course
    )}
    return {cid for cid in chapter_plan.concept_ids if cid not in missing}


def _count_block_type(draft: ChapterDraft, block_type: str) -> int:
    return sum(1 for block in draft.blocks if getattr(block, "type", None) == block_type)


def _source_pages_present(draft: ChapterDraft) -> bool:
    for block in draft.blocks:
        pages = getattr(block, "source_pages", None)
        if isinstance(pages, list) and any(
            isinstance(p, int) and not isinstance(p, bool) and p >= 1 for p in pages
        ):
            return True
    return False


def rewrite_impoverishment_reason(
    original: ChapterDraft,
    rewrite: ChapterDraft,
    *,
    chapter_plan: ChapterPlan,
    course: CourseModel,
) -> str | None:
    """Compare couverture / richesse ; None si la réécriture est acceptable."""
    before = explained_planned_ids(
        original, chapter_plan=chapter_plan, course=course
    )
    after = explained_planned_ids(
        rewrite, chapter_plan=chapter_plan, course=course
    )
    lost = sorted(before - after)
    if lost:
        return (
            "Réécriture appauvrie : explications perdues pour "
            + ", ".join(lost)
        )

    corpus_before = chapter_explanation_corpus(original)
    corpus_after = chapter_explanation_corpus(rewrite)
    len_before = len(_normalize(corpus_before))
    len_after = len(_normalize(corpus_after))
    if len_before >= 80 and len_after < max(40, len_before // 2):
        return (
            "Réécriture appauvrie : contenu explicatif excessivement réduit "
            f"({len_after} caractères utiles vs {len_before} avant)."
        )

    if _count_block_type(original, "example") > 0 and _count_block_type(rewrite, "example") == 0:
        return "Réécriture appauvrie : exemples pédagogiques supprimés."

    if _count_block_type(original, "table") > 0 and _count_block_type(rewrite, "table") == 0:
        return "Réécriture appauvrie : tableau pédagogique supprimé."

    if _source_pages_present(original) and not _source_pages_present(rewrite):
        return "Réécriture appauvrie : références de pages supprimées."

    return None


def _usable_note_lines(notes: str) -> list[str]:
    lines: list[str] = []
    for raw in (notes or "").splitlines():
        line = raw.strip()
        if not line:
            continue
        upper = line.upper()
        if "CONTRADICTION" in upper:
            continue
        if line.casefold().startswith("catégorie llm"):
            continue
        # Retirer un éventuel marqueur d'alias de kind en fin de note.
        line = re.sub(
            r"\s*\(Catégorie LLM d'origine\s*:[^)]+\)\s*$",
            "",
            line,
            flags=re.IGNORECASE,
        ).strip()
        if not line:
            continue
        if len(_normalize(line)) >= _MIN_NOTE_PHRASE or _FORMULA_HINT.search(line):
            lines.append(line)
    return lines


def _paren_aliases(text: str) -> list[str]:
    aliases: list[str] = []
    for match in _PAREN_CHUNK.finditer(text):
        inner = match.group(1).strip()
        if not inner:
            continue
        aliases.append(inner)
        for part in _ALIAS_SPLIT.split(inner):
            part = part.strip(" .;,:")
            if part:
                aliases.append(part)
    return aliases


_ACRONYM_EXPANSION = re.compile(
    r"([A-Za-zÀ-ÿ][\wÀ-ÿ]*(?:\s+[A-Za-zÀ-ÿ][\wÀ-ÿ]*){1,10})\s*\(([A-Z]{2,10})\)"
)
_ACRONYM_THEN_EXPANSION = re.compile(
    r"\b([A-Z]{2,10})\s*\(([A-Za-zÀ-ÿ][^)]{1,80})\)"
)


def _acronym_expansions(text: str) -> list[str]:
    """Extrait les paires « nom complet (ACR) » et « ACR (nom complet) »."""
    pairs: list[str] = []
    for match in _ACRONYM_EXPANSION.finditer(text):
        pairs.append(match.group(1).strip())
        pairs.append(match.group(2).strip())
    for match in _ACRONYM_THEN_EXPANSION.finditer(text):
        pairs.append(match.group(1).strip())
        pairs.append(match.group(2).strip())
    return pairs


def _initials_acronym(label: str) -> str | None:
    words = [w for w in _WORD_SPLIT.split(label) if w]
    if len(words) < 2:
        return None
    initials = "".join(w[0] for w in words if w[0].isalpha())
    if len(initials) >= 2 and initials.isascii():
        return initials.upper()
    return None


def _term_supported_in_corpus(
    term: str, corpus: str, normalized_corpus: str
) -> bool:
    cleaned = term.strip()
    if not cleaned:
        return False
    normalized_term = _normalize(cleaned)
    if not normalized_term:
        return False

    # Phrase issue des notes : sous-chaîne normalisée suffisante.
    if len(normalized_term) >= _MIN_NOTE_PHRASE and normalized_term in normalized_corpus:
        return True

    if not _whole_term_present(cleaned, corpus, normalized_corpus):
        return False

    # Termes très courts (acronymes) : exiger un contexte explicatif à proximité.
    compact = re.sub(r"\s+", "", normalized_term)
    if len(compact) <= _SHORT_TERM_MAX:
        return _has_local_explanation(cleaned, corpus)
    return True


def _whole_term_present(term: str, corpus: str, normalized_corpus: str) -> bool:
    normalized_term = _normalize(term)
    if not normalized_term:
        return False
    if " " in normalized_term or len(normalized_term) >= 8:
        return normalized_term in normalized_corpus
    pattern = re.compile(
        rf"(?<!\w){re.escape(term.strip())}(?!\w)",
        re.IGNORECASE | re.UNICODE,
    )
    if pattern.search(corpus):
        return True
    # Variante normalisée (accents / casse) sur tokens.
    tokens = set(_WORD_SPLIT.split(normalized_corpus))
    return normalized_term in tokens


def _has_local_explanation(term: str, corpus: str) -> bool:
    pattern = re.compile(
        rf"(.{{0,80}})(?<!\w){re.escape(term.strip())}(?!\w)(.{{0,80}})",
        re.IGNORECASE | re.UNICODE | re.DOTALL,
    )
    for match in pattern.finditer(corpus):
        window = f"{match.group(1)}{match.group(2)}"
        if _EXPLAIN_CUES.search(window) or len(_normalize(window)) >= 20:
            return True
    return False


def _kind_evidence_ok(
    concept: Concept,
    corpus: str,
    normalized_corpus: str,
    matched_term: str,
) -> bool:
    kind = concept.kind
    if kind == "formula":
        # La formule des notes, ou un motif mathématique près du terme ancré.
        for line in _usable_note_lines(concept.notes):
            if _FORMULA_HINT.search(line) and _normalize(line) in normalized_corpus:
                return True
        return _has_local_explanation(matched_term, corpus) or bool(
            _FORMULA_HINT.search(corpus)
        )
    if kind in {"mechanism", "method"}:
        # Une ancre + un minimum de prose explicative suffisent.
        return len(normalized_corpus) >= _MIN_EXPLANATORY_CHARS
    if kind == "relation":
        return len(normalized_corpus) >= _MIN_EXPLANATORY_CHARS
    if kind == "definition":
        # Définition : ancre trouvée dans un contexte définitoire, ou notes reprises.
        if any(
            _normalize(line) in normalized_corpus
            for line in _usable_note_lines(concept.notes)
            if len(_normalize(line)) >= _MIN_NOTE_PHRASE
        ):
            return True
        return _has_local_explanation(matched_term, corpus) or len(
            normalized_corpus
        ) >= 40
    # other
    return True


def _normalize(text: str) -> str:
    return " ".join(text.casefold().split())


def _unique_preserve(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        key = _normalize(value)
        if not key or key in seen:
            continue
        seen.add(key)
        result.append(value.strip())
    return result
