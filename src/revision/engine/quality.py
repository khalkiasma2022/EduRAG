"""QualityController : contrôles pédagogiques + réécriture ciblée bornée."""

from __future__ import annotations

import json
import re
from typing import Any

from pydantic import ValidationError

from src.revision.engine.coverage import (
    coverage_failure_message,
    rewrite_impoverishment_reason,
    validate_chapter_concept_coverage,
)
from src.revision.engine.llm_guard import (
    MAX_REWRITE_CHAPTERS,
    TOKENS_REWRITE,
    LlmBudgetGuard,
    ensure_prompt_fits,
    max_attempts_per_step,
)
from src.revision.engine.near_copy import (
    full_source_corpus,
    is_near_copy,
    page_source_corpus,
    whitelist_terms,
)
from src.revision.engine.models import (
    ChapterDraft,
    ChapterPlan,
    ContentBlock,
    CourseModel,
    CoursePage,
    DefinitionBlock,
    ExampleBlock,
    ParagraphBlock,
    PedagogicalDocument,
    PedagogicalPlan,
    QualityIssue,
    QualityReport,
)

_JSON_FENCE = re.compile(r"```(?:json)?\s*([\s\S]*?)```", re.IGNORECASE)
_URL_RE = re.compile(r"https?://|www\.", re.IGNORECASE)
_NUMBER_RE = re.compile(r"(?<![A-Za-z])\d+(?:[.,]\d+)?")
_RESOLVING_RE = re.compile(
    r"\b("
    r"en réalité|la vérité est|on conclut que|donc on ignore|"
    r"in reality|the truth is|we conclude that|"
    r"في الواقع|نستنتج أن"
    r")\b",
    re.IGNORECASE,
)
_FORBIDDEN_SECTION_RE = re.compile(
    r"\b(bibliographie|references|références|contact|remerciements|"
    r"acknowledgements|annexes?|glossary|glossaire)\b",
    re.IGNORECASE,
)
_FILLER_RE = re.compile(
    r"("
    r"dans ce chapitre nous allons|"
    r"il est important de noter que|"
    r"comme indiqué précédemment|"
    r"ceci est un texte de remplissage|"
    r"lorem ipsum"
    r")",
    re.IGNORECASE,
)

# Défauts corrigibles par réécriture ciblée d'un chapitre.
_REWRITABLE_CODES = frozenset(
    {
        "empty_chapter",
        "excessive_repetition",
        "forbidden_url",
        "near_copy",
        "invented_resolution",
        "generic_filler",
        "missing_explanation",
        "unexplained_concept",
        "forbidden_section",
        "invented_number",
        "unmarked_example",
    }
)


class QualityController:
    """Vérifie un PedagogicalDocument et tente des réécritures ciblées bornées."""

    def __init__(self, llm: object, guard: LlmBudgetGuard) -> None:
        if not callable(getattr(llm, "generate", None)):
            raise TypeError("llm doit fournir generate.")
        self.llm = llm
        self.guard = guard

    def review(
        self,
        document: PedagogicalDocument,
        course: CourseModel,
        plan: PedagogicalPlan,
        *,
        source_pages: list[CoursePage] | None = None,
    ) -> tuple[PedagogicalDocument, QualityReport]:
        if not isinstance(document, PedagogicalDocument):
            raise TypeError("document doit être un PedagogicalDocument.")
        if not isinstance(course, CourseModel):
            raise TypeError("course doit être un CourseModel.")
        if not isinstance(plan, PedagogicalPlan):
            raise TypeError("plan doit être un PedagogicalPlan.")

        pages = list(source_pages or [])
        source_text = full_source_corpus(course, pages)
        page_source_text = page_source_corpus(course, pages)
        working = document.model_copy(deep=True)
        rewritten: list[int] = []

        working = _strip_question_blocks(working)
        report = self._evaluate(
            working, course, plan, source_text, page_source_text=page_source_text
        )
        if report.ok:
            return working, report

        if _has_structural_failure(report.issues):
            raise ValueError(
                "Contrôle qualité en échec : le document ne respecte pas le plan "
                f"({_summarize_issues(report.issues)})."
            )

        # Les erreurs non réécriture (ex. ids manquants) bloquent avant toute réécriture.
        if _has_non_rewritable_error(report.issues):
            raise ValueError(
                "Contrôle qualité en échec : "
                f"{_summarize_issues(report.issues)}."
            )

        targets = _select_rewrite_targets(report.issues)
        if not targets:
            raise ValueError(
                "Contrôle qualité en échec : "
                f"{_summarize_issues(report.issues)}."
            )

        for chapter_index in targets:
            if len(rewritten) >= MAX_REWRITE_CHAPTERS:
                break
            # Garantir tentative + retry (run_with_retries) avant de démarrer.
            attempts = max_attempts_per_step()
            need_calls = attempts
            need_tokens = attempts * TOKENS_REWRITE
            if (
                self.guard.remaining_calls < need_calls
                or self.guard.remaining_tokens < need_tokens
            ):
                break
            chapter_plan = _chapter_plan_by_index(plan, chapter_index)
            if chapter_plan is None:
                continue
            original = next(
                (ch for ch in working.chapters if ch.index == chapter_index),
                None,
            )
            try:
                new_draft = self._rewrite_chapter(
                    working,
                    course=course,
                    plan=plan,
                    chapter_plan=chapter_plan,
                    source_pages=pages,
                    issues=[i for i in report.issues if i.chapter_index == chapter_index],
                    original=original,
                )
            except ValueError as error:
                raise ValueError(
                    f"Contrôle qualité en échec pendant la réécriture : {error}"
                ) from error
            working = _replace_chapter(working, new_draft)
            rewritten.append(chapter_index)

        final = self._evaluate(
            working, course, plan, source_text, page_source_text=page_source_text
        )
        final = QualityReport(
            ok=final.ok,
            issues=final.issues,
            rewrite_chapter_indexes=rewritten,
        )
        if not final.ok:
            raise ValueError(
                "Contrôle qualité en échec après réécriture : "
                f"{_summarize_issues(final.issues)}."
            )
        working = _strip_question_blocks(working)
        return working, final

    def _evaluate(
        self,
        document: PedagogicalDocument,
        course: CourseModel,
        plan: PedagogicalPlan,
        source_text: str,
        *,
        page_source_text: str | None = None,
    ) -> QualityReport:
        issues: list[QualityIssue] = []
        page_corpus = page_source_text if page_source_text is not None else source_text

        if len(document.chapters) != len(plan.chapters):
            issues.append(
                QualityIssue(
                    code="missing_chapter",
                    severity="error",
                    message=(
                        "Le nombre de chapitres du document ne correspond pas au plan "
                        f"({len(document.chapters)}/{len(plan.chapters)})."
                    ),
                    chapter_index=None,
                )
            )

        plan_by_index = {ch.index: ch for ch in plan.chapters}
        doc_by_index = {ch.index: ch for ch in document.chapters}

        for chapter_plan in plan.chapters:
            if chapter_plan.index not in doc_by_index:
                issues.append(
                    QualityIssue(
                        code="missing_chapter",
                        severity="error",
                        message=f"Chapitre manquant dans le document : {chapter_plan.title}.",
                        chapter_index=chapter_plan.index,
                    )
                )

        covered_all: set[str] = set()
        for draft in document.chapters:
            covered_all.update(draft.concept_ids_covered)
            chapter_plan = plan_by_index.get(draft.index)
            issues.extend(
                _evaluate_chapter(
                    draft,
                    chapter_plan=chapter_plan,
                    course=course,
                    source_text=source_text,
                    page_source_text=page_corpus,
                )
            )

        planned_essentials = _planned_essential_ids(course, plan)
        missing_essentials = [
            cid for cid in planned_essentials if cid not in covered_all
        ]
        if missing_essentials and not any(i.code == "missing_chapter" for i in issues):
            # Attacher au premier chapitre du plan qui devait les couvrir.
            for chapter_plan in plan.chapters:
                local_missing = [
                    cid for cid in chapter_plan.concept_ids if cid in missing_essentials
                ]
                if not local_missing:
                    continue
                issues.append(
                    QualityIssue(
                        code="missing_essential_in_chapter",
                        severity="error",
                        message=(
                            "Concepts essentiels non couverts : "
                            + ", ".join(local_missing)
                        ),
                        chapter_index=chapter_plan.index,
                    )
                )

        errors = [i for i in issues if i.severity == "error"]
        return QualityReport(
            ok=len(errors) == 0,
            issues=issues,
            rewrite_chapter_indexes=[],
        )

    def _rewrite_chapter(
        self,
        document: PedagogicalDocument,
        *,
        course: CourseModel,
        plan: PedagogicalPlan,
        chapter_plan: ChapterPlan,
        source_pages: list[CoursePage],
        issues: list[QualityIssue],
        original: ChapterDraft | None = None,
    ) -> ChapterDraft:
        prompt = _build_rewrite_prompt(
            document=document,
            course=course,
            plan=plan,
            chapter_plan=chapter_plan,
            source_pages=source_pages,
            issues=issues,
            original=original,
        )

        def produce() -> str:
            ensure_prompt_fits(self.llm, prompt, TOKENS_REWRITE)
            return self.llm.generate(prompt, max_new_tokens=TOKENS_REWRITE)

        def parse(raw: str) -> ChapterDraft:
            draft = _parse_chapter_draft(raw, expected_index=chapter_plan.index)
            unexplained = validate_chapter_concept_coverage(
                draft, chapter_plan=chapter_plan, course=course
            )
            if unexplained:
                raise ValueError(coverage_failure_message(unexplained))
            if original is not None:
                reason = rewrite_impoverishment_reason(
                    original,
                    draft,
                    chapter_plan=chapter_plan,
                    course=course,
                )
                if reason:
                    raise ValueError(reason)
            return draft

        return self.guard.run_with_retries(
            produce,
            tokens=TOKENS_REWRITE,
            parse=parse,
        )


def _strip_question_blocks(document: PedagogicalDocument) -> PedagogicalDocument:
    """Retire les blocs question sans toucher au reste du contenu."""
    chapters: list[ChapterDraft] = []
    changed = False
    for chapter in document.chapters:
        kept = [block for block in chapter.blocks if getattr(block, "type", None) != "question"]
        if len(kept) != len(chapter.blocks):
            changed = True
            chapters.append(chapter.model_copy(update={"blocks": kept}))
        else:
            chapters.append(chapter)
    if not changed:
        return document
    return document.model_copy(update={"chapters": chapters})


def _evaluate_chapter(
    draft: ChapterDraft,
    *,
    chapter_plan: ChapterPlan | None,
    course: CourseModel,
    source_text: str,
    page_source_text: str | None = None,
) -> list[QualityIssue]:
    issues: list[QualityIssue] = []
    index = draft.index
    page_corpus = page_source_text if page_source_text is not None else source_text

    if not draft.blocks:
        issues.append(
            QualityIssue(
                code="empty_chapter",
                severity="error",
                message=f"Chapitre vide : {draft.title}.",
                chapter_index=index,
            )
        )
        return issues

    if chapter_plan is not None:
        required = set(chapter_plan.concept_ids)
        covered = set(draft.concept_ids_covered)
        if required and not required.issubset(covered):
            issues.append(
                QualityIssue(
                    code="missing_essential_in_chapter",
                    severity="error",
                    message=(
                        "Le chapitre ne couvre pas les concepts prévus : "
                        + ", ".join(sorted(required - covered))
                    ),
                    chapter_index=index,
                )
            )
        unexplained = validate_chapter_concept_coverage(
            draft, chapter_plan=chapter_plan, course=course
        )
        if unexplained:
            issues.append(
                QualityIssue(
                    code="unexplained_concept",
                    severity="error",
                    message=coverage_failure_message(unexplained),
                    chapter_index=index,
                )
            )
        if chapter_plan.title.strip() and draft.title.strip() != chapter_plan.title.strip():
            # Tolérance : warning seulement — pas bloquant seul.
            issues.append(
                QualityIssue(
                    code="title_mismatch",
                    severity="warning",
                    message="Le titre du chapitre diffère du plan.",
                    chapter_index=index,
                )
            )

    if not _has_real_explanation(draft):
        issues.append(
            QualityIssue(
                code="missing_explanation",
                severity="error",
                message="Le chapitre ne contient pas d'explication pédagogique suffisante.",
                chapter_index=index,
            )
        )

    texts = [_block_text(block) for block in draft.blocks]
    texts = [t for t in texts if t]
    if _has_excessive_repetition(texts):
        issues.append(
            QualityIssue(
                code="excessive_repetition",
                severity="error",
                message="Répétitions excessives dans le chapitre.",
                chapter_index=index,
            )
        )

    whitelist = whitelist_terms(course)
    concept_notes = " ".join(
        c.notes
        for c in course.concepts
        if c.id in set(draft.concept_ids_covered)
        or (chapter_plan is not None and c.id in set(chapter_plan.concept_ids))
    )
    has_contradiction = "CONTRADICTION" in concept_notes.upper()

    for block in draft.blocks:
        for text in _block_texts(block):
            if _URL_RE.search(text):
                issues.append(
                    QualityIssue(
                        code="forbidden_url",
                        severity="error",
                        message="URL artificielle ou non autorisée détectée.",
                        chapter_index=index,
                    )
                )
            if _FORBIDDEN_SECTION_RE.search(text):
                issues.append(
                    QualityIssue(
                        code="forbidden_section",
                        severity="error",
                        message="Section indésirable détectée (biblio/contact/etc.).",
                        chapter_index=index,
                    )
                )
            if _FILLER_RE.search(text):
                issues.append(
                    QualityIssue(
                        code="generic_filler",
                        severity="error",
                        message="Remplissage générique détecté.",
                        chapter_index=index,
                    )
                )
            if isinstance(block, ParagraphBlock) and is_near_copy(
                text, page_corpus, whitelist=whitelist
            ):
                issues.append(
                    QualityIssue(
                        code="near_copy",
                        severity="error",
                        message="Passage excessivement copié du document source.",
                        chapter_index=index,
                    )
                )
            if has_contradiction and _RESOLVING_RE.search(text):
                issues.append(
                    QualityIssue(
                        code="invented_resolution",
                        severity="error",
                        message="Résolution inventée d'une contradiction du cours.",
                        chapter_index=index,
                    )
                )
            invented = _invented_numbers(text, source_text)
            if invented and not (
                isinstance(block, ExampleBlock)
                and block.origin == "pedagogical_addition"
            ):
                issues.append(
                    QualityIssue(
                        code="invented_number",
                        severity="error",
                        message="Valeur numérique absente des sources du cours.",
                        chapter_index=index,
                    )
                )

        if isinstance(block, ExampleBlock):
            if block.origin not in {"from_course", "pedagogical_addition"}:
                issues.append(
                    QualityIssue(
                        code="unmarked_example",
                        severity="error",
                        message="Exemple sans origine pédagogique claire.",
                        chapter_index=index,
                    )
                )

    return _dedupe_issues(issues)


def _has_real_explanation(draft: ChapterDraft) -> bool:
    for block in draft.blocks:
        if isinstance(block, (ParagraphBlock, DefinitionBlock)):
            if len(_normalize(getattr(block, "text", ""))) >= 24:
                return True
    return False


def _has_excessive_repetition(texts: list[str]) -> bool:
    normalized = [_normalize(t) for t in texts if _normalize(t)]
    if len(normalized) < 3:
        return False
    counts: dict[str, int] = {}
    for text in normalized:
        counts[text] = counts.get(text, 0) + 1
        if counts[text] >= 3:
            return True
    # Similarité forte entre plusieurs blocs
    for i, left in enumerate(normalized):
        similar = 1
        for j, right in enumerate(normalized):
            if i == j:
                continue
            if left == right or (
                len(left) > 40 and (left in right or right in left)
            ):
                similar += 1
        if similar >= 3:
            return True
    return False


def _planned_essential_ids(course: CourseModel, plan: PedagogicalPlan) -> list[str]:
    planned = {cid for chapter in plan.chapters for cid in chapter.concept_ids}
    essentials = set(course.essential_concept_ids) or {
        c.id for c in course.concepts if c.essential
    }
    return sorted(planned & essentials) if essentials else sorted(planned)


def _has_structural_failure(issues: list[QualityIssue]) -> bool:
    return any(issue.code == "missing_chapter" for issue in issues)


def _has_non_rewritable_error(issues: list[QualityIssue]) -> bool:
    return any(
        issue.severity == "error" and issue.code not in _REWRITABLE_CODES
        for issue in issues
    )


def _select_rewrite_targets(issues: list[QualityIssue]) -> list[int]:
    indexes: list[int] = []
    for issue in issues:
        if issue.severity != "error":
            continue
        if issue.code not in _REWRITABLE_CODES:
            continue
        if issue.chapter_index is None:
            continue
        if issue.chapter_index not in indexes:
            indexes.append(issue.chapter_index)
    return indexes[:MAX_REWRITE_CHAPTERS]


def _chapter_plan_by_index(plan: PedagogicalPlan, index: int) -> ChapterPlan | None:
    for chapter in plan.chapters:
        if chapter.index == index:
            return chapter
    return None


def _replace_chapter(
    document: PedagogicalDocument, draft: ChapterDraft
) -> PedagogicalDocument:
    chapters = []
    replaced = False
    for chapter in document.chapters:
        if chapter.index == draft.index:
            chapters.append(draft)
            replaced = True
        else:
            chapters.append(chapter)
    if not replaced:
        chapters.append(draft)
        chapters.sort(key=lambda item: item.index)
    return document.model_copy(update={"chapters": chapters})


def _build_rewrite_prompt(
    *,
    document: PedagogicalDocument,
    course: CourseModel,
    plan: PedagogicalPlan,
    chapter_plan: ChapterPlan,
    source_pages: list[CoursePage],
    issues: list[QualityIssue],
    original: ChapterDraft | None = None,
) -> str:
    issue_lines = "; ".join(f"{i.code}: {i.message}" for i in issues) or "défauts de qualité"
    concepts = [c for c in course.concepts if c.id in set(chapter_plan.concept_ids)]
    excerpts = []
    wanted = set(chapter_plan.page_refs)
    for concept in concepts:
        wanted.update(concept.pages)
    for page in source_pages:
        if page.page_number in wanted:
            excerpts.append(f"[Page {page.page_number}]\n{page.text}")
    previous = [ch.title for ch in document.chapters if ch.index != chapter_plan.index]
    original_json = (
        json.dumps(original.model_dump(), ensure_ascii=False)
        if original is not None
        else "(indisponible)"
    )
    return (
        "Tu es EduRAG. Réécris UNIQUEMENT le chapitre indiqué en JSON valide.\n"
        "Corrige les défauts listés. Ne copie pas le PDF. N'invente rien.\n"
        "Conserve tous les concepts prévus (essentiels et secondaires), les exemples, "
        "formules, tableaux et références de pages déjà présents.\n"
        "Chaque concept prévu doit rester réellement expliqué (pas une simple mention).\n"
        "N'appauvris pas les explications : la couverture réelle doit rester au moins "
        "équivalente au chapitre d'origine.\n"
        "Si une CONTRADICTION est présente, signale-la sans la résoudre.\n"
        "Distingue origin=from_course et origin=pedagogical_addition.\n"
        "Interdit : type question, quiz, questions de révision.\n"
        "Chaque bloc a un seul type parmi : paragraph, definition, example, retain, "
        "table, diagram, synthesis.\n"
        "Schéma : "
        '{"index":0,"title":"...","concept_ids_covered":["..."],'
        '"blocks":[{"type":"paragraph","text":"..."}]}\n'
        f"Langue={plan.language}\n"
        f"Défauts à corriger : {issue_lines}\n"
        f"Chapitre : index={chapter_plan.index} ; titre={chapter_plan.title}\n"
        f"Concepts : {json.dumps([c.model_dump() for c in concepts], ensure_ascii=False)}\n"
        f"Chapitre d'origine : {original_json}\n"
        f"Autres chapitres déjà valides : {', '.join(previous) or '(aucun)'}\n"
        f"Extraits :\n{chr(10).join(excerpts) if excerpts else '(aucun)'}"
    )


def _parse_chapter_draft(raw: str, *, expected_index: int) -> ChapterDraft:
    try:
        payload = _load_json_object(raw)
    except ValueError as error:
        raise ValueError("JSON de réécriture invalide.") from error
    payload["index"] = expected_index
    try:
        if "blocks" in payload and isinstance(payload["blocks"], list):
            normalized = []
            for item in payload["blocks"]:
                if not isinstance(item, dict):
                    raise ValueError("JSON de réécriture invalide.")
                normalized.append(ContentBlock.model_validate(item))
            payload["blocks"] = [block.model_dump() for block in normalized]
        draft = ChapterDraft.model_validate(payload)
    except (ValidationError, ValueError, TypeError) as error:
        raise ValueError("JSON de réécriture invalide.") from error
    if draft.index != expected_index:
        raise ValueError("JSON de réécriture invalide.")
    return draft


def _load_json_object(raw: str) -> dict[str, Any]:
    text = raw.strip()
    fence = _JSON_FENCE.search(text)
    if fence:
        text = fence.group(1).strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError as error:
        raise ValueError("JSON invalide.") from error
    if not isinstance(data, dict):
        raise ValueError("JSON invalide.")
    return data


def _source_corpus(course: CourseModel, pages: list[CoursePage]) -> str:
    """Compatibilité : corpus large (pages + notes + labels)."""
    return full_source_corpus(course, pages)


def _block_text(block: object) -> str:
    text = getattr(block, "text", None)
    return text if isinstance(text, str) else ""


def _block_texts(block: object) -> list[str]:
    texts: list[str] = []
    text = _block_text(block)
    if text.strip():
        texts.append(text)
    headers = getattr(block, "headers", None)
    if isinstance(headers, list):
        texts.extend(str(item) for item in headers)
    rows = getattr(block, "rows", None)
    if isinstance(rows, list):
        for row in rows:
            if isinstance(row, list):
                texts.extend(str(cell) for cell in row)
    return texts


def _invented_numbers(text: str, source_text: str) -> list[str]:
    source_nums = {m.group(0) for m in _NUMBER_RE.finditer(source_text)}
    return [
        match.group(0)
        for match in _NUMBER_RE.finditer(text)
        if match.group(0) not in source_nums
    ]


def _normalize(text: str) -> str:
    return " ".join(text.casefold().split())


def _dedupe_issues(issues: list[QualityIssue]) -> list[QualityIssue]:
    seen: set[tuple[str, int | None, str]] = set()
    unique: list[QualityIssue] = []
    for issue in issues:
        key = (issue.code, issue.chapter_index, issue.message)
        if key in seen:
            continue
        seen.add(key)
        unique.append(issue)
    return unique


def _summarize_issues(issues: list[QualityIssue]) -> str:
    errors = [i for i in issues if i.severity == "error"]
    if not errors:
        return "aucun détail"
    return "; ".join(f"{i.code}: {i.message}" for i in errors[:5])
