"""PedagogicalPlanner : CourseModel → PedagogicalPlan (LLM + fallback heuristique)."""

from __future__ import annotations

import json
import re
from typing import Any

from pydantic import ValidationError

from src.revision.engine.llm_guard import (
    MAX_CHAPTERS_COMPLETE,
    MAX_CHAPTERS_SHORT,
    TOKENS_PLAN,
    LlmBudgetGuard,
    ensure_prompt_fits,
)
from src.revision.engine.models import (
    ChapterPlan,
    Concept,
    CourseModel,
    LearningObjective,
    PedagogicalPlan,
)

_JSON_FENCE = re.compile(r"```(?:json)?\s*([\s\S]*?)```", re.IGNORECASE)
_SHEET_TYPES = frozenset({"short", "complete"})
_LEVELS = frozenset({"beginner", "intermediate", "advanced"})
_LANGUAGES = frozenset({"fr", "en", "ar"})


class PedagogicalPlanner:
    """Produit un plan pédagogique validé, distinct de l'ordre brut du PDF."""

    def __init__(self, llm: object, guard: LlmBudgetGuard) -> None:
        if not callable(getattr(llm, "generate", None)):
            raise TypeError("llm doit fournir generate.")
        self.llm = llm
        self.guard = guard

    def plan(
        self,
        course: CourseModel,
        sheet_type: str,
        level: str,
        language: str,
    ) -> PedagogicalPlan:
        _check_choice(sheet_type, _SHEET_TYPES, "Le type de fiche demandé n'est pas pris en charge.")
        _check_choice(level, _LEVELS, "Le niveau demandé n'est pas pris en charge.")
        _check_choice(language, _LANGUAGES, "La langue demandée n'est pas prise en charge.")
        if not isinstance(course, CourseModel):
            raise TypeError("course doit être un CourseModel.")

        prompt = _build_plan_prompt(course, sheet_type, level, language)

        def produce() -> str:
            if callable(getattr(self.llm, "is_loaded", None)) and self.llm.is_loaded():
                ensure_prompt_fits(self.llm, prompt, TOKENS_PLAN)
            return self.llm.generate(prompt, max_new_tokens=TOKENS_PLAN)

        try:
            payload = self.guard.run_with_retries(
                produce,
                tokens=TOKENS_PLAN,
                parse=_load_plan_payload,
            )
        except ValueError as error:
            return self._fallback_or_raise(course, sheet_type, level, language, error)

        try:
            return _plan_from_payload(
                payload,
                course=course,
                sheet_type=sheet_type,
                level=level,
                language=language,
            )
        except ValueError as error:
            # Erreurs sémantiques (dépendances, plafonds) : pas de fallback.
            message = str(error)
            if "dépendances" in message or "moins de pages" in message:
                raise
            return self._fallback_or_raise(course, sheet_type, level, language, error)

    def _fallback_or_raise(
        self,
        course: CourseModel,
        sheet_type: str,
        level: str,
        language: str,
        error: ValueError,
    ) -> PedagogicalPlan:
        try:
            return _heuristic_plan(course, sheet_type, level, language)
        except ValueError:
            raise error from None


def _check_choice(value: str, accepted: frozenset[str], message: str) -> None:
    if value not in accepted:
        raise ValueError(message)


def _build_plan_prompt(
    course: CourseModel,
    sheet_type: str,
    level: str,
    language: str,
) -> str:
    max_chapters = (
        MAX_CHAPTERS_SHORT if sheet_type == "short" else MAX_CHAPTERS_COMPLETE
    )
    concepts_json = [
        {
            "id": c.id,
            "label": c.label,
            "kind": c.kind,
            "essential": c.essential,
            "depends_on": c.depends_on,
            "pages": c.pages,
        }
        for c in course.concepts
    ]
    relations_json = [
        {
            "source_id": r.source_id,
            "target_id": r.target_id,
            "relation_type": r.relation_type,
        }
        for r in course.relations
    ]
    style = (
        "Fiche short : peu de chapitres, essentiels seulement, concis, "
        "exemples indispensables, pas de questions de révision, depth=short."
        if sheet_type == "short"
        else "Fiche complete : couverture des essentiels, relations, exemples détaillés, "
        "tableaux si utiles, questions de révision, depth=complete."
    )
    return (
        "Tu es EduRAG. Construis un plan pédagogique en JSON uniquement.\n"
        f"Langue={language} ; niveau={level} ; type={sheet_type}.\n"
        f"{style}\n"
        f"Maximum {max_chapters} chapitres. Respecte les dépendances conceptuelles.\n"
        "N'invente aucun concept absent du modèle. Réorganise pédagogiquement "
        "(pas une copie de l'ordre PDF).\n"
        "Schéma : "
        '{"language":"...","level":"...","sheet_type":"...","chapters":['
        '{"index":0,"title":"...","objectives":[{"text":"...","concept_ids":["..."]}],'
        '"concept_ids":["..."],"page_refs":[1],"needs_example":true,'
        '"needs_table":false,"needs_questions":false,"depth":"short|complete"}]}\n'
        f"Thèmes: {json.dumps(course.themes, ensure_ascii=False)}\n"
        f"Essentiels: {json.dumps(course.essential_concept_ids, ensure_ascii=False)}\n"
        f"Concepts: {json.dumps(concepts_json, ensure_ascii=False)}\n"
        f"Relations: {json.dumps(relations_json, ensure_ascii=False)}"
    )


def _load_plan_payload(raw: str) -> dict[str, Any]:
    try:
        return _load_json_object(raw)
    except ValueError as error:
        raise ValueError("JSON de planification invalide.") from error


def _plan_from_payload(
    payload: dict[str, Any],
    *,
    course: CourseModel,
    sheet_type: str,
    level: str,
    language: str,
) -> PedagogicalPlan:
    data = dict(payload)
    data["language"] = language
    data["level"] = level
    data["sheet_type"] = sheet_type
    try:
        plan = PedagogicalPlan.model_validate(data)
    except ValidationError as error:
        raise ValueError("JSON de planification invalide.") from error

    _validate_plan_against_course(plan, course, sheet_type)
    return plan


def _validate_plan_against_course(
    plan: PedagogicalPlan,
    course: CourseModel,
    sheet_type: str,
) -> None:
    max_chapters = (
        MAX_CHAPTERS_SHORT if sheet_type == "short" else MAX_CHAPTERS_COMPLETE
    )
    if len(plan.chapters) == 0:
        raise ValueError("JSON de planification invalide.")
    if len(plan.chapters) > max_chapters:
        raise ValueError(
            "Le cours sélectionné est trop volumineux. "
            "Sélectionnez moins de pages."
        )

    known_ids = {c.id for c in course.concepts}
    depends = {c.id: list(c.depends_on) for c in course.concepts}
    introduced: set[str] = set()

    for chapter in plan.chapters:
        if chapter.depth != sheet_type:
            raise ValueError("JSON de planification invalide.")
        if sheet_type == "short" and chapter.needs_questions:
            raise ValueError("JSON de planification invalide.")
        for cid in chapter.concept_ids:
            if cid not in known_ids:
                raise ValueError("JSON de planification invalide.")
            for dep in depends.get(cid, []):
                if dep in known_ids and dep not in introduced and dep not in chapter.concept_ids:
                    raise ValueError(
                        "Le plan pédagogique ne respecte pas les dépendances conceptuelles."
                    )
        introduced.update(chapter.concept_ids)


def _heuristic_plan(
    course: CourseModel,
    sheet_type: str,
    level: str,
    language: str,
) -> PedagogicalPlan:
    concepts = _select_concepts_for_plan(course, sheet_type)
    if not concepts:
        raise ValueError("Impossible de construire un plan pédagogique fiable.")

    ordered = _topo_sort(concepts)
    max_chapters = (
        MAX_CHAPTERS_SHORT if sheet_type == "short" else MAX_CHAPTERS_COMPLETE
    )
    chunks = _chunk(ordered, max_chapters)
    chapters: list[ChapterPlan] = []
    for index, group in enumerate(chunks):
        page_refs: list[int] = []
        for concept in group:
            for page in concept.pages:
                if page not in page_refs:
                    page_refs.append(page)
        title = _chapter_title(group, index, sheet_type)
        chapters.append(
            ChapterPlan(
                index=index,
                title=title,
                objectives=[
                    LearningObjective(
                        text=f"Comprendre : {concept.label}",
                        concept_ids=[concept.id],
                    )
                    for concept in group
                ],
                concept_ids=[concept.id for concept in group],
                page_refs=page_refs,
                needs_example=True,
                needs_table=sheet_type == "complete" and len(group) > 1,
                needs_questions=sheet_type == "complete",
                depth="short" if sheet_type == "short" else "complete",
            )
        )
    return PedagogicalPlan(
        language=language,
        level=level,
        sheet_type=sheet_type,  # type: ignore[arg-type]
        chapters=chapters,
    )


def _select_concepts_for_plan(course: CourseModel, sheet_type: str) -> list[Concept]:
    by_id = {c.id: c for c in course.concepts}
    essential_ids = list(course.essential_concept_ids) or [
        c.id for c in course.concepts if c.essential
    ]
    selected_ids = list(essential_ids)
    if sheet_type == "complete":
        for concept in course.concepts:
            if concept.id not in selected_ids and concept.essential:
                selected_ids.append(concept.id)
    # Inclure les dépendances des essentiels pour un plan fiable.
    expanded: list[str] = []
    pending = list(selected_ids)
    seen: set[str] = set()
    while pending:
        cid = pending.pop(0)
        if cid in seen or cid not in by_id:
            continue
        seen.add(cid)
        for dep in by_id[cid].depends_on:
            if dep not in seen:
                pending.append(dep)
        expanded.append(cid)
    if not expanded and course.concepts:
        expanded = [course.concepts[0].id]
    return [by_id[cid] for cid in expanded if cid in by_id]


def _topo_sort(concepts: list[Concept]) -> list[Concept]:
    by_id = {c.id: c for c in concepts}
    ids = set(by_id)
    remaining = set(ids)
    ordered: list[Concept] = []
    while remaining:
        ready = [
            cid
            for cid in remaining
            if all(dep not in remaining for dep in by_id[cid].depends_on if dep in ids)
        ]
        if not ready:
            # Cycle : ordre stable sans invention.
            ready = sorted(remaining)
        ready.sort()
        for cid in ready:
            ordered.append(by_id[cid])
            remaining.remove(cid)
    return ordered


def _chunk(concepts: list[Concept], max_chapters: int) -> list[list[Concept]]:
    if max_chapters < 1:
        raise ValueError("Impossible de construire un plan pédagogique fiable.")
    n = min(max_chapters, len(concepts))
    if n == 0:
        return []
    chunks: list[list[Concept]] = [[] for _ in range(n)]
    for index, concept in enumerate(concepts):
        chunks[index % n].append(concept)
    # Réordonner pour préserver la topo globale : remplir séquentiellement.
    size = (len(concepts) + n - 1) // n
    sequential: list[list[Concept]] = []
    for start in range(0, len(concepts), size):
        sequential.append(concepts[start : start + size])
    return [group for group in sequential if group][:max_chapters]


def _chapter_title(group: list[Concept], index: int, sheet_type: str) -> str:
    if len(group) == 1:
        return group[0].label.capitalize()
    labels = ", ".join(c.label for c in group[:2])
    prefix = "Essentiels" if sheet_type == "short" else "Chapitre"
    return f"{prefix} {index + 1} : {labels}"


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
