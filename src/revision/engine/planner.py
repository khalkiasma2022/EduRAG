"""PedagogicalPlanner : CourseModel → PedagogicalPlan (LLM + fallback heuristique).

Mode unique : fiche complète structurée, tous les concepts pertinents, aucune question.
"""

from __future__ import annotations

import json
from typing import Any

from pydantic import ValidationError

from src.revision.engine.json_util import load_json_object
from src.revision.engine.llm_guard import (
    LEGACY_STORE_LEVEL,
    LEGACY_STORE_SHEET_TYPE,
    MAX_CHAPTERS,
    MAX_CONCEPTS_PER_CHAPTER,
    TOKENS_PLAN,
    LlmBudgetGuard,
    ensure_course_concept_capacity,
    ensure_chapter_density,
    ensure_prompt_fits,
)
from src.revision.engine.models import (
    ChapterPlan,
    Concept,
    CourseModel,
    LearningObjective,
    PedagogicalPlan,
)
from src.revision.engine.progress import log_progress

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
        sheet_type: str = LEGACY_STORE_SHEET_TYPE,
        level: str = LEGACY_STORE_LEVEL,
        language: str = "fr",
    ) -> PedagogicalPlan:
        """Construit le plan. ``sheet_type`` / ``level`` sont ignorés pour le contenu."""
        del sheet_type, level
        _check_choice(language, _LANGUAGES, "La langue demandée n'est pas prise en charge.")
        if not isinstance(course, CourseModel):
            raise TypeError("course doit être un CourseModel.")
        # Refus avant appel LLM : pas de densification silencieuse au-delà de 10×4.
        ensure_course_concept_capacity(len(course.concepts))
        # Refus avant appel LLM : graphe cyclique → erreur explicite (pas de fallback).
        _ensure_acyclic_dependencies(course.concepts)

        log_progress(
            f"Planification pédagogique ({len(course.concepts)} concepts)…"
        )
        prompt = _build_plan_prompt(course, language)

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
            return self._fallback_or_raise(course, language, error)

        try:
            return _plan_from_payload(payload, course=course, language=language)
        except ValueError as error:
            message = str(error).casefold()
            # Omission / dépendances : échec dur (ne pas masquer via heuristique).
            # Densité LLM excessive : bascule heuristique si le cours tient en capacité.
            if "dépendances" in message or "omet des concepts" in message:
                raise
            return self._fallback_or_raise(course, language, error)

    def _fallback_or_raise(
        self,
        course: CourseModel,
        language: str,
        error: ValueError,
    ) -> PedagogicalPlan:
        try:
            return _heuristic_plan(course, language)
        except ValueError as fallback_error:
            # Ne pas masquer un cycle (ou autre erreur structurelle) derrière
            # l'erreur LLM d'origine (JSON invalide, etc.).
            if _is_dependency_cycle_error(fallback_error):
                raise fallback_error from None
            raise error from None


def _check_choice(value: str, accepted: frozenset[str], message: str) -> None:
    if value not in accepted:
        raise ValueError(message)


def _build_plan_prompt(course: CourseModel, language: str) -> str:
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
    return (
        "Tu es EduRAG. Construis un plan pédagogique en JSON uniquement.\n"
        f"Langue={language}.\n"
        "Fiche unique complète et structurée : couvre TOUS les concepts fournis "
        "(essentiels et secondaires), relations, exemples, tableaux si utiles.\n"
        "Interdit : questions de révision, quiz, needs_questions=true.\n"
        "Chaque concept fourni doit apparaître dans exactement un chapitre "
        "(pas de perte, pas de doublon).\n"
        f"Regroupe de façon pédagogique (environ {MAX_CONCEPTS_PER_CHAPTER} concepts "
        f"par chapitre, au plus {MAX_CHAPTERS} chapitres si nécessaire).\n"
        "Respecte les dépendances conceptuelles. N'invente aucun concept.\n"
        "Réorganise pédagogiquement (pas une copie de l'ordre PDF).\n"
        "Schéma : "
        '{"language":"...","level":"intermediate","sheet_type":"complete","chapters":['
        '{"index":0,"title":"...","objectives":[{"text":"...","concept_ids":["..."]}],'
        '"concept_ids":["..."],"page_refs":[1],"needs_example":true,'
        '"needs_table":false,"needs_questions":false,"depth":"complete"}]}\n'
        f"Thèmes: {json.dumps(course.themes, ensure_ascii=False)}\n"
        f"Essentiels: {json.dumps(course.essential_concept_ids, ensure_ascii=False)}\n"
        f"Concepts: {json.dumps(concepts_json, ensure_ascii=False)}\n"
        f"Relations: {json.dumps(relations_json, ensure_ascii=False)}"
    )


def _load_plan_payload(raw: str) -> dict[str, Any]:
    try:
        return load_json_object(raw)
    except ValueError as error:
        raise ValueError("JSON de planification invalide.") from error


def _plan_from_payload(
    payload: dict[str, Any],
    *,
    course: CourseModel,
    language: str,
) -> PedagogicalPlan:
    data = dict(payload)
    data["language"] = language
    data["level"] = LEGACY_STORE_LEVEL
    data["sheet_type"] = LEGACY_STORE_SHEET_TYPE
    chapters = data.get("chapters")
    if isinstance(chapters, list):
        normalized = []
        for chapter in chapters:
            if not isinstance(chapter, dict):
                raise ValueError("JSON de planification invalide.")
            item = dict(chapter)
            item["needs_questions"] = False
            item["depth"] = "complete"
            normalized.append(item)
        data["chapters"] = normalized
    try:
        plan = PedagogicalPlan.model_validate(data)
    except ValidationError as error:
        raise ValueError("JSON de planification invalide.") from error

    _validate_plan_against_course(plan, course)
    return plan


def _validate_plan_against_course(plan: PedagogicalPlan, course: CourseModel) -> None:
    if len(plan.chapters) == 0:
        raise ValueError("JSON de planification invalide.")
    if len(plan.chapters) > MAX_CHAPTERS:
        raise ValueError("JSON de planification invalide.")

    known_ids = {c.id for c in course.concepts}
    depends = {c.id: list(c.depends_on) for c in course.concepts}
    introduced: set[str] = set()
    covered: set[str] = set()

    for chapter in plan.chapters:
        if chapter.depth != "complete":
            raise ValueError("JSON de planification invalide.")
        if chapter.needs_questions:
            raise ValueError("JSON de planification invalide.")
        ensure_chapter_density(len(chapter.concept_ids))
        for cid in chapter.concept_ids:
            if cid not in known_ids:
                raise ValueError("JSON de planification invalide.")
            if cid in covered:
                raise ValueError("JSON de planification invalide.")
            for dep in depends.get(cid, []):
                if dep in known_ids and dep not in introduced and dep not in chapter.concept_ids:
                    raise ValueError(
                        "Le plan pédagogique ne respecte pas les dépendances conceptuelles."
                    )
            covered.add(cid)
        introduced.update(chapter.concept_ids)

    if known_ids and covered != known_ids:
        raise ValueError(
            "Le plan pédagogique omet des concepts du cours."
        )


def _heuristic_plan(course: CourseModel, language: str) -> PedagogicalPlan:
    concepts = _select_all_concepts(course)
    if not concepts:
        raise ValueError("Impossible de construire un plan pédagogique fiable.")

    ordered = _topo_sort(concepts)
    # Refus explicite : pas de densification au-delà de MAX_CONCEPTS_PER_CHAPTER
    # (préserve tous les concepts en exigeant un découpage amont).
    ensure_course_concept_capacity(len(ordered))
    chunks = _chunk_all(ordered, MAX_CONCEPTS_PER_CHAPTER)
    # Garantie : len(ordered) ≤ MAX_COURSE_CONCEPTS ⇒ len(chunks) ≤ MAX_CHAPTERS.
    if len(chunks) > MAX_CHAPTERS:
        raise ValueError(
            "Impossible de construire un plan pédagogique fiable sans densification."
        )

    chapters: list[ChapterPlan] = []
    for index, group in enumerate(chunks):
        ensure_chapter_density(len(group))
        page_refs: list[int] = []
        for concept in group:
            for page in concept.pages:
                if page not in page_refs:
                    page_refs.append(page)
        chapters.append(
            ChapterPlan(
                index=index,
                title=_chapter_title(group, index),
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
                needs_table=len(group) > 1,
                needs_questions=False,
                depth="complete",
            )
        )
    return PedagogicalPlan(
        language=language,
        level=LEGACY_STORE_LEVEL,
        sheet_type=LEGACY_STORE_SHEET_TYPE,
        chapters=chapters,
    )


def _select_all_concepts(course: CourseModel) -> list[Concept]:
    """Tous les concepts du modèle, dépendances incluses (pas de filtre short)."""
    by_id = {c.id: c for c in course.concepts}
    if not by_id:
        return []
    expanded: list[str] = []
    pending = list(by_id.keys())
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
    return [by_id[cid] for cid in expanded if cid in by_id]


def _is_dependency_cycle_error(error: BaseException) -> bool:
    message = str(error).casefold()
    return "dépendances cycliques" in message or "cyclique" in message


def _cycle_error_message(concept_ids: list[str]) -> str:
    joined = ", ".join(concept_ids) if concept_ids else "(non identifiés)"
    return (
        "Le modèle du cours contient des dépendances cycliques entre concepts "
        f"({joined}). Corrigez le graphe avant planification."
    )


def _ensure_acyclic_dependencies(concepts: list[Concept]) -> None:
    """Détecte les cycles entre concepts connus ; lève ValueError explicite."""
    if not concepts:
        return
    by_id = {c.id: c for c in concepts}
    ids = set(by_id)
    remaining = set(ids)
    while remaining:
        ready = [
            cid
            for cid in remaining
            if all(dep not in remaining for dep in by_id[cid].depends_on if dep in ids)
        ]
        if not ready:
            raise ValueError(_cycle_error_message(sorted(remaining)))
        for cid in ready:
            remaining.remove(cid)


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
            raise ValueError(_cycle_error_message(sorted(remaining)))
        ready.sort()
        for cid in ready:
            ordered.append(by_id[cid])
            remaining.remove(cid)
    return ordered


def _chunk_all(concepts: list[Concept], max_per_chapter: int) -> list[list[Concept]]:
    if max_per_chapter < 1:
        raise ValueError("Impossible de construire un plan pédagogique fiable.")
    if not concepts:
        return []
    chunks: list[list[Concept]] = []
    for start in range(0, len(concepts), max_per_chapter):
        chunks.append(concepts[start : start + max_per_chapter])
    return chunks


def _chapter_title(group: list[Concept], index: int) -> str:
    if len(group) == 1:
        return group[0].label.capitalize()
    labels = ", ".join(c.label for c in group[:2])
    return f"Chapitre {index + 1} : {labels}"

