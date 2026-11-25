"""Tests du PedagogicalPlanner Smart Revision V2."""

from __future__ import annotations

import json

import pytest

from src.revision.engine.llm_guard import (
    LlmBudgetGuard,
    MAX_CHAPTERS_COMPLETE,
    MAX_CHAPTERS_SHORT,
    TOKENS_PLAN,
)
from src.revision.engine.models import Concept, CourseModel, CourseRelation
from src.revision.engine.planner import PedagogicalPlanner


class ScriptedLLM:
    def __init__(self, replies: list[str]) -> None:
        self.replies = list(replies)
        self.calls = 0
        self.token_budgets: list[int] = []

    def generate(self, prompt: str, max_new_tokens: int = 512) -> str:
        self.calls += 1
        self.token_budgets.append(max_new_tokens)
        if not self.replies:
            raise AssertionError("Plus de réponses LLM scriptées.")
        return self.replies.pop(0)

    def is_loaded(self) -> bool:
        return False


def _course_model() -> CourseModel:
    return CourseModel(
        themes=["Programmation"],
        concepts=[
            Concept(
                id="c_var",
                label="variable",
                kind="definition",
                pages=[1],
                essential=True,
                depends_on=[],
                notes="stocke une valeur",
            ),
            Concept(
                id="c_fun",
                label="fonction",
                kind="method",
                pages=[2],
                essential=True,
                depends_on=["c_var"],
                notes="encapsule un comportement",
            ),
            Concept(
                id="c_rec",
                label="récursivité",
                kind="mechanism",
                pages=[3],
                essential=True,
                depends_on=["c_fun"],
                notes="fonction qui s'appelle",
            ),
            Concept(
                id="c_tip",
                label="astuce locale",
                kind="other",
                pages=[3],
                essential=False,
                depends_on=[],
                notes="détail secondaire",
            ),
        ],
        relations=[
            CourseRelation(
                source_id="c_fun",
                target_id="c_var",
                relation_type="depends_on",
                pages=[2],
            ),
            CourseRelation(
                source_id="c_rec",
                target_id="c_fun",
                relation_type="depends_on",
                pages=[3],
            ),
        ],
        essential_concept_ids=["c_var", "c_fun", "c_rec"],
    )


def _plan_payload(chapters: list[dict], sheet_type: str = "short") -> str:
    return json.dumps(
        {
            "language": "fr",
            "level": "beginner",
            "sheet_type": sheet_type,
            "chapters": chapters,
        },
        ensure_ascii=False,
    )


def test_short_plan_respects_chapter_cap_and_essentials() -> None:
    llm = ScriptedLLM(
        [
            _plan_payload(
                [
                    {
                        "index": 0,
                        "title": "Les variables",
                        "objectives": [
                            {
                                "text": "Comprendre une variable",
                                "concept_ids": ["c_var"],
                            }
                        ],
                        "concept_ids": ["c_var"],
                        "page_refs": [1],
                        "needs_example": True,
                        "needs_table": False,
                        "needs_questions": False,
                        "depth": "short",
                    },
                    {
                        "index": 1,
                        "title": "Les fonctions",
                        "objectives": [
                            {
                                "text": "Utiliser une fonction",
                                "concept_ids": ["c_fun"],
                            }
                        ],
                        "concept_ids": ["c_fun", "c_rec"],
                        "page_refs": [2, 3],
                        "needs_example": True,
                        "needs_table": False,
                        "needs_questions": False,
                        "depth": "short",
                    },
                ],
                sheet_type="short",
            )
        ]
    )
    guard = LlmBudgetGuard()
    plan = PedagogicalPlanner(llm, guard).plan(
        _course_model(),
        sheet_type="short",
        level="beginner",
        language="fr",
    )

    assert plan.sheet_type == "short"
    assert len(plan.chapters) <= MAX_CHAPTERS_SHORT
    assert all(ch.depth == "short" for ch in plan.chapters)
    assert all(not ch.needs_questions for ch in plan.chapters)
    covered = {cid for ch in plan.chapters for cid in ch.concept_ids}
    assert {"c_var", "c_fun", "c_rec"} <= covered
    assert llm.calls == 1
    assert llm.token_budgets == [TOKENS_PLAN]
    assert guard.calls == 1
    assert guard.reserved_tokens == TOKENS_PLAN


def test_complete_plan_is_deeper_than_short() -> None:
    llm = ScriptedLLM(
        [
            _plan_payload(
                [
                    {
                        "index": 0,
                        "title": "Fondations : variables",
                        "objectives": [
                            {"text": "Maîtriser les variables", "concept_ids": ["c_var"]}
                        ],
                        "concept_ids": ["c_var"],
                        "page_refs": [1],
                        "needs_example": True,
                        "needs_table": False,
                        "needs_questions": True,
                        "depth": "complete",
                    },
                    {
                        "index": 1,
                        "title": "Fonctions et dépendances",
                        "objectives": [
                            {
                                "text": "Relier fonctions et variables",
                                "concept_ids": ["c_fun", "c_var"],
                            }
                        ],
                        "concept_ids": ["c_fun"],
                        "page_refs": [2],
                        "needs_example": True,
                        "needs_table": True,
                        "needs_questions": True,
                        "depth": "complete",
                    },
                    {
                        "index": 2,
                        "title": "Récursivité",
                        "objectives": [
                            {
                                "text": "Expliquer la récursivité",
                                "concept_ids": ["c_rec"],
                            }
                        ],
                        "concept_ids": ["c_rec"],
                        "page_refs": [3],
                        "needs_example": True,
                        "needs_table": False,
                        "needs_questions": True,
                        "depth": "complete",
                    },
                ],
                sheet_type="complete",
            )
        ]
    )
    plan = PedagogicalPlanner(ScriptedLLM(llm.replies), LlmBudgetGuard()).plan(
        _course_model(),
        sheet_type="complete",
        level="advanced",
        language="fr",
    )

    assert plan.sheet_type == "complete"
    assert len(plan.chapters) <= MAX_CHAPTERS_COMPLETE
    assert all(ch.depth == "complete" for ch in plan.chapters)
    assert any(ch.needs_questions for ch in plan.chapters)
    assert any(ch.needs_table for ch in plan.chapters)
    assert len(plan.chapters) >= 2


def test_plan_respects_concept_dependencies_order() -> None:
    llm = ScriptedLLM(
        [
            _plan_payload(
                [
                    {
                        "index": 0,
                        "title": "Variables d'abord",
                        "objectives": [],
                        "concept_ids": ["c_var"],
                        "page_refs": [1],
                        "needs_example": False,
                        "needs_table": False,
                        "needs_questions": False,
                        "depth": "short",
                    },
                    {
                        "index": 1,
                        "title": "Puis fonctions",
                        "objectives": [],
                        "concept_ids": ["c_fun"],
                        "page_refs": [2],
                        "needs_example": False,
                        "needs_table": False,
                        "needs_questions": False,
                        "depth": "short",
                    },
                    {
                        "index": 2,
                        "title": "Puis récursivité",
                        "objectives": [],
                        "concept_ids": ["c_rec"],
                        "page_refs": [3],
                        "needs_example": False,
                        "needs_table": False,
                        "needs_questions": False,
                        "depth": "short",
                    },
                ]
            )
        ]
    )
    plan = PedagogicalPlanner(llm, LlmBudgetGuard()).plan(
        _course_model(),
        sheet_type="short",
        level="beginner",
        language="fr",
    )
    positions = {}
    for chapter in plan.chapters:
        for cid in chapter.concept_ids:
            positions.setdefault(cid, chapter.index)
    assert positions["c_var"] <= positions["c_fun"] <= positions["c_rec"]


def test_invalid_dependency_order_is_rejected() -> None:
    llm = ScriptedLLM(
        [
            _plan_payload(
                [
                    {
                        "index": 0,
                        "title": "Récursivité trop tôt",
                        "objectives": [],
                        "concept_ids": ["c_rec"],
                        "page_refs": [3],
                        "needs_example": False,
                        "needs_table": False,
                        "needs_questions": False,
                        "depth": "short",
                    },
                    {
                        "index": 1,
                        "title": "Variables après",
                        "objectives": [],
                        "concept_ids": ["c_var"],
                        "page_refs": [1],
                        "needs_example": False,
                        "needs_table": False,
                        "needs_questions": False,
                        "depth": "short",
                    },
                ]
            )
        ]
    )
    with pytest.raises(ValueError, match="dépendances"):
        PedagogicalPlanner(llm, LlmBudgetGuard()).plan(
            _course_model(),
            sheet_type="short",
            level="beginner",
            language="fr",
        )


def test_too_many_chapters_raises_without_silent_truncation() -> None:
    chapters = [
        {
            "index": i,
            "title": f"Chapitre {i}",
            "objectives": [],
            "concept_ids": ["c_var"] if i == 0 else ["c_fun"] if i == 1 else ["c_rec"],
            "page_refs": [1],
            "needs_example": False,
            "needs_table": False,
            "needs_questions": False,
            "depth": "short",
        }
        for i in range(MAX_CHAPTERS_SHORT + 1)
    ]
    llm = ScriptedLLM([_plan_payload(chapters, sheet_type="short")])
    with pytest.raises(ValueError, match="moins de pages"):
        PedagogicalPlanner(llm, LlmBudgetGuard()).plan(
            _course_model(),
            sheet_type="short",
            level="beginner",
            language="fr",
        )


def test_heuristic_fallback_after_invalid_json() -> None:
    llm = ScriptedLLM(["not-json", "still-bad"])
    guard = LlmBudgetGuard()
    plan = PedagogicalPlanner(llm, guard).plan(
        _course_model(),
        sheet_type="short",
        level="beginner",
        language="fr",
    )

    assert plan.sheet_type == "short"
    assert 1 <= len(plan.chapters) <= MAX_CHAPTERS_SHORT
    covered = {cid for ch in plan.chapters for cid in ch.concept_ids}
    assert {"c_var", "c_fun", "c_rec"} <= covered
    # Retry consommé (2) ; pas d'appel LLM supplémentaire pour le fallback.
    assert llm.calls == 2
    assert guard.calls == 2
    positions = {}
    for chapter in plan.chapters:
        for cid in chapter.concept_ids:
            positions.setdefault(cid, chapter.index)
    assert positions["c_var"] <= positions["c_fun"] <= positions["c_rec"]


def test_fallback_impossible_on_empty_model_reraises() -> None:
    llm = ScriptedLLM(["{}", "encore non"])
    empty = CourseModel()
    with pytest.raises(ValueError, match="JSON|plan"):
        PedagogicalPlanner(llm, LlmBudgetGuard()).plan(
            empty,
            sheet_type="short",
            level="beginner",
            language="fr",
        )


def test_short_and_complete_differ_for_same_course() -> None:
    short_json = _plan_payload(
        [
            {
                "index": 0,
                "title": "Essentiels",
                "objectives": [],
                "concept_ids": ["c_var", "c_fun", "c_rec"],
                "page_refs": [1, 2, 3],
                "needs_example": True,
                "needs_table": False,
                "needs_questions": False,
                "depth": "short",
            }
        ],
        sheet_type="short",
    )
    complete_json = _plan_payload(
        [
            {
                "index": 0,
                "title": "Variables",
                "objectives": [],
                "concept_ids": ["c_var"],
                "page_refs": [1],
                "needs_example": True,
                "needs_table": False,
                "needs_questions": True,
                "depth": "complete",
            },
            {
                "index": 1,
                "title": "Fonctions",
                "objectives": [],
                "concept_ids": ["c_fun"],
                "page_refs": [2],
                "needs_example": True,
                "needs_table": True,
                "needs_questions": True,
                "depth": "complete",
            },
            {
                "index": 2,
                "title": "Récursivité",
                "objectives": [],
                "concept_ids": ["c_rec"],
                "page_refs": [3],
                "needs_example": True,
                "needs_table": False,
                "needs_questions": True,
                "depth": "complete",
            },
        ],
        sheet_type="complete",
    )
    short_plan = PedagogicalPlanner(ScriptedLLM([short_json]), LlmBudgetGuard()).plan(
        _course_model(), sheet_type="short", level="beginner", language="fr"
    )
    complete_plan = PedagogicalPlanner(
        ScriptedLLM([complete_json]), LlmBudgetGuard()
    ).plan(_course_model(), sheet_type="complete", level="beginner", language="fr")

    assert short_plan.sheet_type != complete_plan.sheet_type
    assert len(complete_plan.chapters) >= len(short_plan.chapters)
    assert any(ch.needs_questions for ch in complete_plan.chapters)
    assert all(not ch.needs_questions for ch in short_plan.chapters)
