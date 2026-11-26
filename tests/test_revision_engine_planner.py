"""Tests du PedagogicalPlanner Smart Revision V2 (fiche unique, sans questions)."""

from __future__ import annotations

import json

import pytest

from src.revision.engine.llm_guard import (
    LEGACY_STORE_LEVEL,
    LEGACY_STORE_SHEET_TYPE,
    LlmBudgetGuard,
    MAX_CHAPTERS,
    TOKENS_PLAN,
)
from src.revision.engine.models import Concept, CourseModel, CourseRelation
from src.revision.engine.planner import (
    PedagogicalPlanner,
    _heuristic_plan,
    _topo_sort,
)


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


def _plan_payload(chapters: list[dict]) -> str:
    return json.dumps(
        {
            "language": "fr",
            "level": "beginner",
            "sheet_type": "short",
            "chapters": chapters,
        },
        ensure_ascii=False,
    )


def _full_coverage_chapters(*, needs_questions: bool = False, depth: str = "complete") -> list[dict]:
    return [
        {
            "index": 0,
            "title": "Fondations : variables",
            "objectives": [
                {"text": "Maîtriser les variables", "concept_ids": ["c_var"]}
            ],
            "concept_ids": ["c_var", "c_tip"],
            "page_refs": [1, 3],
            "needs_example": True,
            "needs_table": False,
            "needs_questions": needs_questions,
            "depth": depth,
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
            "needs_questions": needs_questions,
            "depth": depth,
        },
        {
            "index": 2,
            "title": "Récursivité",
            "objectives": [
                {"text": "Expliquer la récursivité", "concept_ids": ["c_rec"]}
            ],
            "concept_ids": ["c_rec"],
            "page_refs": [3],
            "needs_example": True,
            "needs_table": False,
            "needs_questions": needs_questions,
            "depth": depth,
        },
    ]


def test_plan_covers_all_concepts_and_forces_legacy_store_fields() -> None:
    llm = ScriptedLLM([_plan_payload(_full_coverage_chapters(needs_questions=True, depth="short"))])
    guard = LlmBudgetGuard()
    plan = PedagogicalPlanner(llm, guard).plan(
        _course_model(),
        sheet_type="short",
        level="beginner",
        language="fr",
    )

    assert plan.sheet_type == LEGACY_STORE_SHEET_TYPE
    assert plan.level == LEGACY_STORE_LEVEL
    assert len(plan.chapters) <= MAX_CHAPTERS
    assert all(ch.depth == "complete" for ch in plan.chapters)
    assert all(not ch.needs_questions for ch in plan.chapters)
    covered = {cid for ch in plan.chapters for cid in ch.concept_ids}
    assert covered == {"c_var", "c_fun", "c_rec", "c_tip"}
    assert llm.calls == 1
    assert llm.token_budgets == [TOKENS_PLAN]
    assert guard.calls == 1
    assert guard.reserved_tokens == TOKENS_PLAN


def test_plan_ignores_requested_sheet_type_and_level() -> None:
    llm = ScriptedLLM([_plan_payload(_full_coverage_chapters())])
    plan = PedagogicalPlanner(llm, LlmBudgetGuard()).plan(
        _course_model(),
        sheet_type="short",
        level="advanced",
        language="fr",
    )
    assert plan.sheet_type == LEGACY_STORE_SHEET_TYPE
    assert plan.level == LEGACY_STORE_LEVEL
    assert any(ch.needs_table for ch in plan.chapters)


def test_plan_respects_concept_dependencies_order() -> None:
    llm = ScriptedLLM(
        [
            _plan_payload(
                [
                    {
                        "index": 0,
                        "title": "Variables d'abord",
                        "objectives": [],
                        "concept_ids": ["c_var", "c_tip"],
                        "page_refs": [1],
                        "needs_example": False,
                        "needs_table": False,
                        "needs_questions": False,
                        "depth": "complete",
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
                        "depth": "complete",
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
                        "depth": "complete",
                    },
                ]
            )
        ]
    )
    plan = PedagogicalPlanner(llm, LlmBudgetGuard()).plan(
        _course_model(),
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
                        "depth": "complete",
                    },
                    {
                        "index": 1,
                        "title": "Variables après",
                        "objectives": [],
                        "concept_ids": ["c_var", "c_fun", "c_tip"],
                        "page_refs": [1],
                        "needs_example": False,
                        "needs_table": False,
                        "needs_questions": False,
                        "depth": "complete",
                    },
                ]
            )
        ]
    )
    with pytest.raises(ValueError, match="dépendances"):
        PedagogicalPlanner(llm, LlmBudgetGuard()).plan(_course_model(), language="fr")


def test_omitted_concepts_are_rejected() -> None:
    llm = ScriptedLLM(
        [
            _plan_payload(
                [
                    {
                        "index": 0,
                        "title": "Essentiels seulement",
                        "objectives": [],
                        "concept_ids": ["c_var", "c_fun", "c_rec"],
                        "page_refs": [1, 2, 3],
                        "needs_example": True,
                        "needs_table": False,
                        "needs_questions": False,
                        "depth": "complete",
                    }
                ]
            )
        ]
    )
    with pytest.raises(ValueError, match="concepts"):
        PedagogicalPlanner(llm, LlmBudgetGuard()).plan(_course_model(), language="fr")


def test_too_many_chapters_falls_back_to_heuristic_without_losing_concepts() -> None:
    chapters = [
        {
            "index": i,
            "title": f"Chapitre {i}",
            "objectives": [],
            "concept_ids": ["c_var"] if i == 0 else ["c_fun"] if i == 1 else ["c_rec"] if i == 2 else ["c_tip"],
            "page_refs": [1],
            "needs_example": False,
            "needs_table": False,
            "needs_questions": False,
            "depth": "complete",
        }
        for i in range(MAX_CHAPTERS + 1)
    ]
    # Duplique les ids → invalide ; le planner bascule sur l'heuristique.
    llm = ScriptedLLM([_plan_payload(chapters)])
    plan = PedagogicalPlanner(llm, LlmBudgetGuard()).plan(_course_model(), language="fr")
    covered = {cid for ch in plan.chapters for cid in ch.concept_ids}
    assert covered == {"c_var", "c_fun", "c_rec", "c_tip"}
    assert 1 <= len(plan.chapters) <= MAX_CHAPTERS
    assert plan.sheet_type == LEGACY_STORE_SHEET_TYPE


def test_heuristic_fallback_after_invalid_json() -> None:
    llm = ScriptedLLM(["not-json", "still-bad"])
    guard = LlmBudgetGuard()
    plan = PedagogicalPlanner(llm, guard).plan(_course_model(), language="fr")

    assert plan.sheet_type == LEGACY_STORE_SHEET_TYPE
    assert plan.level == LEGACY_STORE_LEVEL
    assert 1 <= len(plan.chapters) <= MAX_CHAPTERS
    covered = {cid for ch in plan.chapters for cid in ch.concept_ids}
    assert covered == {"c_var", "c_fun", "c_rec", "c_tip"}
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
        PedagogicalPlanner(llm, LlmBudgetGuard()).plan(empty, language="fr")


def test_prompt_forbids_questions_and_requires_full_coverage() -> None:
    llm = ScriptedLLM([_plan_payload(_full_coverage_chapters())])
    PedagogicalPlanner(llm, LlmBudgetGuard()).plan(_course_model(), language="fr")
    # ScriptedLLM ne stocke pas le prompt ; on reconstruit via un LLM enregistreur.
    class CapturingLLM(ScriptedLLM):
        def __init__(self) -> None:
            super().__init__([_plan_payload(_full_coverage_chapters())])
            self.prompts: list[str] = []

        def generate(self, prompt: str, max_new_tokens: int = 512) -> str:
            self.prompts.append(prompt)
            return super().generate(prompt, max_new_tokens=max_new_tokens)

    capturing = CapturingLLM()
    PedagogicalPlanner(capturing, LlmBudgetGuard()).plan(_course_model(), language="fr")
    prompt = capturing.prompts[0]
    assert "needs_questions=true" in prompt or "Interdit : questions" in prompt
    assert "TOUS les concepts" in prompt or "tous les concepts" in prompt.casefold()


def _cyclic_course_ab() -> CourseModel:
    return CourseModel(
        themes=["Cycle"],
        concepts=[
            Concept(
                id="c_a",
                label="A",
                kind="other",
                pages=[1],
                essential=True,
                depends_on=["c_b"],
                notes="dépend de B",
            ),
            Concept(
                id="c_b",
                label="B",
                kind="other",
                pages=[1],
                essential=True,
                depends_on=["c_a"],
                notes="dépend de A",
            ),
        ],
        essential_concept_ids=["c_a", "c_b"],
    )


def _cyclic_course_three() -> CourseModel:
    return CourseModel(
        themes=["Cycle"],
        concepts=[
            Concept(
                id="c_a",
                label="A",
                kind="other",
                pages=[1],
                essential=True,
                depends_on=["c_c"],
                notes="A",
            ),
            Concept(
                id="c_b",
                label="B",
                kind="other",
                pages=[1],
                essential=True,
                depends_on=["c_a"],
                notes="B",
            ),
            Concept(
                id="c_c",
                label="C",
                kind="other",
                pages=[1],
                essential=True,
                depends_on=["c_b"],
                notes="C",
            ),
        ],
        essential_concept_ids=["c_a", "c_b", "c_c"],
    )


def test_simple_cycle_a_b_a_is_rejected() -> None:
    with pytest.raises(ValueError, match="dépendances cycliques"):
        PedagogicalPlanner(ScriptedLLM(["{}"]), LlmBudgetGuard()).plan(
            _cyclic_course_ab(), language="fr"
        )


def test_three_concept_cycle_is_rejected() -> None:
    with pytest.raises(ValueError, match="dépendances cycliques"):
        PedagogicalPlanner(ScriptedLLM(["{}"]), LlmBudgetGuard()).plan(
            _cyclic_course_three(), language="fr"
        )


def test_cyclic_graph_does_not_call_llm() -> None:
    llm = ScriptedLLM([_plan_payload(_full_coverage_chapters())])
    with pytest.raises(ValueError, match="dépendances cycliques"):
        PedagogicalPlanner(llm, LlmBudgetGuard()).plan(
            _cyclic_course_ab(), language="fr"
        )
    assert llm.calls == 0


def test_valid_dependency_graph_keeps_topo_order_and_coverage() -> None:
    llm = ScriptedLLM([_plan_payload(_full_coverage_chapters())])
    plan = PedagogicalPlanner(llm, LlmBudgetGuard()).plan(
        _course_model(), language="fr"
    )
    covered = {cid for ch in plan.chapters for cid in ch.concept_ids}
    assert covered == {"c_var", "c_fun", "c_rec", "c_tip"}
    positions = {}
    for chapter in plan.chapters:
        for cid in chapter.concept_ids:
            positions.setdefault(cid, chapter.index)
    assert positions["c_var"] <= positions["c_fun"] <= positions["c_rec"]

    ordered = _topo_sort(list(_course_model().concepts))
    ids = [c.id for c in ordered]
    assert ids.index("c_var") < ids.index("c_fun") < ids.index("c_rec")
    assert set(ids) == {"c_var", "c_fun", "c_rec", "c_tip"}


def test_heuristic_fallback_on_cyclic_graph_raises_without_masking() -> None:
    course = _cyclic_course_ab()
    with pytest.raises(ValueError, match="dépendances cycliques"):
        _heuristic_plan(course, "fr")

    planner = PedagogicalPlanner(ScriptedLLM(["not-json"]), LlmBudgetGuard())
    unrelated = ValueError("JSON de planification invalide.")
    with pytest.raises(ValueError, match="dépendances cycliques") as caught:
        planner._fallback_or_raise(course, "fr", unrelated)
    assert "JSON" not in str(caught.value)
