"""Tests de CourseUnderstanding (analyse multi-fenêtres + fusion)."""

from __future__ import annotations

import json

import pytest

from src.revision.engine.llm_guard import (
    LlmBudgetGuard,
    MAX_ANALYSIS_WINDOWS,
    TOKENS_UNDERSTANDING,
)
from src.revision.engine.models import CoursePage
from src.revision.engine.understanding import (
    CourseUnderstanding,
    _KIND_VALUES,
    _build_understanding_prompt,
    _concept_from_raw,
)


def _page(number: int, text: str, filename: str = "cours.pdf") -> CoursePage:
    return CoursePage(page_number=number, text=text, filename=filename)


class ScriptedLLM:
    """Renvoie des réponses JSON préparées ; compte les appels generate."""

    def __init__(self, replies: list[str]) -> None:
        self.replies = list(replies)
        self.calls = 0
        self.prompts: list[str] = []
        self.token_budgets: list[int] = []

    def generate(self, prompt: str, max_new_tokens: int = 512) -> str:
        self.calls += 1
        self.prompts.append(prompt)
        self.token_budgets.append(max_new_tokens)
        if not self.replies:
            raise AssertionError("Plus de réponses LLM scriptées.")
        return self.replies.pop(0)

    def is_loaded(self) -> bool:
        return False


def _window_payload(
    *,
    themes: list[str],
    concepts: list[dict],
    relations: list[dict] | None = None,
    contradictions: list[dict] | None = None,
) -> str:
    return json.dumps(
        {
            "themes": themes,
            "concepts": concepts,
            "relations": relations or [],
            "contradictions": contradictions or [],
        },
        ensure_ascii=False,
    )


def test_analyzes_multiple_windows_and_extracts_concepts_relations() -> None:
    llm = ScriptedLLM(
        [
            _window_payload(
                themes=["Variables"],
                concepts=[
                    {
                        "id": "c_var",
                        "label": "variable",
                        "kind": "definition",
                        "pages": [1],
                        "essential": True,
                        "depends_on": [],
                        "notes": "Une variable stocke une valeur.",
                    }
                ],
                relations=[],
            ),
            _window_payload(
                themes=["Fonctions"],
                concepts=[
                    {
                        "id": "c_fun",
                        "label": "fonction",
                        "kind": "method",
                        "pages": [2],
                        "essential": True,
                        "depends_on": ["c_var"],
                        "notes": "Une fonction encapsule un comportement.",
                    }
                ],
                relations=[
                    {
                        "source_id": "c_fun",
                        "target_id": "c_var",
                        "relation_type": "depends_on",
                        "pages": [2],
                    }
                ],
            ),
        ]
    )
    guard = LlmBudgetGuard()
    understanding = CourseUnderstanding(llm, guard)
    windows = [
        [_page(1, "Une variable stocke une valeur.")],
        [_page(2, "Une fonction encapsule un comportement.")],
    ]

    model = understanding.analyze(windows)

    assert "Variables" in model.themes
    assert "Fonctions" in model.themes
    assert {c.id for c in model.concepts} == {"c_var", "c_fun"}
    assert "c_var" in model.essential_concept_ids
    assert "c_fun" in model.essential_concept_ids
    assert len(model.relations) == 1
    assert model.relations[0].source_id == "c_fun"
    assert llm.calls == 2
    assert all(budget == TOKENS_UNDERSTANDING for budget in llm.token_budgets)
    assert guard.calls == 2
    assert guard.reserved_tokens == 2 * TOKENS_UNDERSTANDING


def test_merges_truly_identical_concepts_and_unions_pages() -> None:
    llm = ScriptedLLM(
        [
            _window_payload(
                themes=["Mémoire"],
                concepts=[
                    {
                        "id": "c_var",
                        "label": "variable",
                        "kind": "definition",
                        "pages": [1],
                        "essential": True,
                        "depends_on": [],
                        "notes": "Nomme une valeur en mémoire.",
                    }
                ],
            ),
            _window_payload(
                themes=["Mémoire"],
                concepts=[
                    {
                        "id": "c_var_alias",
                        "label": "Variable",
                        "kind": "definition",
                        "pages": [3],
                        "essential": True,
                        "depends_on": [],
                        "notes": "Nomme une valeur en mémoire.",
                        "same_as": "c_var",
                    }
                ],
            ),
        ]
    )
    model = CourseUnderstanding(llm, LlmBudgetGuard()).analyze(
        [[_page(1, "var p1")], [_page(3, "var p3")]]
    )

    assert len(model.concepts) == 1
    concept = model.concepts[0]
    assert concept.id == "c_var"
    assert concept.pages == [1, 3]
    assert "CONTRADICTION" not in concept.notes.upper()


def test_keeps_near_but_distinct_concepts() -> None:
    llm = ScriptedLLM(
        [
            _window_payload(
                themes=["Portée"],
                concepts=[
                    {
                        "id": "c_var",
                        "label": "variable",
                        "kind": "definition",
                        "pages": [1],
                        "essential": True,
                        "depends_on": [],
                        "notes": "Notion générale de variable.",
                    },
                    {
                        "id": "c_var_global",
                        "label": "variable globale",
                        "kind": "definition",
                        "pages": [1],
                        "essential": True,
                        "depends_on": ["c_var"],
                        "notes": "Variable visible dans tout le programme.",
                    },
                ],
            )
        ]
    )
    model = CourseUnderstanding(llm, LlmBudgetGuard()).analyze(
        [[_page(1, "variable et variable globale")]]
    )

    assert {c.id for c in model.concepts} == {"c_var", "c_var_global"}
    labels = {c.label.casefold() for c in model.concepts}
    assert "variable" in labels
    assert "variable globale" in labels


def test_marks_contradiction_without_inventing_resolution() -> None:
    llm = ScriptedLLM(
        [
            _window_payload(
                themes=["Signe"],
                concepts=[
                    {
                        "id": "c_x",
                        "label": "x",
                        "kind": "formula",
                        "pages": [1],
                        "essential": True,
                        "depends_on": [],
                        "notes": "x est toujours positif.",
                    }
                ],
            ),
            _window_payload(
                themes=["Signe"],
                concepts=[
                    {
                        "id": "c_x",
                        "label": "x",
                        "kind": "formula",
                        "pages": [2],
                        "essential": True,
                        "depends_on": [],
                        "notes": "x est toujours négatif.",
                    }
                ],
                contradictions=[
                    {
                        "concept_id": "c_x",
                        "description": "Les pages 1 et 2 affirment des signes opposés pour x.",
                        "pages": [1, 2],
                    }
                ],
            ),
        ]
    )
    model = CourseUnderstanding(llm, LlmBudgetGuard()).analyze(
        [[_page(1, "x > 0")], [_page(2, "x < 0")]]
    )

    concept = next(c for c in model.concepts if c.id == "c_x")
    assert concept.pages == [1, 2]
    assert "CONTRADICTION" in concept.notes.upper()
    assert "positif" in concept.notes or "négatif" in concept.notes
    # Pas de résolution inventée : les deux formulations restent visibles
    assert "positif" in concept.notes and "négatif" in concept.notes
    assert "donc" not in concept.notes.casefold()
    assert "en réalité" not in concept.notes.casefold()


def test_invalid_json_after_retries_raises_clean_error() -> None:
    llm = ScriptedLLM(["pas-json", "{toujours invalide"])
    guard = LlmBudgetGuard()
    understanding = CourseUnderstanding(llm, guard)

    with pytest.raises(ValueError, match="JSON"):
        understanding.analyze([[_page(1, "texte")]])

    assert llm.calls == 2  # 1 + MAX_RETRIES_PER_CALL
    assert guard.calls == 2
    assert guard.reserved_tokens == 2 * TOKENS_UNDERSTANDING


def test_respects_call_and_token_ceilings() -> None:
    # Prépare assez de fenêtres pour consommer le budget, puis une de trop.
    replies = [
        _window_payload(
            themes=[f"T{i}"],
            concepts=[
                {
                    "id": f"c{i}",
                    "label": f"n{i}",
                    "kind": "other",
                    "pages": [i + 1],
                    "essential": False,
                    "depends_on": [],
                    "notes": "",
                }
            ],
        )
        for i in range(MAX_ANALYSIS_WINDOWS)
    ]
    llm = ScriptedLLM(replies)
    guard = LlmBudgetGuard()
    # Saturation artificielle du plafond d'appels avant analyse.
    for _ in range(16):
        guard.reserve(1)

    understanding = CourseUnderstanding(llm, guard)
    with pytest.raises(ValueError, match="appels"):
        understanding.analyze([[_page(1, "x")]])


def test_too_many_windows_refused_without_truncation() -> None:
    llm = ScriptedLLM([])
    windows = [[_page(i, f"p{i}")] for i in range(1, MAX_ANALYSIS_WINDOWS + 2)]
    with pytest.raises(ValueError, match="moins de pages"):
        CourseUnderstanding(llm, LlmBudgetGuard()).analyze(windows)
    assert llm.calls == 0


def test_prompt_forbids_invented_kind_categories() -> None:
    prompt = _build_understanding_prompt([_page(1, "PaaS et IaaS")])
    assert "definition" in prompt
    assert "formula" in prompt
    assert "method" in prompt
    assert "mechanism" in prompt
    assert "relation" in prompt
    assert "other" in prompt
    assert 'kind="other"' in prompt or 'kind=\\"other\\"' in prompt or "kind=\"other\"" in prompt
    assert "PaaS" in prompt
    assert "IaaS" in prompt
    assert "SaaS" in prompt
    assert "service" in prompt.casefold()
    assert "technology" in prompt.casefold()
    assert "sans Markdown" in prompt
    assert "service" not in _KIND_VALUES


@pytest.mark.parametrize(
    ("label", "concept_id"),
    [("PaaS", "c_paas"), ("IaaS", "c_iaas")],
)
def test_cloud_service_notions_accepted_with_kind_other(label: str, concept_id: str) -> None:
    llm = ScriptedLLM(
        [
            _window_payload(
                themes=["Cloud"],
                concepts=[
                    {
                        "id": concept_id,
                        "label": label,
                        "kind": "other",
                        "pages": [1],
                        "essential": True,
                        "depends_on": [],
                        "notes": f"{label} est un modèle de service cloud.",
                    }
                ],
            )
        ]
    )
    model = CourseUnderstanding(llm, LlmBudgetGuard()).analyze([[_page(1, label)]])
    concept = next(c for c in model.concepts if c.id == concept_id)
    assert concept.kind == "other"
    assert concept.label == label


@pytest.mark.parametrize("bad_kind", ["service", "technology"])
def test_invented_kind_is_rejected_with_diagnostic(bad_kind: str) -> None:
    with pytest.raises(ValueError, match=rf"kind invalide.*c_bad.*{bad_kind}"):
        _concept_from_raw(
            {
                "id": "c_bad",
                "label": "PaaS",
                "kind": bad_kind,
                "pages": [1],
                "essential": True,
                "depends_on": [],
                "notes": "",
            }
        )


@pytest.mark.parametrize("kind", sorted(_KIND_VALUES))
def test_all_allowed_kinds_are_accepted(kind: str) -> None:
    concept = _concept_from_raw(
        {
            "id": f"c_{kind}",
            "label": kind,
            "kind": kind,
            "pages": [1],
            "essential": False,
            "depends_on": [],
            "notes": "",
        }
    )
    assert concept.kind == kind


def test_invalid_kind_error_lists_allowed_values() -> None:
    with pytest.raises(ValueError) as caught:
        _concept_from_raw(
            {
                "id": "c_tool",
                "label": "outil",
                "kind": "tool",
                "pages": [1],
            }
        )
    message = str(caught.value)
    assert "c_tool" in message
    assert "tool" in message
    for allowed in _KIND_VALUES:
        assert allowed in message
