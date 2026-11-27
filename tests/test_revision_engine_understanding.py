"""Tests de CourseUnderstanding (analyse multi-fenêtres + fusion)."""

from __future__ import annotations

import json

import pytest

from src.revision.engine.llm_guard import (
    LlmBudgetGuard,
    MAX_ANALYSIS_WINDOWS,
    MAX_LLM_CALLS,
    MAX_TOTAL_NEW_TOKENS,
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


def test_merges_complementary_notes_across_windows_without_false_contradiction() -> None:
    """Deux explications complémentaires du même concept → notes fusionnées, pages unies."""
    llm = ScriptedLLM(
        [
            _window_payload(
                themes=["Biologie"],
                concepts=[
                    {
                        "id": "c_photo",
                        "label": "photosynthèse",
                        "kind": "definition",
                        "pages": [1],
                        "essential": True,
                        "depends_on": [],
                        "notes": "Processus par lequel les plantes convertissent la lumière en énergie chimique.",
                    }
                ],
            ),
            _window_payload(
                themes=["Biologie"],
                concepts=[
                    {
                        "id": "c_photo_alias",
                        "label": "photosynthèse",
                        "kind": "definition",
                        "pages": [4, 5],
                        "essential": True,
                        "depends_on": [],
                        "notes": (
                            "Étapes principales : captation de la lumière, fixation du CO2, "
                            "production de glucose. Aussi appelée assimilation chlorophyllienne."
                        ),
                        "same_as": "c_photo",
                    }
                ],
            ),
        ]
    )
    model = CourseUnderstanding(llm, LlmBudgetGuard()).analyze(
        [[_page(1, "intro photo")], [_page(4, "etapes"), _page(5, "alias")]]
    )
    assert len(model.concepts) == 1
    concept = model.concepts[0]
    assert concept.id == "c_photo"
    assert concept.pages == [1, 4, 5]
    assert "CONTRADICTION" not in concept.notes.upper()
    assert "lumière" in concept.notes.casefold() or "lumiere" in concept.notes.casefold()
    assert "glucose" in concept.notes.casefold()
    assert "assimilation chlorophyllienne" in concept.notes.casefold()


def test_empty_then_rich_notes_preserve_multi_window_content() -> None:
    """Fenêtre titre (notes vides) puis fenêtre définition → conserver le fond pédagogique."""
    llm = ScriptedLLM(
        [
            _window_payload(
                themes=["Droit"],
                concepts=[
                    {
                        "id": "habeas-corpus",
                        "label": "Habeas Corpus",
                        "kind": "definition",
                        "pages": [1],
                        "essential": True,
                        "depends_on": [],
                        "notes": "",
                    }
                ],
            ),
            _window_payload(
                themes=["Droit"],
                concepts=[
                    {
                        "id": "habeas-corpus",
                        "label": "Habeas Corpus",
                        "kind": "definition",
                        "pages": [8, 9],
                        "essential": True,
                        "depends_on": [],
                        "notes": (
                            "Garantie procédurale contre la détention arbitraire. "
                            "Aussi appelée writ of habeas corpus."
                        ),
                    }
                ],
            ),
        ]
    )
    model = CourseUnderstanding(llm, LlmBudgetGuard()).analyze(
        [[_page(1, "Sommaire Habeas Corpus")], [_page(8, "def"), _page(9, "suite")]]
    )
    concept = next(c for c in model.concepts if c.id == "habeas-corpus")
    assert concept.pages == [1, 8, 9]
    assert "détention" in concept.notes.casefold() or "detention" in concept.notes.casefold()
    assert "CONTRADICTION" not in concept.notes.upper()


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


def test_respects_call_ceiling() -> None:
    llm = ScriptedLLM(
        [
            _window_payload(
                themes=["T0"],
                concepts=[
                    {
                        "id": "c0",
                        "label": "n0",
                        "kind": "other",
                        "pages": [1],
                        "essential": False,
                        "depends_on": [],
                        "notes": "",
                    }
                ],
            )
        ]
    )
    guard = LlmBudgetGuard()
    for _ in range(MAX_LLM_CALLS):
        guard.reserve(1)

    understanding = CourseUnderstanding(llm, guard)
    with pytest.raises(ValueError, match="appels"):
        understanding.analyze([[_page(1, "x")]])
    assert llm.calls == 0
    assert guard.calls == MAX_LLM_CALLS
    assert guard.reserved_tokens == MAX_LLM_CALLS


def test_respects_token_ceiling() -> None:
    llm = ScriptedLLM(
        [
            _window_payload(
                themes=["Tokens"],
                concepts=[
                    {
                        "id": "c_tok",
                        "label": "notion",
                        "kind": "other",
                        "pages": [1],
                        "essential": True,
                        "depends_on": [],
                        "notes": "note",
                    }
                ],
            )
        ]
    )
    guard = LlmBudgetGuard()
    # 1 fenêtre ⇒ besoin garanti = 2 * TOKENS_UNDERSTANDING ; laisser 1 de moins.
    tokens_to_leave = 2 * TOKENS_UNDERSTANDING - 1
    calls_to_leave = 10  # largement au-dessus du besoin d'appels (2)
    tokens_to_burn = MAX_TOTAL_NEW_TOKENS - tokens_to_leave
    calls_to_burn = MAX_LLM_CALLS - calls_to_leave
    base = tokens_to_burn // calls_to_burn
    rem = tokens_to_burn % calls_to_burn
    for index in range(calls_to_burn):
        guard.reserve(base + (1 if index < rem else 0))

    assert guard.remaining_tokens == tokens_to_leave
    assert guard.remaining_calls == calls_to_leave
    calls_before = guard.calls
    tokens_before = guard.reserved_tokens

    with pytest.raises(ValueError, match="tokens|plafond de tokens"):
        CourseUnderstanding(llm, guard).analyze([[_page(1, "texte")]])

    assert llm.calls == 0
    assert guard.calls == calls_before
    assert guard.reserved_tokens == tokens_before


def test_too_many_windows_refused_without_truncation() -> None:
    llm = ScriptedLLM([])
    windows = [[_page(i, f"p{i}")] for i in range(1, MAX_ANALYSIS_WINDOWS + 2)]
    with pytest.raises(ValueError, match="capacité d'analyse|fenêtres"):
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
    assert "expliquent vraiment" in prompt.casefold() or "explique" in prompt.casefold()
    assert "synonymes" in prompt.casefold() or "formulations alternatives" in prompt.casefold()
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


@pytest.mark.parametrize(
    ("raw_kind", "expected"),
    [
        ("service", "other"),
        ("Service", "other"),
        ("technology", "other"),
        ("tool", "other"),
        ("platform", "other"),
        ("process", "method"),
        ("equation", "formula"),
        ("link", "relation"),
    ],
)
def test_kind_aliases_are_normalized(raw_kind: str, expected: str) -> None:
    concept = _concept_from_raw(
        {
            "id": "c_alias",
            "label": "PaaS",
            "kind": raw_kind,
            "pages": [1],
            "essential": True,
            "depends_on": [],
            "notes": "Modèle cloud.",
        }
    )
    assert concept.kind == expected
    assert "Catégorie LLM d'origine :" in concept.notes
    assert raw_kind in concept.notes or raw_kind.casefold() in concept.notes.casefold()
    assert "Modèle cloud." in concept.notes


def test_paas_with_kind_service_is_accepted_as_other() -> None:
    llm = ScriptedLLM(
        [
            _window_payload(
                themes=["Cloud"],
                concepts=[
                    {
                        "id": "c_paas",
                        "label": "PaaS",
                        "kind": "service",
                        "pages": [1],
                        "essential": True,
                        "depends_on": [],
                        "notes": "Platform as a Service.",
                    }
                ],
            )
        ]
    )
    model = CourseUnderstanding(llm, LlmBudgetGuard()).analyze([[_page(1, "PaaS")]])
    concept = next(c for c in model.concepts if c.id == "c_paas")
    assert concept.kind == "other"
    assert concept.label == "PaaS"
    assert "service" in concept.notes.casefold()


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
    assert concept.notes == ""


def test_invalid_kind_error_lists_allowed_values() -> None:
    with pytest.raises(ValueError) as caught:
        _concept_from_raw(
            {
                "id": "c_weird",
                "label": "outil",
                "kind": "banana",
                "pages": [1],
            }
        )
    message = str(caught.value)
    assert "c_weird" in message
    assert "banana" in message
    for allowed in _KIND_VALUES:
        assert allowed in message


def _assert_no_orphan_ids(model) -> None:
    known = {c.id for c in model.concepts}
    for concept in model.concepts:
        for dep in concept.depends_on:
            assert dep in known
            assert dep != concept.id
    for relation in model.relations:
        assert relation.source_id in known
        assert relation.target_id in known


def test_same_as_towards_already_known_concept_merges_pages_and_aliases() -> None:
    llm = ScriptedLLM(
        [
            _window_payload(
                themes=["Cloud"],
                concepts=[
                    {
                        "id": "c_iaas",
                        "label": "IaaS",
                        "kind": "other",
                        "pages": [1],
                        "essential": True,
                        "depends_on": [],
                        "notes": "Infrastructure as a Service.",
                    }
                ],
            ),
            _window_payload(
                themes=["Cloud"],
                concepts=[
                    {
                        "id": "c_iaas_local",
                        "label": "IaaS",
                        "kind": "other",
                        "pages": [2],
                        "essential": False,
                        "depends_on": [],
                        "notes": "Infrastructure as a Service.",
                        "same_as": "c_iaas",
                    }
                ],
            ),
        ]
    )
    model = CourseUnderstanding(llm, LlmBudgetGuard()).analyze(
        [[_page(1, "IaaS p1")], [_page(2, "IaaS p2")]]
    )
    assert {c.id for c in model.concepts} == {"c_iaas"}
    concept = model.concepts[0]
    assert concept.pages == [1, 2]
    assert concept.essential is True
    assert "c_iaas_local" not in {c.id for c in model.concepts}
    _assert_no_orphan_ids(model)


def test_same_as_towards_future_window_concept() -> None:
    llm = ScriptedLLM(
        [
            _window_payload(
                themes=["Cloud"],
                concepts=[
                    {
                        "id": "c_local_vm",
                        "label": "machine virtuelle",
                        "kind": "mechanism",
                        "pages": [1],
                        "essential": True,
                        "depends_on": [],
                        "notes": "Ordinateur logiciel isole.",
                        "same_as": "c_vm",
                    }
                ],
            ),
            _window_payload(
                themes=["Cloud"],
                concepts=[
                    {
                        "id": "c_vm",
                        "label": "VM",
                        "kind": "mechanism",
                        "pages": [2],
                        "essential": True,
                        "depends_on": [],
                        "notes": "Virtual machine.",
                    }
                ],
            ),
        ]
    )
    model = CourseUnderstanding(llm, LlmBudgetGuard()).analyze(
        [[_page(1, "VM locale")], [_page(2, "VM canonique")]]
    )
    assert {c.id for c in model.concepts} == {"c_vm"}
    concept = model.concepts[0]
    assert concept.pages == [1, 2]
    assert "CONTRADICTION" in concept.notes.upper() or "Virtual machine" in concept.notes
    _assert_no_orphan_ids(model)


def test_successive_aliases_collapse_to_one_canonical() -> None:
    llm = ScriptedLLM(
        [
            _window_payload(
                themes=["Reseau"],
                concepts=[
                    {
                        "id": "c_vpc",
                        "label": "VPC",
                        "kind": "other",
                        "pages": [1],
                        "essential": True,
                        "depends_on": [],
                        "notes": "Virtual Private Cloud.",
                    }
                ],
            ),
            _window_payload(
                themes=["Reseau"],
                concepts=[
                    {
                        "id": "c_vpc_alias",
                        "label": "cloud prive virtuel",
                        "kind": "other",
                        "pages": [2],
                        "essential": False,
                        "depends_on": [],
                        "notes": "Reseau logique isole.",
                        "same_as": "c_vpc",
                    }
                ],
            ),
            _window_payload(
                themes=["Reseau"],
                concepts=[
                    {
                        "id": "c_vpc_alias2",
                        "label": "VPC",
                        "kind": "other",
                        "pages": [3],
                        "essential": True,
                        "depends_on": [],
                        "notes": "Encore le VPC.",
                        "same_as": "c_vpc_alias",
                    }
                ],
            ),
        ]
    )
    model = CourseUnderstanding(llm, LlmBudgetGuard()).analyze(
        [[_page(1, "vpc1")], [_page(2, "vpc2")], [_page(3, "vpc3")]]
    )
    assert {c.id for c in model.concepts} == {"c_vpc"}
    assert model.concepts[0].pages == [1, 2, 3]
    assert model.concepts[0].essential is True
    _assert_no_orphan_ids(model)


def test_depends_on_remapped_after_same_as_fusion() -> None:
    llm = ScriptedLLM(
        [
            _window_payload(
                themes=["Stockage"],
                concepts=[
                    {
                        "id": "c_object",
                        "label": "stockage objet",
                        "kind": "definition",
                        "pages": [1],
                        "essential": True,
                        "depends_on": [],
                        "notes": "Blobs identifies par cle.",
                    }
                ],
            ),
            _window_payload(
                themes=["Stockage"],
                concepts=[
                    {
                        "id": "c_object_alias",
                        "label": "object storage",
                        "kind": "definition",
                        "pages": [2],
                        "essential": False,
                        "depends_on": [],
                        "notes": "Blobs identifies par cle.",
                        "same_as": "c_object",
                    },
                    {
                        "id": "c_bucket",
                        "label": "bucket",
                        "kind": "other",
                        "pages": [2],
                        "essential": True,
                        "depends_on": ["c_object_alias"],
                        "notes": "Conteneur logique de stockage objet.",
                    },
                ],
            ),
        ]
    )
    model = CourseUnderstanding(llm, LlmBudgetGuard()).analyze(
        [[_page(1, "objet")], [_page(2, "bucket")]]
    )
    by_id = {c.id: c for c in model.concepts}
    assert set(by_id) == {"c_object", "c_bucket"}
    assert by_id["c_bucket"].depends_on == ["c_object"]
    _assert_no_orphan_ids(model)


def test_cross_window_relations_use_canonical_ids() -> None:
    llm = ScriptedLLM(
        [
            _window_payload(
                themes=["Securite"],
                concepts=[
                    {
                        "id": "c_iam",
                        "label": "IAM",
                        "kind": "method",
                        "pages": [1],
                        "essential": True,
                        "depends_on": [],
                        "notes": "Identity and Access Management.",
                    }
                ],
            ),
            _window_payload(
                themes=["Securite"],
                concepts=[
                    {
                        "id": "c_iam_alias",
                        "label": "gestion des identites",
                        "kind": "method",
                        "pages": [2],
                        "essential": False,
                        "depends_on": [],
                        "notes": "Identity and Access Management.",
                        "same_as": "c_iam",
                    },
                    {
                        "id": "c_policy",
                        "label": "politique IAM",
                        "kind": "other",
                        "pages": [2],
                        "essential": True,
                        "depends_on": ["c_iam_alias"],
                        "notes": "Regles de moindre privilege.",
                    },
                ],
                relations=[
                    {
                        "source_id": "c_policy",
                        "target_id": "c_iam_alias",
                        "relation_type": "depends_on",
                        "pages": [2],
                    }
                ],
            ),
        ]
    )
    model = CourseUnderstanding(llm, LlmBudgetGuard()).analyze(
        [[_page(1, "IAM")], [_page(2, "politique")]]
    )
    assert len(model.relations) == 1
    assert model.relations[0].source_id == "c_policy"
    assert model.relations[0].target_id == "c_iam"
    _assert_no_orphan_ids(model)


def test_orphan_relations_and_depends_are_dropped() -> None:
    llm = ScriptedLLM(
        [
            _window_payload(
                themes=["Reseau"],
                concepts=[
                    {
                        "id": "c_lb",
                        "label": "equilibreur",
                        "kind": "mechanism",
                        "pages": [1],
                        "essential": True,
                        "depends_on": ["c_missing_dep"],
                        "notes": "Repartit le trafic.",
                    }
                ],
                relations=[
                    {
                        "source_id": "c_lb",
                        "target_id": "c_ghost",
                        "relation_type": "related",
                        "pages": [1],
                    },
                    {
                        "source_id": "c_ghost",
                        "target_id": "c_lb",
                        "relation_type": "related",
                        "pages": [1],
                    },
                ],
            )
        ]
    )
    model = CourseUnderstanding(llm, LlmBudgetGuard()).analyze([[_page(1, "LB")]])
    assert {c.id for c in model.concepts} == {"c_lb"}
    assert model.concepts[0].depends_on == []
    assert model.relations == []
    _assert_no_orphan_ids(model)


def test_duplicate_relations_are_merged_with_page_union() -> None:
    llm = ScriptedLLM(
        [
            _window_payload(
                themes=["Cloud"],
                concepts=[
                    {
                        "id": "c_a",
                        "label": "A",
                        "kind": "other",
                        "pages": [1],
                        "essential": True,
                        "depends_on": [],
                        "notes": "A",
                    },
                    {
                        "id": "c_b",
                        "label": "B",
                        "kind": "other",
                        "pages": [1],
                        "essential": True,
                        "depends_on": ["c_a"],
                        "notes": "B",
                    },
                ],
                relations=[
                    {
                        "source_id": "c_b",
                        "target_id": "c_a",
                        "relation_type": "depends_on",
                        "pages": [1],
                    },
                    {
                        "source_id": "c_b",
                        "target_id": "c_a",
                        "relation_type": "depends_on",
                        "pages": [2],
                    },
                ],
            )
        ]
    )
    model = CourseUnderstanding(llm, LlmBudgetGuard()).analyze([[_page(1, "A B")]])
    assert len(model.relations) == 1
    assert model.relations[0].pages == [1, 2]
    _assert_no_orphan_ids(model)


def test_final_ids_are_internally_consistent_after_complex_merge() -> None:
    llm = ScriptedLLM(
        [
            _window_payload(
                themes=["Virtualisation"],
                concepts=[
                    {
                        "id": "c_hyp",
                        "label": "hyperviseur",
                        "kind": "mechanism",
                        "pages": [1],
                        "essential": True,
                        "depends_on": [],
                        "notes": "Abstrait le materiel.",
                    }
                ],
            ),
            _window_payload(
                themes=["Virtualisation"],
                concepts=[
                    {
                        "id": "c_hyp2",
                        "label": "hyperviseur",
                        "kind": "mechanism",
                        "pages": [2],
                        "essential": False,
                        "depends_on": [],
                        "notes": "Abstrait le materiel.",
                        "same_as": "c_hyp",
                    },
                    {
                        "id": "c_vm",
                        "label": "VM",
                        "kind": "mechanism",
                        "pages": [2],
                        "essential": True,
                        "depends_on": ["c_hyp2"],
                        "notes": "Machine virtuelle.",
                    },
                ],
                relations=[
                    {
                        "source_id": "c_vm",
                        "target_id": "c_hyp2",
                        "relation_type": "depends_on",
                        "pages": [2],
                    }
                ],
            ),
        ]
    )
    model = CourseUnderstanding(llm, LlmBudgetGuard()).analyze(
        [[_page(1, "hyp")], [_page(2, "vm")]]
    )
    known = {c.id for c in model.concepts}
    assert known == {"c_hyp", "c_vm"}
    assert next(c for c in model.concepts if c.id == "c_vm").depends_on == ["c_hyp"]
    assert model.relations[0].target_id == "c_hyp"
    assert "c_hyp" in model.essential_concept_ids
    _assert_no_orphan_ids(model)


def test_contradiction_on_future_alias_applies_to_canonical() -> None:
    """Contradiction vers un alias découvert plus tard → notes du concept canonique."""
    llm = ScriptedLLM(
        [
            _window_payload(
                themes=["Cloud"],
                concepts=[
                    {
                        "id": "c_main",
                        "label": "IaaS",
                        "kind": "other",
                        "pages": [1],
                        "essential": True,
                        "depends_on": [],
                        "notes": "Infrastructure as a Service.",
                    }
                ],
                contradictions=[
                    {
                        "concept_id": "c_alias",
                        "description": "Deux formulations concurrentes pour IaaS.",
                        "pages": [1, 3],
                    }
                ],
            ),
            _window_payload(
                themes=["Cloud"],
                concepts=[
                    {
                        "id": "c_alias",
                        "label": "infrastructure as a service",
                        "kind": "other",
                        "pages": [2],
                        "essential": False,
                        "depends_on": [],
                        "notes": "Infrastructure as a Service.",
                        "same_as": "c_main",
                    }
                ],
            ),
        ]
    )
    model = CourseUnderstanding(llm, LlmBudgetGuard()).analyze(
        [[_page(1, "IaaS p1")], [_page(2, "IaaS p2")]]
    )
    assert {c.id for c in model.concepts} == {"c_main"}
    concept = model.concepts[0]
    assert "CONTRADICTION" in concept.notes.upper()
    assert "Deux formulations concurrentes pour IaaS." in concept.notes
    assert 1 in concept.pages
    assert 3 in concept.pages
    assert 2 in concept.pages
    _assert_no_orphan_ids(model)


def test_contradiction_on_unknown_id_is_silently_ignored() -> None:
    llm = ScriptedLLM(
        [
            _window_payload(
                themes=["Reseau"],
                concepts=[
                    {
                        "id": "c_vpc",
                        "label": "VPC",
                        "kind": "other",
                        "pages": [1],
                        "essential": True,
                        "depends_on": [],
                        "notes": "Virtual Private Cloud.",
                    }
                ],
                contradictions=[
                    {
                        "concept_id": "c_ghost_unknown",
                        "description": "Contradiction orpheline sans concept cible.",
                        "pages": [9],
                    }
                ],
            )
        ]
    )
    model = CourseUnderstanding(llm, LlmBudgetGuard()).analyze([[_page(1, "VPC")]])
    assert {c.id for c in model.concepts} == {"c_vpc"}
    concept = model.concepts[0]
    assert "CONTRADICTION" not in concept.notes.upper()
    assert "orpheline" not in concept.notes.casefold()
    assert concept.pages == [1]
    _assert_no_orphan_ids(model)
