"""Tests de la vérification réelle de couverture pédagogique (Correction 3)."""

from __future__ import annotations

import json

import pytest

from src.revision.engine.coverage import (
    concept_is_explained,
    validate_chapter_concept_coverage,
)
from src.revision.engine.generator import OriginalContentGenerator
from src.revision.engine.llm_guard import LlmBudgetGuard, max_attempts_per_step
from src.revision.engine.models import (
    ChapterDraft,
    ChapterPlan,
    Concept,
    CourseModel,
    CoursePage,
    DefinitionBlock,
    LearningObjective,
    ParagraphBlock,
    PedagogicalDocument,
    PedagogicalPlan,
    SynthesisBlock,
)
from src.revision.engine.quality import QualityController


class _TokenIds:
    def __init__(self, count: int) -> None:
        self.shape = (-1, count)


class _FakeTokenizer:
    chat_template = None

    def __call__(self, prompt: str, return_tensors: str = "pt") -> dict[str, _TokenIds]:
        return {"input_ids": _TokenIds(max(1, len(prompt) // 8))}


class _FakeConfig:
    max_position_embeddings = 32768
    use_sliding_window = False
    rope_parameters = {"rope_type": "default"}


class _FakeModel:
    config = _FakeConfig()


class ScriptedLLM:
    def __init__(self, replies: list[str] | None = None) -> None:
        self.replies = list(replies or [])
        self.calls = 0
        self.prompts: list[str] = []
        self.tokenizer = _FakeTokenizer()
        self.model = _FakeModel()

    def is_loaded(self) -> bool:
        return True

    def generate(self, prompt: str, max_new_tokens: int = 512) -> str:
        self.calls += 1
        self.prompts.append(prompt)
        if not self.replies:
            raise AssertionError("Plus de réponses LLM scriptées.")
        return self.replies.pop(0)


def _pages() -> list[CoursePage]:
    return [
        CoursePage(
            page_number=1,
            text="IaaS fournit des ressources. E = mc2. L'hyperviseur abstrait le materiel.",
            filename="cours.pdf",
        )
    ]


def _course_with_secondary() -> CourseModel:
    return CourseModel(
        concepts=[
            Concept(
                id="c_iaas",
                label="IaaS",
                kind="definition",
                pages=[1],
                essential=True,
                notes="Infrastructure as a Service. Fournit des ressources virtualisées.",
            ),
            Concept(
                id="c_hyp",
                label="hyperviseur",
                kind="mechanism",
                pages=[1],
                essential=False,
                notes="Couche qui abstrait le matériel pour créer des machines virtuelles.",
            ),
            Concept(
                id="c_energy",
                label="énergie",
                kind="formula",
                pages=[1],
                essential=True,
                notes="E = mc2",
            ),
        ],
        themes=["Cloud"],
        essential_concept_ids=["c_iaas", "c_energy"],
    )


def _plan_single(
    concept_ids: list[str],
    *,
    title: str = "Cloud",
) -> PedagogicalPlan:
    return PedagogicalPlan(
        language="fr",
        level="beginner",
        sheet_type="short",
        chapters=[
            ChapterPlan(
                index=0,
                title=title,
                objectives=[
                    LearningObjective(text="Comprendre le cloud", concept_ids=concept_ids)
                ],
                concept_ids=concept_ids,
                page_refs=[1],
                depth="short",
            )
        ],
    )


def _document(
    *,
    concept_ids: list[str],
    blocks: list,
    title: str = "Cloud",
) -> PedagogicalDocument:
    return PedagogicalDocument(
        filename="cours.pdf",
        page_start=1,
        page_end=1,
        language="fr",
        level="beginner",
        sheet_type="short",
        chapters=[
            ChapterDraft(
                index=0,
                title=title,
                concept_ids_covered=concept_ids,
                blocks=blocks,
            )
        ],
    )


def _chapter_json(*, concept_ids: list[str], blocks: list[dict], title: str = "Cloud") -> str:
    return json.dumps(
        {
            "index": 0,
            "title": title,
            "concept_ids_covered": concept_ids,
            "blocks": blocks,
        },
        ensure_ascii=False,
    )


def test_all_ids_present_but_empty_text_is_rejected() -> None:
    document = _document(
        concept_ids=["c_iaas", "c_hyp"],
        blocks=[
            ParagraphBlock(text="   ", source_pages=[1]),
            DefinitionBlock(text="", source_pages=[1]),
        ],
    )
    course = _course_with_secondary()
    plan = _plan_single(["c_iaas", "c_hyp"])
    unexplained = validate_chapter_concept_coverage(
        document.chapters[0], chapter_plan=plan.chapters[0], course=course
    )
    assert {c.id for c in unexplained} == {"c_iaas", "c_hyp"}
    still_empty = _chapter_json(
        concept_ids=["c_iaas", "c_hyp"],
        blocks=[
            {"type": "paragraph", "text": "   ", "source_pages": [1]},
            {"type": "definition", "text": "", "source_pages": [1]},
        ],
    )
    with pytest.raises(ValueError, match="Couverture|expliqué|qualité"):
        QualityController(
            ScriptedLLM([still_empty] * max_attempts_per_step()),
            LlmBudgetGuard(),
        ).review(document, course, plan, source_pages=_pages())


def test_secondary_declared_but_unexplained_is_rejected() -> None:
    document = _document(
        concept_ids=["c_iaas", "c_hyp"],
        blocks=[
            ParagraphBlock(
                text=(
                    "IaaS, ou Infrastructure as a Service, fournit des ressources "
                    "virtualisées à la demande pour les équipes techniques."
                ),
                source_pages=[1],
            ),
            DefinitionBlock(
                text="Infrastructure as a Service. Fournit des ressources virtualisées.",
                source_pages=[1],
            ),
            # Aucune explication de l'hyperviseur (secondaire).
            SynthesisBlock(
                text="Retenir le rôle d'IaaS dans le cloud public.",
                source_pages=[1],
            ),
        ],
    )
    course = _course_with_secondary()
    plan = _plan_single(["c_iaas", "c_hyp"])
    unexplained = validate_chapter_concept_coverage(
        document.chapters[0], chapter_plan=plan.chapters[0], course=course
    )
    assert [c.id for c in unexplained] == ["c_hyp"]
    assert unexplained[0].essential is False
    still_missing_secondary = _chapter_json(
        concept_ids=["c_iaas", "c_hyp"],
        blocks=[
            {
                "type": "paragraph",
                "text": (
                    "IaaS, ou Infrastructure as a Service, fournit des ressources "
                    "virtualisées à la demande pour les équipes techniques."
                ),
                "source_pages": [1],
            },
            {
                "type": "definition",
                "text": "Infrastructure as a Service. Fournit des ressources virtualisées.",
                "source_pages": [1],
            },
            {
                "type": "synthesis",
                "text": "Retenir le rôle d'IaaS dans le cloud public.",
                "source_pages": [1],
            },
        ],
    )
    with pytest.raises(ValueError, match="secondaires non expliqués|c_hyp|qualité"):
        QualityController(
            ScriptedLLM([still_missing_secondary] * max_attempts_per_step()),
            LlmBudgetGuard(),
        ).review(document, course, plan, source_pages=_pages())


def test_relevant_definition_is_accepted() -> None:
    concept = Concept(
        id="c_iaas",
        label="IaaS",
        kind="definition",
        pages=[1],
        essential=True,
        notes="Infrastructure as a Service. Fournit des ressources virtualisées.",
    )
    corpus = (
        "Dans le cloud, IaaS désigne une offre qui fournit des machines et du stockage "
        "à la demande, sans que l'équipe gère le matériel physique."
    )
    assert concept_is_explained(concept, corpus) is True


def test_synonym_explanation_counts_as_coverage() -> None:
    concept = Concept(
        id="c_iaas",
        label="IaaS",
        kind="definition",
        pages=[1],
        essential=True,
        notes="Infrastructure as a Service. Fournit des ressources virtualisées.",
    )
    # Pas le label exact, mais la formulation des notes / synonyme long.
    corpus = (
        "Infrastructure as a Service fournit des ressources virtualisées que l'on "
        "provisionne sans acheter de serveurs."
    )
    assert concept_is_explained(concept, corpus) is True


def test_acronym_and_full_name_both_count() -> None:
    full = Concept(
        id="c_iam",
        label="Identity and Access Management",
        kind="method",
        pages=[1],
        essential=True,
        notes="Identity and Access Management (IAM) contrôle les droits.",
    )
    short = Concept(
        id="c_iam2",
        label="IAM",
        kind="method",
        pages=[1],
        essential=True,
        notes="Identity and Access Management (IAM) contrôle les droits.",
    )
    assert concept_is_explained(
        full,
        "IAM is a method that controls who can access which cloud resources safely.",
    )
    assert concept_is_explained(
        short,
        "Identity and Access Management controls permissions across accounts and roles.",
    )


def test_concept_explained_inside_larger_paragraph() -> None:
    concept = Concept(
        id="c_hyp",
        label="hyperviseur",
        kind="mechanism",
        pages=[1],
        essential=False,
        notes="Couche qui abstrait le matériel pour créer des machines virtuelles.",
    )
    corpus = (
        "Avant de parler du stockage objet, rappelons que l'hyperviseur abstrait le "
        "matériel afin de créer plusieurs machines virtuelles isolées sur un même hôte ; "
        "cette couche reste invisible pour la plupart des applications métier."
    )
    assert concept_is_explained(concept, corpus) is True


def test_formula_or_mechanism_explanation_is_accepted() -> None:
    formula = Concept(
        id="c_energy",
        label="énergie",
        kind="formula",
        pages=[1],
        essential=True,
        notes="E = mc2",
    )
    mechanism = Concept(
        id="c_hyp",
        label="hyperviseur",
        kind="mechanism",
        pages=[1],
        essential=False,
        notes="Couche qui abstrait le matériel pour créer des machines virtuelles.",
    )
    assert concept_is_explained(
        formula,
        "La relation E = mc2 relie l'énergie à la masse dans ce chapitre de révision.",
    )
    assert concept_is_explained(
        mechanism,
        "L'hyperviseur est le mécanisme qui abstrait le matériel pour créer des VMs.",
    )


def test_rewrite_keeping_ids_but_removing_explanation_fails() -> None:
    course = _course_with_secondary()
    plan = _plan_single(["c_iaas"])
    # Déclenche une réécriture via URL interdite, puis renvoie des ids sans explication.
    document = _document(
        concept_ids=["c_iaas"],
        blocks=[
            ParagraphBlock(
                text="Voir https://exemple.invente/iaas pour IaaS et Infrastructure as a Service.",
                source_pages=[1],
            ),
            DefinitionBlock(
                text="Infrastructure as a Service. Fournit des ressources virtualisées.",
                source_pages=[1],
            ),
        ],
    )
    bad_rewrite = _chapter_json(
        concept_ids=["c_iaas"],
        blocks=[
            {
                "type": "paragraph",
                "text": "Texte vague sans aucune notion cloud réellement expliquée ici.",
                "source_pages": [1],
            },
            {
                "type": "synthesis",
                "text": "Conclusion générique sans ancrage conceptuel utile.",
                "source_pages": [1],
            },
        ],
    )
    with pytest.raises(ValueError, match="Couverture|expliqué|qualité"):
        QualityController(
            ScriptedLLM([bad_rewrite] * max_attempts_per_step()),
            LlmBudgetGuard(),
        ).review(document, course, plan, source_pages=_pages())


def test_complete_document_accepted_without_false_reject() -> None:
    course = _course_with_secondary()
    plan = _plan_single(["c_iaas", "c_hyp", "c_energy"])
    document = _document(
        concept_ids=["c_iaas", "c_hyp", "c_energy"],
        blocks=[
            ParagraphBlock(
                text=(
                    "Dans une offre IaaS, l'équipe réserve processeurs et disques "
                    "sans acheter de serveur. L'hyperviseur isole plusieurs systèmes "
                    "sur la même machine physique. On rappelle aussi la relation "
                    "E = mc2 pour lier énergie et masse."
                ),
                source_pages=[1],
            ),
            DefinitionBlock(
                text=(
                    "IaaS signifie Infrastructure as a Service : mise à disposition "
                    "de ressources informatiques à la demande."
                ),
                source_pages=[1],
            ),
            SynthesisBlock(
                text=(
                    "Retenir IaaS, le rôle de l'hyperviseur et la formule E = mc2 "
                    "comme points clés."
                ),
                source_pages=[1],
            ),
        ],
    )
    reviewed, report = QualityController(ScriptedLLM(), LlmBudgetGuard()).review(
        document, course, plan, source_pages=_pages()
    )
    assert report.ok is True
    assert report.issues == []
    assert reviewed.chapters[0].concept_ids_covered == ["c_iaas", "c_hyp", "c_energy"]


def test_generator_rejects_declared_ids_without_real_explanation() -> None:
    course = _course_with_secondary()
    plan = _plan_single(["c_iaas"])
    bad = _chapter_json(
        concept_ids=["c_iaas"],
        blocks=[
            {
                "type": "paragraph",
                "text": "Introduction générale au chapitre sans développer la notion.",
                "source_pages": [1],
            },
            {
                "type": "synthesis",
                "text": "Une conclusion trop large pour valider la couverture.",
                "source_pages": [1],
            },
        ],
    )
    llm = ScriptedLLM([bad])
    llm.tokenizer = _FakeTokenizer()
    llm.model = _FakeModel()
    with pytest.raises(ValueError, match="Couverture|expliqué"):
        OriginalContentGenerator(llm, LlmBudgetGuard()).generate(
            course,
            plan,
            filename="cours.pdf",
            page_start=1,
            page_end=1,
            source_pages=_pages(),
        )
