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
    TableBlock,
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
    # Validation dans parse → chaque tentative consomme une réponse.
    llm = ScriptedLLM([bad] * max_attempts_per_step())
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


def _generic_concept(
    *,
    cid: str,
    label: str,
    kind: str = "definition",
    notes: str = "",
    essential: bool = True,
) -> Concept:
    return Concept(
        id=cid,
        label=label,
        kind=kind,  # type: ignore[arg-type]
        pages=[1],
        essential=essential,
        notes=notes,
    )


def test_hyphen_and_dash_variants_of_multiword_label_are_accepted() -> None:
    concept = _generic_concept(
        cid="cloud-computing",
        label="Cloud Computing",
        notes=(
            "Modele d acces a la demande a des ressources informatiques partagees. "
            "Aussi appele informatique en nuage."
        ),
    )
    assert concept_is_explained(
        concept,
        "Le cloud-computing est un modele qui fournit des ressources a la demande via Internet.",
    )
    assert concept_is_explained(
        concept,
        "Le Cloud–Computing est un modele qui fournit des ressources a la demande via Internet.",
    )
    assert concept_is_explained(
        concept,
        "Cloud Computing: acces ubiquitaire a un ensemble partage de ressources configurables.",
    )


def test_note_alias_synonym_is_accepted_without_exact_label() -> None:
    concept = _generic_concept(
        cid="cloud-computing",
        label="Cloud Computing",
        notes=(
            "Modele d acces a la demande a des ressources informatiques partagees. "
            "Aussi appele informatique en nuage."
        ),
    )
    assert concept_is_explained(
        concept,
        "L'informatique en nuage designe un mode de fourniture de ressources a la demande.",
    )


def test_nbsp_and_punctuation_variants_are_accepted() -> None:
    concept = _generic_concept(
        cid="machine-learning",
        label="Machine Learning",
        notes="Apprentissage automatique a partir de donnees.",
    )
    assert concept_is_explained(
        concept,
        "Le Machine\u00a0Learning est une methode qui apprend des motifs a partir de donnees.",
    )
    assert concept_is_explained(
        concept,
        "«Machine Learning» : methode qui apprend des motifs a partir de donnees.",
    )


def test_english_and_arabic_formulations_are_accepted() -> None:
    en = _generic_concept(
        cid="load-balancer",
        label="load balancer",
        kind="mechanism",
        notes="Also known as traffic distributor. Spreads requests across servers.",
    )
    ar = _generic_concept(
        cid="khawarizmiya",
        label="خوارزمية",
        kind="method",
        notes="سلسلة خطوات لحل مسألة.",
    )
    assert concept_is_explained(
        en,
        "A traffic distributor spreads incoming requests across several backend servers.",
    )
    assert concept_is_explained(
        ar,
        "الخوارزمية هي سلسلة خطوات واضحة تسمح بحل مسألة بشكل منظم.",
    )


def test_table_and_definition_blocks_count_as_explanation() -> None:
    concept = _generic_concept(
        cid="iaas",
        label="IaaS",
        notes="Infrastructure as a Service. Fournit des ressources virtualisees.",
    )
    draft = ChapterDraft(
        index=0,
        title="Cloud",
        concept_ids_covered=["iaas"],
        blocks=[
            DefinitionBlock(
                text="Infrastructure as a Service fournit des machines a la demande.",
                source_pages=[1],
            ),
            TableBlock(
                headers=["Offre", "Role"],
                rows=[["IaaS", "ressources virtualisees"]],
                source_pages=[1],
            ),
        ],
    )
    plan = ChapterPlan(
        index=0,
        title="Cloud",
        concept_ids=["iaas"],
        page_refs=[1],
        depth="complete",
    )
    course = CourseModel(concepts=[concept], essential_concept_ids=["iaas"])
    assert validate_chapter_concept_coverage(
        draft, chapter_plan=plan, course=course
    ) == []


def test_mere_mention_or_title_only_is_still_rejected() -> None:
    concept = _generic_concept(
        cid="cloud-computing",
        label="Cloud Computing",
        notes="Modele d acces a la demande a des ressources informatiques partagees.",
    )
    assert not concept_is_explained(
        concept,
        "Titre seul : Cloud Computing",
    )
    assert not concept_is_explained(
        concept,
        "Le cloud transforme l'informatique en service consomme et facture.",
    )


def test_declared_id_without_explanation_still_fails_validation() -> None:
    concept = _generic_concept(
        cid="photosynthesis",
        label="photosynthesis",
        notes="Process by which plants convert light into chemical energy.",
    )
    draft = ChapterDraft(
        index=0,
        title="Biology",
        concept_ids_covered=["photosynthesis"],
        blocks=[
            ParagraphBlock(
                text="This chapter introduces several ideas about living systems in general.",
                source_pages=[1],
            )
        ],
    )
    plan = ChapterPlan(
        index=0,
        title="Biology",
        concept_ids=["photosynthesis"],
        page_refs=[1],
        depth="complete",
    )
    course = CourseModel(concepts=[concept], essential_concept_ids=["photosynthesis"])
    missing = validate_chapter_concept_coverage(
        draft, chapter_plan=plan, course=course
    )
    assert [c.id for c in missing] == ["photosynthesis"]


def test_generic_multiword_slug_concept_regression() -> None:
    """Régression générique : id slug + label multi-mots + formulation tiretée."""
    concept = _generic_concept(
        cid="operating-system",
        label="Operating System",
        notes="Software layer that manages hardware and processes. Also known as OS kernel host.",
    )
    assert concept_is_explained(
        concept,
        "An operating-system manages hardware resources and schedules processes for applications.",
    )
    assert concept_is_explained(
        concept,
        "The OS kernel host coordinates memory, CPU and devices for user programs.",
    )


def test_title_only_in_long_paragraph_is_still_rejected_for_definition() -> None:
    """Mention du label dans un long texte sans marqueur définitoire → non couvert."""
    concept = _generic_concept(
        cid="mitosis",
        label="mitosis",
        notes="Cell division that produces two genetically identical daughter cells.",
    )
    corpus = (
        "This chapter surveys several school topics. Mitosis. Later we discuss geography, "
        "literature history and unrelated administrative procedures without explaining the term."
    )
    assert not concept_is_explained(concept, corpus)


def test_synonym_reformulation_from_merged_notes_counts_as_coverage() -> None:
    concept = _generic_concept(
        cid="photosynthesis",
        label="photosynthesis",
        notes=(
            "Process by which plants convert light into chemical energy.\n"
            "Aussi appelée assimilation chlorophyllienne."
        ),
    )
    corpus = (
        "L'assimilation chlorophyllienne permet aux végétaux de transformer la lumière "
        "en énergie chimique stockée sous forme de glucose."
    )
    assert concept_is_explained(concept, corpus) is True


def test_generated_chapter_covers_essential_with_real_explanation() -> None:
    course = CourseModel(
        concepts=[
            Concept(
                id="ohm-law",
                label="loi d'Ohm",
                kind="formula",
                pages=[2, 3],
                essential=True,
                notes="U = R * I. Relie tension, résistance et intensité.",
            )
        ],
        essential_concept_ids=["ohm-law"],
    )
    plan = _plan_single(["ohm-law"], title="Électricité")
    good = _chapter_json(
        concept_ids=["ohm-law"],
        title="Électricité",
        blocks=[
            {
                "type": "paragraph",
                "text": (
                    "La loi d'Ohm relie la tension U à la résistance R et à l'intensité I "
                    "selon la relation U = R * I présentée dans le cours."
                ),
                "source_pages": [2, 3],
            },
            {
                "type": "definition",
                "text": "U = R * I. Relie tension, résistance et intensité.",
                "source_pages": [2],
            },
            {
                "type": "synthesis",
                "text": "Retenir que tension, résistance et courant sont liés par Ohm.",
                "source_pages": [3],
            },
        ],
    )
    document = OriginalContentGenerator(
        ScriptedLLM([good]),
        LlmBudgetGuard(),
    ).generate(
        course,
        plan,
        filename="cours.pdf",
        page_start=2,
        page_end=3,
        source_pages=[
            CoursePage(
                page_number=2,
                text="U = R * I. Relie tension, résistance et intensité.",
                filename="cours.pdf",
            ),
            CoursePage(
                page_number=3,
                text="La loi d'Ohm structure l'analyse des circuits simples.",
                filename="cours.pdf",
            ),
        ],
    )
    unexplained = validate_chapter_concept_coverage(
        document.chapters[0],
        chapter_plan=plan.chapters[0],
        course=course,
    )
    assert unexplained == []
