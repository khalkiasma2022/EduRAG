"""Tests de OriginalContentGenerator (Phase 4 — rédaction pédagogique)."""

from __future__ import annotations

import json

import pytest

from src.revision.engine.generator import OriginalContentGenerator, _build_chapter_prompt
from src.revision.engine.llm_guard import (
    LEGACY_STORE_LEVEL,
    LEGACY_STORE_SHEET_TYPE,
    LlmBudgetGuard,
    MAX_LLM_CALLS,
    TOKENS_CHAPTER,
)
from src.revision.engine.models import (
    ChapterPlan,
    Concept,
    CourseModel,
    CoursePage,
    CourseRelation,
    CourseUnit,
    ExampleBlock,
    LearningObjective,
    PedagogicalPlan,
)


class _TokenIds:
    def __init__(self, count: int) -> None:
        self.shape = (-1, count)


class _FakeTokenizer:
    def __init__(self) -> None:
        self.chat_template = None

    def __call__(self, prompt: str, return_tensors: str = "pt") -> dict[str, _TokenIds]:
        # Prompt court pour laisser de la marge au contexte.
        return {"input_ids": _TokenIds(max(1, len(prompt) // 8))}


class _FakeConfig:
    max_position_embeddings = 32768
    use_sliding_window = False
    rope_parameters = {"rope_type": "default"}


class _FakeModel:
    config = _FakeConfig()


class ScriptedLLM:
    """LLM scripté chargé : active ensure_prompt_fits + generate."""

    def __init__(self, replies: list[str]) -> None:
        self.replies = list(replies)
        self.calls = 0
        self.prompts: list[str] = []
        self.token_budgets: list[int] = []
        self.tokenizer = _FakeTokenizer()
        self.model = _FakeModel()

    def is_loaded(self) -> bool:
        return True

    def generate(self, prompt: str, max_new_tokens: int = 512) -> str:
        self.calls += 1
        self.prompts.append(prompt)
        self.token_budgets.append(max_new_tokens)
        if not self.replies:
            raise AssertionError("Plus de réponses LLM scriptées.")
        return self.replies.pop(0)


def _pages() -> list[CoursePage]:
    return [
        CoursePage(
            page_number=1,
            text="Une variable stocke une valeur. Formule : E = 12.",
            filename="cours.pdf",
        ),
        CoursePage(
            page_number=2,
            text="Une fonction encapsule un comportement. "
            "[CONTRADICTION] Page A dit x>0 ; page B dit x<0.",
            filename="cours.pdf",
        ),
    ]


def _course() -> CourseModel:
    return CourseModel(
        units=[
            CourseUnit(text="Une variable stocke une valeur. Formule : E = 12.", page_start=1, page_end=1),
            CourseUnit(
                text="Une fonction encapsule un comportement.",
                page_start=2,
                page_end=2,
            ),
        ],
        concepts=[
            Concept(
                id="c_var",
                label="variable",
                kind="definition",
                pages=[1],
                essential=True,
                notes="Une variable stocke une valeur.",
            ),
            Concept(
                id="c_fun",
                label="fonction",
                kind="method",
                pages=[2],
                essential=True,
                depends_on=["c_var"],
                notes="Une fonction encapsule un comportement.\n"
                "[CONTRADICTION] Les pages affirment des signes opposés pour x.",
            ),
        ],
        relations=[
            CourseRelation(
                source_id="c_fun",
                target_id="c_var",
                relation_type="depends_on",
                pages=[2],
            )
        ],
        themes=["Programmation"],
        essential_concept_ids=["c_var", "c_fun"],
    )


def _plan(language: str = "fr", *, needs_table: bool = False) -> PedagogicalPlan:
    return PedagogicalPlan(
        language=language,
        level=LEGACY_STORE_LEVEL,
        sheet_type=LEGACY_STORE_SHEET_TYPE,
        chapters=[
            ChapterPlan(
                index=0,
                title="Variables",
                objectives=[
                    LearningObjective(text="Comprendre les variables", concept_ids=["c_var"])
                ],
                concept_ids=["c_var"],
                page_refs=[1],
                needs_example=True,
                needs_table=False,
                needs_questions=False,
                depth="complete",
            ),
            ChapterPlan(
                index=1,
                title="Fonctions",
                objectives=[
                    LearningObjective(text="Utiliser les fonctions", concept_ids=["c_fun"])
                ],
                concept_ids=["c_fun"],
                page_refs=[2],
                needs_example=True,
                needs_table=needs_table,
                needs_questions=False,
                depth="complete",
            ),
        ],
    )


def _chapter_json(
    *,
    index: int,
    title: str,
    concept_ids: list[str],
    blocks: list[dict],
) -> str:
    return json.dumps(
        {
            "index": index,
            "title": title,
            "concept_ids_covered": concept_ids,
            "blocks": blocks,
        },
        ensure_ascii=False,
    )


def _short_chapter_var() -> str:
    return _chapter_json(
        index=0,
        title="Variables",
        concept_ids=["c_var"],
        blocks=[
            {
                "type": "paragraph",
                "text": "On introduit la notion de variable pour nommer une information.",
                "source_pages": [1],
            },
            {
                "type": "definition",
                "text": "Une variable stocke une valeur.",
                "source_pages": [1],
            },
            {
                "type": "example",
                "text": "Dans le cours : E = 12.",
                "origin": "from_course",
                "source_pages": [1],
            },
            {
                "type": "example",
                "text": "Comme une boîte étiquetée qui contient un objet.",
                "origin": "pedagogical_addition",
                "source_pages": [],
            },
            {
                "type": "synthesis",
                "text": "Retenir qu'une variable nomme une valeur.",
                "source_pages": [1],
            },
        ],
    )


def _short_chapter_fun() -> str:
    return _chapter_json(
        index=1,
        title="Fonctions",
        concept_ids=["c_fun"],
        blocks=[
            {
                "type": "paragraph",
                "text": "Après les variables, on organise le comportement dans une fonction.",
                "source_pages": [2],
            },
            {
                "type": "definition",
                "text": "Une fonction encapsule un comportement.",
                "source_pages": [2],
            },
            {
                "type": "example",
                "text": "Analogie : une recette qui utilise des ingrédients nommés.",
                "origin": "pedagogical_addition",
                "source_pages": [],
            },
            {
                "type": "retain",
                "text": "Le cours présente des affirmations contradictoires sur le signe de x ; "
                "elles sont signalées sans être tranchées.",
                "source_pages": [2],
            },
            {
                "type": "synthesis",
                "text": "Une fonction s'appuie sur les variables déjà vues.",
                "source_pages": [2],
            },
        ],
    )


def _complete_chapter_var() -> str:
    return _chapter_json(
        index=0,
        title="Variables",
        concept_ids=["c_var"],
        blocks=[
            {
                "type": "paragraph",
                "text": "Développement : une variable associe un nom à une valeur manipulable.",
                "source_pages": [1],
            },
            {
                "type": "definition",
                "text": "Une variable stocke une valeur.",
                "source_pages": [1],
            },
            {
                "type": "paragraph",
                "text": "Cette idée prépare l'usage des fonctions qui consomment des valeurs nommées.",
                "source_pages": [1],
            },
            {
                "type": "example",
                "text": "Dans le cours : E = 12.",
                "origin": "from_course",
                "source_pages": [1],
            },
            {
                "type": "example",
                "text": "Image mentale : un casier numéroté dans une bibliothèque.",
                "origin": "pedagogical_addition",
                "source_pages": [],
            },
            {
                "type": "table",
                "headers": ["Notion", "Rôle"],
                "rows": [["Variable", "Nomme une valeur"]],
                "source_pages": [1],
            },
            {
                "type": "synthesis",
                "text": "La variable est la brique de base avant les fonctions.",
                "source_pages": [1],
            },
        ],
    )


def _complete_chapter_fun() -> str:
    return _chapter_json(
        index=1,
        title="Fonctions",
        concept_ids=["c_fun"],
        blocks=[
            {
                "type": "paragraph",
                "text": "On relie maintenant fonctions et variables : la fonction réutilise des valeurs nommées.",
                "source_pages": [2],
            },
            {
                "type": "definition",
                "text": "Une fonction encapsule un comportement.",
                "source_pages": [2],
            },
            {
                "type": "diagram",
                "text": "variable → fonction → résultat",
                "source_pages": [2],
            },
            {
                "type": "example",
                "text": "Analogie pédagogique : atelier qui transforme des pièces étiquetées.",
                "origin": "pedagogical_addition",
                "source_pages": [],
            },
            {
                "type": "retain",
                "text": "Contradiction signalée sur x : le cours ne permet pas de trancher.",
                "source_pages": [2],
            },
            {
                "type": "synthesis",
                "text": "Les fonctions organisent les comportements autour des variables.",
                "source_pages": [2],
            },
        ],
    )


def test_generates_chapters_in_plan_order_covering_essentials() -> None:
    llm = ScriptedLLM([_short_chapter_var(), _short_chapter_fun()])
    guard = LlmBudgetGuard()
    document = OriginalContentGenerator(llm, guard).generate(
        _course(),
        _plan(),
        filename="cours.pdf",
        page_start=1,
        page_end=2,
        source_pages=_pages(),
    )

    assert [ch.title for ch in document.chapters] == ["Variables", "Fonctions"]
    assert [ch.index for ch in document.chapters] == [0, 1]
    covered = {cid for ch in document.chapters for cid in ch.concept_ids_covered}
    assert {"c_var", "c_fun"} <= covered
    assert document.level == LEGACY_STORE_LEVEL
    assert document.sheet_type == LEGACY_STORE_SHEET_TYPE
    assert llm.calls == 2
    assert llm.token_budgets == [TOKENS_CHAPTER, TOKENS_CHAPTER]
    assert guard.calls == 2
    assert guard.reserved_tokens == 2 * TOKENS_CHAPTER


def test_complete_sheet_includes_rich_blocks_without_questions() -> None:
    llm = ScriptedLLM([_complete_chapter_var(), _complete_chapter_fun()])
    document = OriginalContentGenerator(llm, LlmBudgetGuard()).generate(
        _course(),
        _plan(needs_table=True),
        filename="cours.pdf",
        page_start=1,
        page_end=2,
        source_pages=_pages(),
    )

    assert llm.token_budgets == [TOKENS_CHAPTER, TOKENS_CHAPTER]
    assert "complète" in llm.prompts[0].casefold() or "complete" in llm.prompts[0].casefold()
    assert "question" in llm.prompts[0].casefold()
    assert "interdit" in llm.prompts[0].casefold()
    assert any(
        isinstance(block, ExampleBlock) and block.origin == "pedagogical_addition"
        for ch in document.chapters
        for block in ch.blocks
    )
    assert all(block.type != "question" for ch in document.chapters for block in ch.blocks)


def test_prompt_adapts_language_only() -> None:
    llm = ScriptedLLM([_short_chapter_var(), _short_chapter_fun()])
    OriginalContentGenerator(llm, LlmBudgetGuard()).generate(
        _course(),
        _plan(language="en"),
        filename="cours.pdf",
        page_start=1,
        page_end=2,
        source_pages=_pages(),
    )
    prompt = llm.prompts[0]
    assert "english" in prompt.casefold() or "write entirely in english" in prompt.casefold()

def test_distinguishes_course_and_pedagogical_examples() -> None:
    llm = ScriptedLLM([_short_chapter_var(), _short_chapter_fun()])
    document = OriginalContentGenerator(llm, LlmBudgetGuard()).generate(
        _course(),
        _plan(),
        filename="cours.pdf",
        page_start=1,
        page_end=2,
        source_pages=_pages(),
    )
    examples = [
        block
        for chapter in document.chapters
        for block in chapter.blocks
        if isinstance(block, ExampleBlock)
    ]
    origins = {ex.origin for ex in examples}
    assert "from_course" in origins
    assert "pedagogical_addition" in origins


def test_rejects_question_blocks() -> None:
    bad = _chapter_json(
        index=0,
        title="Variables",
        concept_ids=["c_var"],
        blocks=[
            {
                "type": "paragraph",
                "text": "On introduit la notion de variable pour nommer une information.",
                "source_pages": [1],
            },
            {
                "type": "question",
                "text": "Pourquoi nommer une valeur ?",
                "source_pages": [1],
            },
        ],
    )
    llm = ScriptedLLM([bad, bad])
    with pytest.raises(ValueError, match="question"):
        OriginalContentGenerator(llm, LlmBudgetGuard()).generate(
            _course(),
            _plan(),
            filename="cours.pdf",
            page_start=1,
            page_end=2,
            source_pages=_pages(),
        )


def test_rejects_invented_url_attributed_to_source() -> None:
    bad = _chapter_json(
        index=0,
        title="Variables",
        concept_ids=["c_var"],
        blocks=[
            {
                "type": "paragraph",
                "text": "Voir https://exemple.invente/docs pour la définition.",
                "source_pages": [1],
            },
            {
                "type": "definition",
                "text": "Une variable stocke une valeur.",
                "source_pages": [1],
            },
        ],
    )
    llm = ScriptedLLM([bad, bad])
    with pytest.raises(ValueError, match="invent|URL|interdit"):
        OriginalContentGenerator(llm, LlmBudgetGuard()).generate(
            _course(),
            _plan(),
            filename="cours.pdf",
            page_start=1,
            page_end=2,
            source_pages=_pages(),
        )


def test_rejects_invented_numeric_fact_not_in_source() -> None:
    bad = _chapter_json(
        index=0,
        title="Variables",
        concept_ids=["c_var"],
        blocks=[
            {
                "type": "definition",
                "text": "Une variable stocke une valeur. La constante secrète vaut 987654.",
                "source_pages": [1],
            }
        ],
    )
    llm = ScriptedLLM([bad, bad])
    with pytest.raises(ValueError, match="invent"):
        OriginalContentGenerator(llm, LlmBudgetGuard()).generate(
            _course(),
            _plan(),
            filename="cours.pdf",
            page_start=1,
            page_end=2,
            source_pages=_pages(),
        )


def test_keeps_contradiction_without_invented_resolution() -> None:
    resolving = _chapter_json(
        index=1,
        title="Fonctions",
        concept_ids=["c_fun"],
        blocks=[
            {
                "type": "paragraph",
                "text": "En réalité x est positif, donc on ignore la page 2.",
                "source_pages": [2],
            },
            {
                "type": "definition",
                "text": "Une fonction encapsule un comportement.",
                "source_pages": [2],
            },
        ],
    )
    llm = ScriptedLLM([_short_chapter_var(), resolving, resolving])
    with pytest.raises(ValueError, match="contradiction|résolution|trancher"):
        OriginalContentGenerator(llm, LlmBudgetGuard()).generate(
            _course(),
            _plan(),
            filename="cours.pdf",
            page_start=1,
            page_end=2,
            source_pages=_pages(),
        )


def test_invalid_json_retries_then_fails_without_partial_document() -> None:
    llm = ScriptedLLM(["pas-json", "{toujours bad", "ne-doit-pas-etre-lu"])
    guard = LlmBudgetGuard()
    with pytest.raises(ValueError, match="JSON"):
        OriginalContentGenerator(llm, guard).generate(
            _course(),
            _plan(),
            filename="cours.pdf",
            page_start=1,
            page_end=2,
            source_pages=_pages(),
        )
    assert llm.calls == 2
    assert guard.calls == 2
    assert guard.reserved_tokens == 2 * TOKENS_CHAPTER


def test_incomplete_second_chapter_does_not_return_partial_success() -> None:
    empty = _chapter_json(
        index=1,
        title="Fonctions",
        concept_ids=["c_fun"],
        blocks=[],
    )
    llm = ScriptedLLM([_short_chapter_var(), empty, empty])
    with pytest.raises(ValueError, match="incomplet|vide|chapitre"):
        OriginalContentGenerator(llm, LlmBudgetGuard()).generate(
            _course(),
            _plan(),
            filename="cours.pdf",
            page_start=1,
            page_end=2,
            source_pages=_pages(),
        )


def test_missing_essential_concept_is_rejected() -> None:
    missing = _chapter_json(
        index=0,
        title="Variables",
        concept_ids=[],
        blocks=[
            {
                "type": "paragraph",
                "text": "Texte vague sans couvrir la notion essentielle.",
                "source_pages": [1],
            }
        ],
    )
    llm = ScriptedLLM([missing, missing])
    with pytest.raises(ValueError, match="concept|essentiel"):
        OriginalContentGenerator(llm, LlmBudgetGuard()).generate(
            _course(),
            _plan(),
            filename="cours.pdf",
            page_start=1,
            page_end=2,
            source_pages=_pages(),
        )


def test_budget_ceiling_is_enforced() -> None:
    llm = ScriptedLLM([_short_chapter_var(), _short_chapter_fun()])
    guard = LlmBudgetGuard()
    for _ in range(MAX_LLM_CALLS):
        guard.reserve(1)
    with pytest.raises(ValueError, match="appels"):
        OriginalContentGenerator(llm, guard).generate(
            _course(),
            _plan(),
            filename="cours.pdf",
            page_start=1,
            page_end=2,
            source_pages=_pages(),
        )
    assert llm.calls == 0


def test_near_copy_of_source_paragraph_is_rejected() -> None:
    copied = _chapter_json(
        index=0,
        title="Variables",
        concept_ids=["c_var"],
        blocks=[
            {
                "type": "paragraph",
                "text": "Une variable stocke une valeur. Formule : E = 12.",
                "source_pages": [1],
            },
            {
                "type": "definition",
                "text": "Une variable stocke une valeur.",
                "source_pages": [1],
            },
        ],
    )
    llm = ScriptedLLM([copied, copied])
    with pytest.raises(ValueError, match="paraphrase|copie|original"):
        OriginalContentGenerator(llm, LlmBudgetGuard()).generate(
            _course(),
            _plan(),
            filename="cours.pdf",
            page_start=1,
            page_end=2,
            source_pages=_pages(),
        )


def test_near_copy_retries_then_accepts_original_rewrite() -> None:
    """Une 1ʳᵉ tentative trop proche du PDF peut être corrigée au retry."""
    copied = _chapter_json(
        index=0,
        title="Variables",
        concept_ids=["c_var"],
        blocks=[
            {
                "type": "paragraph",
                "text": "Une variable stocke une valeur. Formule : E = 12.",
                "source_pages": [1],
            },
            {
                "type": "definition",
                "text": "Une variable stocke une valeur.",
                "source_pages": [1],
            },
            {
                "type": "synthesis",
                "text": "Retenir qu'une variable nomme une valeur.",
                "source_pages": [1],
            },
        ],
    )
    fixed = _short_chapter_var()
    fun = _short_chapter_fun()
    llm = ScriptedLLM([copied, fixed, fun])
    document = OriginalContentGenerator(llm, LlmBudgetGuard()).generate(
        _course(),
        _plan(),
        filename="cours.pdf",
        page_start=1,
        page_end=2,
        source_pages=_pages(),
    )
    assert len(document.chapters) == 2
    assert llm.calls >= 2


def test_concept_notes_alone_do_not_trigger_near_copy() -> None:
    """Les notes de concepts (souvent reprises du PDF) ne forment plus le corpus anti-copie."""
    from src.revision.engine.near_copy import is_near_copy, page_source_corpus, whitelist_terms

    course = _course()
    # Notes = phrase longue du cours ; absente des pages fournies à ce test.
    course.concepts[0].notes = (
        "Formulation technique issue de l'analyse : une variable associe un identifiant "
        "durable a une valeur stockee en memoire pendant l'execution."
    )
    pages = _pages()
    page_corpus = page_source_corpus(course, pages)
    paragraph = (
        "Formulation technique issue de l'analyse : une variable associe un identifiant "
        "durable a une valeur stockee en memoire pendant l'execution."
    )
    # Corpus pages-only : la phrase n'est pas dans les pages → pas near-copy.
    assert paragraph.casefold() not in page_corpus.casefold()
    assert not is_near_copy(
        paragraph, page_corpus, whitelist=whitelist_terms(course)
    )


def test_generator_accepts_original_paragraph_despite_long_concept_notes() -> None:
    """Notes ≈ PDF ne doivent pas faire échouer une rédaction originale."""
    course = _course()
    course.concepts[0].notes = (
        "Une variable stocke une valeur. Formule : E = 12. "
        "On retient aussi qu'un identifiant durable pointe vers une case memoire."
    )
    original = _short_chapter_var()
    fun = _short_chapter_fun()
    llm = ScriptedLLM([original, fun])
    document = OriginalContentGenerator(llm, LlmBudgetGuard()).generate(
        course,
        _plan(),
        filename="cours.pdf",
        page_start=1,
        page_end=2,
        source_pages=_pages(),
    )
    assert len(document.chapters) == 2


def test_chapter_prompt_documents_exact_block_types_without_pipe_union() -> None:
    plan = _plan()
    prompt = _build_chapter_prompt(
        course=_course(),
        plan=plan,
        chapter_plan=plan.chapters[0],
        previous_titles=[],
        source_pages=_pages(),
    )

    assert (
        '"type":"paragraph|definition|example|retain|table|diagram|synthesis|question"'
        not in prompt
    )
    assert "paragraph|definition|example|retain|table|diagram|synthesis|question" not in prompt
    assert '"type":"question"' not in prompt
    assert "Interdit : type question" in prompt or "type question" in prompt.casefold()

    for block_type in (
        "paragraph",
        "definition",
        "example",
        "retain",
        "table",
        "diagram",
        "synthesis",
    ):
        assert f'"type":"{block_type}"' in prompt

    assert "from_course" in prompt
    assert "pedagogical_addition" in prompt
    assert "headers" in prompt
    assert "rows" in prompt
    assert "concept_ids_covered" in prompt
    assert "sans Markdown" in prompt
    assert "needs_questions=false" in prompt
