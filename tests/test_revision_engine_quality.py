"""Tests du QualityController Smart Revision V2."""

from __future__ import annotations

import json

import pytest

from src.revision.engine.llm_guard import (
    LlmBudgetGuard,
    MAX_LLM_CALLS,
    MAX_REWRITE_CHAPTERS,
    MAX_TOTAL_NEW_TOKENS,
    TOKENS_REWRITE,
    max_attempts_per_step,
)
from src.revision.engine.models import (
    ChapterDraft,
    ChapterPlan,
    Concept,
    CourseModel,
    CoursePage,
    CourseRelation,
    DefinitionBlock,
    ExampleBlock,
    LearningObjective,
    ParagraphBlock,
    PedagogicalDocument,
    PedagogicalPlan,
    QuestionBlock,
    RetainBlock,
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
            text="Une fonction encapsule un comportement.",
            filename="cours.pdf",
        ),
    ]


def _course(*, contradiction: bool = False) -> CourseModel:
    fun_notes = "Une fonction encapsule un comportement."
    if contradiction:
        fun_notes += (
            "\n[CONTRADICTION] Les pages affirment des signes opposés pour x."
        )
    return CourseModel(
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
                notes=fun_notes,
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


def _plan() -> PedagogicalPlan:
    return PedagogicalPlan(
        language="fr",
        level="beginner",
        sheet_type="short",
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
                depth="short",
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
                depth="short",
            ),
        ],
    )


def _valid_document() -> PedagogicalDocument:
    return PedagogicalDocument(
        filename="cours.pdf",
        page_start=1,
        page_end=2,
        language="fr",
        level="beginner",
        sheet_type="short",
        chapters=[
            ChapterDraft(
                index=0,
                title="Variables",
                concept_ids_covered=["c_var"],
                blocks=[
                    ParagraphBlock(
                        text="On nomme une information pour pouvoir la réutiliser plus tard.",
                        source_pages=[1],
                    ),
                    DefinitionBlock(
                        text="Une variable stocke une valeur.",
                        source_pages=[1],
                    ),
                    ExampleBlock(
                        text="Dans le cours : E = 12.",
                        origin="from_course",
                        source_pages=[1],
                    ),
                    ExampleBlock(
                        text="Comme une boîte étiquetée.",
                        origin="pedagogical_addition",
                        source_pages=[],
                    ),
                    SynthesisBlock(
                        text="Retenir le rôle de nommage d'une variable.",
                        source_pages=[1],
                    ),
                ],
            ),
            ChapterDraft(
                index=1,
                title="Fonctions",
                concept_ids_covered=["c_fun"],
                blocks=[
                    ParagraphBlock(
                        text="Après avoir nommé des valeurs, on regroupe un comportement.",
                        source_pages=[2],
                    ),
                    DefinitionBlock(
                        text="Une fonction encapsule un comportement.",
                        source_pages=[2],
                    ),
                    ExampleBlock(
                        text="Analogie : une recette qui utilise des ingrédients nommés.",
                        origin="pedagogical_addition",
                        source_pages=[],
                    ),
                    SynthesisBlock(
                        text="La fonction s'appuie sur les variables déjà vues.",
                        source_pages=[2],
                    ),
                ],
            ),
        ],
    )


def _rewrite_chapter_json(
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


def _fixed_functions_chapter() -> str:
    return _rewrite_chapter_json(
        index=1,
        title="Fonctions",
        concept_ids=["c_fun"],
        blocks=[
            {
                "type": "paragraph",
                "text": "On organise maintenant un comportement réutilisable autour des valeurs nommées.",
                "source_pages": [2],
            },
            {
                "type": "definition",
                "text": "Une fonction encapsule un comportement.",
                "source_pages": [2],
            },
            {
                "type": "example",
                "text": "Image mentale : un atelier qui transforme des pièces étiquetées.",
                "origin": "pedagogical_addition",
                "source_pages": [],
            },
            {
                "type": "synthesis",
                "text": "Les fonctions prolongent l'usage des variables.",
                "source_pages": [2],
            },
        ],
    )


def test_strips_question_blocks_without_rewrite() -> None:
    llm = ScriptedLLM()
    document = _valid_document()
    with_question = document.model_copy(
        update={
            "chapters": [
                document.chapters[0].model_copy(
                    update={
                        "blocks": list(document.chapters[0].blocks)
                        + [QuestionBlock(text="Pourquoi nommer une valeur ?", source_pages=[1])]
                    }
                ),
                document.chapters[1],
            ]
        }
    )
    reviewed, report = QualityController(llm, LlmBudgetGuard()).review(
        with_question,
        _course(),
        _plan(),
        source_pages=_pages(),
    )
    assert report.ok
    assert llm.calls == 0
    assert all(block.type != "question" for ch in reviewed.chapters for block in ch.blocks)
    assert any(block.type == "definition" for block in reviewed.chapters[0].blocks)


def test_valid_document_passes_without_llm_rewrite() -> None:
    llm = ScriptedLLM()
    guard = LlmBudgetGuard()
    document, report = QualityController(llm, guard).review(
        _valid_document(),
        _course(),
        _plan(),
        source_pages=_pages(),
    )
    assert report.ok is True
    assert report.issues == []
    assert report.rewrite_chapter_indexes == []
    assert len(document.chapters) == 2
    assert llm.calls == 0
    assert guard.calls == 0


def test_missing_essential_concept_fails() -> None:
    document = _valid_document()
    document.chapters[0].concept_ids_covered = []
    document.chapters[0].blocks = [
        ParagraphBlock(text="Texte vague sans couvrir la notion.", source_pages=[1])
    ]
    llm = ScriptedLLM()
    with pytest.raises(ValueError, match="essentiel|concept|qualité"):
        QualityController(llm, LlmBudgetGuard()).review(
            document, _course(), _plan(), source_pages=_pages()
        )


def test_missing_chapter_fails_without_partial_success() -> None:
    document = _valid_document()
    document.chapters = document.chapters[:1]
    with pytest.raises(ValueError, match="chapitre|plan|qualité"):
        QualityController(ScriptedLLM(), LlmBudgetGuard()).review(
            document, _course(), _plan(), source_pages=_pages()
        )


def test_excessive_repetition_triggers_targeted_rewrite() -> None:
    document = _valid_document()
    repeated = "On organise un comportement réutilisable autour des valeurs nommées."
    document.chapters[1].blocks = [
        ParagraphBlock(text=repeated, source_pages=[2]),
        ParagraphBlock(text=repeated, source_pages=[2]),
        ParagraphBlock(text=repeated, source_pages=[2]),
        DefinitionBlock(text="Une fonction encapsule un comportement.", source_pages=[2]),
        SynthesisBlock(text=repeated, source_pages=[2]),
    ]
    llm = ScriptedLLM([_fixed_functions_chapter()])
    guard = LlmBudgetGuard()
    document, report = QualityController(llm, guard).review(
        document, _course(), _plan(), source_pages=_pages()
    )
    assert report.ok is True
    assert 1 in report.rewrite_chapter_indexes
    assert llm.calls == 1
    assert llm.token_budgets == [TOKENS_REWRITE]
    assert guard.calls == 1
    assert guard.reserved_tokens == TOKENS_REWRITE
    texts = [b.text for b in document.chapters[1].blocks if hasattr(b, "text")]
    assert len(set(texts)) > 1


def test_artificial_url_is_detected_and_rewritten() -> None:
    document = _valid_document()
    document.chapters[1].blocks.insert(
        0,
        ParagraphBlock(
            text="Voir https://exemple.invente/docs pour la suite.",
            source_pages=[2],
        ),
    )
    llm = ScriptedLLM([_fixed_functions_chapter()])
    document, report = QualityController(llm, LlmBudgetGuard()).review(
        document, _course(), _plan(), source_pages=_pages()
    )
    assert report.ok is True
    assert 1 in report.rewrite_chapter_indexes
    joined = " ".join(
        getattr(block, "text", "") for block in document.chapters[1].blocks
    )
    assert "https://" not in joined.casefold()


def test_near_copy_is_detected_but_jargon_alone_is_allowed() -> None:
    # Jargon / définition courte du modèle : ne doit pas échouer à lui seul.
    jargon_ok = _valid_document()
    llm = ScriptedLLM()
    document, report = QualityController(llm, LlmBudgetGuard()).review(
        jargon_ok, _course(), _plan(), source_pages=_pages()
    )
    assert report.ok is True

    # Copie longue d'un paragraphe source → réécriture.
    copied = _valid_document()
    copied.chapters[0].blocks[0] = ParagraphBlock(
        text="Une variable stocke une valeur. Formule : E = 12.",
        source_pages=[1],
    )
    fixed_var = _rewrite_chapter_json(
        index=0,
        title="Variables",
        concept_ids=["c_var"],
        blocks=[
            {
                "type": "paragraph",
                "text": "On apprend à associer un nom clair à une information du cours.",
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
                "text": "Comme une boîte étiquetée.",
                "origin": "pedagogical_addition",
                "source_pages": [],
            },
            {
                "type": "synthesis",
                "text": "Retenir le rôle de nommage d'une variable.",
                "source_pages": [1],
            },
        ],
    )
    llm2 = ScriptedLLM([fixed_var])
    document, report = QualityController(llm2, LlmBudgetGuard()).review(
        copied, _course(), _plan(), source_pages=_pages()
    )
    assert report.ok is True
    assert 0 in report.rewrite_chapter_indexes


def test_contradiction_without_invented_resolution_is_accepted() -> None:
    document = _valid_document()
    document.chapters[1].blocks.insert(
        0,
        RetainBlock(
            text="Le cours signale une contradiction sur le signe de x ; elle n'est pas tranchée.",
            source_pages=[2],
        ),
    )
    document, report = QualityController(ScriptedLLM(), LlmBudgetGuard()).review(
        document, _course(contradiction=True), _plan(), source_pages=_pages()
    )
    assert report.ok is True


def test_invented_resolution_of_contradiction_fails_or_is_rewritten() -> None:
    document = _valid_document()
    document.chapters[1].blocks.insert(
        0,
        ParagraphBlock(
            text="En réalité x est positif, donc on ignore l'autre page.",
            source_pages=[2],
        ),
    )
    # Réécriture qui persiste à résoudre → échec final (garde exemples/pages).
    still_bad = _rewrite_chapter_json(
        index=1,
        title="Fonctions",
        concept_ids=["c_fun"],
        blocks=[
            {
                "type": "paragraph",
                "text": "En réalité la vérité est que x reste positif.",
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
                "type": "synthesis",
                "text": "La fonction s'appuie sur les variables déjà vues.",
                "source_pages": [2],
            },
        ],
    )
    llm = ScriptedLLM([still_bad])
    with pytest.raises(ValueError, match="qualité|contradiction|résolution"):
        QualityController(llm, LlmBudgetGuard()).review(
            document,
            _course(contradiction=True),
            _plan(),
            source_pages=_pages(),
        )


def test_rewrite_limit_is_two_chapters() -> None:
    document = _valid_document()
    document.chapters[0].blocks.insert(
        0,
        ParagraphBlock(text="Voir https://a.example/x", source_pages=[1]),
    )
    document.chapters[1].blocks.insert(
        0,
        ParagraphBlock(text="Voir https://b.example/y", source_pages=[2]),
    )
    # Un troisième chapitre défectueux dépasserait la limite si on en ajoutait ;
    # ici 2 chapitres URL → exactement MAX_REWRITE_CHAPTERS réécritures.
    fixed0 = _rewrite_chapter_json(
        index=0,
        title="Variables",
        concept_ids=["c_var"],
        blocks=[
            {
                "type": "paragraph",
                "text": "On introduit le nommage d'une information utile.",
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
                "type": "synthesis",
                "text": "La variable prépare les fonctions.",
                "source_pages": [1],
            },
        ],
    )
    llm = ScriptedLLM([fixed0, _fixed_functions_chapter()])
    document, report = QualityController(llm, LlmBudgetGuard()).review(
        document, _course(), _plan(), source_pages=_pages()
    )
    assert report.ok is True
    assert sorted(report.rewrite_chapter_indexes) == [0, 1]
    assert len(report.rewrite_chapter_indexes) <= MAX_REWRITE_CHAPTERS
    assert llm.calls == 2


def test_persistent_defects_after_rewrite_raise() -> None:
    document = _valid_document()
    document.chapters[1].blocks.insert(
        0,
        ParagraphBlock(text="Voir https://toujours.mauvais/x", source_pages=[2]),
    )
    still_url = _rewrite_chapter_json(
        index=1,
        title="Fonctions",
        concept_ids=["c_fun"],
        blocks=[
            {
                "type": "paragraph",
                "text": "Toujours https://toujours.mauvais/x dans le texte.",
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
                "type": "synthesis",
                "text": "La fonction s'appuie sur les variables déjà vues.",
                "source_pages": [2],
            },
        ],
    )
    with pytest.raises(ValueError, match="qualité"):
        QualityController(ScriptedLLM([still_url]), LlmBudgetGuard()).review(
            document, _course(), _plan(), source_pages=_pages()
        )


def test_budget_exhaustion_raises_explicit_error() -> None:
    document = _valid_document()
    document.chapters[1].blocks.insert(
        0,
        ParagraphBlock(text="Voir https://budget.example/x", source_pages=[2]),
    )
    guard = LlmBudgetGuard()
    for _ in range(MAX_LLM_CALLS):
        guard.reserve(1)
    with pytest.raises(ValueError, match="appels|budget|qualité"):
        QualityController(ScriptedLLM([_fixed_functions_chapter()]), guard).review(
            document, _course(), _plan(), source_pages=_pages()
        )


def test_does_not_return_partial_document_as_valid_on_failure() -> None:
    document = _valid_document()
    document.chapters = []
    with pytest.raises(ValueError):
        QualityController(ScriptedLLM(), LlmBudgetGuard()).review(
            document, _course(), _plan(), source_pages=_pages()
        )


def _guard_with_remaining(*, tokens: int, calls: int) -> LlmBudgetGuard:
    """Construit un garde avec exactement `tokens` / `calls` restants."""
    if tokens < 0 or calls < 0:
        raise AssertionError("remaining budget invalide")
    if tokens > MAX_TOTAL_NEW_TOKENS or calls > MAX_LLM_CALLS:
        raise AssertionError("remaining dépasse les plafonds")
    guard = LlmBudgetGuard()
    tokens_to_burn = MAX_TOTAL_NEW_TOKENS - tokens
    calls_to_burn = MAX_LLM_CALLS - calls
    if calls_to_burn == 0:
        assert tokens_to_burn == 0
        return guard
    # Chaque reserve consomme 1 appel : répartir tokens_to_burn sur calls_to_burn.
    assert tokens_to_burn >= calls_to_burn, (
        "Impossible : trop peu de tokens à brûler pour le nombre d'appels à consommer"
    )
    base = tokens_to_burn // calls_to_burn
    rem = tokens_to_burn % calls_to_burn
    for index in range(calls_to_burn):
        chunk = base + (1 if index < rem else 0)
        guard.reserve(chunk)
    assert guard.remaining_tokens == tokens
    assert guard.remaining_calls == calls
    return guard


def _document_with_url_defect(chapter_index: int = 1) -> PedagogicalDocument:
    document = _valid_document()
    document.chapters[chapter_index].blocks.insert(
        0,
        ParagraphBlock(
            text=f"Voir https://exemple.invente/ch{chapter_index} pour la suite.",
            source_pages=[chapter_index + 1],
        ),
    )
    return document


def test_rewrite_starts_when_retry_budget_fully_available() -> None:
    """768 tokens + 2 appels = attempts * TOKENS_REWRITE → une réécriture peut démarrer."""
    attempts = max_attempts_per_step()
    assert attempts * TOKENS_REWRITE == 768
    document = _document_with_url_defect(1)
    guard = _guard_with_remaining(tokens=768, calls=2)
    llm = ScriptedLLM([_fixed_functions_chapter()])
    reviewed, report = QualityController(llm, guard).review(
        document, _course(), _plan(), source_pages=_pages()
    )
    assert report.ok is True
    assert 1 in report.rewrite_chapter_indexes
    assert llm.calls == 1
    # Succès du premier essai : 1 réservation consommée.
    assert guard.calls == MAX_LLM_CALLS - 2 + 1
    assert guard.reserved_tokens == MAX_TOTAL_NEW_TOKENS - 768 + TOKENS_REWRITE


def test_rewrite_does_not_start_with_767_tokens() -> None:
    document = _document_with_url_defect(1)
    guard = _guard_with_remaining(tokens=767, calls=2)
    llm = ScriptedLLM([_fixed_functions_chapter()])
    with pytest.raises(ValueError, match="qualité"):
        QualityController(llm, guard).review(
            document, _course(), _plan(), source_pages=_pages()
        )
    assert llm.calls == 0
    assert guard.remaining_tokens == 767
    assert guard.remaining_calls == 2


def test_rewrite_does_not_start_with_only_one_call() -> None:
    document = _document_with_url_defect(1)
    guard = _guard_with_remaining(tokens=768, calls=1)
    llm = ScriptedLLM([_fixed_functions_chapter()])
    with pytest.raises(ValueError, match="qualité"):
        QualityController(llm, guard).review(
            document, _course(), _plan(), source_pages=_pages()
        )
    assert llm.calls == 0
    assert guard.remaining_calls == 1


def test_two_rewrites_only_when_budget_covers_each_retry_envelope() -> None:
    """Après une 1ère réécriture réussie, il faut encore 768 tokens / 2 appels."""
    document = _valid_document()
    document.chapters[0].blocks.insert(
        0, ParagraphBlock(text="Voir https://a.example/x", source_pages=[1])
    )
    document.chapters[1].blocks.insert(
        0, ParagraphBlock(text="Voir https://b.example/y", source_pages=[2])
    )
    fixed0 = _rewrite_chapter_json(
        index=0,
        title="Variables",
        concept_ids=["c_var"],
        blocks=[
            {
                "type": "paragraph",
                "text": "On introduit le nommage d'une information utile.",
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
                "type": "synthesis",
                "text": "La variable prépare les fonctions.",
                "source_pages": [1],
            },
        ],
    )
    # Budget : 1ère rewrite (succès 1 essai) + réserve complète pour la 2ᵉ.
    # tokens = 384 + 768 = 1152 ; calls = 1 + 2 = 3.
    guard = _guard_with_remaining(tokens=1152, calls=3)
    llm = ScriptedLLM([fixed0, _fixed_functions_chapter()])
    reviewed, report = QualityController(llm, guard).review(
        document, _course(), _plan(), source_pages=_pages()
    )
    assert report.ok is True
    assert sorted(report.rewrite_chapter_indexes) == [0, 1]
    assert llm.calls == 2

    # Avec seulement 768 tokens / 2 appels : une seule réécriture, puis échec final.
    guard_tight = _guard_with_remaining(tokens=768, calls=2)
    llm_tight = ScriptedLLM([fixed0, _fixed_functions_chapter()])
    with pytest.raises(ValueError, match="qualité"):
        QualityController(llm_tight, guard_tight).review(
            document, _course(), _plan(), source_pages=_pages()
        )
    assert llm_tight.calls == 1
    assert guard_tight.remaining_calls == 1
    assert guard_tight.remaining_tokens == 768 - TOKENS_REWRITE


def test_failed_rewrite_attempt_consumes_retry_budget() -> None:
    document = _document_with_url_defect(1)
    attempts = max_attempts_per_step()
    guard = _guard_with_remaining(
        tokens=attempts * TOKENS_REWRITE,
        calls=attempts,
    )
    # 1ʳᵉ tentative : couverture réelle invalide → retry ; 2ᵉ : chapitre correct.
    bare = _rewrite_chapter_json(
        index=1,
        title="Fonctions",
        concept_ids=["c_fun"],
        blocks=[
            {
                "type": "paragraph",
                "text": "Texte vague sans expliquer la fonction du tout.",
                "source_pages": [2],
            }
        ],
    )
    llm = ScriptedLLM([bare, _fixed_functions_chapter()])
    reviewed, report = QualityController(llm, guard).review(
        document, _course(), _plan(), source_pages=_pages()
    )
    assert report.ok is True
    assert 1 in report.rewrite_chapter_indexes
    assert llm.calls == 2
    assert guard.remaining_calls == 0
    assert guard.remaining_tokens == 0


def test_final_quality_control_always_runs_even_without_rewrite() -> None:
    """Budget insuffisant → pas de rewrite, mais l'évaluation finale reste obligatoire."""
    document = _document_with_url_defect(1)
    guard = _guard_with_remaining(tokens=767, calls=10)
    llm = ScriptedLLM()
    with pytest.raises(ValueError, match="après réécriture|qualité"):
        QualityController(llm, guard).review(
            document, _course(), _plan(), source_pages=_pages()
        )
    assert llm.calls == 0
    # Document toujours défectueux : le contrôle final a bien rejeté.
    assert guard.remaining_tokens == 767
