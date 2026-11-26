"""Budgets, retries et contrôle de contexte pour le moteur Smart Revision V2."""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar

from src.generation.llm import MAX_NEW_TOKENS_LIMIT
from src.revision.sheet import (
    _check_prompt_text,
    _check_reserved_tokens,
    _context_limit,
    _count_prompt_tokens,
    _require_loaded_model,
)

# ---------------------------------------------------------------------------
# Enveloppe GARANTIE (pire cas : chaque étape utilise tentative + retry).
#
# attempts = 1 + MAX_RETRIES_PER_CALL = 2
# tokens_worst = attempts * (W*384 + 512 + C*640 + R*384)
# calls_worst  = attempts * (W + 1 + C + R)
#
# Avec W=C=10 et R=2 :
#   nominal = 3840 + 512 + 6400 + 768 = 11520
#   worst   = 2 * 11520 = 23040 <= 24000
#   calls   = 2 * (10 + 1 + 10 + 2) = 46 <= 48
# Marge restante au pire cas : 960 tokens et 2 appels.
#
# 20 fenêtres + 20 chapitres n'est PAS garanti avec retries
# (worst ≈ 43520 tokens / 86 appels).
# ---------------------------------------------------------------------------
MAX_COURSE_TOKENS_PER_WINDOW = 1800
MAX_ANALYSIS_WINDOWS = 10
MAX_CHAPTERS = 10
MAX_CONCEPTS_PER_CHAPTER = 4
TOKENS_UNDERSTANDING = 384
TOKENS_PLAN = 512
TOKENS_CHAPTER = 640
# Alias legacy (tests / anciens imports).
MAX_CHAPTERS_SHORT = MAX_CHAPTERS
MAX_CHAPTERS_COMPLETE = MAX_CHAPTERS
TOKENS_CHAPTER_SHORT = TOKENS_CHAPTER
TOKENS_CHAPTER_COMPLETE = TOKENS_CHAPTER
TOKENS_REWRITE = 384
MAX_RETRIES_PER_CALL = 1
MAX_REWRITE_CHAPTERS = 2
MAX_LLM_CALLS = 48
MAX_TOTAL_NEW_TOKENS = 24000

# Densité pédagogique : 640 tokens / 4 concepts ≈ 160 tokens utiles par notion.
# Au-delà, la génération produit typiquement des fiches incomplètes.
# Capacité cours sans densification forcée : 10 × 4 = 40 concepts.
MIN_TOKENS_PER_CONCEPT = TOKENS_CHAPTER // MAX_CONCEPTS_PER_CHAPTER
MAX_COURSE_CONCEPTS = MAX_CHAPTERS * MAX_CONCEPTS_PER_CHAPTER

# Valeurs techniques persistées (compatibilité SQLite) — non exposées à l'UI.
LEGACY_STORE_LEVEL = "intermediate"
LEGACY_STORE_SHEET_TYPE = "complete"

T = TypeVar("T")


def max_attempts_per_step() -> int:
    """Nombre maximal de réservations par étape LLM (essai + retries)."""
    return 1 + MAX_RETRIES_PER_CALL


def pipeline_token_cost_nominal(
    windows: int,
    chapters: int,
    *,
    rewrites: int = MAX_REWRITE_CHAPTERS,
) -> int:
    """Coût nominal (1 tentative / étape) en nouveaux tokens."""
    if windows < 0 or chapters < 0 or rewrites < 0:
        raise ValueError("Les tailles de pipeline doivent être positives ou nulles.")
    return (
        windows * TOKENS_UNDERSTANDING
        + TOKENS_PLAN
        + chapters * TOKENS_CHAPTER
        + rewrites * TOKENS_REWRITE
    )


def pipeline_call_cost_nominal(
    windows: int,
    chapters: int,
    *,
    rewrites: int = MAX_REWRITE_CHAPTERS,
) -> int:
    """Coût nominal (1 tentative / étape) en appels LLM."""
    if windows < 0 or chapters < 0 or rewrites < 0:
        raise ValueError("Les tailles de pipeline doivent être positives ou nulles.")
    return windows + 1 + chapters + rewrites


def pipeline_token_cost(
    windows: int,
    chapters: int,
    *,
    rewrites: int = MAX_REWRITE_CHAPTERS,
) -> int:
    """Coût GARANTI pire cas (retries inclus) en nouveaux tokens."""
    return max_attempts_per_step() * pipeline_token_cost_nominal(
        windows, chapters, rewrites=rewrites
    )


def pipeline_call_cost(
    windows: int,
    chapters: int,
    *,
    rewrites: int = MAX_REWRITE_CHAPTERS,
) -> int:
    """Coût GARANTI pire cas (retries inclus) en appels LLM."""
    return max_attempts_per_step() * pipeline_call_cost_nominal(
        windows, chapters, rewrites=rewrites
    )


def ensure_pipeline_capacity(
    windows: int,
    chapters: int = MAX_CHAPTERS,
    *,
    rewrites: int = MAX_REWRITE_CHAPTERS,
) -> None:
    """Refuse avant démarrage si le pire cas (retries inclus) dépasse les plafonds."""
    if not isinstance(windows, int) or isinstance(windows, bool) or windows < 1:
        raise ValueError("Les fenêtres du cours sont invalides.")
    if not isinstance(chapters, int) or isinstance(chapters, bool) or chapters < 1:
        raise ValueError("Le nombre de chapitres est invalide.")
    if windows > MAX_ANALYSIS_WINDOWS:
        raise ValueError(
            "Le cours sélectionné dépasse la capacité d'analyse "
            f"({MAX_ANALYSIS_WINDOWS} fenêtres). "
            "Réduisez la taille du document ou découpez le cours."
        )
    if chapters > MAX_CHAPTERS:
        raise ValueError(
            "Le plan pédagogique dépasse la capacité de génération "
            f"({MAX_CHAPTERS} chapitres)."
        )

    tokens = pipeline_token_cost(windows, chapters, rewrites=rewrites)
    calls = pipeline_call_cost(windows, chapters, rewrites=rewrites)
    if tokens > MAX_TOTAL_NEW_TOKENS or calls > MAX_LLM_CALLS:
        raise ValueError(
            "Le cours sélectionné dépasse le budget LLM garanti pour une fiche "
            f"complète avec retries (besoin ≤ {tokens} tokens / {calls} appels ; "
            f"plafonds {MAX_TOTAL_NEW_TOKENS} tokens / {MAX_LLM_CALLS} appels). "
            "Réduisez la taille du document."
        )


def ensure_generation_capacity(guard: LlmBudgetGuard, chapters: int) -> None:
    """Vérifie le budget restant pour génération + QC, retries inclus."""
    if not isinstance(chapters, int) or isinstance(chapters, bool) or chapters < 1:
        raise ValueError("Le nombre de chapitres est invalide.")
    if chapters > MAX_CHAPTERS:
        raise ValueError(
            "Le plan pédagogique dépasse la capacité de génération "
            f"({MAX_CHAPTERS} chapitres)."
        )

    attempts = max_attempts_per_step()
    need_tokens = attempts * (
        chapters * TOKENS_CHAPTER + MAX_REWRITE_CHAPTERS * TOKENS_REWRITE
    )
    need_calls = attempts * (chapters + MAX_REWRITE_CHAPTERS)
    if need_calls > guard.remaining_calls or need_tokens > guard.remaining_tokens:
        raise ValueError(
            "Le budget LLM restant ne permet pas de générer une fiche complète "
            f"({chapters} chapitres + réserve QC, retries inclus). "
            f"Reste : {guard.remaining_tokens} tokens / {guard.remaining_calls} appels ; "
            f"besoin garanti : {need_tokens} tokens / {need_calls} appels."
        )


def ensure_course_concept_capacity(concept_count: int) -> None:
    """Refuse un cours trop vaste pour 10×4 notions sans perte ni densification."""
    if not isinstance(concept_count, int) or isinstance(concept_count, bool):
        raise ValueError("Le nombre de concepts est invalide.")
    if concept_count < 0:
        raise ValueError("Le nombre de concepts est invalide.")
    if concept_count > MAX_COURSE_CONCEPTS:
        raise ValueError(
            f"Le cours contient {concept_count} concepts, au-delà de la capacité "
            f"pédagogique garantie ({MAX_COURSE_CONCEPTS} = "
            f"{MAX_CHAPTERS} chapitres × {MAX_CONCEPTS_PER_CHAPTER} concepts, "
            f"budget {TOKENS_CHAPTER} tokens/chapitre). "
            "Découpez le document : aucun concept ne sera omis silencieusement."
        )


def ensure_chapter_density(concept_count: int) -> None:
    """Refuse un chapitre trop dense pour le budget de génération (640 tokens)."""
    if not isinstance(concept_count, int) or isinstance(concept_count, bool):
        raise ValueError("Le nombre de concepts du chapitre est invalide.")
    if concept_count < 0:
        raise ValueError("Le nombre de concepts du chapitre est invalide.")
    if concept_count > MAX_CONCEPTS_PER_CHAPTER:
        raise ValueError(
            f"Densité pédagogique excessive : {concept_count} concepts dans un "
            f"chapitre (max {MAX_CONCEPTS_PER_CHAPTER}). "
            f"Le budget de génération ({TOKENS_CHAPTER} tokens, soit environ "
            f"{MIN_TOKENS_PER_CONCEPT} tokens/concept) est insuffisant pour une "
            "explication complète. Découpez le cours."
        )
    needed = concept_count * MIN_TOKENS_PER_CONCEPT
    if concept_count > 0 and needed > TOKENS_CHAPTER:
        raise ValueError(
            f"Budget insuffisant pour un chapitre dense : {concept_count} concepts "
            f"exigeraient environ {needed} tokens, plafond {TOKENS_CHAPTER}."
        )


class LlmBudgetGuard:
    """Fait respecter simultanément le plafond d'appels et le plafond de tokens."""

    def __init__(self) -> None:
        self.calls = 0
        self.reserved_tokens = 0

    @property
    def remaining_calls(self) -> int:
        return max(0, MAX_LLM_CALLS - self.calls)

    @property
    def remaining_tokens(self) -> int:
        return max(0, MAX_TOTAL_NEW_TOKENS - self.reserved_tokens)

    def reserve(self, tokens: int) -> None:
        """Réserve un appel et son budget de nouveaux tokens."""
        if isinstance(tokens, bool) or not isinstance(tokens, int):
            raise ValueError(
                f"max_new_tokens doit être compris entre 1 et {MAX_NEW_TOKENS_LIMIT}."
            )
        if tokens <= 0 or tokens > MAX_NEW_TOKENS_LIMIT:
            raise ValueError(
                f"max_new_tokens doit être compris entre 1 et {MAX_NEW_TOKENS_LIMIT}."
            )
        if self.calls >= MAX_LLM_CALLS:
            raise ValueError(
                f"Le plafond d'appels LLM ({MAX_LLM_CALLS}) est atteint."
            )
        if self.reserved_tokens + tokens > MAX_TOTAL_NEW_TOKENS:
            raise ValueError(
                f"Le plafond de tokens générés ({MAX_TOTAL_NEW_TOKENS}) est atteint."
            )
        self.calls += 1
        self.reserved_tokens += tokens

    def run_with_retries(
        self,
        produce: Callable[[], str],
        tokens: int,
        parse: Callable[[str], T],
    ) -> T:
        """Exécute un appel LLM avec retries ; chaque tentative consomme les budgets."""
        attempts = max_attempts_per_step()
        last_error: Exception | None = None
        for _ in range(attempts):
            self.reserve(tokens)
            try:
                raw = produce()
                return parse(raw)
            except Exception as error:  # noqa: BLE001 — retries bornés sur parse/LLM
                last_error = error
        assert last_error is not None
        raise last_error


def ensure_prompt_fits(llm: object, prompt: str, max_new_tokens: int) -> dict[str, int]:
    """Vérifie qu'un appel tient dans le contexte, sans tronquer le prompt."""
    _require_loaded_model(llm)
    _check_reserved_tokens(max_new_tokens)
    _check_prompt_text(prompt)
    limit = _context_limit(llm.model)
    prompt_tokens = _count_prompt_tokens(llm.tokenizer, prompt)
    total_tokens = prompt_tokens + max_new_tokens
    if total_tokens > limit:
        raise ValueError(
            "Le cours sélectionné est trop volumineux. "
            "Sélectionnez moins de pages."
        )
    return {
        "prompt_tokens": prompt_tokens,
        "reserved_tokens": max_new_tokens,
        "context_limit": limit,
        "total_tokens": total_tokens,
    }
