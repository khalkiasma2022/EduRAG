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

# Limites numériques V2 (spec §8.3) — maximums, pas des cibles de consommation.
MAX_COURSE_TOKENS_PER_WINDOW = 1800
MAX_ANALYSIS_WINDOWS = 5
MAX_CHAPTERS_SHORT = 4
MAX_CHAPTERS_COMPLETE = 6
TOKENS_UNDERSTANDING = 384
TOKENS_PLAN = 512
TOKENS_CHAPTER_SHORT = 384
TOKENS_CHAPTER_COMPLETE = 640
TOKENS_REWRITE = 384
MAX_RETRIES_PER_CALL = 1
MAX_REWRITE_CHAPTERS = 2
MAX_LLM_CALLS = 16
MAX_TOTAL_NEW_TOKENS = 7500

T = TypeVar("T")


class LlmBudgetGuard:
    """Fait respecter simultanément le plafond d'appels et le plafond de tokens."""

    def __init__(self) -> None:
        self.calls = 0
        self.reserved_tokens = 0

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
        attempts = 1 + MAX_RETRIES_PER_CALL
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
