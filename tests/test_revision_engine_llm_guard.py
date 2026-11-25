"""Tests des budgets et retries du garde LLM Smart Revision V2."""

from __future__ import annotations

import pytest

from src.generation.llm import MAX_NEW_TOKENS_LIMIT
from src.revision.engine import llm_guard
from src.revision.engine.llm_guard import LlmBudgetGuard, ensure_prompt_fits


def test_step_budget_constants_match_spec() -> None:
    assert llm_guard.MAX_ANALYSIS_WINDOWS == 5
    assert llm_guard.MAX_CHAPTERS_SHORT == 4
    assert llm_guard.MAX_CHAPTERS_COMPLETE == 6
    assert llm_guard.TOKENS_UNDERSTANDING == 384
    assert llm_guard.TOKENS_PLAN == 512
    assert llm_guard.TOKENS_CHAPTER_SHORT == 384
    assert llm_guard.TOKENS_CHAPTER_COMPLETE == 640
    assert llm_guard.TOKENS_REWRITE == 384
    assert llm_guard.MAX_RETRIES_PER_CALL == 1
    assert llm_guard.MAX_REWRITE_CHAPTERS == 2
    assert llm_guard.MAX_LLM_CALLS == 16
    assert llm_guard.MAX_TOTAL_NEW_TOKENS == 7500
    assert llm_guard.MAX_COURSE_TOKENS_PER_WINDOW == 1800


def test_short_and_complete_theoretical_budgets_fit_global_cap() -> None:
    short_total = (
        llm_guard.MAX_ANALYSIS_WINDOWS * llm_guard.TOKENS_UNDERSTANDING
        + llm_guard.TOKENS_PLAN
        + llm_guard.MAX_CHAPTERS_SHORT * llm_guard.TOKENS_CHAPTER_SHORT
        + llm_guard.MAX_REWRITE_CHAPTERS * llm_guard.TOKENS_REWRITE
    )
    complete_total = (
        llm_guard.MAX_ANALYSIS_WINDOWS * llm_guard.TOKENS_UNDERSTANDING
        + llm_guard.TOKENS_PLAN
        + llm_guard.MAX_CHAPTERS_COMPLETE * llm_guard.TOKENS_CHAPTER_COMPLETE
        + llm_guard.MAX_REWRITE_CHAPTERS * llm_guard.TOKENS_REWRITE
    )
    assert short_total == 4736
    assert complete_total == 7040
    assert short_total <= llm_guard.MAX_TOTAL_NEW_TOKENS
    assert complete_total <= llm_guard.MAX_TOTAL_NEW_TOKENS


def test_reserve_accepts_complete_theoretical_maximum() -> None:
    """Le plafond théorique complete (7040) est accepté via réservations par étape."""
    guard = LlmBudgetGuard()
    step_budgets = (
        [llm_guard.TOKENS_UNDERSTANDING] * llm_guard.MAX_ANALYSIS_WINDOWS
        + [llm_guard.TOKENS_PLAN]
        + [llm_guard.TOKENS_CHAPTER_COMPLETE] * llm_guard.MAX_CHAPTERS_COMPLETE
        + [llm_guard.TOKENS_REWRITE] * llm_guard.MAX_REWRITE_CHAPTERS
    )
    for tokens in step_budgets:
        guard.reserve(tokens)
    assert guard.calls == len(step_budgets)
    assert guard.reserved_tokens == 7040


def test_reserve_rejects_above_global_token_cap() -> None:
    guard = LlmBudgetGuard()
    with pytest.raises(ValueError, match="tokens"):
        # Un seul appel ne peut dépasser MAX_NEW_TOKENS_LIMIT ; on réserve
        # jusqu'au plafond global puis on tente un dépassement.
        remaining = llm_guard.MAX_TOTAL_NEW_TOKENS
        while remaining > 0:
            chunk = min(remaining, MAX_NEW_TOKENS_LIMIT)
            guard.reserve(chunk)
            remaining -= chunk
        guard.reserve(1)


def test_reserve_rejects_when_sum_would_exceed_global_cap() -> None:
    guard = LlmBudgetGuard()
    # 7000 via chunks ≤ MAX_NEW_TOKENS_LIMIT, puis 600 qui ferait dépasser 7500.
    for _ in range(3):
        guard.reserve(2048)
    guard.reserve(856)  # 3×2048 + 856 = 7000
    with pytest.raises(ValueError, match="tokens"):
        guard.reserve(600)


def test_reserve_rejects_invalid_step_budget() -> None:
    guard = LlmBudgetGuard()
    with pytest.raises(ValueError, match="max_new_tokens"):
        guard.reserve(0)
    with pytest.raises(ValueError, match="max_new_tokens"):
        guard.reserve(MAX_NEW_TOKENS_LIMIT + 1)


def test_reserve_rejects_when_call_cap_reached() -> None:
    guard = LlmBudgetGuard()
    for _ in range(llm_guard.MAX_LLM_CALLS):
        guard.reserve(1)
    with pytest.raises(ValueError, match="appels"):
        guard.reserve(1)


def test_retries_consume_both_call_and_token_budgets() -> None:
    guard = LlmBudgetGuard()
    attempts = {"n": 0}

    def flaky() -> str:
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise ValueError("json invalide")
        return '{"ok": true}'

    def parse(raw: str) -> dict:
        if "ok" not in raw:
            raise ValueError("parse")
        return {"ok": True}

    result = guard.run_with_retries(flaky, tokens=384, parse=parse)
    assert result == {"ok": True}
    assert attempts["n"] == 2
    assert guard.calls == 2
    assert guard.reserved_tokens == 768


def test_retries_exhausted_raise_without_success() -> None:
    guard = LlmBudgetGuard()

    def always_bad() -> str:
        return "not-json"

    def parse(raw: str) -> dict:
        raise ValueError("json invalide")

    with pytest.raises(ValueError, match="json invalide"):
        guard.run_with_retries(always_bad, tokens=100, parse=parse)

    assert guard.calls == 1 + llm_guard.MAX_RETRIES_PER_CALL
    assert guard.reserved_tokens == 100 * (1 + llm_guard.MAX_RETRIES_PER_CALL)


class _TokenIds:
    def __init__(self, count: int) -> None:
        self.shape = (-1, count)


class _FakeTokenizer:
    def __init__(self, prompt_tokens: int) -> None:
        self.prompt_tokens = prompt_tokens
        self.chat_template = None

    def __call__(self, prompt: str, return_tensors: str = "pt") -> dict[str, _TokenIds]:
        return {"input_ids": _TokenIds(self.prompt_tokens)}


class _FakeConfig:
    def __init__(self, max_position_embeddings: int) -> None:
        self.max_position_embeddings = max_position_embeddings
        self.use_sliding_window = False
        self.rope_parameters = {"rope_type": "default"}


class _FakeModel:
    def __init__(self, limit: int) -> None:
        self.config = _FakeConfig(limit)


class _FakeLLM:
    def __init__(self, prompt_tokens: int, limit: int) -> None:
        self.tokenizer = _FakeTokenizer(prompt_tokens)
        self.model = _FakeModel(limit)

    def is_loaded(self) -> bool:
        return True


def test_ensure_prompt_fits_accepts_when_total_fits() -> None:
    llm = _FakeLLM(prompt_tokens=10, limit=20)
    measured = ensure_prompt_fits(llm, "prompt", max_new_tokens=5)
    assert measured["prompt_tokens"] == 10
    assert measured["reserved_tokens"] == 5
    assert measured["total_tokens"] == 15


def test_ensure_prompt_fits_refuses_without_truncation() -> None:
    llm = _FakeLLM(prompt_tokens=10, limit=14)
    prompt = "cours trop long"
    with pytest.raises(ValueError, match="moins de pages"):
        ensure_prompt_fits(llm, prompt, max_new_tokens=5)
    assert prompt == "cours trop long"
