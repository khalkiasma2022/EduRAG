"""Tests des budgets et retries du garde LLM Smart Revision V2."""

from __future__ import annotations

import pytest

from src.generation.llm import MAX_NEW_TOKENS_LIMIT
from src.revision.engine import llm_guard
from src.revision.engine.llm_guard import (
    LlmBudgetGuard,
    ensure_generation_capacity,
    ensure_pipeline_capacity,
    ensure_prompt_fits,
    max_attempts_per_step,
    pipeline_call_cost,
    pipeline_call_cost_nominal,
    pipeline_token_cost,
    pipeline_token_cost_nominal,
)


def test_step_budget_constants_match_guaranteed_envelope() -> None:
    assert llm_guard.MAX_ANALYSIS_WINDOWS == 10
    assert llm_guard.MAX_CHAPTERS == 10
    assert llm_guard.MAX_CHAPTERS_SHORT == llm_guard.MAX_CHAPTERS
    assert llm_guard.MAX_CHAPTERS_COMPLETE == llm_guard.MAX_CHAPTERS
    assert llm_guard.TOKENS_UNDERSTANDING == 1024
    assert llm_guard.TOKENS_PLAN == 512
    assert llm_guard.TOKENS_CHAPTER == 768
    assert llm_guard.TOKENS_CHAPTER_SHORT == llm_guard.TOKENS_CHAPTER
    assert llm_guard.TOKENS_CHAPTER_COMPLETE == llm_guard.TOKENS_CHAPTER
    assert llm_guard.TOKENS_REWRITE == 384
    assert llm_guard.MAX_RETRIES_PER_CALL == 1
    assert max_attempts_per_step() == 2
    assert llm_guard.MAX_REWRITE_CHAPTERS == 2
    assert llm_guard.MAX_CONCEPTS_PER_CHAPTER == 4
    assert llm_guard.MIN_TOKENS_PER_CONCEPT == 192
    assert llm_guard.MAX_COURSE_CONCEPTS == 40
    assert llm_guard.MAX_LLM_CALLS == 48
    assert llm_guard.MAX_TOTAL_NEW_TOKENS == 39000
    assert llm_guard.MAX_COURSE_TOKENS_PER_WINDOW == 1800
    assert llm_guard.LEGACY_STORE_LEVEL == "intermediate"
    assert llm_guard.LEGACY_STORE_SHEET_TYPE == "complete"


def test_guaranteed_worst_case_pipeline_fits_global_caps() -> None:
    """Pire cas : chaque étape consomme tentative + retry autorisé."""
    tokens = pipeline_token_cost(
        llm_guard.MAX_ANALYSIS_WINDOWS,
        llm_guard.MAX_CHAPTERS,
    )
    calls = pipeline_call_cost(
        llm_guard.MAX_ANALYSIS_WINDOWS,
        llm_guard.MAX_CHAPTERS,
    )
    nominal = pipeline_token_cost_nominal(10, 10)
    assert nominal == 10 * 1024 + 512 + 10 * 768 + 2 * 384  # 19200
    assert tokens == 2 * nominal
    assert tokens == 38400
    assert tokens <= llm_guard.MAX_TOTAL_NEW_TOKENS
    assert pipeline_call_cost_nominal(10, 10) == 23  # 10+1+10+2
    assert calls == 46
    assert calls <= llm_guard.MAX_LLM_CALLS


def test_former_20_20_envelope_is_rejected_under_retry_guarantee() -> None:
    with pytest.raises(ValueError, match="budget|capacité|tokens|appels|fenêtres"):
        ensure_pipeline_capacity(windows=20, chapters=20)


def test_ensure_pipeline_capacity_accepts_reasonable_pdf() -> None:
    ensure_pipeline_capacity(windows=4, chapters=3)


def test_ensure_pipeline_capacity_refuses_too_many_windows() -> None:
    with pytest.raises(ValueError, match="capacité|fenêtres"):
        ensure_pipeline_capacity(
            windows=llm_guard.MAX_ANALYSIS_WINDOWS + 1,
            chapters=llm_guard.MAX_CHAPTERS,
        )


def test_ensure_pipeline_capacity_skip_understanding_ignores_window_cap() -> None:
    too_many = llm_guard.MAX_ANALYSIS_WINDOWS + 5
    with pytest.raises(ValueError, match="capacité|fenêtres"):
        ensure_pipeline_capacity(windows=too_many, chapters=3)
    ensure_pipeline_capacity(
        windows=too_many,
        chapters=3,
        skip_understanding=True,
    )
    full = pipeline_token_cost(too_many, 3)
    skipped = pipeline_token_cost(0, 3)
    assert skipped < full


def test_ensure_generation_capacity_includes_rewrite_and_retry_reserve() -> None:
    guard = LlmBudgetGuard()
    # Understanding+plan au pire cas pour 10 fenêtres.
    for _ in range(10 * max_attempts_per_step()):
        guard.reserve(llm_guard.TOKENS_UNDERSTANDING)
    for _ in range(max_attempts_per_step()):
        guard.reserve(llm_guard.TOKENS_PLAN)

    ensure_generation_capacity(guard, chapters=llm_guard.MAX_CHAPTERS)

    # Consomme la marge restante : le pire cas génération+QC (retries inclus) ne passe plus.
    while guard.remaining_calls >= max_attempts_per_step() * (
        llm_guard.MAX_CHAPTERS + llm_guard.MAX_REWRITE_CHAPTERS
    ):
        guard.reserve(1)
    with pytest.raises(ValueError, match="budget|capacité|chapitres"):
        ensure_generation_capacity(guard, chapters=llm_guard.MAX_CHAPTERS)


def test_guard_reports_remaining_budget() -> None:
    guard = LlmBudgetGuard()
    guard.reserve(1000)
    assert guard.remaining_calls == llm_guard.MAX_LLM_CALLS - 1
    assert guard.remaining_tokens == llm_guard.MAX_TOTAL_NEW_TOKENS - 1000


def test_reserve_accepts_worst_case_full_pipeline() -> None:
    guard = LlmBudgetGuard()
    attempts = max_attempts_per_step()
    step_budgets = (
        [llm_guard.TOKENS_UNDERSTANDING] * llm_guard.MAX_ANALYSIS_WINDOWS * attempts
        + [llm_guard.TOKENS_PLAN] * attempts
        + [llm_guard.TOKENS_CHAPTER] * llm_guard.MAX_CHAPTERS * attempts
        + [llm_guard.TOKENS_REWRITE] * llm_guard.MAX_REWRITE_CHAPTERS * attempts
    )
    for tokens in step_budgets:
        guard.reserve(tokens)
    assert guard.calls == len(step_budgets) == 46
    assert guard.reserved_tokens == 38400


def test_understanding_retry_consumes_two_reservations() -> None:
    guard = LlmBudgetGuard()
    attempts = {"n": 0}

    def flaky() -> str:
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise ValueError("json understanding invalide")
        return '{"themes":[],"concepts":[]}'

    def parse(raw: str) -> dict:
        if "themes" not in raw:
            raise ValueError("parse")
        return {"themes": [], "concepts": []}

    result = guard.run_with_retries(
        flaky, tokens=llm_guard.TOKENS_UNDERSTANDING, parse=parse
    )
    assert result == {"themes": [], "concepts": []}
    assert attempts["n"] == 2
    assert guard.calls == 2
    assert guard.reserved_tokens == 2 * llm_guard.TOKENS_UNDERSTANDING


def test_plan_retry_consumes_two_reservations() -> None:
    guard = LlmBudgetGuard()
    attempts = {"n": 0}

    def flaky() -> str:
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise ValueError("json plan invalide")
        return '{"ok": true}'

    result = guard.run_with_retries(
        flaky,
        tokens=llm_guard.TOKENS_PLAN,
        parse=lambda raw: {"ok": True} if "ok" in raw else (_ for _ in ()).throw(ValueError("x")),
    )
    assert result == {"ok": True}
    assert guard.calls == 2
    assert guard.reserved_tokens == 2 * llm_guard.TOKENS_PLAN


def test_chapter_retry_consumes_two_reservations() -> None:
    guard = LlmBudgetGuard()
    attempts = {"n": 0}

    def flaky() -> str:
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise ValueError("json chapitre invalide")
        return '{"ok": true}'

    result = guard.run_with_retries(
        flaky,
        tokens=llm_guard.TOKENS_CHAPTER,
        parse=lambda raw: {"ok": True} if "ok" in raw else (_ for _ in ()).throw(ValueError("x")),
    )
    assert result == {"ok": True}
    assert guard.calls == 2
    assert guard.reserved_tokens == 2 * llm_guard.TOKENS_CHAPTER


def test_rewrite_retry_consumes_two_reservations() -> None:
    guard = LlmBudgetGuard()
    attempts = {"n": 0}

    def flaky() -> str:
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise ValueError("json rewrite invalide")
        return '{"ok": true}'

    result = guard.run_with_retries(
        flaky,
        tokens=llm_guard.TOKENS_REWRITE,
        parse=lambda raw: {"ok": True} if "ok" in raw else (_ for _ in ()).throw(ValueError("x")),
    )
    assert result == {"ok": True}
    assert guard.calls == 2
    assert guard.reserved_tokens == 2 * llm_guard.TOKENS_REWRITE


def test_budget_exhaustion_during_retries_is_explicit() -> None:
    guard = LlmBudgetGuard()
    # Laisse exactement une réservation possible.
    while guard.remaining_calls > 1:
        guard.reserve(1)

    def always_bad() -> str:
        return "bad"

    def parse(raw: str) -> dict:
        raise ValueError("json invalide")

    with pytest.raises(ValueError, match="json invalide|appels|tokens"):
        guard.run_with_retries(always_bad, tokens=100, parse=parse)

    assert guard.calls == llm_guard.MAX_LLM_CALLS or guard.reserved_tokens <= (
        llm_guard.MAX_TOTAL_NEW_TOKENS
    )


def test_reserve_rejects_above_global_token_cap() -> None:
    guard = LlmBudgetGuard()
    with pytest.raises(ValueError, match="tokens"):
        remaining = llm_guard.MAX_TOTAL_NEW_TOKENS
        while remaining > 0:
            chunk = min(remaining, MAX_NEW_TOKENS_LIMIT)
            guard.reserve(chunk)
            remaining -= chunk
        guard.reserve(1)


def test_reserve_rejects_when_sum_would_exceed_global_cap() -> None:
    guard = LlmBudgetGuard()
    for _ in range(19):
        guard.reserve(2048)  # 19×2048 = 38912
    with pytest.raises(ValueError, match="tokens"):
        guard.reserve(1500)


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
