"""Tests du routeur d'intentions du chatbot.

Le vrai modèle Qwen n'est pas chargé : on injecte un faux LLM.
"""

from __future__ import annotations

import json

import pytest

from src.tutor.intent_router import (
    GENERAL_CHAT,
    COURSE_QUESTION,
    WEB_QUESTION,
    IntentRouter,
    RouteDecision,
)


class ScriptedLLM:
    """LLM de test : file d'attente de réponses ou exception."""

    def __init__(
        self,
        replies: list[str] | None = None,
        *,
        error: Exception | None = None,
    ) -> None:
        self.replies = list(replies or [])
        self.error = error
        self.prompts: list[str] = []
        self.max_new_tokens_calls: list[int] = []

    def generate(self, prompt: str, max_new_tokens: int = 512) -> str:
        self.prompts.append(prompt)
        self.max_new_tokens_calls.append(max_new_tokens)
        if self.error is not None:
            raise self.error
        if not self.replies:
            raise RuntimeError("Plus de réponses scriptées.")
        return self.replies.pop(0)


def _json_intent(intent: str, confidence: float = 0.9, reason: str = "test") -> str:
    return json.dumps(
        {"intent": intent, "confidence": confidence, "reason": reason},
        ensure_ascii=False,
    )


def test_classify_course_question() -> None:
    llm = ScriptedLLM([_json_intent(COURSE_QUESTION)])
    router = IntentRouter(llm)

    decision = router.classify("Explique le concept de virtualisation dans mon cours.")

    assert isinstance(decision, RouteDecision)
    assert decision.intent == COURSE_QUESTION
    assert decision.used_fallback is False
    assert 0.0 <= decision.confidence <= 1.0
    assert llm.prompts and "virtualisation" in llm.prompts[0]


def test_classify_web_question() -> None:
    llm = ScriptedLLM([_json_intent(WEB_QUESTION)])
    router = IntentRouter(llm)

    decision = router.classify("Quelles sont les dernières nouveautés de Python ?")

    assert decision.intent == WEB_QUESTION
    assert decision.used_fallback is False


def test_classify_general_chat() -> None:
    llm = ScriptedLLM([_json_intent(GENERAL_CHAT)])
    router = IntentRouter(llm)

    decision = router.classify("Bonjour")

    assert decision.intent == GENERAL_CHAT
    assert decision.used_fallback is False


def test_short_messages_can_use_llm_or_fallback_safely() -> None:
    llm = ScriptedLLM([_json_intent(GENERAL_CHAT, confidence=0.95)])
    router = IntentRouter(llm)

    decision = router.classify("Merci")

    assert decision.intent == GENERAL_CHAT
    assert isinstance(decision.reason, str)


def test_ambiguous_question_returns_valid_intent() -> None:
    llm = ScriptedLLM([_json_intent(COURSE_QUESTION, confidence=0.55, reason="ambigu")])
    router = IntentRouter(llm)

    decision = router.classify("Peux-tu m'expliquer ça ?")

    assert decision.intent in {COURSE_QUESTION, WEB_QUESTION, GENERAL_CHAT}


def test_invalid_json_uses_fallback_without_raising() -> None:
    llm = ScriptedLLM(["ceci n'est pas du JSON"])
    router = IntentRouter(llm)

    decision = router.classify("Bonjour")

    assert decision.intent == GENERAL_CHAT
    assert decision.used_fallback is True


def test_llm_error_uses_fallback_without_raising() -> None:
    llm = ScriptedLLM(error=RuntimeError("GPU HS"))
    router = IntentRouter(llm)

    decision = router.classify("Résume le chapitre 2 de mon PDF.")

    assert decision.intent == COURSE_QUESTION
    assert decision.used_fallback is True


def test_unknown_intent_uses_fallback() -> None:
    llm = ScriptedLLM([_json_intent("SMALL_TALK")])
    router = IntentRouter(llm)

    decision = router.classify("Salut")

    assert decision.intent in {COURSE_QUESTION, WEB_QUESTION, GENERAL_CHAT}
    assert decision.used_fallback is True


def test_history_none_is_accepted() -> None:
    llm = ScriptedLLM([_json_intent(GENERAL_CHAT)])
    router = IntentRouter(llm)

    decision = router.classify("Comment vas-tu ?", history=None)

    assert decision.intent == GENERAL_CHAT
    assert decision.used_fallback is False


def test_history_is_included_in_prompt_when_provided() -> None:
    llm = ScriptedLLM([_json_intent(GENERAL_CHAT)])
    router = IntentRouter(llm)
    history = [
        {"role": "user", "content": "Explique IaaS"},
        {"role": "assistant", "content": "IaaS est..."},
    ]

    decision = router.classify("Peux-tu reformuler ta réponse ?", history=history)

    assert decision.intent == GENERAL_CHAT
    prompt = llm.prompts[0]
    assert "IaaS" in prompt
    assert "reformuler" in prompt.lower() or "reformuler" in prompt


def test_web_heuristic_fallback_on_invalid_json() -> None:
    llm = ScriptedLLM(["{broken"])
    router = IntentRouter(llm)

    decision = router.classify("Trouve-moi la documentation officielle de FastAPI.")

    assert decision.intent == WEB_QUESTION
    assert decision.used_fallback is True


def test_default_fallback_is_general_for_ambiguous_failure() -> None:
    llm = ScriptedLLM(["not-json"])
    router = IntentRouter(llm)

    decision = router.classify("Peux-tu m'expliquer davantage ce point ?")

    assert decision.intent == GENERAL_CHAT
    assert decision.used_fallback is True


def test_fallback_general_explanations_without_course_markers() -> None:
    llm = ScriptedLLM(["not-json"])
    router = IntentRouter(llm)

    assert router.classify("Bonjour").intent == GENERAL_CHAT
    assert router.classify("Explique-moi Python").intent == GENERAL_CHAT
    assert (
        router.classify("Donne-moi trois exemples de programmation").intent
        == GENERAL_CHAT
    )


def test_fallback_course_when_pdf_or_chapter_mentioned() -> None:
    llm = ScriptedLLM(["not-json"])
    router = IntentRouter(llm)

    assert router.classify("Explique la page 5 de mon PDF").intent == COURSE_QUESTION
    assert router.classify("Résume mon chapitre 2").intent == COURSE_QUESTION


def test_fallback_web_for_news_and_official_docs() -> None:
    llm = ScriptedLLM(["not-json"])
    router = IntentRouter(llm)

    assert (
        router.classify("Quelles sont les dernières nouveautés de Python ?").intent
        == WEB_QUESTION
    )
    assert (
        router.classify("Trouve la documentation officielle de React").intent
        == WEB_QUESTION
    )


def test_fallback_follow_up_inherits_course_from_history() -> None:
    llm = ScriptedLLM(["not-json"])
    router = IntentRouter(llm)
    history = [
        {"role": "user", "content": "Explique la page 5 de mon PDF."},
        {"role": "assistant", "content": "La page 5…"},
    ]

    decision = router.classify("Donne-moi trois exemples.", history=history)

    assert decision.intent == COURSE_QUESTION
    assert decision.used_fallback is True


def test_fallback_follow_up_inherits_general_from_history() -> None:
    llm = ScriptedLLM(["not-json"])
    router = IntentRouter(llm)
    history = [
        {"role": "user", "content": "Explique-moi Python."},
        {"role": "assistant", "content": "Python est…"},
    ]

    decision = router.classify("Donne-moi trois exemples.", history=history)

    assert decision.intent == GENERAL_CHAT
    assert decision.used_fallback is True


def test_json_in_markdown_fence_is_accepted() -> None:
    payload = _json_intent(WEB_QUESTION)
    llm = ScriptedLLM([f"```json\n{payload}\n```"])
    router = IntentRouter(llm)

    decision = router.classify("Donne-moi des ressources fiables pour apprendre React.")

    assert decision.intent == WEB_QUESTION
    assert decision.used_fallback is False


def test_empty_question_falls_back_to_general_chat() -> None:
    llm = ScriptedLLM([_json_intent(COURSE_QUESTION)])
    router = IntentRouter(llm)

    decision = router.classify("   ")

    assert decision.intent == GENERAL_CHAT
    assert decision.used_fallback is True
    assert llm.prompts == []


def test_confidence_out_of_range_is_clamped() -> None:
    llm = ScriptedLLM([_json_intent(COURSE_QUESTION, confidence=3.5)])
    router = IntentRouter(llm)

    decision = router.classify("Quelle est la différence entre IaaS et PaaS dans mon document ?")

    assert decision.intent == COURSE_QUESTION
    assert decision.confidence == 1.0


def test_follow_up_classification_uses_history_in_prompt() -> None:
    llm = ScriptedLLM([_json_intent(COURSE_QUESTION, reason="suivi cours")])
    router = IntentRouter(llm)
    history = [
        {"role": "user", "content": "Explique le concept de virtualisation dans mon cours."},
        {"role": "assistant", "content": "La virtualisation…"},
    ]

    decision = router.classify("Donne-moi trois exemples.", history=history)

    assert decision.intent == COURSE_QUESTION
    assert "virtualisation" in llm.prompts[0]
    assert "Donne-moi trois exemples." in llm.prompts[0]
    assert "suivi" in llm.prompts[0].lower() or "historique" in llm.prompts[0].lower()


def test_follow_up_fallback_inherits_course_from_history() -> None:
    llm = ScriptedLLM(["not-json"])
    router = IntentRouter(llm)
    history = [
        {"role": "user", "content": "Résume le chapitre 2 de mon PDF."},
        {"role": "assistant", "content": "Le chapitre 2…"},
    ]

    decision = router.classify("Lequel est le plus simple ?", history=history)

    assert decision.intent == COURSE_QUESTION
    assert decision.used_fallback is True
