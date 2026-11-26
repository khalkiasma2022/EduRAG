"""Tests de DirectChatResponder.

Le vrai modèle Qwen n'est pas chargé : on injecte un faux LLM.
"""

from __future__ import annotations

import pytest

from src.tutor.direct_chat import DirectChatResponder


class ScriptedLLM:
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


def test_reply_returns_answer_and_empty_sources() -> None:
    llm = ScriptedLLM(["Bonjour ! Je suis EduRAG."])
    responder = DirectChatResponder(llm)

    result = responder.reply("Bonjour")

    assert result["answer"] == "Bonjour ! Je suis EduRAG."
    assert result["sources"] == []
    assert llm.prompts and "Bonjour" in llm.prompts[0]


def test_reply_uses_french_instruction_for_french_message() -> None:
    llm = ScriptedLLM(["Avec plaisir."])
    responder = DirectChatResponder(llm)

    responder.reply("Merci beaucoup")

    prompt = llm.prompts[0].lower()
    assert "français" in prompt or "francais" in prompt


def test_reply_uses_english_instruction_for_english_message() -> None:
    llm = ScriptedLLM(["You're welcome."])
    responder = DirectChatResponder(llm)

    responder.reply("Thank you very much")

    prompt = llm.prompts[0].lower()
    assert "english" in prompt


def test_reply_never_asks_for_pdf_sources() -> None:
    llm = ScriptedLLM(["Je vais reformuler."])
    responder = DirectChatResponder(llm)

    result = responder.reply("Peux-tu reformuler ta réponse ?")

    prompt = llm.prompts[0].lower()
    assert "filename" not in prompt
    assert "page_number" not in prompt
    assert result["sources"] == []


def test_llm_error_raises_runtime_error() -> None:
    llm = ScriptedLLM(error=RuntimeError("GPU HS"))
    responder = DirectChatResponder(llm)

    with pytest.raises(RuntimeError, match="génération"):
        responder.reply("Comment vas-tu ?")


def test_empty_llm_answer_raises_runtime_error() -> None:
    llm = ScriptedLLM(["   "])
    responder = DirectChatResponder(llm)

    with pytest.raises(RuntimeError, match="génération"):
        responder.reply("Hello")


def test_empty_question_raises_value_error() -> None:
    llm = ScriptedLLM(["Ignoré"])
    responder = DirectChatResponder(llm)

    with pytest.raises(ValueError):
        responder.reply("   ")
    assert llm.prompts == []


def test_reply_injects_history_without_duplicating_current_message() -> None:
    llm = ScriptedLLM(["Voici trois exemples."])
    responder = DirectChatResponder(llm)
    history = [
        {"role": "user", "content": "Explique-moi Python."},
        {"role": "assistant", "content": "Python est un langage."},
    ]

    result = responder.reply("Donne-moi trois exemples.", history=history)

    prompt = llm.prompts[0]
    assert "Explique-moi Python." in prompt
    assert "Python est un langage." in prompt
    assert prompt.count("Donne-moi trois exemples.") == 1
    assert result["sources"] == []


def test_reply_with_empty_history_still_works() -> None:
    llm = ScriptedLLM(["Bonjour !"])
    responder = DirectChatResponder(llm)

    result = responder.reply("Bonjour", history=[])

    assert "Historique récent" in llm.prompts[0] or "Recent history" in llm.prompts[0]
    assert result["answer"] == "Bonjour !"
