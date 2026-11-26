"""Réponses conversationnelles directes (sans RAG ni Web).

Utilisé pour GENERAL_CHAT : salutations, remerciements, discussion générale.
"""

from __future__ import annotations

from typing import Any

DEFAULT_MAX_NEW_TOKENS = 512

_FREQUENT_FRENCH = {
    "le",
    "la",
    "les",
    "de",
    "des",
    "un",
    "une",
    "et",
    "est",
    "que",
    "qui",
    "dans",
    "pour",
    "pas",
    "plus",
    "avec",
    "sur",
    "je",
    "tu",
    "nous",
    "vous",
    "merci",
    "bonjour",
    "salut",
    "comment",
    "peux",
    "peut",
    "réponse",
    "reponse",
}

_FREQUENT_ENGLISH = {
    "the",
    "a",
    "an",
    "and",
    "is",
    "are",
    "you",
    "your",
    "me",
    "my",
    "to",
    "of",
    "in",
    "for",
    "with",
    "on",
    "thank",
    "thanks",
    "hello",
    "hi",
    "how",
    "what",
    "please",
    "can",
    "could",
}


class DirectChatResponder:
    """Génère une réponse LLM sans retrieval ni sources PDF."""

    def __init__(
        self,
        llm: object,
        *,
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    ) -> None:
        if not callable(getattr(llm, "generate", None)):
            raise TypeError("llm doit fournir generate.")
        if not isinstance(max_new_tokens, int) or max_new_tokens < 1:
            raise ValueError("max_new_tokens doit être un entier positif.")
        self.llm = llm
        self.max_new_tokens = max_new_tokens

    def reply(
        self,
        question: str,
        history: list[dict[str, Any]] | None = None,
    ) -> dict[str, str | list]:
        """Retourne ``{"answer": str, "sources": []}``.

        ``history`` contient les messages antérieurs déjà bornés (sans le message actuel).
        """
        if not isinstance(question, str) or not question.strip():
            raise ValueError("La question ne peut pas être vide.")

        cleaned = question.strip()
        language = _detect_language(cleaned) or "fr"
        prompt = _build_prompt(cleaned, language, history)

        try:
            raw = self.llm.generate(prompt, max_new_tokens=self.max_new_tokens)
        except Exception as error:
            raise RuntimeError("La génération de la réponse a échoué.") from error

        if not isinstance(raw, str) or not raw.strip():
            raise RuntimeError("La génération de la réponse a échoué.")

        return {"answer": raw.strip(), "sources": []}


def _format_history(history: list[dict[str, Any]] | None) -> str:
    if not history:
        return "(aucun)"
    lines: list[str] = []
    for item in history:
        if not isinstance(item, dict):
            continue
        role = str(item.get("role") or "unknown").strip() or "unknown"
        content = str(item.get("content") or "").strip()
        if not content:
            continue
        if len(content) > 400:
            content = content[:400] + "…"
        lines.append(f"- {role}: {content}")
    return "\n".join(lines) if lines else "(aucun)"


def _build_prompt(
    question: str,
    language: str,
    history: list[dict[str, Any]] | None,
) -> str:
    history_block = _format_history(history)
    if language == "en":
        return (
            "You are EduRAG, a friendly pedagogical assistant.\n"
            "Answer the user's message naturally and clearly in English.\n"
            "Use the recent conversation history when the message is a follow-up.\n"
            "Rules:\n"
            "- Do not invent PDF sources, page numbers, filenames, or citations.\n"
            "- Do not claim you searched the web or a document corpus.\n"
            "- Keep a helpful, concise, student-friendly tone.\n"
            "- If the user greets or thanks you, reply politely.\n"
            "- If the user asks what EduRAG is, explain briefly that you help "
            "students study from their course PDFs.\n\n"
            f"Recent history:\n{history_block}\n\n"
            f"User message:\n{question}\n"
        )
    if language == "ar":
        return (
            "أنت EduRAG، مساعد تعليمي ودود.\n"
            "أجب على رسالة المستخدم بوضوح وبالعربية.\n"
            "استخدم سجل المحادثة الأخير إذا كانت الرسالة متابعة لسياق سابق.\n"
            "القواعد:\n"
            "- لا تخترع مصادر PDF أو أرقام صفحات أو أسماء ملفات أو اقتباسات.\n"
            "- لا تدّع أنك بحثت في الويب أو في مجموعة وثائق.\n"
            "- حافظ على أسلوب مفيد ومختصر ومناسب للطلاب.\n"
            "- إذا حيّاك المستخدم أو شكرك، رد بأدب.\n"
            "- إذا سألك عن EduRAG، اشرح باختصار أنك تساعد الطلاب "
            "على الدراسة من ملفات PDF الخاصة بدروسهم.\n\n"
            f"السجل الأخير:\n{history_block}\n\n"
            f"رسالة المستخدم:\n{question}\n"
        )
    return (
        "Tu es EduRAG, un assistant pédagogique bienveillant.\n"
        "Réponds au message de l'utilisateur naturellement et clairement en français.\n"
        "Utilise l'historique récent si le message est une question de suivi.\n"
        "Règles :\n"
        "- N'invente aucune source PDF, numéro de page, nom de fichier ou citation.\n"
        "- Ne prétends pas avoir cherché sur le web ou dans un corpus de documents.\n"
        "- Garde un ton utile, concis et adapté aux étudiants.\n"
        "- Si l'utilisateur te salue ou te remercie, réponds poliment.\n"
        "- S'il demande ce qu'est EduRAG, explique brièvement que tu aides "
        "les étudiants à travailler à partir de leurs PDF de cours.\n\n"
        f"Historique récent :\n{history_block}\n\n"
        f"Message utilisateur :\n{question}\n"
    )


def _detect_language(text: str) -> str | None:
    letters = [char for char in text if char.isalpha()]
    if not letters:
        return None
    arabic = sum(1 for char in letters if _is_arabic_letter(char))
    if arabic > len(letters) - arabic:
        return "ar"
    french_hits, english_hits = _latin_language_hits(text)
    if french_hits == 0 and english_hits == 0:
        return None
    if french_hits > english_hits:
        return "fr"
    if english_hits > french_hits:
        return "en"
    return None


def _latin_language_hits(text: str) -> tuple[int, int]:
    words = [token.strip(".,;:!?\"'()[]{}").lower() for token in text.split()]
    french = sum(1 for word in words if word in _FREQUENT_FRENCH)
    english = sum(1 for word in words if word in _FREQUENT_ENGLISH)
    return french, english


def _is_arabic_letter(char: str) -> bool:
    code = ord(char)
    return (
        0x0600 <= code <= 0x06FF
        or 0x0750 <= code <= 0x077F
        or 0x08A0 <= code <= 0x08FF
        or 0xFB50 <= code <= 0xFDFF
        or 0xFE70 <= code <= 0xFEFF
    )
