"""Fenêtre d'historique conversationnel pour le chatbot EduRAG.

Borne le nombre de messages et la taille en caractères pour ne pas
saturer le contexte du modèle.
"""

from __future__ import annotations

import re
from typing import Any

DEFAULT_MAX_MESSAGES = 6
DEFAULT_MAX_CHARS = 2400

_FOLLOW_UP_PATTERNS = (
    re.compile(
        r"^\s*(et|donc|alors|aussi|également|also|and|so|then)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(donne[- ]moi|donnez[- ]moi|give\s+me|montre[- ]moi|show\s+me|"
        r"explique\s+(ça|cela|ceci)|explain\s+(that|this|it)|"
        r"plus\s+d['’]?exemples?|more\s+examples?|"
        r"trois\s+exemples|three\s+examples|"
        r"lequel|laquelle|lesquels|which\s+one|"
        r"plus\s+simple|plus\s+clair|simplifie|précise|precise|"
        r"reformule|continue|pourquoi|why\s+is\s+that|"
        r"et\s+pour|what\s+about)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"^\s*(et\s+)?(après|ensuite|next)\b",
        re.IGNORECASE,
    ),
)

_STANDALONE_GENERAL = (
    re.compile(
        r"^\s*(bonjour|salut|hello|hi|hey|bonsoir|merci|thanks|thank you|"
        r"au revoir|bye|ok|d['’]accord)\b",
        re.IGNORECASE,
    ),
)


class HistoryWindow:
    """Sélectionne un sous-ensemble borné des messages d'une conversation."""

    def __init__(
        self,
        *,
        max_messages: int = DEFAULT_MAX_MESSAGES,
        max_chars: int = DEFAULT_MAX_CHARS,
    ) -> None:
        if not isinstance(max_messages, int) or max_messages < 1:
            raise ValueError("max_messages doit être un entier positif.")
        if not isinstance(max_chars, int) or max_chars < 1:
            raise ValueError("max_chars doit être un entier positif.")
        self.max_messages = max_messages
        self.max_chars = max_chars

    def build(
        self,
        messages: list[dict[str, Any]] | None,
        *,
        exclude_message_id: object | None = None,
    ) -> list[dict[str, str]]:
        """Retourne les derniers messages utiles, sans le message courant."""
        if not messages:
            return []

        selected: list[dict[str, str]] = []
        for item in messages:
            if not isinstance(item, dict):
                continue
            if exclude_message_id is not None and item.get("id") == exclude_message_id:
                continue
            role = str(item.get("role") or "").strip()
            content = str(item.get("content") or "").strip()
            if role not in {"user", "assistant"} or not content:
                continue
            selected.append({"role": role, "content": content})

        if not selected:
            return []

        selected = selected[-self.max_messages :]
        return _trim_to_char_budget(selected, self.max_chars)

    def is_follow_up(self, question: str, history: list[dict[str, str]] | None) -> bool:
        """Indique si la question dépend vraisemblablement du contexte récent."""
        if not history:
            return False
        text = question if isinstance(question, str) else ""
        cleaned = text.strip()
        if not cleaned:
            return False
        if any(pattern.search(cleaned) for pattern in _STANDALONE_GENERAL):
            return False
        if len(cleaned) <= 80 and any(pattern.search(cleaned) for pattern in _FOLLOW_UP_PATTERNS):
            return True
        # Courte question sans sujet lexical riche après un échange : suivi probable.
        if len(cleaned) <= 48 and len(cleaned.split()) <= 8:
            return True
        return any(pattern.search(cleaned) for pattern in _FOLLOW_UP_PATTERNS)

    def should_enrich_rag(
        self,
        question: str,
        history: list[dict[str, str]] | None,
    ) -> bool:
        """N'enrichit le RAG que pour les suivis avec un historique non vide."""
        return self.is_follow_up(question, history)

    def enrich_question(
        self,
        question: str,
        history: list[dict[str, str]] | None,
    ) -> str:
        """Ajoute un contexte compact sans dupliquer inutilement la question."""
        cleaned = (question or "").strip()
        if not history:
            return cleaned
        lines = []
        for item in history:
            role = item.get("role", "unknown")
            content = str(item.get("content") or "").strip()
            if not content:
                continue
            if len(content) > 320:
                content = content[:320] + "…"
            lines.append(f"- {role}: {content}")
        if not lines:
            return cleaned
        block = "\n".join(lines)
        return (
            f"{cleaned}\n\n"
            "[Contexte de la conversation]\n"
            f"{block}"
        )


def _trim_to_char_budget(
    messages: list[dict[str, str]],
    max_chars: int,
) -> list[dict[str, str]]:
    """Retire les messages les plus anciens tant que le budget est dépassé."""
    kept = list(messages)
    while kept and _total_chars(kept) > max_chars:
        # Si un seul message dépasse le budget, on le tronque.
        if len(kept) == 1:
            content = kept[0]["content"]
            if len(content) > max_chars:
                kept[0] = {
                    "role": kept[0]["role"],
                    "content": content[: max(0, max_chars - 1)] + "…",
                }
            break
        kept.pop(0)
    return kept


def _total_chars(messages: list[dict[str, str]]) -> int:
    return sum(len(item.get("content") or "") for item in messages)
