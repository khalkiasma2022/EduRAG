"""Routeur d'intentions pour le chatbot EduRAG.

Classe chaque message en COURSE_QUESTION, WEB_QUESTION ou GENERAL_CHAT.
Aucune recherche FAISS ni Web : classification seule, avec repli prudent.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Literal

Intent = Literal["COURSE_QUESTION", "WEB_QUESTION", "GENERAL_CHAT"]

COURSE_QUESTION: Intent = "COURSE_QUESTION"
WEB_QUESTION: Intent = "WEB_QUESTION"
GENERAL_CHAT: Intent = "GENERAL_CHAT"

VALID_INTENTS: frozenset[str] = frozenset(
    {COURSE_QUESTION, WEB_QUESTION, GENERAL_CHAT}
)

DEFAULT_MAX_NEW_TOKENS = 128
MAX_HISTORY_TURNS = 6

_JSON_FENCE = re.compile(r"```(?:json)?\s*([\s\S]*?)```", re.IGNORECASE)

_GENERAL_PATTERNS = (
    re.compile(
        r"^\s*(bonjour|salut|hello|hi|hey|bonsoir|coucou)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"^\s*(merci|thanks|thank you|ok|d['’]accord|au revoir|bye)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(comment\s+vas[- ]tu|how\s+are\s+you|ça\s+va)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(reformule|reformuler|réécris|réécrire|rephrase)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(comment\s+fonctionne\s+edurag|what\s+is\s+edurag|"
        r"qui\s+es[- ]tu|who\s+are\s+you)\b",
        re.IGNORECASE,
    ),
)

_COURSE_PATTERNS = (
    re.compile(
        r"\b(cours|pdf|document|chapitre|page|pages|syllabus|handout)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(dans\s+mon\s+(cours|pdf|document)|from\s+my\s+(course|pdf|document))\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(fiche\s+de\s+révision|smart\s+revision)\b",
        re.IGNORECASE,
    ),
)

_WEB_PATTERNS = (
    re.compile(
        r"\b(derni[eè]res?\s+nouveaut[eé]s|latest\s+news|actualité|actualités)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(documentation\s+officielle|official\s+docs?|ressources?\s+fiables)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(recherches?\s+sur|research\s+on|sur\s+le\s+web|on\s+the\s+web|"
        r"internet|en\s+ligne|online)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(trouve[- ]moi|find\s+me|cherche\s+sur)\b",
        re.IGNORECASE,
    ),
)


@dataclass(frozen=True)
class RouteDecision:
    """Résultat de classification d'un message utilisateur."""

    intent: Intent
    confidence: float
    reason: str
    used_fallback: bool = False


class IntentRouter:
    """Classifie une question via le LLM existant, avec repli heuristique."""

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

    def classify(
        self,
        question: str,
        history: list[dict[str, Any]] | None = None,
    ) -> RouteDecision:
        """Retourne toujours une RouteDecision ; n'interrompt jamais l'appelant."""
        text = question if isinstance(question, str) else ""
        cleaned = text.strip()
        if not cleaned:
            return RouteDecision(
                intent=GENERAL_CHAT,
                confidence=0.0,
                reason="Question vide : repli GENERAL_CHAT.",
                used_fallback=True,
            )

        prompt = _build_prompt(cleaned, history)
        try:
            raw = self.llm.generate(prompt, max_new_tokens=self.max_new_tokens)
        except Exception:
            return _fallback_decision(
                cleaned,
                history=history,
                reason="Erreur LLM : repli heuristique.",
            )

        if not isinstance(raw, str):
            return _fallback_decision(
                cleaned,
                history=history,
                reason="Réponse LLM non textuelle : repli heuristique.",
            )
        return _parse_decision(raw, cleaned, history=history)


def _build_prompt(question: str, history: list[dict[str, Any]] | None) -> str:
    history_block = _format_history(history)
    return (
        "Tu es un routeur d'intentions pour EduRAG, un assistant pédagogique "
        "basé sur des PDF de cours.\n"
        "Classe le message utilisateur dans EXACTEMENT une catégorie.\n\n"
        "Catégories :\n"
        "- COURSE_QUESTION : question sur le contenu des PDF / cours / pages / "
        "chapitres / documents de l'étudiant.\n"
        "- WEB_QUESTION : besoin d'informations externes, actualité, "
        "documentation officielle, ressources web fiables.\n"
        "- GENERAL_CHAT : salutation, remerciement, discussion générale, "
        "question sur EduRAG lui-même, reformulation, sans recherche PDF ni web.\n\n"
        "Si le message est une question de suivi courte (ex. « Donne-moi trois "
        "exemples », « Lequel est le plus simple ? »), hérite de l'intention "
        "du sujet discuté dans l'historique.\n\n"
        "Réponds UNIQUEMENT avec un objet JSON valide, sans markdown, de la forme :\n"
        '{"intent":"COURSE_QUESTION|WEB_QUESTION|GENERAL_CHAT",'
        '"confidence":0.0,'
        '"reason":"courte justification"}\n'
        "confidence est un nombre entre 0 et 1.\n\n"
        f"Historique récent :\n{history_block}\n\n"
        f"Message utilisateur :\n{question}\n"
    )


def _format_history(history: list[dict[str, Any]] | None) -> str:
    if not history:
        return "(aucun)"
    lines: list[str] = []
    for item in history[-MAX_HISTORY_TURNS:]:
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


def _parse_decision(
    raw: str,
    question: str,
    *,
    history: list[dict[str, Any]] | None = None,
) -> RouteDecision:
    try:
        payload = _load_json_object(raw)
    except ValueError:
        return _fallback_decision(
            question,
            history=history,
            reason="JSON invalide : repli heuristique.",
        )

    intent_raw = payload.get("intent")
    if not isinstance(intent_raw, str) or intent_raw not in VALID_INTENTS:
        return _fallback_decision(
            question,
            history=history,
            reason=f"Intention inconnue {intent_raw!r} : repli heuristique.",
        )

    confidence = _clamp_confidence(payload.get("confidence", 0.5))
    reason = payload.get("reason")
    if not isinstance(reason, str) or not reason.strip():
        reason = "Classification LLM."
    else:
        reason = reason.strip()

    return RouteDecision(
        intent=intent_raw,  # type: ignore[arg-type]
        confidence=confidence,
        reason=reason,
        used_fallback=False,
    )


def _clamp_confidence(value: object) -> float:
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0.5
    if number < 0.0:
        return 0.0
    if number > 1.0:
        return 1.0
    return number


def _load_json_object(raw: str) -> dict[str, Any]:
    text = raw.strip()
    fence = _JSON_FENCE.search(text)
    if fence:
        text = fence.group(1).strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError as error:
        raise ValueError("JSON invalide.") from error
    if not isinstance(data, dict):
        raise ValueError("JSON invalide.")
    return data


def _fallback_decision(
    question: str,
    *,
    reason: str,
    history: list[dict[str, Any]] | None = None,
) -> RouteDecision:
    return RouteDecision(
        intent=_heuristic_intent(question, history),
        confidence=0.0,
        reason=reason,
        used_fallback=True,
    )


def _heuristic_intent(
    question: str,
    history: list[dict[str, Any]] | None = None,
) -> Intent:
    text = (question or "").strip()
    if not text:
        return GENERAL_CHAT
    if any(pattern.search(text) for pattern in _GENERAL_PATTERNS):
        return GENERAL_CHAT
    if any(pattern.search(text) for pattern in _WEB_PATTERNS):
        return WEB_QUESTION
    if any(pattern.search(text) for pattern in _COURSE_PATTERNS):
        return COURSE_QUESTION

    inherited = _intent_from_history(history)
    if inherited is not None and _looks_like_follow_up(text):
        return inherited

    # Sans indice cours/web : discussion générale (évite un RAG inutile).
    return GENERAL_CHAT


def _looks_like_follow_up(text: str) -> bool:
    if len(text) <= 80:
        return True
    lowered = text.lower()
    markers = (
        "exemple",
        "example",
        "lequel",
        "laquelle",
        "which",
        "plus simple",
        "donne-moi",
        "give me",
        "et pour",
        "pourquoi",
        "why",
    )
    return any(marker in lowered for marker in markers)


def _intent_from_history(history: list[dict[str, Any]] | None) -> Intent | None:
    if not history:
        return None
    # Parcourir du plus récent au plus ancien pour hériter du dernier sujet clair.
    for item in reversed(history):
        if not isinstance(item, dict):
            continue
        if str(item.get("role") or "") != "user":
            continue
        content = str(item.get("content") or "").strip()
        if not content:
            continue
        if any(pattern.search(content) for pattern in _WEB_PATTERNS):
            return WEB_QUESTION
        if any(pattern.search(content) for pattern in _COURSE_PATTERNS):
            return COURSE_QUESTION
        if any(pattern.search(content) for pattern in _GENERAL_PATTERNS):
            # Salutation seule : continuer à chercher un sujet plus substantiel.
            continue
        # Sujet pédagogique générique précédent → chat général (pas le RAG).
        return GENERAL_CHAT
    return None
