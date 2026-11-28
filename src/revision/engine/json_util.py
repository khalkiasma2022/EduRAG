"""Extraction robuste d'objets JSON renvoyés par le LLM local."""

from __future__ import annotations

import json
import re
from typing import Any

_JSON_FENCE = re.compile(r"```(?:json)?\s*([\s\S]*?)```", re.IGNORECASE)
_TRAILING_COMMA = re.compile(r",\s*([}\]])")


def load_json_object(raw: str) -> dict[str, Any]:
    """Charge un objet JSON malgré Markdown, texte autour, virgules ou troncature."""
    if not isinstance(raw, str):
        raise ValueError("JSON invalide.")
    text = _prepare_text(raw)
    if text == "":
        raise ValueError("JSON invalide.")

    candidates: list[str] = [text]
    extracted = _extract_first_object(text)
    if extracted is not None and extracted not in candidates:
        candidates.append(extracted)
    repaired = repair_truncated_json(text)
    if repaired is not None and repaired not in candidates:
        candidates.append(repaired)

    last_error: Exception | None = None
    for candidate in candidates:
        for variant in (candidate, _TRAILING_COMMA.sub(r"\1", candidate)):
            try:
                data = json.loads(variant)
            except json.JSONDecodeError as error:
                last_error = error
                continue
            if isinstance(data, dict):
                return data
            last_error = ValueError("JSON invalide.")

    raise ValueError("JSON invalide.") from last_error


def repair_truncated_json(raw: str) -> str | None:
    """Referme un JSON coupé en gardant la dernière valeur complète utilisable."""
    text = _prepare_text(raw)
    start = text.find("{")
    if start < 0:
        return None
    chunk = text[start:]

    bases: list[str] = []
    cut = _last_safe_cut(chunk)
    if cut > 0:
        bases.append(chunk[:cut].rstrip())
    # Coupe en milieu de chaîne : on referme la string puis les conteneurs.
    bases.append(chunk.rstrip())

    seen: set[str] = set()
    for base in bases:
        cleaned = re.sub(r",\s*$", "", base)
        if not cleaned or cleaned in seen:
            continue
        seen.add(cleaned)
        repaired = _force_close_json(cleaned)
        if repaired is None:
            continue
        repaired = _TRAILING_COMMA.sub(r"\1", repaired)
        try:
            data = json.loads(repaired)
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict):
            return repaired
    return None


def _force_close_json(text: str) -> str | None:
    """Referme guillemets / crochets / accolades ouverts ; None si incohérent."""
    stack: list[str] = []
    in_string = False
    escape = False
    for char in text:
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
            continue
        if char in "{[":
            stack.append(char)
            continue
        if char in "}]":
            if not stack:
                return None
            opening = stack.pop()
            if (char == "}" and opening != "{") or (char == "]" and opening != "["):
                return None
    closed = text
    if in_string:
        closed += '"'
    closers = "".join("]" if kind == "[" else "}" for kind in reversed(stack))
    return closed + closers


def _prepare_text(raw: str) -> str:
    text = raw.strip()
    fence = _JSON_FENCE.search(text)
    if fence:
        text = fence.group(1).strip()
    return text


def _extract_first_object(text: str) -> str | None:
    """Retourne le premier objet ``{...}`` équilibré, s'il existe."""
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_string = False
    escape = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return None


def _last_safe_cut(text: str) -> int:
    """Index après le dernier ``}`` ou ``]`` hors chaîne (valeur complète)."""
    last = 0
    in_string = False
    escape = False
    for index, char in enumerate(text):
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
            continue
        if char in "}]":
            last = index + 1
    return last


