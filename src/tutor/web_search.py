"""Recherche Web via Tavily (httpx), sans second LLM ni framework d'agents."""

from __future__ import annotations

import os
from typing import Any
from urllib.parse import urlparse

import httpx
from dotenv import load_dotenv

TAVILY_SEARCH_URL = "https://api.tavily.com/search"
DEFAULT_PROVIDER = "tavily"
DEFAULT_MAX_RESULTS = 5
DEFAULT_TIMEOUT_SECONDS = 20.0

_OFFICIAL_HOST_HINTS = (
    "docs.",
    "developer.",
    "developers.",
    "learn.",
    "help.",
    "support.",
    "www.python.org",
    "fastapi.tiangolo.com",
    "react.dev",
    "nodejs.org",
    "developer.mozilla.org",
    "wikipedia.org",
    "arxiv.org",
    "nature.com",
    "ieee.org",
    "acm.org",
    "gov",
    "edu",
    "ac.uk",
    "gouv.fr",
)


class WebSearcher:
    """Interroge Tavily et renvoie des sources Web structurées."""

    def __init__(
        self,
        *,
        search_token: str | None = None,
        provider: str | None = None,
        max_results: int = DEFAULT_MAX_RESULTS,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        load_dotenv()
        if not isinstance(max_results, int) or max_results < 1:
            raise ValueError("max_results doit être un entier positif.")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds doit être positif.")
        self.provider = (provider or os.environ.get("EDURAG_WEB_SEARCH_PROVIDER") or DEFAULT_PROVIDER).strip().lower()
        configured = (
            search_token
            if search_token is not None
            else os.environ.get("EDURAG_WEB_SEARCH_API_KEY")
        )
        self._tavily_token = configured.strip() if isinstance(configured, str) else ""
        self.max_results = max_results
        self.timeout_seconds = timeout_seconds

    def search(self, query: str) -> list[dict[str, str]]:
        """Retourne des hits ``{type, title, url, snippet}`` (jamais inventés)."""
        cleaned = query.strip() if isinstance(query, str) else ""
        if not cleaned:
            raise ValueError("La requête de recherche Web ne peut pas être vide.")
        if not self._tavily_token:
            raise ValueError(
                "La recherche Web est indisponible : définissez EDURAG_WEB_SEARCH_API_KEY "
                "dans votre fichier .env (aucune recherche n'a été effectuée)."
            )
        if self.provider != "tavily":
            raise ValueError(
                f"Fournisseur de recherche Web non pris en charge : {self.provider!r}. "
                "Utilisez EDURAG_WEB_SEARCH_PROVIDER=tavily."
            )

        try:
            with httpx.Client(timeout=self.timeout_seconds) as client:
                response = client.post(
                    TAVILY_SEARCH_URL,
                    headers={
                        "Authorization": f"Bearer {self._tavily_token}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "query": cleaned,
                        "max_results": self.max_results,
                        "search_depth": "basic",
                        "include_answer": False,
                    },
                )
                response.raise_for_status()
                payload = response.json()
        except httpx.TimeoutException as error:
            raise RuntimeError(
                "La recherche Web a échoué : délai d'attente dépassé."
            ) from error
        except httpx.HTTPError as error:
            raise RuntimeError(
                "La recherche Web a échoué : erreur HTTP."
            ) from error
        except ValueError as error:
            raise RuntimeError(
                "La recherche Web a échoué : réponse invalide."
            ) from error

        return _hits_from_payload(payload, max_results=self.max_results)


def _hits_from_payload(payload: object, *, max_results: int) -> list[dict[str, str]]:
    if not isinstance(payload, dict) or not isinstance(payload.get("results"), list):
        raise RuntimeError("La recherche Web a échoué : réponse invalide.")

    hits: list[dict[str, str]] = []
    for item in payload["results"]:
        if not isinstance(item, dict):
            continue
        url = _valid_http_url(item.get("url"))
        if url is None:
            continue
        title = str(item.get("title") or "").strip() or url
        snippet = str(item.get("content") or item.get("snippet") or "").strip()
        hits.append(
            {
                "type": "web",
                "title": title,
                "url": url,
                "snippet": snippet,
            }
        )

    hits.sort(key=_official_rank)
    return hits[:max_results]


def _valid_http_url(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text:
        return None
    parsed = urlparse(text)
    if parsed.scheme not in {"http", "https"}:
        return None
    if not parsed.netloc:
        return None
    return text


def _official_rank(hit: dict[str, str]) -> tuple[int, str]:
    host = (urlparse(hit["url"]).netloc or "").lower()
    score = 0
    for hint in _OFFICIAL_HOST_HINTS:
        if hint in host:
            score -= 10
            break
    if host.endswith(".gov") or host.endswith(".edu") or host.endswith(".gouv.fr"):
        score -= 5
    if host.startswith("docs.") or ".docs." in host:
        score -= 5
    return (score, host)
