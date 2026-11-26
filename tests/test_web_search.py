"""Tests de WebSearcher (Tavily via httpx, sans réseau réel)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import httpx
import pytest

from src.tutor.web_search import WebSearcher


def _tavily_payload(results: list[dict]) -> dict:
    return {"results": results}


def _result(title: str, url: str, content: str = "extrait") -> dict:
    return {"title": title, "url": url, "content": content}


def test_search_returns_structured_hits() -> None:
    response = MagicMock()
    response.status_code = 200
    response.json.return_value = _tavily_payload(
        [
            _result("FastAPI docs", "https://fastapi.tiangolo.com/", "Official docs"),
            _result("Python news", "https://www.python.org/", "Release notes"),
        ]
    )
    response.raise_for_status = MagicMock()

    with patch("src.tutor.web_search.httpx.Client") as client_cls:
        client = MagicMock()
        client.__enter__.return_value = client
        client.__exit__.return_value = False
        client.post.return_value = response
        client_cls.return_value = client

        searcher = WebSearcher(search_token="test-key", max_results=3)
        hits = searcher.search("documentation officielle FastAPI")

    assert len(hits) == 2
    assert hits[0] == {
        "type": "web",
        "title": "FastAPI docs",
        "url": "https://fastapi.tiangolo.com/",
        "snippet": "Official docs",
    }
    assert hits[1]["type"] == "web"
    posted = client.post.call_args
    assert posted.args[0] == "https://api.tavily.com/search"
    assert posted.kwargs["headers"]["Authorization"] == "Bearer test-key"
    assert posted.kwargs["json"]["query"] == "documentation officielle FastAPI"
    assert posted.kwargs["json"]["max_results"] == 3
    assert "api_key" not in posted.kwargs["json"]


def test_missing_api_key_raises_clear_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("EDURAG_WEB_SEARCH_API_KEY", raising=False)
    searcher = WebSearcher(search_token="")

    with pytest.raises(ValueError, match="EDURAG_WEB_SEARCH_API_KEY"):
        searcher.search("actualité Python")


def test_empty_results_returns_empty_list() -> None:
    response = MagicMock()
    response.status_code = 200
    response.json.return_value = _tavily_payload([])
    response.raise_for_status = MagicMock()

    with patch("src.tutor.web_search.httpx.Client") as client_cls:
        client = MagicMock()
        client.__enter__.return_value = client
        client.__exit__.return_value = False
        client.post.return_value = response
        client_cls.return_value = client

        hits = WebSearcher(search_token="k").search("sujet obscur")

    assert hits == []


def test_http_error_raises_runtime_error() -> None:
    response = MagicMock()
    response.raise_for_status.side_effect = httpx.HTTPStatusError(
        "boom",
        request=MagicMock(),
        response=MagicMock(status_code=401),
    )

    with patch("src.tutor.web_search.httpx.Client") as client_cls:
        client = MagicMock()
        client.__enter__.return_value = client
        client.__exit__.return_value = False
        client.post.return_value = response
        client_cls.return_value = client

        with pytest.raises(RuntimeError, match="recherche Web"):
            WebSearcher(search_token="k").search("React")


def test_timeout_raises_runtime_error() -> None:
    with patch("src.tutor.web_search.httpx.Client") as client_cls:
        client = MagicMock()
        client.__enter__.return_value = client
        client.__exit__.return_value = False
        client.post.side_effect = httpx.TimeoutException("slow")
        client_cls.return_value = client

        with pytest.raises(RuntimeError, match="recherche Web"):
            WebSearcher(search_token="k").search("IA")


def test_invalid_urls_are_dropped_never_invented() -> None:
    response = MagicMock()
    response.status_code = 200
    response.json.return_value = _tavily_payload(
        [
            _result("Bad", "ftp://example.com/x", "no"),
            _result("Also bad", "not-a-url", "no"),
            _result("Good", "https://docs.python.org/3/", "yes"),
        ]
    )
    response.raise_for_status = MagicMock()

    with patch("src.tutor.web_search.httpx.Client") as client_cls:
        client = MagicMock()
        client.__enter__.return_value = client
        client.__exit__.return_value = False
        client.post.return_value = response
        client_cls.return_value = client

        hits = WebSearcher(search_token="k").search("Python")

    assert len(hits) == 1
    assert hits[0]["url"] == "https://docs.python.org/3/"


def test_invalid_json_payload_raises() -> None:
    response = MagicMock()
    response.raise_for_status = MagicMock()
    response.json.return_value = {"unexpected": True}

    with patch("src.tutor.web_search.httpx.Client") as client_cls:
        client = MagicMock()
        client.__enter__.return_value = client
        client.__exit__.return_value = False
        client.post.return_value = response
        client_cls.return_value = client

        with pytest.raises(RuntimeError, match="recherche Web"):
            WebSearcher(search_token="k").search("x")


def test_prefers_official_domains_when_ranking() -> None:
    response = MagicMock()
    response.status_code = 200
    response.json.return_value = _tavily_payload(
        [
            _result("Blog", "https://random-blog.example/post", "blog"),
            _result("Docs", "https://docs.python.org/3/whatsnew/", "official"),
            _result("Edu", "https://mit.edu/ai", "edu"),
        ]
    )
    response.raise_for_status = MagicMock()

    with patch("src.tutor.web_search.httpx.Client") as client_cls:
        client = MagicMock()
        client.__enter__.return_value = client
        client.__exit__.return_value = False
        client.post.return_value = response
        client_cls.return_value = client

        hits = WebSearcher(search_token="k", max_results=3).search("Python")

    urls = [hit["url"] for hit in hits]
    assert urls[0] in {
        "https://docs.python.org/3/whatsnew/",
        "https://mit.edu/ai",
    }
    assert "https://random-blog.example/post" in urls
