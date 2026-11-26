"""Tests de HistoryWindow : fenêtre d'historique conversationnel."""

from __future__ import annotations

from src.tutor.history_window import HistoryWindow


def _msg(role: str, content: str, message_id: int = 1) -> dict:
    return {"id": message_id, "role": role, "content": content, "sources": []}


def test_empty_history_returns_empty_list() -> None:
    window = HistoryWindow()
    assert window.build([]) == []
    assert window.build(None) == []  # type: ignore[arg-type]


def test_build_keeps_recent_messages_only() -> None:
    window = HistoryWindow(max_messages=4, max_chars=10_000)
    messages = [
        _msg("user", "A", 1),
        _msg("assistant", "B", 2),
        _msg("user", "C", 3),
        _msg("assistant", "D", 4),
        _msg("user", "E", 5),
        _msg("assistant", "F", 6),
    ]

    history = window.build(messages)

    assert [item["content"] for item in history] == ["C", "D", "E", "F"]
    assert all(set(item) == {"role", "content"} for item in history)


def test_build_excludes_current_message_by_id() -> None:
    window = HistoryWindow(max_messages=10, max_chars=10_000)
    messages = [
        _msg("user", "Explique-moi Python.", 1),
        _msg("assistant", "Python est…", 2),
        _msg("user", "Donne-moi trois exemples.", 3),
    ]

    history = window.build(messages, exclude_message_id=3)

    assert [item["content"] for item in history] == [
        "Explique-moi Python.",
        "Python est…",
    ]
    assert "Donne-moi trois exemples." not in [item["content"] for item in history]


def test_build_respects_max_chars_from_the_oldest() -> None:
    window = HistoryWindow(max_messages=10, max_chars=35)
    messages = [
        _msg("user", "aaaaaaaaaa", 1),  # 10
        _msg("assistant", "bbbbbbbbbb", 2),  # 10
        _msg("user", "cccccccccc", 3),  # 10
        _msg("assistant", "dddddddddd", 4),  # 10
    ]

    history = window.build(messages)

    joined = "".join(item["content"] for item in history)
    assert len(joined) <= 35
    assert history[-1]["content"] == "dddddddddd"
    assert history[0]["content"] != "aaaaaaaaaa"
    assert "aaaaaaaaaa" not in [item["content"] for item in history]


def test_follow_up_detection() -> None:
    window = HistoryWindow()
    history = [
        {"role": "user", "content": "Explique-moi Python."},
        {"role": "assistant", "content": "Python est un langage."},
    ]

    assert window.is_follow_up("Donne-moi trois exemples.", history) is True
    assert window.is_follow_up("Lequel est le plus simple ?", history) is True
    assert window.is_follow_up("Bonjour", history) is False
    assert window.is_follow_up("Explique la virtualisation dans mon cours.", []) is False


def test_enrich_question_includes_context_without_duplicating_current() -> None:
    window = HistoryWindow()
    history = [
        {"role": "user", "content": "Explique-moi Python."},
        {"role": "assistant", "content": "Python est un langage."},
    ]
    current = "Donne-moi trois exemples."

    enriched = window.enrich_question(current, history)

    assert "Explique-moi Python." in enriched
    assert "Python est un langage." in enriched
    assert enriched.count(current) == 1


def test_should_enrich_rag_only_for_follow_ups_with_history() -> None:
    window = HistoryWindow()
    history = [
        {"role": "user", "content": "Explique mon cours sur les listes."},
        {"role": "assistant", "content": "Les listes…"},
    ]

    assert window.should_enrich_rag("Donne un exemple.", history) is True
    assert window.should_enrich_rag("Donne un exemple.", []) is False
    assert window.should_enrich_rag("Bonjour", history) is False
