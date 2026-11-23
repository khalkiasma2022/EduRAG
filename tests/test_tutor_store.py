"""Tests de l'enregistrement SQLite des conversations.

Les bases sont temporaires. Aucun modèle n'est chargé.
"""

import json
import sqlite3
from pathlib import Path

import pytest

from src.tutor.store import ConversationStore


def _source(**changes: object) -> dict[str, object]:
    """Reproduit une source renvoyée par RAGChain.ask()."""
    source = {
        "text": "La mitochondrie produit l'énergie.",
        "filename": "cellule.pdf",
        "page_number": 2,
        "chunk_id": 4,
    }
    source.update(changes)
    return source


def test_store_creates_the_database_and_the_tables(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "conversations.sqlite"
    assert not path.exists()

    ConversationStore(path)

    assert path.is_file()
    connection = sqlite3.connect(path)
    try:
        tables = connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name IN ('conversations', 'messages')"
        ).fetchall()
        conversation_columns = [row[1] for row in connection.execute("PRAGMA table_info(conversations)")]
        message_columns = [row[1] for row in connection.execute("PRAGMA table_info(messages)")]
        message_sql = connection.execute(
            "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'messages'"
        ).fetchone()[0]
    finally:
        connection.close()

    assert tables == [("conversations",), ("messages",)]
    assert conversation_columns == ["id", "title", "created_at"]
    assert message_columns == [
        "id",
        "conversation_id",
        "role",
        "content",
        "sources",
        "created_at",
    ]
    assert "ON DELETE CASCADE" in message_sql


def test_create_conversation_returns_a_reliable_id(tmp_path: Path) -> None:
    store = ConversationStore(tmp_path / "conversations.sqlite")

    created = store.create_conversation("  Cours de biologie  ")

    assert created["title"] == "Cours de biologie"
    assert isinstance(created["id"], int)
    assert created["id"] > 0
    assert isinstance(created["created_at"], str)
    assert created["created_at"] != ""
    assert store.get_conversation(created["id"])["messages"] == []


def test_messages_keep_their_role_and_their_chronological_order(tmp_path: Path) -> None:
    store = ConversationStore(tmp_path / "conversations.sqlite")
    conversation = store.create_conversation("Mitochondrie")

    question = store.add_message(conversation["id"], "user", "Que produit la mitochondrie ?")
    answer = store.add_message(
        conversation["id"],
        "assistant",
        "Elle produit l'énergie de la cellule.",
        sources=[_source()],
    )
    follow_up = store.add_message(conversation["id"], "user", "Sur quelle page ?")

    loaded = store.get_conversation(conversation["id"])

    assert [message["id"] for message in loaded["messages"]] == [
        question["id"],
        answer["id"],
        follow_up["id"],
    ]
    assert [message["role"] for message in loaded["messages"]] == ["user", "assistant", "user"]
    assert [message["content"] for message in loaded["messages"]] == [
        "Que produit la mitochondrie ?",
        "Elle produit l'énergie de la cellule.",
        "Sur quelle page ?",
    ]
    assert question["sources"] == []
    assert follow_up["sources"] == []


def test_sources_are_reloaded_from_json_without_calling_a_model(tmp_path: Path) -> None:
    store = ConversationStore(tmp_path / "conversations.sqlite")
    conversation = store.create_conversation("Cellule")
    sources = [_source(), _source(text="Page suivante.", page_number=3, chunk_id=5)]

    saved = store.add_message(
        conversation["id"],
        "assistant",
        "Réponse appuyée sur le cours.",
        sources=sources,
    )

    connection = sqlite3.connect(store.path)
    try:
        raw_sources = connection.execute("SELECT sources FROM messages WHERE id = ?", (saved["id"],)).fetchone()[0]
    finally:
        connection.close()

    assert json.loads(raw_sources) == sources
    assert "mitochondrie" in raw_sources
    assert store.get_conversation(conversation["id"])["messages"][0]["sources"] == sources


def test_list_conversations_returns_the_newest_first(tmp_path: Path) -> None:
    store = ConversationStore(tmp_path / "conversations.sqlite")
    first = store.create_conversation("Ancienne")
    second = store.create_conversation("Récente")

    listed = store.list_conversations()

    assert [conversation["id"] for conversation in listed] == [second["id"], first["id"]]
    assert [conversation["title"] for conversation in listed] == ["Récente", "Ancienne"]
    assert "messages" not in listed[0]


def test_conversation_remains_after_the_store_is_reopened(tmp_path: Path) -> None:
    path = tmp_path / "conversations.sqlite"
    store = ConversationStore(path)
    conversation = store.create_conversation("À conserver")
    message = store.add_message(
        conversation["id"],
        "assistant",
        "Réponse conservée.",
        sources=[_source()],
    )
    saved = store.get_conversation(conversation["id"])

    reopened = ConversationStore(path)

    assert reopened.get_conversation(conversation["id"]) == saved
    assert reopened.list_conversations() == [conversation]
    assert reopened.get_conversation(conversation["id"])["messages"] == [message]


def test_delete_conversation_removes_its_messages_and_keeps_the_others(tmp_path: Path) -> None:
    store = ConversationStore(tmp_path / "conversations.sqlite")
    removed = store.create_conversation("À supprimer")
    kept = store.create_conversation("À garder")
    store.add_message(removed["id"], "user", "Question supprimée")
    store.add_message(removed["id"], "assistant", "Réponse supprimée", sources=[_source()])
    kept_message = store.add_message(kept["id"], "user", "Question conservée")

    store.delete_conversation(removed["id"])

    connection = sqlite3.connect(store.path)
    try:
        remaining_messages = connection.execute("SELECT conversation_id, content FROM messages").fetchall()
    finally:
        connection.close()

    assert remaining_messages == [(kept["id"], "Question conservée")]
    assert store.list_conversations() == [kept]
    assert store.get_conversation(kept["id"])["messages"] == [kept_message]
    with pytest.raises(ValueError, match="introuvable"):
        store.get_conversation(removed["id"])


def test_deleted_conversation_stays_deleted_after_reopening(tmp_path: Path) -> None:
    path = tmp_path / "conversations.sqlite"
    store = ConversationStore(path)
    conversation = store.create_conversation("Temporaire")
    store.add_message(conversation["id"], "user", "Une question")
    store.delete_conversation(conversation["id"])

    reopened = ConversationStore(path)

    assert reopened.list_conversations() == []
    with pytest.raises(ValueError, match="introuvable"):
        reopened.get_conversation(conversation["id"])


@pytest.mark.parametrize("conversation_id", [0, -1, True, "1", None])
def test_invalid_conversation_id_is_rejected(tmp_path: Path, conversation_id: object) -> None:
    store = ConversationStore(tmp_path / "conversations.sqlite")
    created = store.create_conversation("Valide")

    with pytest.raises(ValueError, match="invalide"):
        store.get_conversation(conversation_id)
    with pytest.raises(ValueError, match="invalide"):
        store.add_message(conversation_id, "user", "Question")
    with pytest.raises(ValueError, match="invalide"):
        store.delete_conversation(conversation_id)

    assert store.list_conversations() == [created]
    assert store.get_conversation(created["id"])["messages"] == []


def test_missing_conversation_rejects_the_message_and_the_deletion(tmp_path: Path) -> None:
    store = ConversationStore(tmp_path / "conversations.sqlite")

    with pytest.raises(ValueError, match="introuvable"):
        store.add_message(4, "user", "Question")
    with pytest.raises(ValueError, match="introuvable"):
        store.delete_conversation(4)

    assert store.list_conversations() == []


@pytest.mark.parametrize("title", ["", "   ", None, 12])
def test_invalid_title_is_rejected(tmp_path: Path, title: object) -> None:
    store = ConversationStore(tmp_path / "conversations.sqlite")

    with pytest.raises(ValueError, match="titre"):
        store.create_conversation(title)

    assert store.list_conversations() == []


def test_invalid_messages_are_rejected(tmp_path: Path) -> None:
    store = ConversationStore(tmp_path / "conversations.sqlite")
    conversation = store.create_conversation("Contrôles")
    rejected = [
        ("visitor", "Question", None),
        ("user", "   ", None),
        ("user", "Question", [_source()]),
        ("assistant", "Réponse", "pas une liste"),
        ("assistant", "Réponse", [_source(page_number=True)]),
        ("assistant", "Réponse", [_source(text="")]),
    ]

    for role, content, sources in rejected:
        with pytest.raises(ValueError):
            store.add_message(conversation["id"], role, content, sources)

    assert store.get_conversation(conversation["id"])["messages"] == []


def test_directory_path_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="fichier"):
        ConversationStore(tmp_path)

    with pytest.raises(ValueError, match="vide"):
        ConversationStore("   ")
