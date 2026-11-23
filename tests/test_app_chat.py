"""Tests des fonctions de l'interface conversationnelle.

Ils utilisent un faux assistant.
Ils n'appellent pas get_assistant() et ne chargent aucun modèle.
"""

import pytest

import app as app_module


class _Assistant:
    """Imite les méthodes de conversation sans créer EduRAGAssistant."""

    def __init__(self) -> None:
        self.created: list[str] = []
        self.sent: list[tuple[object, str, int]] = []
        self.loaded: list[object] = []
        self.deleted: list[object] = []
        self.removed: list[str] = []
        self.asked: list[str] = []
        self.fail = False
        self._rows: list[dict[str, str | int]] = []
        self._messages: dict[int, list[dict[str, object]]] = {}

    def create_conversation(self, title: str) -> int:
        conversation_id = len(self.created) + 1
        self.created.append(title)
        self._rows.insert(0, {"id": conversation_id, "title": title, "created_at": "2026-11-23"})
        self._messages[conversation_id] = []
        return conversation_id

    def list_conversations(self) -> list[dict[str, str | int]]:
        return [dict(row) for row in self._rows]

    def get_conversation(self, conversation_id: object) -> dict[str, object]:
        self.loaded.append(conversation_id)
        for row in self._rows:
            if row["id"] == conversation_id:
                return {**row, "messages": list(self._messages[int(conversation_id)])}
        raise ValueError("Conversation introuvable.")

    def delete_conversation(self, conversation_id: object) -> None:
        self.deleted.append(conversation_id)
        self._rows = [row for row in self._rows if row["id"] != conversation_id]
        if isinstance(conversation_id, int):
            self._messages.pop(conversation_id, None)

    def send_message(
        self,
        conversation_id: object,
        question: str,
        top_k: int = 3,
    ) -> dict[str, object]:
        self.sent.append((conversation_id, question, top_k))
        if self.fail:
            raise RuntimeError("La génération de la réponse a échoué.")
        messages = self._messages[int(conversation_id)]
        question_message = {"role": "user", "content": question, "sources": []}
        answer_message = {
            "role": "assistant",
            "content": "La mitochondrie produit l'énergie.",
            "sources": [
                {
                    "text": "passage interne",
                    "filename": "cellule.pdf",
                    "page_number": 2,
                    "chunk_id": 9,
                }
            ],
        }
        messages.extend([question_message, answer_message])
        return {
            "conversation_id": conversation_id,
            "question": question_message,
            "answer": answer_message,
        }

    def remove_document(self, document_id: str) -> None:
        self.removed.append(document_id)

    def ask(self, question: str, top_k: int = 3) -> dict[str, object]:
        self.asked.append(question)
        raise AssertionError("L'interface doit appeler send_message, pas ask.")


@pytest.fixture(autouse=True)
def _block_real_assistant(monkeypatch: pytest.MonkeyPatch) -> None:
    """Interdit le chargement du véritable assistant."""

    def forbidden() -> None:
        raise AssertionError("get_assistant ne doit pas être appelé.")

    monkeypatch.setattr(app_module, "get_assistant", forbidden)


def test_new_conversation_clears_the_open_id_without_creating_a_row() -> None:
    assistant = _Assistant()
    state: dict[str, object] = {"conversation_id": 4, "revision_id": 2}

    app_module._begin_new_conversation(state)

    assert state["conversation_id"] is None
    assert state["screen"] == "chat"
    assert state["revision_id"] == 2
    assert assistant.created == []
    assert assistant.asked == []


def test_history_keeps_the_newest_first_and_marks_the_open_id() -> None:
    conversations = [
        {"id": 8, "title": "Récente", "created_at": "2026-11-23T12:02:00"},
        {"id": 3, "title": "Récente", "created_at": "2026-11-23T12:01:00"},
    ]

    entries = app_module._conversation_entries(conversations, 3)

    assert [entry["id"] for entry in entries] == [8, 3]
    assert [entry["current"] for entry in entries] == [False, True]
    assert entries[0]["title"] == "Récente"
    assert entries[1]["title"] == "Récente"


def test_selecting_a_conversation_stores_its_id() -> None:
    state: dict[str, object] = {"screen": "courses"}

    app_module._select_conversation(state, 8)

    assert state["conversation_id"] == 8
    assert state["screen"] == "chat"


def test_first_question_creates_one_conversation_and_sends_it_once() -> None:
    assistant = _Assistant()
    state: dict[str, object] = {"conversation_id": None}
    question = "Que produit\nla mitochondrie ?"

    result = app_module._submit_chat(assistant, state, question)

    assert assistant.created == ["Que produit la mitochondrie ?"]
    assert assistant.sent == [(1, question, 3)]
    assert state["conversation_id"] == 1
    assert result["answer"]["content"] == "La mitochondrie produit l'énergie."
    assert assistant.asked == []


def test_next_question_reuses_the_same_conversation() -> None:
    assistant = _Assistant()
    state: dict[str, object] = {"conversation_id": None}
    app_module._submit_chat(assistant, state, "Première question")

    app_module._submit_chat(assistant, state, "Deuxième question")

    assert assistant.created == ["Première question"]
    assert [call[0] for call in assistant.sent] == [1, 1]
    assert [call[1] for call in assistant.sent] == ["Première question", "Deuxième question"]
    assert assistant.asked == []


def test_reopening_reads_sqlite_and_does_not_send_a_message() -> None:
    assistant = _Assistant()
    state: dict[str, object] = {"conversation_id": None}
    app_module._submit_chat(assistant, state, "Question enregistrée")
    assistant.sent.clear()

    loaded = app_module._load_conversation(assistant, state["conversation_id"])

    assert loaded["title"] == "Question enregistrée"
    assert [message["role"] for message in loaded["messages"]] == ["user", "assistant"]
    assert assistant.loaded == [1]
    assert assistant.sent == []
    assert assistant.asked == []


def test_deleting_the_open_conversation_clears_only_its_id() -> None:
    assistant = _Assistant()
    state: dict[str, object] = {"conversation_id": None, "revision_id": 6}
    app_module._submit_chat(assistant, state, "À supprimer")
    kept = assistant.create_conversation("À garder")
    state["conversation_id"] = 1

    app_module._delete_conversation(assistant, state, 1)

    assert assistant.deleted == [1]
    assert state["conversation_id"] is None
    assert state["revision_id"] == 6
    assert [row["id"] for row in assistant.list_conversations()] == [kept]
    assert assistant.asked == []


def test_deleting_another_conversation_keeps_the_open_one() -> None:
    assistant = _Assistant()
    state: dict[str, object] = {"conversation_id": 2}

    app_module._delete_conversation(assistant, state, 1)

    assert state["conversation_id"] == 2


def test_sources_show_the_file_and_the_page_only() -> None:
    sources = [
        {
            "text": "passage interne",
            "filename": "cellule.pdf",
            "page_number": 2,
            "chunk_id": 9,
        },
        {
            "text": "autre passage",
            "filename": "neurone.pdf",
            "page_number": 5,
            "chunk_id": 1,
        },
    ]

    label = app_module._format_message_sources(sources)

    assert label == "cellule.pdf, page 2 · neurone.pdf, page 5"
    assert "passage interne" not in label
    assert "chunk_id" not in label


def test_empty_question_does_not_create_or_send() -> None:
    assistant = _Assistant()
    state: dict[str, object] = {"conversation_id": None}

    with pytest.raises(ValueError, match="vide"):
        app_module._submit_chat(assistant, state, "   ")

    assert assistant.created == []
    assert assistant.sent == []
    assert state["conversation_id"] is None


def test_failed_answer_calls_send_message_once_and_not_ask() -> None:
    assistant = _Assistant()
    assistant.fail = True
    state: dict[str, object] = {"conversation_id": None}

    with pytest.raises(RuntimeError, match="génération"):
        app_module._submit_chat(assistant, state, "Question interrompue")

    assert assistant.sent == [(1, "Question interrompue", 3)]
    assert assistant.asked == []


def test_course_removal_uses_the_document_id() -> None:
    assistant = _Assistant()

    app_module._remove_course(assistant, "doc-7")

    assert assistant.removed == ["doc-7"]
    assert assistant.asked == []
