"""Enregistre les conversations dans une base SQLite locale.

Le fichier prévu pour l'application est data/documents/conversations.sqlite.
Il reste distinct de revisions.sqlite. Le chemin est choisi par l'appelant.
"""

import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

_ROLES = frozenset({"user", "assistant"})

_CREATE_CONVERSATIONS = """
CREATE TABLE IF NOT EXISTS conversations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL,
    created_at TEXT NOT NULL
)
"""

_CREATE_MESSAGES = """
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id INTEGER NOT NULL,
    role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
    content TEXT NOT NULL,
    sources TEXT NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
)
"""

_INSERT_CONVERSATION = """
INSERT INTO conversations (title, created_at) VALUES (?, ?)
"""

_INSERT_MESSAGE = """
INSERT INTO messages (
    conversation_id, role, content, sources, created_at
) VALUES (?, ?, ?, ?, ?)
"""

_SELECT_CONVERSATION = """
SELECT id, title, created_at
FROM conversations
WHERE id = ?
"""

_SELECT_CONVERSATIONS = """
SELECT id, title, created_at
FROM conversations
ORDER BY created_at DESC, id DESC
"""

_SELECT_MESSAGE = """
SELECT id, conversation_id, role, content, sources, created_at
FROM messages
WHERE id = ?
"""

_SELECT_MESSAGES = """
SELECT id, conversation_id, role, content, sources, created_at
FROM messages
WHERE conversation_id = ?
ORDER BY created_at ASC, id ASC
"""

_DELETE_MESSAGES = """
DELETE FROM messages WHERE conversation_id = ?
"""

_DELETE_CONVERSATION = """
DELETE FROM conversations WHERE id = ?
"""


class ConversationStore:
    """Conserve les conversations et leurs messages dans un fichier SQLite."""

    def __init__(self, path: str | Path) -> None:
        """Ouvre ou crée la base, puis crée les tables si elles manquent."""
        if isinstance(path, str) and path.strip() == "":
            raise ValueError("Le chemin de la base ne peut pas être vide.")
        if not isinstance(path, (str, Path)):
            raise TypeError("Le chemin de la base doit être un chemin de fichier.")

        self.path = Path(path)
        if self.path.exists() and self.path.is_dir():
            raise ValueError("Le chemin de la base doit désigner un fichier.")

        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connection() as connection:
            connection.execute(_CREATE_CONVERSATIONS)
            connection.execute(_CREATE_MESSAGES)

    def create_conversation(self, title: object) -> dict[str, str | int]:
        """Crée une conversation et renvoie la ligne enregistrée."""
        stored_title = _required_text(title, "Le titre de la conversation est invalide.")
        created_at = _now()
        with self._connection() as connection:
            cursor = connection.execute(_INSERT_CONVERSATION, (stored_title, created_at))
            conversation_id = _inserted_id(cursor.lastrowid, "La conversation n'a pas pu être enregistrée.")
            row = connection.execute(_SELECT_CONVERSATION, (conversation_id,)).fetchone()
        if row is None:
            raise RuntimeError("La conversation n'a pas pu être enregistrée.")
        return _conversation_to_dict(row)

    def add_message(
        self,
        conversation_id: object,
        role: object,
        content: object,
        sources: object = None,
    ) -> dict[str, str | int | list]:
        """Ajoute un message utilisateur ou une réponse, avec ses sources."""
        _check_conversation_id(conversation_id)
        stored_role = _required_role(role)
        stored_content = _required_text(content, "Le message est invalide.")
        stored_sources = _validate_sources(stored_role, sources)
        created_at = _now()
        encoded_sources = json.dumps(stored_sources, ensure_ascii=False, separators=(",", ":"))
        with self._connection() as connection:
            if connection.execute(_SELECT_CONVERSATION, (conversation_id,)).fetchone() is None:
                raise ValueError("Conversation introuvable.")
            cursor = connection.execute(
                _INSERT_MESSAGE,
                (conversation_id, stored_role, stored_content, encoded_sources, created_at),
            )
            message_id = _inserted_id(cursor.lastrowid, "Le message n'a pas pu être enregistré.")
            row = connection.execute(_SELECT_MESSAGE, (message_id,)).fetchone()
        if row is None:
            raise RuntimeError("Le message n'a pas pu être enregistré.")
        return _message_to_dict(row)

    def get_conversation(self, conversation_id: object) -> dict[str, str | int | list]:
        """Retrouve une conversation et ses messages, du plus ancien au plus récent."""
        _check_conversation_id(conversation_id)
        with self._connection() as connection:
            row = connection.execute(_SELECT_CONVERSATION, (conversation_id,)).fetchone()
            if row is None:
                raise ValueError("Conversation introuvable.")
            messages = connection.execute(_SELECT_MESSAGES, (conversation_id,)).fetchall()
        conversation = _conversation_to_dict(row)
        conversation["messages"] = [_message_to_dict(message) for message in messages]
        return conversation

    def list_conversations(self) -> list[dict[str, str | int]]:
        """Renvoie les conversations, de la plus récente à la plus ancienne."""
        with self._connection() as connection:
            rows = connection.execute(_SELECT_CONVERSATIONS).fetchall()
        return [_conversation_to_dict(row) for row in rows]

    def delete_conversation(self, conversation_id: object) -> None:
        """Supprime une conversation et tous ses messages dans une seule transaction."""
        _check_conversation_id(conversation_id)
        with self._connection() as connection:
            exists = connection.execute(_SELECT_CONVERSATION, (conversation_id,)).fetchone()
            if exists is None:
                raise ValueError("Conversation introuvable.")
            connection.execute(_DELETE_MESSAGES, (conversation_id,))
            deleted = connection.execute(_DELETE_CONVERSATION, (conversation_id,))
            if deleted.rowcount != 1:
                raise RuntimeError("La conversation n'a pas pu être supprimée.")

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        """Ouvre une transaction, puis ferme la connexion dans tous les cas."""
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        try:
            connection.execute("PRAGMA foreign_keys = ON")
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()


def _now() -> str:
    """Horodate en UTC, avec les microsecondes, pour garder l'ordre des écritures."""
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def _required_text(value: object, message: str) -> str:
    """Exige un texte qui contient autre chose que des espaces."""
    if not isinstance(value, str) or value.strip() == "":
        raise ValueError(message)
    return value.strip()


def _required_role(role: object) -> str:
    """N'accepte que les deux rôles prévus."""
    if not isinstance(role, str) or role not in _ROLES:
        raise ValueError("Le rôle du message est invalide.")
    return role


def _check_conversation_id(conversation_id: object) -> None:
    """Refuse un identifiant qui ne vient pas de SQLite."""
    if isinstance(conversation_id, bool) or not isinstance(conversation_id, int) or conversation_id <= 0:
        raise ValueError("L'identifiant de la conversation est invalide.")


def _inserted_id(value: object, message: str) -> int:
    """Garde uniquement un identifiant SQLite positif."""
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise RuntimeError(message)
    return value


def _validate_sources(role: str, sources: object) -> list[dict[str, str | int]]:
    """Vérifie les sources avant de les sérialiser en JSON.

    Un message d'étudiant n'a pas de sources. Une réponse peut en avoir une liste,
    éventuellement vide lorsque aucun passage n'a été retenu.
    Formats acceptés :
    - PDF : text, filename, page_number, chunk_id
    - Web : type=web, title, url, snippet
    """
    if sources is None:
        sources = []
    if not isinstance(sources, list):
        raise ValueError("Les sources du message sont invalides.")
    if role == "user" and sources:
        raise ValueError("Les sources du message sont invalides.")

    stored: list[dict[str, str | int]] = []
    for source in sources:
        if not isinstance(source, dict):
            raise ValueError("Les sources du message sont invalides.")
        if source.get("type") == "web" or (
            "url" in source and "filename" not in source and "page_number" not in source
        ):
            stored.append(_validate_web_source(source))
            continue
        text = source.get("text")
        filename = source.get("filename")
        page_number = source.get("page_number")
        chunk_id = source.get("chunk_id")
        if not isinstance(text, str) or text.strip() == "":
            raise ValueError("Les sources du message sont invalides.")
        if not isinstance(filename, str) or filename.strip() == "":
            raise ValueError("Les sources du message sont invalides.")
        if not _is_positive_int(page_number) or not _is_positive_int(chunk_id):
            raise ValueError("Les sources du message sont invalides.")
        stored.append(
            {
                "text": text,
                "filename": filename,
                "page_number": page_number,
                "chunk_id": chunk_id,
            }
        )
    return stored


def _validate_web_source(source: dict) -> dict[str, str | int]:
    """Valide une source Web sans inventer d'URL."""
    url = source.get("url")
    title = source.get("title")
    snippet = source.get("snippet", "")
    if not isinstance(url, str) or not url.startswith(("http://", "https://")):
        raise ValueError("Les sources du message sont invalides.")
    if not isinstance(title, str) or title.strip() == "":
        raise ValueError("Les sources du message sont invalides.")
    if snippet is None:
        snippet = ""
    if not isinstance(snippet, str):
        raise ValueError("Les sources du message sont invalides.")
    return {
        "type": "web",
        "title": title.strip(),
        "url": url.strip(),
        "snippet": snippet,
    }


def _is_positive_int(value: object) -> bool:
    """Reconnaît un entier positif. Un booléen n'en est pas un."""
    return not isinstance(value, bool) and isinstance(value, int) and value >= 1


def _conversation_to_dict(row: sqlite3.Row) -> dict[str, str | int]:
    """Recopie une conversation dans un dictionnaire Python."""
    return {
        "id": int(row["id"]),
        "title": str(row["title"]),
        "created_at": str(row["created_at"]),
    }


def _message_to_dict(row: sqlite3.Row) -> dict[str, str | int | list]:
    """Recopie un message et décode ses sources JSON."""
    try:
        sources = json.loads(str(row["sources"]))
    except json.JSONDecodeError as error:
        raise RuntimeError("Les sources enregistrées sont illisibles.") from error
    if not isinstance(sources, list):
        raise RuntimeError("Les sources enregistrées sont illisibles.")
    return {
        "id": int(row["id"]),
        "conversation_id": int(row["conversation_id"]),
        "role": str(row["role"]),
        "content": str(row["content"]),
        "sources": sources,
        "created_at": str(row["created_at"]),
    }
