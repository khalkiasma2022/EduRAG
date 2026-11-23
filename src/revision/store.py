"""Enregistre les fiches de révision dans une base SQLite locale.

Chaque fiche reste en base même si le PDF d'origine quitte la bibliothèque.
Le chemin du fichier est choisi par l'appelant.
"""

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

_LANGUAGES = frozenset({"fr", "en", "ar"})
_LEVELS = frozenset({"beginner", "intermediate", "advanced"})
_SHEET_TYPES = frozenset({"short", "complete"})

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS revision_sheets (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    document_id TEXT NOT NULL,
    filename TEXT NOT NULL,
    page_start INTEGER NOT NULL,
    page_end INTEGER NOT NULL,
    language TEXT NOT NULL,
    level TEXT NOT NULL,
    sheet_type TEXT NOT NULL,
    content TEXT NOT NULL,
    created_at TEXT NOT NULL
)
"""

_INSERT_SHEET = """
INSERT INTO revision_sheets (
    document_id, filename, page_start, page_end,
    language, level, sheet_type, content, created_at
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
"""

_SELECT_SHEET = """
SELECT id, document_id, filename, page_start, page_end,
       language, level, sheet_type, content, created_at
FROM revision_sheets
WHERE id = ?
"""

_SELECT_SHEETS = """
SELECT id, document_id, filename, page_start, page_end,
       language, level, sheet_type, content, created_at
FROM revision_sheets
ORDER BY created_at DESC, id DESC
"""


class RevisionStore:
    """Conserve les fiches générées dans un fichier SQLite."""

    def __init__(self, path: str | Path) -> None:
        """Ouvre ou crée la base, puis crée la table si elle manque."""
        if isinstance(path, str) and path.strip() == "":
            raise ValueError("Le chemin de la base ne peut pas être vide.")
        if not isinstance(path, (str, Path)):
            raise TypeError("Le chemin de la base doit être un chemin de fichier.")

        self.path = Path(path)
        if self.path.exists() and self.path.is_dir():
            raise ValueError("Le chemin de la base doit désigner un fichier.")

        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connection() as connection:
            connection.execute(_CREATE_TABLE)

    def save(self, sheet: object) -> dict[str, str | int]:
        """Enregistre le dictionnaire produit par RevisionSheet.generate()."""
        fields = _validate_sheet(sheet)
        created_at = datetime.now(timezone.utc).isoformat(timespec="microseconds")
        with self._connection() as connection:
            cursor = connection.execute(
                _INSERT_SHEET,
                (
                    fields["document_id"],
                    fields["filename"],
                    fields["page_start"],
                    fields["page_end"],
                    fields["language"],
                    fields["level"],
                    fields["sheet_type"],
                    fields["content"],
                    created_at,
                ),
            )
            sheet_id = cursor.lastrowid
        if not isinstance(sheet_id, int) or sheet_id <= 0:
            raise RuntimeError("La fiche n'a pas pu être enregistrée.")
        return self.get(sheet_id)

    def get(self, sheet_id: object) -> dict[str, str | int]:
        """Retrouve une fiche à partir de son identifiant SQLite."""
        _check_sheet_id(sheet_id)
        with self._connection() as connection:
            row = connection.execute(_SELECT_SHEET, (sheet_id,)).fetchone()
        if row is None:
            raise ValueError("Fiche introuvable.")
        return _row_to_dict(row)

    def list_sheets(self) -> list[dict[str, str | int]]:
        """Renvoie les fiches, de la plus récente à la plus ancienne."""
        with self._connection() as connection:
            rows = connection.execute(_SELECT_SHEETS).fetchall()
        return [_row_to_dict(row) for row in rows]

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        """Ouvre une transaction, puis ferme la connexion dans tous les cas."""
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()


def _validate_sheet(sheet: object) -> dict[str, str | int]:
    """Vérifie le dictionnaire avant toute écriture."""
    if not isinstance(sheet, dict):
        raise ValueError("La fiche à enregistrer est invalide.")

    document_id = _required_text(sheet.get("document_id"), "L'identifiant du document est invalide.")
    filename = _required_text(sheet.get("filename"), "Le nom du fichier est invalide.")
    page_start = sheet.get("page_start")
    page_end = sheet.get("page_end")
    if not _is_page_number(page_start) or not _is_page_number(page_end) or page_end < page_start:
        raise ValueError("L'intervalle de pages est invalide.")
    language = _required_choice(sheet.get("language"), _LANGUAGES, "La langue demandée n'est pas prise en charge.")
    level = _required_choice(sheet.get("level"), _LEVELS, "Le niveau demandé n'est pas pris en charge.")
    sheet_type = _required_choice(
        sheet.get("sheet_type"),
        _SHEET_TYPES,
        "Le type de fiche demandé n'est pas pris en charge.",
    )
    content = _required_text(sheet.get("content"), "Le contenu de la fiche est invalide.")
    return {
        "document_id": document_id,
        "filename": filename,
        "page_start": page_start,
        "page_end": page_end,
        "language": language,
        "level": level,
        "sheet_type": sheet_type,
        "content": content.strip(),
    }


def _required_text(value: object, message: str) -> str:
    """Exige un texte qui contient autre chose que des espaces."""
    if not isinstance(value, str) or value.strip() == "":
        raise ValueError(message)
    return value.strip()


def _required_choice(value: object, accepted: frozenset[str], message: str) -> str:
    """Exige une valeur prévue, sans la corriger."""
    if not isinstance(value, str) or value not in accepted:
        raise ValueError(message)
    return value


def _is_page_number(value: object) -> bool:
    """Reconnaît un numéro de page. Un booléen n'en est pas un."""
    return not isinstance(value, bool) and isinstance(value, int) and value >= 1


def _check_sheet_id(sheet_id: object) -> None:
    """Refuse un identifiant qui ne vient pas de SQLite."""
    if isinstance(sheet_id, bool) or not isinstance(sheet_id, int) or sheet_id <= 0:
        raise ValueError("L'identifiant de la fiche est invalide.")


def _row_to_dict(row: sqlite3.Row) -> dict[str, str | int]:
    """Recopie une ligne SQL dans un dictionnaire Python."""
    return {
        "id": int(row["id"]),
        "document_id": str(row["document_id"]),
        "filename": str(row["filename"]),
        "page_start": int(row["page_start"]),
        "page_end": int(row["page_end"]),
        "language": str(row["language"]),
        "level": str(row["level"]),
        "sheet_type": str(row["sheet_type"]),
        "content": str(row["content"]),
        "created_at": str(row["created_at"]),
    }
