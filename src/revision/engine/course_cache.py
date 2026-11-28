"""Cache disque de CourseModel pour éviter de relancer la compréhension."""

from __future__ import annotations

import json
import re
import tempfile
from pathlib import Path

from pydantic import ValidationError

from src.revision.engine.models import CourseModel

_SAFE_ID = re.compile(r"[^A-Za-z0-9._-]+")


class CourseUnderstandingCache:
    """Persiste un CourseModel par document_id sous forme de JSON atomique."""

    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    def path_for(self, document_id: str) -> Path:
        safe = _SAFE_ID.sub("_", (document_id or "").strip())
        if not safe:
            raise ValueError("L'identifiant du document est invalide.")
        return self.directory / f"{safe}.json"

    def load(self, document_id: str) -> CourseModel | None:
        path = self.path_for(document_id)
        if not path.is_file():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            return None
        if not isinstance(payload, dict):
            return None
        course_raw = payload.get("course")
        if not isinstance(course_raw, dict):
            return None
        try:
            return CourseModel.model_validate(course_raw)
        except ValidationError:
            return None

    def save(
        self,
        document_id: str,
        *,
        filename: str,
        page_start: int,
        page_end: int,
        course: CourseModel,
    ) -> None:
        if not isinstance(course, CourseModel):
            raise TypeError("course doit être un CourseModel.")
        path = self.path_for(document_id)
        payload = {
            "document_id": document_id,
            "filename": filename,
            "page_start": int(page_start),
            "page_end": int(page_end),
            "course": course.model_dump(mode="json"),
        }
        text = json.dumps(payload, ensure_ascii=False, indent=2)
        self.directory.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(
            prefix=f".{path.stem}.",
            suffix=".tmp",
            dir=self.directory,
        )
        tmp_path = Path(tmp_name)
        try:
            with open(fd, "w", encoding="utf-8") as handle:
                handle.write(text)
            tmp_path.replace(path)
        except Exception:
            tmp_path.unlink(missing_ok=True)
            raise
