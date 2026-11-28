"""Journalisation du pipeline Smart Revision dans le terminal serveur."""

from __future__ import annotations

import sys
from datetime import datetime, timezone


def log_progress(message: str) -> None:
    """Affiche une étape visible pendant la génération (stdout, flush immédiat)."""
    if not isinstance(message, str) or message.strip() == "":
        return
    stamp = datetime.now(timezone.utc).astimezone().strftime("%H:%M:%S")
    print(f"[EduRAG {stamp}] {message.strip()}", flush=True, file=sys.stdout)
