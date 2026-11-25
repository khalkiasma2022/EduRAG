"""Routes HTTP des cours PDF.

Ces routes appellent l'agent déjà enregistré.
Elles ne construisent ni Qwen, ni MiniLM, ni l'index FAISS.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path, PureWindowsPath
from typing import Protocol

from dotenv import load_dotenv
from fastapi import APIRouter, Depends, HTTPException, Request, UploadFile
from pydantic import BaseModel, Field

from src.api.dependencies import get_assistant
from src.ingestion.pdf_loader import load_pdf

# 20 Mio : un cours tient largement, un envoi accidentel est arrêté.
DEFAULT_MAX_UPLOAD_BYTES = 20 * 1024 * 1024
_DUPLICATE = "Ce document est déjà dans la bibliothèque."
_MISSING = "Document introuvable dans la bibliothèque."
_PDF_MEDIA_TYPES = {"", "application/pdf", "application/octet-stream", "application/x-pdf"}


class DocumentAssistant(Protocol):
    """Méthodes de cours déjà fournies par EduRAGAssistant."""

    def list_documents(self) -> list[dict[str, str]]:
        """Retourne les cours enregistrés."""

    def add_pdf(self, file_path: str | Path) -> dict[str, str]:
        """Ajoute un PDF à partir d'un chemin."""

    def remove_document(self, document_id: str) -> None:
        """Retire un cours par son identifiant."""


class DocumentPublic(BaseModel):
    """Cours renvoyé au client."""

    document_id: str = Field(description="Identifiant stable du cours.")
    filename: str = Field(description="Nom d'origine du PDF.")


router = APIRouter(prefix="/documents", tags=["documents"])


def max_upload_bytes() -> int:
    """Lit la taille maximale d'un PDF.

    EDURAG_MAX_UPLOAD_BYTES est un nombre d'octets.
    Sans cette variable, la limite vaut 20 Mio.
    """
    load_dotenv()
    raw = os.environ.get("EDURAG_MAX_UPLOAD_BYTES", "").strip()
    if raw == "":
        return DEFAULT_MAX_UPLOAD_BYTES
    if not raw.isdecimal() or int(raw) < 1:
        raise HTTPException(
            status_code=500,
            detail="EDURAG_MAX_UPLOAD_BYTES doit être un entier positif.",
        )
    return int(raw)


def document_assistant(request: Request) -> DocumentAssistant:
    """Retourne l'unique agent enregistré au démarrage."""
    return get_assistant(request)  # type: ignore[return-value]


def _public_document(entry: object) -> DocumentPublic:
    """Garde l'identifiant et le nom. Le chemin interne reste côté serveur."""
    if not isinstance(entry, dict):
        raise HTTPException(status_code=500, detail="La bibliothèque a renvoyé un document incomplet.")
    document_id = entry.get("document_id")
    filename = entry.get("filename")
    if not isinstance(document_id, str) or document_id.strip() == "":
        raise HTTPException(status_code=500, detail="La bibliothèque a renvoyé un document incomplet.")
    if not isinstance(filename, str) or filename.strip() == "":
        raise HTTPException(status_code=500, detail="La bibliothèque a renvoyé un document incomplet.")
    return DocumentPublic(document_id=document_id, filename=filename)


def _pdf_name(upload: UploadFile) -> str:
    """Réduit le nom au fichier et refuse un autre type."""
    name = PureWindowsPath(upload.filename or "").name
    media = (upload.content_type or "").split(";", 1)[0].strip().lower()
    if name in {"", ".", ".."} or not name.lower().endswith(".pdf") or media not in _PDF_MEDIA_TYPES:
        raise HTTPException(status_code=400, detail="Le fichier sélectionné doit être un PDF.")
    return name


def _pdf_bytes(upload: UploadFile, filename: str) -> bytes:
    """Lit le fichier jusqu'à la limite, puis contrôle qu'il ressemble à un PDF."""
    limit = max_upload_bytes()
    payload = upload.file.read(limit + 1)
    if not isinstance(payload, bytes):
        raise HTTPException(status_code=400, detail="Le fichier PDF est illisible.")
    if len(payload) > limit:
        raise HTTPException(status_code=413, detail="Le PDF dépasse la taille autorisée.")
    if len(payload) == 0:
        raise HTTPException(status_code=400, detail="Le fichier PDF est vide.")
    if b"%PDF-" not in payload[:1024]:
        raise HTTPException(status_code=400, detail=f"Fichier PDF invalide : {filename}")
    return payload


def _add_pdf(assistant: DocumentAssistant, filename: str, payload: bytes) -> DocumentPublic:
    """Vérifie le contenu réel, puis transmet le fichier à l'agent."""
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / filename
        path.write_bytes(payload)
        try:
            load_pdf(path)
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        try:
            added = assistant.add_pdf(path)
        except ValueError as error:
            status = 409 if str(error) == _DUPLICATE else 400
            raise HTTPException(status_code=status, detail=str(error)) from error
    return _public_document(added)


@router.get(
    "",
    response_model=list[DocumentPublic],
    summary="Lister les cours PDF",
    responses={
        200: {
            "description": (
                "Cours enregistrés, du plus ancien au plus récent. "
                "Chaque élément contient document_id et filename."
            )
        }
    },
)
def list_documents(
    assistant: DocumentAssistant = Depends(document_assistant),
) -> list[DocumentPublic]:
    """Retourne les cours sans chemin interne ni empreinte."""
    return [_public_document(entry) for entry in assistant.list_documents()]


@router.post(
    "",
    status_code=201,
    response_model=DocumentPublic,
    summary="Importer un cours PDF",
    responses={
        201: {"description": "Cours enregistré. La réponse contient document_id et filename."},
        400: {"description": "Fichier vide, type refusé, nom invalide ou PDF illisible."},
        409: {"description": "Ce document est déjà dans la bibliothèque."},
        413: {"description": "Le PDF dépasse EDURAG_MAX_UPLOAD_BYTES, ou 20 Mio par défaut."},
    },
)
def create_document(
    file: UploadFile,
    assistant: DocumentAssistant = Depends(document_assistant),
) -> DocumentPublic:
    """Reçoit un PDF, vérifie son contenu, puis appelle assistant.add_pdf."""
    filename = _pdf_name(file)
    payload = _pdf_bytes(file, filename)
    return _add_pdf(assistant, filename, payload)


@router.delete(
    "/{document_id}",
    status_code=204,
    summary="Retirer un cours PDF",
    responses={
        204: {"description": "Cours retiré. La réponse n'a pas de corps."},
        400: {"description": "L'identifiant du document est invalide."},
        404: {"description": "Document introuvable dans la bibliothèque."},
    },
)
def delete_document(
    document_id: str,
    assistant: DocumentAssistant = Depends(document_assistant),
) -> None:
    """Retire le cours avec assistant.remove_document."""
    try:
        assistant.remove_document(document_id)
    except ValueError as error:
        status = 404 if str(error) == _MISSING else 400
        raise HTTPException(status_code=status, detail=str(error)) from error
