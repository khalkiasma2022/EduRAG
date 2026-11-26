"""Routes HTTP des fiches de révision.

Ces routes appellent l'agent déjà enregistré.
Elles ne construisent ni Qwen, ni MiniLM, ni l'index FAISS.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import NoReturn, Protocol

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import Response
from pydantic import BaseModel, Field

from src.api.dependencies import get_assistant

_MISSING_DOCUMENT = "Document introuvable dans la bibliothèque."
_MISSING_SHEET = "Fiche introuvable."
_UNREADABLE = "La fiche enregistrée est illisible."


class RevisionAssistant(Protocol):
    """Méthodes de fiches déjà fournies par EduRAGAssistant."""

    def create_revision(
        self,
        document_id: str,
        page_start: int | None = None,
        page_end: int | None = None,
        language: str = "fr",
        level: str = "beginner",
        sheet_type: str = "complete",
    ) -> dict[str, str | int]:
        """Produit une fiche, puis l'enregistre."""

    def list_revisions(self) -> list[dict[str, str | int]]:
        """Retourne les fiches dans l'ordre du stockage."""

    def get_revision(self, sheet_id: object) -> dict[str, str | int]:
        """Relit une fiche enregistrée."""

    def export_revision_markdown(self, sheet_id: object, output_path: str | Path) -> Path:
        """Écrit une fiche déjà enregistrée en Markdown."""

    def export_revision_pdf(self, sheet_id: object, output_path: str | Path) -> Path:
        """Écrit une fiche déjà enregistrée en PDF."""


class RevisionPublic(BaseModel):
    """Fiche visible par le client."""

    id: int
    document_id: str
    filename: str
    page_start: int
    page_end: int
    language: str
    level: str
    sheet_type: str
    content: str
    created_at: str


class RevisionCreate(BaseModel):
    """Paramètres d'une nouvelle fiche.

    Les nouveaux clients n'envoient que ``document_id`` et ``language``.
    ``level``, ``sheet_type``, ``page_start`` et ``page_end`` restent acceptés
    pour compatibilité mais sont ignorés : l'analyse porte sur tout le PDF et
    le stockage utilise les valeurs legacy ``intermediate`` / ``complete``.
    """

    document_id: str = Field(description="Identifiant du cours PDF.")
    language: str = Field(default="fr", description="Langue : fr, en ou ar.")
    level: str = Field(
        default="beginner",
        description="Ignoré (compatibilité). Valeur persistée : intermediate.",
    )
    sheet_type: str = Field(
        default="complete",
        description="Ignoré (compatibilité). Valeur persistée : complete.",
    )
    page_start: int | None = Field(
        default=None,
        description="Ignoré (compatibilité). L'analyse couvre toujours tout le PDF.",
    )
    page_end: int | None = Field(
        default=None,
        description="Ignoré (compatibilité). L'analyse couvre toujours tout le PDF.",
    )


router = APIRouter(prefix="/revisions", tags=["revisions"])


def revision_assistant(request: Request) -> RevisionAssistant:
    """Retourne l'unique agent enregistré au démarrage."""
    return get_assistant(request)  # type: ignore[return-value]


def _raise_revision_error(error: ValueError) -> NoReturn:
    """Traduit une erreur métier en réponse HTTP."""
    message = str(error)
    if message in {_MISSING_DOCUMENT, _MISSING_SHEET}:
        status = 404
    else:
        status = 400
    raise HTTPException(status_code=status, detail=message) from error


def _required_text(value: object) -> str:
    """Garde un texte non vide."""
    if not isinstance(value, str) or value.strip() == "":
        raise HTTPException(status_code=500, detail=_UNREADABLE)
    return value


def _required_id(value: object) -> int:
    """Garde un identifiant SQLite positif."""
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise HTTPException(status_code=500, detail=_UNREADABLE)
    return value


def _required_page(value: object) -> int:
    """Garde un numéro de page positif."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise HTTPException(status_code=500, detail=_UNREADABLE)
    return value


def _revision_public(entry: object) -> RevisionPublic:
    """Recopie la fiche sans chemin local."""
    if not isinstance(entry, dict):
        raise HTTPException(status_code=500, detail=_UNREADABLE)
    return RevisionPublic(
        id=_required_id(entry.get("id")),
        document_id=_required_text(entry.get("document_id")),
        filename=_required_text(entry.get("filename")),
        page_start=_required_page(entry.get("page_start")),
        page_end=_required_page(entry.get("page_end")),
        language=_required_text(entry.get("language")),
        level=_required_text(entry.get("level")),
        sheet_type=_required_text(entry.get("sheet_type")),
        content=_required_text(entry.get("content")),
        created_at=_required_text(entry.get("created_at")),
    )


def _export_file(
    assistant: RevisionAssistant,
    sheet_id: int,
    kind: str,
) -> Response:
    """Écrit l'export dans un dossier temporaire, puis renvoie ses octets."""
    suffix = ".md" if kind == "markdown" else ".pdf"
    media_type = "text/markdown; charset=utf-8" if kind == "markdown" else "application/pdf"
    filename = f"fiche-{sheet_id}{suffix}"
    with tempfile.TemporaryDirectory() as folder:
        target = Path(folder) / filename
        try:
            if kind == "markdown":
                written = assistant.export_revision_markdown(sheet_id, target)
            else:
                written = assistant.export_revision_pdf(sheet_id, target)
        except ValueError as error:
            _raise_revision_error(error)
        payload = Path(written).read_bytes()
    return Response(
        content=payload,
        media_type=media_type,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.post(
    "",
    status_code=201,
    response_model=RevisionPublic,
    summary="Générer une fiche",
    responses={
        201: {"description": "Fiche enregistrée, avec son contenu et ses paramètres."},
        400: {"description": "Langue refusée ou document inexploitable."},
        404: {"description": "Document introuvable dans la bibliothèque."},
        500: {"description": "La génération de la fiche a échoué."},
    },
)
def create_revision(
    body: RevisionCreate,
    assistant: RevisionAssistant = Depends(revision_assistant),
) -> RevisionPublic:
    """Appelle create_revision une seule fois, avec les paramètres reçus."""
    try:
        saved = assistant.create_revision(
            body.document_id,
            page_start=body.page_start,
            page_end=body.page_end,
            language=body.language,
            level=body.level,
            sheet_type=body.sheet_type,
        )
    except ValueError as error:
        _raise_revision_error(error)
    except RuntimeError as error:
        raise HTTPException(status_code=500, detail=str(error)) from error
    return _revision_public(saved)


@router.get(
    "",
    response_model=list[RevisionPublic],
    summary="Lister les fiches",
    responses={
        200: {
            "description": (
                "Fiches enregistrées, de la plus récente à la plus ancienne. "
                "Aucune fiche n'est régénérée."
            )
        }
    },
)
def list_revisions(
    assistant: RevisionAssistant = Depends(revision_assistant),
) -> list[RevisionPublic]:
    """Retourne l'historique sans appeler le modèle."""
    return [_revision_public(entry) for entry in assistant.list_revisions()]


@router.get(
    "/{revision_id}",
    response_model=RevisionPublic,
    summary="Lire une fiche",
    responses={
        200: {"description": "Fiche enregistrée. Elle n'est pas régénérée."},
        400: {"description": "L'identifiant de la fiche est invalide."},
        404: {"description": "Fiche introuvable."},
    },
)
def read_revision(
    revision_id: int,
    assistant: RevisionAssistant = Depends(revision_assistant),
) -> RevisionPublic:
    """Relit la fiche enregistrée."""
    try:
        stored = assistant.get_revision(revision_id)
    except ValueError as error:
        _raise_revision_error(error)
    return _revision_public(stored)


@router.get(
    "/{revision_id}/markdown",
    summary="Télécharger une fiche Markdown",
    responses={
        200: {"description": "Fichier Markdown de la fiche enregistrée.", "content": {"text/markdown": {}}},
        400: {"description": "L'identifiant de la fiche est invalide."},
        404: {"description": "Fiche introuvable."},
    },
)
def export_revision_markdown(
    revision_id: int,
    assistant: RevisionAssistant = Depends(revision_assistant),
) -> Response:
    """Exporte la fiche déjà enregistrée, sans la régénérer."""
    return _export_file(assistant, revision_id, "markdown")


@router.get(
    "/{revision_id}/pdf",
    summary="Télécharger une fiche PDF",
    responses={
        200: {"description": "PDF de la fiche enregistrée, y compris l'arabe de droite à gauche.", "content": {"application/pdf": {}}},
        400: {"description": "L'identifiant de la fiche est invalide."},
        404: {"description": "Fiche introuvable."},
    },
)
def export_revision_pdf(
    revision_id: int,
    assistant: RevisionAssistant = Depends(revision_assistant),
) -> Response:
    """Exporte la fiche déjà enregistrée avec le générateur PDF existant."""
    return _export_file(assistant, revision_id, "pdf")
