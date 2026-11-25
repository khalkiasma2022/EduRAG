"""Routes HTTP des conversations.

Ces routes appellent l'agent déjà enregistré.
Elles ne construisent ni Qwen, ni MiniLM, ni l'index FAISS.
"""

from __future__ import annotations

from typing import NoReturn, Protocol

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from src.api.dependencies import get_assistant

_MISSING = "Conversation introuvable."


class ConversationAssistant(Protocol):
    """Méthodes de conversation déjà fournies par EduRAGAssistant."""

    def create_conversation(self, title: str) -> int:
        """Crée une conversation vide."""

    def list_conversations(self) -> list[dict[str, str | int]]:
        """Retourne les conversations dans l'ordre du stockage."""

    def get_conversation(self, conversation_id: object) -> dict[str, str | int | list]:
        """Relit une conversation et ses messages."""

    def delete_conversation(self, conversation_id: object) -> None:
        """Supprime une conversation et ses messages."""

    def send_message(
        self,
        conversation_id: object,
        question: str,
        top_k: int = 3,
    ) -> dict[str, str | int | dict]:
        """Enregistre la question, interroge le cours une fois, puis enregistre la réponse."""


class ConversationPublic(BaseModel):
    """Conversation visible par le client."""

    id: int = Field(description="Identifiant SQLite de la conversation.")
    title: str = Field(description="Titre enregistré.")
    created_at: str = Field(description="Date de création, fournie par le stockage.")


class SourcePublic(BaseModel):
    """Passage cité dans une réponse."""

    text: str
    filename: str
    page_number: int
    chunk_id: int


class MessagePublic(BaseModel):
    """Message enregistré."""

    id: int
    conversation_id: int
    role: str
    content: str
    sources: list[SourcePublic]
    created_at: str


class ConversationDetail(ConversationPublic):
    """Conversation avec ses messages, du plus ancien au plus récent."""

    messages: list[MessagePublic]


class MessageExchange(BaseModel):
    """Question enregistrée et réponse enregistrée."""

    conversation_id: int
    question: MessagePublic
    answer: MessagePublic


class ConversationCreate(BaseModel):
    """Titre demandé pour une nouvelle conversation."""

    title: str = Field(description="Titre de la conversation.")


class QuestionCreate(BaseModel):
    """Question posée dans une conversation existante."""

    question: str = Field(description="Question de l'étudiant.")


router = APIRouter(prefix="/conversations", tags=["conversations"])


def conversation_assistant(request: Request) -> ConversationAssistant:
    """Retourne l'unique agent enregistré au démarrage."""
    return get_assistant(request)  # type: ignore[return-value]


def _raise_store_error(error: ValueError) -> NoReturn:
    """Traduit une erreur du stockage en réponse HTTP."""
    status = 404 if str(error) == _MISSING else 400
    raise HTTPException(status_code=status, detail=str(error)) from error


def _require_text(value: object, detail: str) -> str:
    """Garde un texte non vide."""
    if not isinstance(value, str) or value.strip() == "":
        raise HTTPException(status_code=500, detail=detail)
    return value


def _conversation_public(entry: object) -> ConversationPublic:
    """Garde l'identifiant, le titre et la date."""
    if not isinstance(entry, dict):
        raise HTTPException(status_code=500, detail="La conversation enregistrée est illisible.")
    conversation_id = entry.get("id")
    if isinstance(conversation_id, bool) or not isinstance(conversation_id, int) or conversation_id <= 0:
        raise HTTPException(status_code=500, detail="La conversation enregistrée est illisible.")
    return ConversationPublic(
        id=conversation_id,
        title=_require_text(entry.get("title"), "La conversation enregistrée est illisible."),
        created_at=_require_text(entry.get("created_at"), "La conversation enregistrée est illisible."),
    )


def _source_public(source: object) -> SourcePublic:
    """Garde le passage cité, sans chemin local."""
    if not isinstance(source, dict):
        raise HTTPException(status_code=500, detail="Les sources enregistrées sont illisibles.")
    page_number = source.get("page_number")
    chunk_id = source.get("chunk_id")
    if isinstance(page_number, bool) or not isinstance(page_number, int) or page_number < 1:
        raise HTTPException(status_code=500, detail="Les sources enregistrées sont illisibles.")
    if isinstance(chunk_id, bool) or not isinstance(chunk_id, int) or chunk_id < 1:
        raise HTTPException(status_code=500, detail="Les sources enregistrées sont illisibles.")
    return SourcePublic(
        text=_require_text(source.get("text"), "Les sources enregistrées sont illisibles."),
        filename=_require_text(source.get("filename"), "Les sources enregistrées sont illisibles."),
        page_number=page_number,
        chunk_id=chunk_id,
    )


def _message_public(entry: object) -> MessagePublic:
    """Recopie un message et ses sources publiques."""
    if not isinstance(entry, dict):
        raise HTTPException(status_code=500, detail="Le message enregistré est illisible.")
    message_id = entry.get("id")
    conversation_id = entry.get("conversation_id")
    role = entry.get("role")
    sources = entry.get("sources")
    if isinstance(message_id, bool) or not isinstance(message_id, int) or message_id <= 0:
        raise HTTPException(status_code=500, detail="Le message enregistré est illisible.")
    if isinstance(conversation_id, bool) or not isinstance(conversation_id, int) or conversation_id <= 0:
        raise HTTPException(status_code=500, detail="Le message enregistré est illisible.")
    if role not in {"user", "assistant"}:
        raise HTTPException(status_code=500, detail="Le message enregistré est illisible.")
    if not isinstance(sources, list):
        raise HTTPException(status_code=500, detail="Les sources enregistrées sont illisibles.")
    return MessagePublic(
        id=message_id,
        conversation_id=conversation_id,
        role=role,
        content=_require_text(entry.get("content"), "Le message enregistré est illisible."),
        sources=[_source_public(source) for source in sources],
        created_at=_require_text(entry.get("created_at"), "Le message enregistré est illisible."),
    )


def _conversation_detail(entry: object) -> ConversationDetail:
    """Ajoute les messages dans l'ordre déjà fourni par le stockage."""
    if not isinstance(entry, dict) or not isinstance(entry.get("messages"), list):
        raise HTTPException(status_code=500, detail="La conversation enregistrée est illisible.")
    public = _conversation_public(entry)
    return ConversationDetail(
        id=public.id,
        title=public.title,
        created_at=public.created_at,
        messages=[_message_public(message) for message in entry["messages"]],
    )


def _exchange(result: object) -> MessageExchange:
    """Retourne la question et la réponse enregistrées."""
    if not isinstance(result, dict):
        raise HTTPException(status_code=500, detail="La réponse enregistrée est illisible.")
    conversation_id = result.get("conversation_id")
    if isinstance(conversation_id, bool) or not isinstance(conversation_id, int) or conversation_id <= 0:
        raise HTTPException(status_code=500, detail="La réponse enregistrée est illisible.")
    return MessageExchange(
        conversation_id=conversation_id,
        question=_message_public(result.get("question")),
        answer=_message_public(result.get("answer")),
    )


@router.post(
    "",
    status_code=201,
    response_model=ConversationPublic,
    summary="Créer une conversation",
    responses={
        201: {"description": "Conversation créée. La réponse contient id, title et created_at."},
        400: {"description": "Le titre de la conversation est invalide."},
    },
)
def create_conversation(
    body: ConversationCreate,
    assistant: ConversationAssistant = Depends(conversation_assistant),
) -> ConversationPublic:
    """Crée une conversation vide et relit le titre enregistré."""
    try:
        conversation_id = assistant.create_conversation(body.title)
        stored = assistant.get_conversation(conversation_id)
    except ValueError as error:
        _raise_store_error(error)
    return _conversation_public(stored)


@router.get(
    "",
    response_model=list[ConversationPublic],
    summary="Lister les conversations",
    responses={
        200: {
            "description": (
                "Conversations dans l'ordre du stockage : "
                "de la plus récente à la plus ancienne."
            )
        }
    },
)
def list_conversations(
    assistant: ConversationAssistant = Depends(conversation_assistant),
) -> list[ConversationPublic]:
    """Retourne l'historique sans le réordonner."""
    return [_conversation_public(entry) for entry in assistant.list_conversations()]


@router.get(
    "/{conversation_id}",
    response_model=ConversationDetail,
    summary="Lire une conversation",
    responses={
        200: {"description": "Conversation, messages chronologiques et sources des réponses."},
        400: {"description": "L'identifiant de la conversation est invalide."},
        404: {"description": "Conversation introuvable."},
    },
)
def read_conversation(
    conversation_id: int,
    assistant: ConversationAssistant = Depends(conversation_assistant),
) -> ConversationDetail:
    """Relit la conversation sans interroger le modèle."""
    try:
        stored = assistant.get_conversation(conversation_id)
    except ValueError as error:
        _raise_store_error(error)
    return _conversation_detail(stored)


@router.delete(
    "/{conversation_id}",
    status_code=204,
    summary="Supprimer une conversation",
    responses={
        204: {"description": "Conversation et messages supprimés. La réponse n'a pas de corps."},
        400: {"description": "L'identifiant de la conversation est invalide."},
        404: {"description": "Conversation introuvable."},
    },
)
def delete_conversation(
    conversation_id: int,
    assistant: ConversationAssistant = Depends(conversation_assistant),
) -> None:
    """Supprime la conversation avec ses messages."""
    try:
        assistant.delete_conversation(conversation_id)
    except ValueError as error:
        _raise_store_error(error)


@router.post(
    "/{conversation_id}/messages",
    response_model=MessageExchange,
    summary="Envoyer une question",
    responses={
        200: {"description": "Question et réponse enregistrées, avec les sources de la réponse."},
        400: {"description": "Question vide ou invalide, ou identifiant invalide."},
        404: {"description": "Conversation introuvable."},
        500: {"description": "La génération de la réponse a échoué."},
    },
)
def create_message(
    conversation_id: int,
    body: QuestionCreate,
    assistant: ConversationAssistant = Depends(conversation_assistant),
) -> MessageExchange:
    """Appelle send_message une seule fois.

    Les messages précédents ne sont pas ajoutés ici :
    send_message ne les transmet pas au modèle.
    """
    try:
        result = assistant.send_message(conversation_id, body.question)
    except ValueError as error:
        _raise_store_error(error)
    except RuntimeError as error:
        raise HTTPException(status_code=500, detail=str(error)) from error
    return _exchange(result)
