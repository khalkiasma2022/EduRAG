"""Objet central d'EduRAG.

Il crée les modèles une seule fois, puis les partage
entre la bibliothèque de PDF, la chaîne de réponse, les fiches et les conversations.
"""

from pathlib import Path

from src.generation.llm import LLM
from src.generation.rag_chain import RAGChain
from src.ingestion.library import DocumentLibrary
from src.retrieval.embeddings import EmbeddingModel
from src.retrieval.vector_store import (
    VectorStore,
    _directory_to_load,
    _require_trusted_manifest,
)
from src.revision.export import RevisionExport
from src.revision.sheet import RevisionSheet
from src.revision.store import RevisionStore
from src.tutor.direct_chat import DirectChatResponder
from src.tutor.history_window import HistoryWindow
from src.tutor.intent_router import GENERAL_CHAT, WEB_QUESTION, IntentRouter
from src.tutor.store import ConversationStore
from src.tutor.web_answer import WebAnswerComposer
from src.tutor.web_search import WebSearcher


class EduRAGAssistant:
    """Relie l'ajout des cours et les questions de l'apprenant."""

    def __init__(
        self,
        documents_directory: str | Path,
        index_directory: str | Path,
        revision_database_path: str | Path | None = None,
        conversation_database_path: str | Path | None = None,
    ) -> None:
        """Crée les composants, puis ouvre ou reconstruit l'index si besoin."""
        self.documents_directory = Path(documents_directory).resolve()
        self.index_directory = Path(index_directory).resolve()
        if revision_database_path is None:
            revision_database_path = self.documents_directory / "revisions.sqlite"
        if conversation_database_path is None:
            conversation_database_path = Path(revision_database_path).parent / "conversations.sqlite"

        # Une seule création : les mêmes objets sont ensuite réutilisés.
        self.embedding_model = EmbeddingModel()
        self.vector_store = VectorStore(self.embedding_model)
        self.llm = LLM()
        self.library = DocumentLibrary(
            self.documents_directory,
            self.index_directory,
            self.embedding_model,
            self.vector_store,
        )
        self.rag_chain = RAGChain(
            self.embedding_model,
            self.vector_store,
            self.llm,
        )
        self.intent_router = IntentRouter(self.llm)
        self.direct_chat = DirectChatResponder(self.llm)
        self.history_window = HistoryWindow()
        self.web_searcher = WebSearcher()
        self.web_answer = WebAnswerComposer(self.llm)
        self.revision_sheet = RevisionSheet(self.library, self.llm)
        self.revision_store = RevisionStore(revision_database_path)
        self.revision_export = RevisionExport()
        self.conversation_store = ConversationStore(conversation_database_path)
        self._open_existing_index()

    def create_revision(
        self,
        document_id: str,
        page_start: int | None = None,
        page_end: int | None = None,
        language: str = "fr",
        level: str = "beginner",
        sheet_type: str = "complete",
    ) -> dict[str, str | int]:
        """Produit une fiche avec le modèle déjà chargé, puis l'enregistre."""
        generated = self.revision_sheet.generate(
            document_id,
            page_start=page_start,
            page_end=page_end,
            language=language,
            level=level,
            sheet_type=sheet_type,
        )
        return self.revision_store.save(generated)

    def list_revisions(self) -> list[dict[str, str | int]]:
        """Retourne les fiches enregistrées, de la plus récente à la plus ancienne."""
        return self.revision_store.list_sheets()

    def get_revision(self, sheet_id: object) -> dict[str, str | int]:
        """Retrouve une fiche enregistrée, même si son PDF a quitté la bibliothèque."""
        return self.revision_store.get(sheet_id)

    def export_revision_markdown(self, sheet_id: object, output_path: str | Path) -> Path:
        """Écrit une fiche déjà enregistrée en Markdown."""
        sheet = self.get_revision(sheet_id)
        return self.revision_export.export_markdown(sheet, output_path)

    def export_revision_pdf(self, sheet_id: object, output_path: str | Path) -> Path:
        """Écrit une fiche déjà enregistrée en PDF."""
        sheet = self.get_revision(sheet_id)
        return self.revision_export.export_pdf(sheet, output_path)

    def add_pdf(self, file_path: str | Path) -> dict[str, str]:
        """Ajoute un PDF avec la bibliothèque déjà ouverte."""
        return self.library.add_pdf(file_path)

    def list_documents(self) -> list[dict[str, str]]:
        """Retourne les cours enregistrés."""
        return self.library.list_documents()

    def remove_document(self, document_id: str) -> None:
        """Retire un cours par son identifiant.

        Quand il reste des PDF, la bibliothèque reconstruit l'index et
        remplace celui qui est en mémoire. Quand le dernier PDF part,
        elle efface seulement le dossier : l'objet FAISS garderait les
        anciens vecteurs. Il n'existe pas de méthode publique pour le
        vider, donc on le remplace par un index vide partagé.
        """
        self.library.remove_document(document_id)
        if self.library.list_documents():
            return
        self.vector_store = VectorStore(self.embedding_model)
        self.library.vector_store = self.vector_store
        self.rag_chain.vector_store = self.vector_store

    def ask(
        self,
        question: str,
        top_k: int = 3,
    ) -> dict[str, str | list[dict[str, str | int]]]:
        """Pose une question sans recharger les modèles."""
        return self.rag_chain.ask(question, top_k=top_k)

    def create_conversation(self, title: str) -> int:
        """Crée une conversation vide et renvoie son identifiant."""
        created = self.conversation_store.create_conversation(title)
        return int(created["id"])

    def list_conversations(self) -> list[dict[str, str | int]]:
        """Retourne les conversations, de la plus récente à la plus ancienne."""
        return self.conversation_store.list_conversations()

    def get_conversation(self, conversation_id: object) -> dict[str, str | int | list]:
        """Relit une conversation enregistrée, sans interroger le modèle."""
        return self.conversation_store.get_conversation(conversation_id)

    def delete_conversation(self, conversation_id: object) -> None:
        """Supprime une conversation et ses messages."""
        self.conversation_store.delete_conversation(conversation_id)

    def send_message(
        self,
        conversation_id: object,
        question: str,
        top_k: int = 3,
    ) -> dict[str, str | int | dict]:
        """Enregistre la question, route l'intention, puis enregistre la réponse.

        GENERAL_CHAT → DirectChatResponder (sans FAISS ni Web).
        COURSE_QUESTION → RAG existant (enrichi seulement pour les suivis).
        WEB_QUESTION → WebSearcher puis WebAnswerComposer (jamais un repli RAG silencieux).
        Si la réponse échoue, elle n'est pas enregistrée.
        """
        conversation = self.get_conversation(conversation_id)
        history = self.history_window.build(conversation["messages"])
        question_message = self.conversation_store.add_message(
            conversation_id,
            "user",
            question,
        )
        decision = self.intent_router.classify(question, history=history)
        if decision.intent == GENERAL_CHAT:
            result = self.direct_chat.reply(question, history=history)
        elif decision.intent == WEB_QUESTION:
            result = self._answer_web_question(question, history=history)
        else:
            # COURSE_QUESTION (et intentions inattendues traitées comme cours).
            rag_question = question
            if self.history_window.should_enrich_rag(question, history):
                rag_question = self.history_window.enrich_question(question, history)
            result = self.ask(rag_question, top_k=top_k)
        answer_message = self.conversation_store.add_message(
            conversation_id,
            "assistant",
            result["answer"],
            sources=result["sources"],
        )
        return {
            "conversation_id": question_message["conversation_id"],
            "question": question_message,
            "answer": answer_message,
        }

    def _answer_web_question(
        self,
        question: str,
        *,
        history: list,
    ) -> dict[str, str | list]:
        """Recherche Web réelle puis composition ; pas de repli RAG."""
        hits = self.web_searcher.search(question)
        return self.web_answer.compose(question, hits, history=history)

    def _open_existing_index(self) -> None:
        """Charge un index valide, ou le reconstruit si des PDF n'ont pas d'index.

        Une bibliothèque vide n'ouvre rien, même si un ancien dossier d'index
        est encore présent.
        """
        if not self.library.list_documents():
            return
        if _index_is_valid(self.index_directory):
            self.vector_store.load(self.index_directory)
            return
        self.library.rebuild_index()


def _index_is_valid(directory: Path) -> bool:
    """Indique si VectorStore.load accepterait ce dossier."""
    try:
        folder = _directory_to_load(directory)
        _require_trusted_manifest(folder)
    except (OSError, ValueError, TypeError):
        return False
    return True
