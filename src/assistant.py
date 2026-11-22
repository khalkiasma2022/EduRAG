"""Objet central d'EduRAG.

Il crée les modèles une seule fois, puis les partage
entre la bibliothèque de PDF et la chaîne de réponse.
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


class EduRAGAssistant:
    """Relie l'ajout des cours et les questions de l'apprenant."""

    def __init__(
        self,
        documents_directory: str | Path,
        index_directory: str | Path,
    ) -> None:
        """Crée les composants, puis ouvre ou reconstruit l'index si besoin."""
        self.documents_directory = Path(documents_directory).resolve()
        self.index_directory = Path(index_directory).resolve()

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
        self._open_existing_index()

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
