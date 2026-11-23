"""Objet central d'EduRAG.

Il crée les modèles une seule fois, puis les partage
entre la bibliothèque de PDF, la chaîne de réponse et les fiches.
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


class EduRAGAssistant:
    """Relie l'ajout des cours et les questions de l'apprenant."""

    def __init__(
        self,
        documents_directory: str | Path,
        index_directory: str | Path,
        revision_database_path: str | Path | None = None,
    ) -> None:
        """Crée les composants, puis ouvre ou reconstruit l'index si besoin."""
        self.documents_directory = Path(documents_directory).resolve()
        self.index_directory = Path(index_directory).resolve()
        if revision_database_path is None:
            revision_database_path = self.documents_directory / "revisions.sqlite"

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
        self.revision_sheet = RevisionSheet(self.library, self.llm)
        self.revision_store = RevisionStore(revision_database_path)
        self.revision_export = RevisionExport()
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
