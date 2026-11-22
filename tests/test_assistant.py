"""Tests de l'objet qui relie la bibliothèque, FAISS et la réponse.

Les PDF sont créés dans un dossier temporaire.
Les embeddings et Qwen sont simulés : aucun modèle n'est téléchargé.
"""

import json
from pathlib import Path

import pytest
from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

from src.generation.rag_chain import RAGChain
from src.ingestion.library import DocumentLibrary
from src.retrieval.vector_store import MANIFEST_NAME

import src.assistant as assistant_module


def _install_fakes(monkeypatch: pytest.MonkeyPatch) -> dict[str, list]:
    """Remplace les trois modèles par des compteurs, sans téléchargement."""
    created: dict[str, list] = {"embedding": [], "store": [], "llm": []}

    class FakeEmbeddingModel:
        """Compte les créations et les vectorisations."""

        def __init__(self, model_name: str = "fake") -> None:
            created["embedding"].append(self)
            self.embed_chunk_calls = 0
            self.query_calls = 0

        def embed_chunks(self, chunks: list[dict]) -> list[list[float]]:
            self.embed_chunk_calls += 1
            return [[1.0, float(index)] for index, _chunk in enumerate(chunks)]

        def embed_query(self, question: str) -> list[float]:
            self.query_calls += 1
            return [0.2, 0.8]

    class FakeVectorStore:
        """Garde les chunks en mémoire et note chaque chargement."""

        def __init__(self, embedding_model: object) -> None:
            created["store"].append(self)
            self.embedding_model = embedding_model
            self.chunks: list[dict] = []
            self.loaded: list[Path] = []

        def create(self, chunks: list[dict], embeddings: list[list[float]]) -> None:
            self.chunks = [dict(chunk) for chunk in chunks]

        def save(self, directory: str | Path) -> None:
            folder = Path(directory)
            folder.mkdir(parents=True, exist_ok=True)
            (folder / "index.faiss").write_text(str(len(self.chunks)), encoding="utf-8")
            (folder / MANIFEST_NAME).write_text(
                json.dumps({"creator": "EduRAG", "format_version": 1}),
                encoding="utf-8",
            )

        def load(self, directory: str | Path) -> None:
            self.loaded.append(Path(directory))

        def search_by_vector(self, vector: list[float], top_k: int = 3) -> list[dict]:
            if not self.chunks:
                raise ValueError("Impossible de rechercher dans une base vide.")
            return self.chunks[:top_k]

    class FakeLLM:
        """Répond sans charger Qwen et conserve le prompt."""

        def __init__(self, model_name: str | None = None, device: str | None = None) -> None:
            created["llm"].append(self)
            self.prompts: list[str] = []

        def generate(self, prompt: str, max_new_tokens: int = 512) -> str:
            self.prompts.append(prompt)
            if "Réponds par un seul mot" in prompt:
                return "oui"
            return "Réponse de test."

    monkeypatch.setattr(assistant_module, "EmbeddingModel", FakeEmbeddingModel)
    monkeypatch.setattr(assistant_module, "VectorStore", FakeVectorStore)
    monkeypatch.setattr(assistant_module, "LLM", FakeLLM)
    return created


def _write_text_pdf(path: Path, pages: list[str]) -> Path:
    """Crée un petit PDF dont le texte de chaque page reste extractible."""
    writer = PdfWriter()
    for text in pages:
        page = writer.add_blank_page(width=612, height=792)
        page[NameObject("/Resources")] = DictionaryObject(
            {
                NameObject("/Font"): DictionaryObject(
                    {
                        NameObject("/F1"): DictionaryObject(
                            {
                                NameObject("/Type"): NameObject("/Font"),
                                NameObject("/Subtype"): NameObject("/Type1"),
                                NameObject("/BaseFont"): NameObject("/Helvetica"),
                            }
                        )
                    }
                )
            }
        )
        stream = DecodedStreamObject()
        stream.set_data(f"BT /F1 24 Tf 100 700 Td ({text}) Tj ET".encode("ascii"))
        page[NameObject("/Contents")] = stream
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        writer.write(handle)
    return path


def _write_catalog(documents: Path, stored: Path) -> None:
    """Inscrit un PDF déjà présent, comme après un ajout précédent."""
    entry = {
        "document_id": "courspython001",
        "filename": stored.name,
        "stored_path": str(stored.resolve()),
        "sha256": "a" * 64,
    }
    documents.mkdir(parents=True, exist_ok=True)
    (documents / "catalog.json").write_text(
        json.dumps([entry], ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _write_valid_index(index: Path) -> None:
    """Pose le témoin qu'un index EduRAG est déjà enregistré."""
    index.mkdir(parents=True, exist_ok=True)
    (index / MANIFEST_NAME).write_text(
        json.dumps({"creator": "EduRAG", "format_version": 1}),
        encoding="utf-8",
    )


def _assistant(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    created = _install_fakes(monkeypatch)
    assistant = assistant_module.EduRAGAssistant(
        tmp_path / "documents",
        tmp_path / "index",
    )
    return assistant, created


def test_creates_each_component_once(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    assistant, created = _assistant(tmp_path, monkeypatch)

    assert len(created["embedding"]) == 1
    assert len(created["store"]) == 1
    assert len(created["llm"]) == 1
    assert isinstance(assistant.library, DocumentLibrary)
    assert isinstance(assistant.rag_chain, RAGChain)
    assert assistant.embedding_model is created["embedding"][0]
    assert assistant.vector_store is created["store"][0]
    assert assistant.llm is created["llm"][0]


def test_library_and_chain_share_the_vector_store(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assistant, _created = _assistant(tmp_path, monkeypatch)

    assert assistant.library.vector_store is assistant.vector_store
    assert assistant.rag_chain.vector_store is assistant.vector_store


def test_library_store_and_chain_share_the_embedding_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assistant, _created = _assistant(tmp_path, monkeypatch)

    assert assistant.library.embedding_model is assistant.embedding_model
    assert assistant.vector_store.embedding_model is assistant.embedding_model
    assert assistant.rag_chain.embedding_model is assistant.embedding_model


def test_add_pdf_registers_the_document_and_rebuilds_the_index(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assistant, created = _assistant(tmp_path, monkeypatch)
    source = _write_text_pdf(tmp_path / "cours.pdf", ["Cours de Python"])

    added = assistant.add_pdf(source)

    assert assistant.list_documents() == [added]
    assert added["filename"] == "cours.pdf"
    assert created["embedding"][0].embed_chunk_calls == 1
    assert assistant.vector_store.chunks
    assert (tmp_path / "index" / MANIFEST_NAME).is_file()


def test_ask_returns_the_answer_and_the_sources(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assistant, created = _assistant(tmp_path, monkeypatch)
    source = _write_text_pdf(tmp_path / "cours.pdf", ["Cours de Python"])
    assistant.add_pdf(source)

    result = assistant.ask("Qu'est-ce qu'une variable dans ce cours ?")

    assert result["answer"] == "Réponse de test."
    assert result["sources"]
    assert result["sources"][0]["filename"] == "cours.pdf"
    assert result["sources"][0]["page_number"] == 1
    assert "Cours de Python" in result["sources"][0]["text"]
    assert created["llm"][0].prompts


def test_startup_loads_an_existing_index(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    documents = tmp_path / "documents"
    index = tmp_path / "index"
    stored = _write_text_pdf(documents / "cours.pdf", ["Cours de Python"])
    _write_catalog(documents, stored)
    _write_valid_index(index)
    assistant, created = _assistant(tmp_path, monkeypatch)

    assert assistant.vector_store.loaded == [index.resolve()]
    assert created["embedding"][0].embed_chunk_calls == 0
    assert assistant.list_documents()[0]["filename"] == "cours.pdf"


def test_startup_rebuilds_when_the_pdfs_have_no_index(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    documents = tmp_path / "documents"
    stored = _write_text_pdf(documents / "cours.pdf", ["Cours de Python"])
    _write_catalog(documents, stored)
    assistant, created = _assistant(tmp_path, monkeypatch)

    assert assistant.vector_store.loaded == []
    assert created["embedding"][0].embed_chunk_calls == 1
    assert assistant.vector_store.chunks
    assert (tmp_path / "index" / MANIFEST_NAME).is_file()


def test_empty_library_does_not_load_or_rebuild(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    index = tmp_path / "index"
    _write_valid_index(index)
    assistant, created = _assistant(tmp_path, monkeypatch)

    assert assistant.list_documents() == []
    assert assistant.vector_store.loaded == []
    assert created["embedding"][0].embed_chunk_calls == 0
    with pytest.raises(ValueError, match="vide"):
        assistant.ask("Que dit le cours ?")


def test_questions_do_not_reload_the_models(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assistant, created = _assistant(tmp_path, monkeypatch)
    source = _write_text_pdf(tmp_path / "cours.pdf", ["Cours de Python"])
    assistant.add_pdf(source)
    embedding = assistant.embedding_model
    store = assistant.vector_store
    llm = assistant.llm

    assistant.ask("Qu'est-ce qu'une variable dans ce cours ?")
    assistant.ask("Et une fonction dans ce cours ?")

    assert created["embedding"] == [embedding]
    assert created["store"] == [store]
    assert created["llm"] == [llm]
    assert assistant.embedding_model is embedding
    assert assistant.vector_store is store
    assert assistant.llm is llm
    assert embedding.query_calls == 2


def test_remove_document_uses_the_same_library(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assistant, _created = _assistant(tmp_path, monkeypatch)
    source = _write_text_pdf(tmp_path / "cours.pdf", ["Cours de Python"])
    added = assistant.add_pdf(source)

    assistant.remove_document(added["document_id"])

    assert assistant.list_documents() == []
