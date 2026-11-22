"""Parcours réel : PDF, découpage, FAISS sur disque, recherche.

EmbeddingModel et LLM sont simulés. Aucun modèle n'est téléchargé.
DocumentLibrary, VectorStore et RAGChain sont les vraies classes.
"""

import json
from pathlib import Path

import pytest
from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

import src.assistant as assistant_module
from src.ingestion.pdf_loader import load_pdf
from src.ingestion.text_splitter import split_documents
from src.retrieval.vector_store import MANIFEST_NAME, VectorStore


_NEURONE = "Le neurone transmet un signal electrique."
_MITOCHONDRIE = "La mitochondrie produit de l energie."


def _install_fakes(monkeypatch: pytest.MonkeyPatch) -> None:
    """Remplace seulement les deux modèles lourds."""

    class FakeEmbeddingModel:
        """Produit des vecteurs stables, sans Sentence Transformers."""

        def __init__(self, model_name: str = "fake") -> None:
            self.embed_chunk_calls = 0
            self.query_calls = 0

        def embed_chunks(self, chunks: list[dict]) -> list[list[float]]:
            self.embed_chunk_calls += 1
            return [_vector_for(str(chunk["text"])) for chunk in chunks]

        def embed_query(self, question: str) -> list[float]:
            self.query_calls += 1
            return _vector_for(question)

    class FakeLLM:
        """Répond sans charger Qwen et conserve le prompt."""

        def __init__(self, model_name: str | None = None, device: str | None = None) -> None:
            self.prompts: list[str] = []

        def generate(self, prompt: str, max_new_tokens: int = 512) -> str:
            self.prompts.append(prompt)
            if "Réponds par un seul mot" in prompt:
                return "oui"
            return "Reponse simulee."

    monkeypatch.setattr(assistant_module, "EmbeddingModel", FakeEmbeddingModel)
    monkeypatch.setattr(assistant_module, "LLM", FakeLLM)


def _vector_for(text: str) -> list[float]:
    """Place le neurone et la mitochondrie dans deux directions distinctes."""
    if "neurone" in text:
        return [1.0, 0.0, 0.0, 0.0]
    if "mitochondrie" in text:
        return [0.0, 1.0, 0.0, 0.0]
    return [0.0, 0.0, 1.0, 0.0]


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


def _assistant(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    _install_fakes(monkeypatch)
    assistant = assistant_module.EduRAGAssistant(
        tmp_path / "documents",
        tmp_path / "index",
    )
    return assistant


def _stored_chunks(stored_path: str) -> list[dict]:
    """Relit le PDF copié avec le chargeur et le découpeur réels."""
    return split_documents(load_pdf(stored_path))


def test_added_pdf_is_split_and_stored_in_real_faiss(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assistant = _assistant(tmp_path, monkeypatch)
    source = _write_text_pdf(tmp_path / "neurone.pdf", [_NEURONE])

    added = assistant.add_pdf(source)
    chunks = _stored_chunks(added["stored_path"])
    index = assistant.vector_store._store.index

    assert isinstance(assistant.vector_store, VectorStore)
    assert chunks
    assert _NEURONE in chunks[0]["text"]
    assert chunks[0]["page_number"] == 1
    assert index.ntotal == len(chunks)
    assert (tmp_path / "index" / "index.faiss").is_file()
    manifest = json.loads((tmp_path / "index" / MANIFEST_NAME).read_text(encoding="utf-8"))
    assert manifest == {"creator": "EduRAG", "format_version": 1}

    result = assistant.ask("Que fait le neurone dans ce cours ?")

    assert result["answer"] == "Reponse simulee."
    assert result["sources"][0]["filename"] == "neurone.pdf"
    assert result["sources"][0]["page_number"] == 1
    assert _NEURONE in result["sources"][0]["text"]
    prompt = assistant.llm.prompts[-1]
    assert "Fichier : neurone.pdf" in prompt
    assert "Page : 1" in prompt


def test_new_assistant_loads_the_saved_faiss_index(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assistant = _assistant(tmp_path, monkeypatch)
    source = _write_text_pdf(tmp_path / "neurone.pdf", [_NEURONE])
    assistant.add_pdf(source)
    saved_vectors = assistant.vector_store._store.index.ntotal

    restarted = _assistant(tmp_path, monkeypatch)

    assert restarted.embedding_model.embed_chunk_calls == 0
    assert restarted.vector_store._store.index.ntotal == saved_vectors
    result = restarted.ask("Que fait le neurone dans ce cours ?")
    assert result["sources"][0]["filename"] == "neurone.pdf"
    assert result["sources"][0]["page_number"] == 1
    assert _NEURONE in result["sources"][0]["text"]
    assert "neurone.pdf" in restarted.llm.prompts[-1]
    assert "Page : 1" in restarted.llm.prompts[-1]


def test_deletion_updates_the_real_index_and_a_restart_finds_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assistant = _assistant(tmp_path, monkeypatch)
    neurone = _write_text_pdf(tmp_path / "neurone.pdf", [_NEURONE])
    mitochondrie = _write_text_pdf(tmp_path / "mitochondrie.pdf", [_MITOCHONDRIE])
    first = assistant.add_pdf(neurone)
    second = assistant.add_pdf(mitochondrie)

    assistant.remove_document(first["document_id"])
    remaining = assistant.ask("Que produit la mitochondrie dans ce cours ?")

    assert assistant.list_documents() == [second]
    assert not Path(first["stored_path"]).exists()
    assert remaining["sources"][0]["filename"] == "mitochondrie.pdf"
    assert remaining["sources"][0]["page_number"] == 1
    assert _MITOCHONDRIE in remaining["sources"][0]["text"]
    assert assistant.vector_store._store.index.ntotal == len(_stored_chunks(second["stored_path"]))
    assert all(source["filename"] != "neurone.pdf" for source in remaining["sources"])

    assistant.remove_document(second["document_id"])

    assert assistant.list_documents() == []
    assert not Path(second["stored_path"]).exists()
    assert not (tmp_path / "index").exists()
    with pytest.raises(ValueError, match="vide"):
        assistant.ask("Que produit la mitochondrie dans ce cours ?")

    restarted = _assistant(tmp_path, monkeypatch)
    assert restarted.embedding_model.embed_chunk_calls == 0
    assert restarted.list_documents() == []
    with pytest.raises(ValueError, match="vide"):
        restarted.ask("Que fait le neurone dans ce cours ?")
