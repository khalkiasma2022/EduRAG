"""Parcours réel : PDF, découpage, FAISS sur disque, recherche et fiches.

EmbeddingModel et LLM sont simulés. Aucun modèle n'est téléchargé.
DocumentLibrary, VectorStore, RAGChain et Smart Revision sont les vraies classes.
"""

import json
from pathlib import Path

import pytest
from pypdf import PdfReader, PdfWriter
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

    class _TokenIds:
        """Imite le tenseur lu par le contrôle de longueur."""

        def __init__(self, count: int) -> None:
            self.shape = (1, count)

    class _Tokenizer:
        """Compte un prompt court sans télécharger le tokenizer de Qwen."""

        def __init__(self) -> None:
            self.chat_template = None

        def __call__(self, prompt: str, return_tensors: str = "pt") -> dict[str, _TokenIds]:
            return {"input_ids": _TokenIds(4)}

    class _Config:
        """Expose une limite large et un RoPE standard."""

        def __init__(self) -> None:
            self.max_position_embeddings = 8192
            self.rope_parameters = {"rope_type": "default"}
            self.use_sliding_window = False

    class _Model:
        def __init__(self) -> None:
            self.config = _Config()

    class FakeLLM:
        """Répond sans charger Qwen et conserve le prompt."""

        def __init__(self, model_name: str | None = None, device: str | None = None) -> None:
            self.prompts: list[str] = []
            self.calls = 0
            self.tokenizer = _Tokenizer()
            self.model = _Model()

        def is_loaded(self) -> bool:
            return True

        def generate(self, prompt: str, max_new_tokens: int = 512) -> str:
            self.calls += 1
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


_PAGE_EXCLUE = "Le neurone transmet un signal electrique. PAGE EXCLUE ALPHA"
_PAGE_RETENUE = "La mitochondrie produit de l energie. PAGE RETENUE BETA"
_PAGE_AUTRE = "PAGE EXCLUE GAMMA hors de la selection"


def _count_searches(store: VectorStore) -> list[int]:
    """Compte les recherches FAISS sans remplacer le véritable index."""
    calls: list[int] = []
    original = store.search_by_vector

    def wrapped(vector: list[float], top_k: int = 3) -> list[dict]:
        calls.append(top_k)
        return original(vector, top_k=top_k)

    store.search_by_vector = wrapped
    return calls


def _pdf_text(path: Path) -> str:
    """Relit le texte d'un PDF exporté."""
    return "\n".join(page.extract_text() or "" for page in PdfReader(path).pages)


def test_revision_runs_from_the_pdf_to_the_exports(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assistant = _assistant(tmp_path, monkeypatch)
    source = _write_text_pdf(
        tmp_path / "cellule.pdf",
        [_PAGE_EXCLUE, _PAGE_RETENUE, _PAGE_AUTRE],
    )
    added = assistant.add_pdf(source)
    searches = _count_searches(assistant.vector_store)
    vectors_before = assistant.vector_store._store.index.ntotal
    chunks_before = assistant.embedding_model.embed_chunk_calls
    queries_before = assistant.embedding_model.query_calls
    calls_before = assistant.llm.calls

    saved = assistant.create_revision(
        added["document_id"],
        page_start=2,
        page_end=2,
        language="en",
        level="advanced",
        sheet_type="short",
    )

    prompt = assistant.llm.prompts[-1]
    assert assistant.llm.calls == calls_before + 1
    assert _PAGE_RETENUE in prompt
    assert _PAGE_EXCLUE not in prompt
    assert _PAGE_AUTRE not in prompt
    assert isinstance(saved["id"], int) and saved["id"] > 0
    assert isinstance(saved["created_at"], str) and saved["created_at"].strip() != ""
    assert saved["content"] == "Reponse simulee."
    assert saved["page_start"] == 2
    assert saved["page_end"] == 2
    assert saved["language"] == "en"
    assert saved["level"] == "advanced"
    assert saved["sheet_type"] == "short"
    assert assistant.get_revision(saved["id"]) == saved
    assert assistant.list_revisions()[0] == saved
    assert assistant.vector_store._store.index.ntotal == vectors_before
    assert assistant.embedding_model.embed_chunk_calls == chunks_before
    assert assistant.embedding_model.query_calls == queries_before
    assert searches == []

    markdown_path = tmp_path / "exports" / "fiche.md"
    pdf_path = tmp_path / "exports" / "fiche.pdf"
    assert assistant.export_revision_markdown(saved["id"], markdown_path) == markdown_path
    assert assistant.export_revision_pdf(saved["id"], pdf_path) == pdf_path
    markdown = markdown_path.read_text(encoding="utf-8")
    pdf_text = _pdf_text(pdf_path)
    assert markdown_path.is_file()
    assert saved["content"] in markdown
    assert saved["filename"] in markdown
    assert saved["created_at"] in markdown
    assert "Anglais" in markdown
    assert "Avancé" in markdown
    assert "Fiche courte" in markdown
    assert pdf_path.read_bytes().startswith(b"%PDF")
    assert pdf_path.stat().st_size > 0
    assert saved["content"] in pdf_text
    assert saved["filename"] in pdf_text
    assert saved["created_at"] in pdf_text
    assert "Anglais" in pdf_text
    assert assistant.llm.calls == calls_before + 1
    assert searches == []

    result = assistant.ask("Que produit la mitochondrie dans ce cours ?")

    assert result["answer"] == "Reponse simulee."
    assert result["sources"][0]["filename"] == "cellule.pdf"
    assert result["sources"][0]["page_number"] == 2
    assert _PAGE_RETENUE in result["sources"][0]["text"]
    assert searches


def test_revision_survives_a_new_assistant(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assistant = _assistant(tmp_path, monkeypatch)
    source = _write_text_pdf(tmp_path / "neurone.pdf", [_NEURONE, _MITOCHONDRIE])
    added = assistant.add_pdf(source)
    saved = assistant.create_revision(
        added["document_id"],
        page_start=1,
        page_end=1,
        language="fr",
        level="beginner",
        sheet_type="complete",
    )
    vectors = assistant.vector_store._store.index.ntotal

    restarted = _assistant(tmp_path, monkeypatch)

    assert restarted.list_revisions() == [saved]
    assert restarted.get_revision(saved["id"]) == saved
    assert restarted.llm.calls == 0
    assert restarted.llm.prompts == []
    assert restarted.embedding_model.embed_chunk_calls == 0
    assert restarted.vector_store._store.index.ntotal == vectors
    found = restarted.vector_store.search_by_vector(_vector_for(_NEURONE), top_k=1)
    assert found[0]["filename"] == "neurone.pdf"
    assert _NEURONE in found[0]["text"]
    assert restarted.llm.calls == 0


def test_revision_survives_document_removal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assistant = _assistant(tmp_path, monkeypatch)
    neurone = _write_text_pdf(tmp_path / "neurone.pdf", [_NEURONE])
    mitochondrie = _write_text_pdf(tmp_path / "mitochondrie.pdf", [_MITOCHONDRIE])
    first = assistant.add_pdf(neurone)
    second = assistant.add_pdf(mitochondrie)
    saved = assistant.create_revision(
        first["document_id"],
        page_start=1,
        page_end=1,
        language="fr",
        level="intermediate",
        sheet_type="short",
    )

    assistant.remove_document(first["document_id"])
    calls_before = assistant.llm.calls

    assert not Path(first["stored_path"]).exists()
    assert assistant.list_documents() == [second]
    assert assistant.get_revision(saved["id"]) == saved
    assert assistant.list_revisions() == [saved]
    markdown_path = tmp_path / "exports" / "conservee.md"
    pdf_path = tmp_path / "exports" / "conservee.pdf"
    assert assistant.export_revision_markdown(saved["id"], markdown_path) == markdown_path
    assert assistant.export_revision_pdf(saved["id"], pdf_path) == pdf_path
    assert saved["content"] in markdown_path.read_text(encoding="utf-8")
    assert saved["filename"] in markdown_path.read_text(encoding="utf-8")
    assert pdf_path.read_bytes().startswith(b"%PDF")
    assert saved["content"] in _pdf_text(pdf_path)
    assert assistant.llm.calls == calls_before
    remaining = assistant.vector_store.search_by_vector(_vector_for(_NEURONE), top_k=3)
    assert remaining
    assert all(item["filename"] != "neurone.pdf" for item in remaining)
    assert _MITOCHONDRIE in remaining[0]["text"]
    assert assistant.vector_store._store.index.ntotal == len(_stored_chunks(second["stored_path"]))
