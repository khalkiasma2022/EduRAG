"""Tests de l'objet qui relie la bibliothèque, FAISS et la réponse.

Les PDF sont créés dans un dossier temporaire.
Les embeddings et Qwen sont simulés : aucun modèle n'est téléchargé.
"""

import json
from pathlib import Path

import pytest
from pypdf import PdfReader, PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

from src.generation.rag_chain import RAGChain
from src.ingestion.library import DocumentLibrary
from src.retrieval.vector_store import MANIFEST_NAME
from src.revision.export import RevisionExport
from src.revision.sheet import RevisionSheet
from src.revision.store import RevisionStore
from src.tutor.store import ConversationStore

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
            self.search_calls = 0

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
            self.search_calls += 1
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


def test_revision_components_share_the_library_and_the_llm(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assistant, created = _assistant(tmp_path, monkeypatch)
    database = (tmp_path / "documents").resolve() / "revisions.sqlite"

    assert isinstance(assistant.revision_sheet, RevisionSheet)
    assert isinstance(assistant.revision_store, RevisionStore)
    assert isinstance(assistant.revision_export, RevisionExport)
    assert assistant.revision_sheet.library is assistant.library
    assert assistant.revision_sheet.llm is assistant.llm
    assert assistant.revision_sheet.llm is created["llm"][0]
    assert assistant.revision_store.path == database
    assert database.is_file()
    assert len(created["embedding"]) == 1
    assert len(created["llm"]) == 1


def test_revision_database_path_can_be_chosen(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    created = _install_fakes(monkeypatch)
    database = tmp_path / "bases" / "fiches.sqlite"

    assistant = assistant_module.EduRAGAssistant(
        tmp_path / "documents",
        tmp_path / "index",
        database,
    )

    assert assistant.revision_store.path == database
    assert database.is_file()
    assert not ((tmp_path / "documents").resolve() / "revisions.sqlite").exists()
    assert len(created["embedding"]) == 1
    assert len(created["llm"]) == 1
    assert assistant.revision_sheet.library is assistant.library
    assert assistant.revision_sheet.llm is assistant.llm


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


class _RevisionTokenIds:
    """Imite le tenseur lu par le contrôle de longueur."""

    def __init__(self, count: int) -> None:
        self.shape = (1, count)


class _RevisionTokenizer:
    """Compte un prompt court sans télécharger le tokenizer de Qwen."""

    def __init__(self) -> None:
        self.chat_template = None

    def __call__(self, prompt: str, return_tensors: str = "pt") -> dict[str, _RevisionTokenIds]:
        return {"input_ids": _RevisionTokenIds(4)}


class _RevisionConfig:
    """Expose une limite large et un RoPE standard."""

    def __init__(self) -> None:
        self.max_position_embeddings = 8192
        self.rope_parameters = {"rope_type": "default"}
        self.use_sliding_window = False


def _arm_revision_model(llm: object) -> None:
    """Rend le faux Qwen déjà chargé, sans créer un second modèle."""
    llm.tokenizer = _RevisionTokenizer()
    llm.model = type("RevisionModel", (), {"config": _RevisionConfig()})()
    llm.is_loaded = lambda: True


def test_create_revision_saves_one_sheet_without_searching(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assistant, created = _assistant(tmp_path, monkeypatch)
    source = _write_text_pdf(tmp_path / "cours.pdf", ["Page un", "Page deux", "Page trois"])
    added = assistant.add_pdf(source)
    _arm_revision_model(assistant.llm)
    generate_calls: list[dict[str, object]] = []
    original_generate = assistant.revision_sheet.generate

    def spy_generate(
        document_id: str,
        page_start: int | None = None,
        page_end: int | None = None,
        language: str = "fr",
        level: str = "beginner",
        sheet_type: str = "complete",
    ) -> dict[str, str | int]:
        generate_calls.append(
            {
                "document_id": document_id,
                "page_start": page_start,
                "page_end": page_end,
                "language": language,
                "level": level,
                "sheet_type": sheet_type,
            }
        )
        return original_generate(
            document_id,
            page_start=page_start,
            page_end=page_end,
            language=language,
            level=level,
            sheet_type=sheet_type,
        )

    assistant.revision_sheet.generate = spy_generate
    ask_calls: list[tuple] = []
    original_ask = assistant.rag_chain.ask

    def spy_ask(*args: object, **kwargs: object) -> dict:
        ask_calls.append((args, kwargs))
        return original_ask(*args, **kwargs)

    assistant.rag_chain.ask = spy_ask
    queries_before = assistant.embedding_model.query_calls
    chunks_before = assistant.embedding_model.embed_chunk_calls
    searches_before = assistant.vector_store.search_calls

    saved = assistant.create_revision(
        added["document_id"],
        page_start=2,
        page_end=3,
        language="en",
        level="advanced",
        sheet_type="short",
    )

    assert generate_calls == [
        {
            "document_id": added["document_id"],
            "page_start": 2,
            "page_end": 3,
            "language": "en",
            "level": "advanced",
            "sheet_type": "short",
        }
    ]
    assert isinstance(saved["id"], int) and saved["id"] > 0
    assert isinstance(saved["created_at"], str) and saved["created_at"].strip() != ""
    assert saved["content"] == "Réponse de test."
    assert saved["document_id"] == added["document_id"]
    assert saved["filename"] == "cours.pdf"
    assert saved["page_start"] == 2
    assert saved["page_end"] == 3
    assert saved["language"] == "en"
    assert saved["level"] == "advanced"
    assert saved["sheet_type"] == "short"
    assert assistant.revision_store.get(saved["id"]) == saved
    assert ask_calls == []
    assert assistant.embedding_model.query_calls == queries_before
    assert assistant.embedding_model.embed_chunk_calls == chunks_before
    assert assistant.vector_store.search_calls == searches_before
    assert len(created["llm"]) == 1
    assert len(created["embedding"]) == 1
    assert len(created["store"]) == 1
    assert assistant.revision_sheet.llm is assistant.llm


def test_create_revision_does_not_save_when_generation_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assistant, _created = _assistant(tmp_path, monkeypatch)

    with pytest.raises(ValueError, match="introuvable"):
        assistant.create_revision("document-absent")

    assert assistant.revision_store.list_sheets() == []

    def fail_generation(*args: object, **kwargs: object) -> dict[str, str | int]:
        raise RuntimeError("La génération a échoué.")

    assistant.revision_sheet.generate = fail_generation

    with pytest.raises(RuntimeError, match="La génération a échoué"):
        assistant.create_revision("doc-python", page_start=1, page_end=1)

    assert assistant.revision_store.list_sheets() == []


def _forbid_model_use(assistant: assistant_module.EduRAGAssistant) -> dict[str, int]:
    """Mémorise l'activité des modèles, puis refuse tout nouvel appel."""

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("La consultation ne doit pas interroger les modèles.")

    assistant.revision_sheet.generate = forbidden
    assistant.rag_chain.ask = forbidden
    assistant.embedding_model.embed_query = forbidden
    assistant.embedding_model.embed_chunks = forbidden
    assistant.vector_store.search_by_vector = forbidden
    return {
        "prompts": len(assistant.llm.prompts),
        "queries": assistant.embedding_model.query_calls,
        "chunks": assistant.embedding_model.embed_chunk_calls,
        "searches": assistant.vector_store.search_calls,
    }


def test_list_and_get_revision_return_saved_sheets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assistant, created = _assistant(tmp_path, monkeypatch)
    assert assistant.list_revisions() == []
    source = _write_text_pdf(tmp_path / "cours.pdf", ["Page un", "Page deux", "Page trois"])
    added = assistant.add_pdf(source)
    _arm_revision_model(assistant.llm)
    first = assistant.create_revision(
        added["document_id"],
        page_start=1,
        page_end=1,
        language="fr",
        level="beginner",
        sheet_type="short",
    )
    second = assistant.create_revision(
        added["document_id"],
        page_start=2,
        page_end=3,
        language="en",
        level="advanced",
        sheet_type="complete",
    )
    before = _forbid_model_use(assistant)

    listed = assistant.list_revisions()
    found = assistant.get_revision(first["id"])

    assert listed == [second, first]
    assert found == first
    assert found["document_id"] == added["document_id"]
    assert found["filename"] == "cours.pdf"
    assert found["page_start"] == 1
    assert found["page_end"] == 1
    assert found["language"] == "fr"
    assert found["level"] == "beginner"
    assert found["sheet_type"] == "short"
    assert found["content"] == "Réponse de test."
    assert found["created_at"]
    assert len(assistant.llm.prompts) == before["prompts"]
    assert assistant.embedding_model.query_calls == before["queries"]
    assert assistant.embedding_model.embed_chunk_calls == before["chunks"]
    assert assistant.vector_store.search_calls == before["searches"]
    assert len(created["llm"]) == 1
    assert len(created["embedding"]) == 1


@pytest.mark.parametrize("sheet_id", [0, -1, True, "1", None])
def test_get_revision_rejects_an_invalid_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, sheet_id: object
) -> None:
    assistant, _created = _assistant(tmp_path, monkeypatch)
    before = _forbid_model_use(assistant)

    with pytest.raises(ValueError, match="invalide"):
        assistant.get_revision(sheet_id)

    assert assistant.list_revisions() == []
    assert len(assistant.llm.prompts) == before["prompts"]


def test_get_revision_rejects_an_unknown_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assistant, _created = _assistant(tmp_path, monkeypatch)

    with pytest.raises(ValueError, match="introuvable"):
        assistant.get_revision(4)

    assert assistant.list_revisions() == []


def test_get_revision_keeps_a_sheet_after_the_pdf_is_removed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assistant, _created = _assistant(tmp_path, monkeypatch)
    source = _write_text_pdf(tmp_path / "cours.pdf", ["Page un"])
    added = assistant.add_pdf(source)
    _arm_revision_model(assistant.llm)
    saved = assistant.create_revision(added["document_id"], language="ar", level="intermediate")

    assistant.remove_document(added["document_id"])
    before = _forbid_model_use(assistant)

    assert assistant.list_documents() == []
    assert assistant.get_revision(saved["id"]) == saved
    assert assistant.list_revisions() == [saved]
    assert len(assistant.llm.prompts) == before["prompts"]
    assert assistant.embedding_model.query_calls == before["queries"]
    assert assistant.embedding_model.embed_chunk_calls == before["chunks"]
    assert assistant.vector_store.search_calls == before["searches"]


def _spy_export(exporter: RevisionExport, method_name: str) -> list[tuple[object, object]]:
    """Enregistre les appels, puis laisse l'exporteur réel écrire le fichier."""
    calls: list[tuple[object, object]] = []
    original = getattr(exporter, method_name)

    def spy(sheet: object, output_path: object) -> Path:
        calls.append((sheet, output_path))
        return original(sheet, output_path)

    setattr(exporter, method_name, spy)
    return calls


def test_export_revision_writes_markdown_and_pdf_from_sqlite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assistant, _created = _assistant(tmp_path, monkeypatch)
    source = _write_text_pdf(tmp_path / "cours.pdf", ["Page un"])
    added = assistant.add_pdf(source)
    _arm_revision_model(assistant.llm)
    saved = assistant.create_revision(
        added["document_id"],
        page_start=1,
        page_end=1,
        language="fr",
        level="beginner",
        sheet_type="complete",
    )
    markdown_path = tmp_path / "exports" / "fiche.md"
    pdf_path = tmp_path / "exports" / "fiche.pdf"
    before = _forbid_model_use(assistant)
    markdown_calls = _spy_export(assistant.revision_export, "export_markdown")
    pdf_calls = _spy_export(assistant.revision_export, "export_pdf")

    written_markdown = assistant.export_revision_markdown(saved["id"], markdown_path)
    written_pdf = assistant.export_revision_pdf(saved["id"], pdf_path)

    markdown = markdown_path.read_text(encoding="utf-8")
    pdf_text = "\n".join(page.extract_text() or "" for page in PdfReader(pdf_path).pages)
    assert written_markdown == markdown_path
    assert markdown_path.is_file()
    assert saved["content"] in markdown
    assert saved["filename"] in markdown
    assert saved["created_at"] in markdown
    assert "Français" in markdown
    assert "Débutant" in markdown
    assert "Fiche complète" in markdown
    assert markdown_calls == [(saved, markdown_path)]
    assert written_pdf == pdf_path
    assert pdf_path.read_bytes().startswith(b"%PDF")
    assert pdf_path.stat().st_size > 0
    assert saved["content"] in pdf_text
    assert saved["filename"] in pdf_text
    assert saved["created_at"] in pdf_text
    assert "Français" in pdf_text
    assert "Débutant" in pdf_text
    assert "Fiche complète" in pdf_text
    assert pdf_calls == [(saved, pdf_path)]
    assert len(assistant.llm.prompts) == before["prompts"]
    assert assistant.embedding_model.query_calls == before["queries"]
    assert assistant.embedding_model.embed_chunk_calls == before["chunks"]
    assert assistant.vector_store.search_calls == before["searches"]


@pytest.mark.parametrize("sheet_id,message", [(0, "invalide"), (4, "introuvable")])
def test_export_revision_rejects_a_bad_sheet_id(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    sheet_id: object,
    message: str,
) -> None:
    assistant, _created = _assistant(tmp_path, monkeypatch)
    markdown_path = tmp_path / "fiche.md"
    pdf_path = tmp_path / "fiche.pdf"
    before = _forbid_model_use(assistant)
    markdown_calls = _spy_export(assistant.revision_export, "export_markdown")
    pdf_calls = _spy_export(assistant.revision_export, "export_pdf")

    with pytest.raises(ValueError, match=message):
        assistant.export_revision_markdown(sheet_id, markdown_path)
    with pytest.raises(ValueError, match=message):
        assistant.export_revision_pdf(sheet_id, pdf_path)

    assert not markdown_path.exists()
    assert not pdf_path.exists()
    assert markdown_calls == []
    assert pdf_calls == []
    assert len(assistant.llm.prompts) == before["prompts"]


def _source() -> dict[str, str | int]:
    """Reproduit une source acceptée par le magasin des conversations."""
    return {
        "text": "La mitochondrie produit l'énergie.",
        "filename": "cellule.pdf",
        "page_number": 2,
        "chunk_id": 4,
    }


def _answer_for(question: str, top_k: int = 3) -> dict[str, str | list[dict[str, str | int]]]:
    """Imite le dictionnaire renvoyé par ask(), sans interroger le modèle."""
    return {
        "answer": f"Réponse à : {question}",
        "sources": [_source()] if top_k > 0 else [],
    }


def test_conversation_database_follows_the_revision_database(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assistant, created = _assistant(tmp_path, monkeypatch)
    documents = (tmp_path / "documents").resolve()

    assert isinstance(assistant.conversation_store, ConversationStore)
    assert assistant.conversation_store.path == documents / "conversations.sqlite"
    assert assistant.revision_store.path == documents / "revisions.sqlite"
    assert (documents / "conversations.sqlite").is_file()
    assert len(created["llm"]) == 1
    assert len(created["embedding"]) == 1

    custom_revision = tmp_path / "bases" / "fiches.sqlite"
    chosen = assistant_module.EduRAGAssistant(
        tmp_path / "documents",
        tmp_path / "index",
        custom_revision,
    )

    assert chosen.revision_store.path == custom_revision
    assert chosen.conversation_store.path == tmp_path / "bases" / "conversations.sqlite"
    assert (tmp_path / "bases" / "conversations.sqlite").is_file()


def test_conversation_database_path_can_be_chosen(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    created = _install_fakes(monkeypatch)
    database = tmp_path / "echanges" / "conversations.sqlite"
    documents = (tmp_path / "documents").resolve()

    assistant = assistant_module.EduRAGAssistant(
        tmp_path / "documents",
        tmp_path / "index",
        conversation_database_path=database,
    )

    assert assistant.conversation_store.path == database
    assert database.is_file()
    assert assistant.revision_store.path == documents / "revisions.sqlite"
    assert not (documents / "conversations.sqlite").exists()
    assert len(created["llm"]) == 1
    assert len(created["embedding"]) == 1


def test_conversations_can_be_created_listed_read_and_deleted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assistant, _created = _assistant(tmp_path, monkeypatch)
    assistant.ask = lambda *args, **kwargs: (_ for _ in ()).throw(
        AssertionError("La gestion des conversations ne doit pas interroger le modèle.")
    )

    assert assistant.list_conversations() == []
    older = assistant.create_conversation("Ancienne")
    newer = assistant.create_conversation("Récente")

    assert isinstance(older, int) and older > 0
    assert isinstance(newer, int) and newer != older
    assert [conversation["id"] for conversation in assistant.list_conversations()] == [newer, older]
    assert assistant.get_conversation(older)["title"] == "Ancienne"
    assert assistant.get_conversation(older)["messages"] == []

    assistant.delete_conversation(older)

    assert [conversation["id"] for conversation in assistant.list_conversations()] == [newer]
    with pytest.raises(ValueError, match="introuvable"):
        assistant.get_conversation(older)


def test_send_message_asks_once_and_saves_the_question_the_answer_and_the_sources(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assistant, _created = _assistant(tmp_path, monkeypatch)
    conversation_id = assistant.create_conversation("Mitochondrie")
    calls: list[tuple[str, int]] = []

    def spy_ask(question: str, top_k: int = 3) -> dict[str, str | list[dict[str, str | int]]]:
        stored = assistant.get_conversation(conversation_id)
        assert stored["messages"][-1]["role"] == "user"
        assert stored["messages"][-1]["content"] == question
        calls.append((question, top_k))
        return _answer_for(question, top_k)

    assistant.ask = spy_ask

    first = assistant.send_message(conversation_id, "Que produit la mitochondrie ?")
    second = assistant.send_message(conversation_id, "Sur quelle page ?", top_k=2)
    assistant.ask = lambda *args, **kwargs: (_ for _ in ()).throw(
        AssertionError("La consultation ne doit pas interroger le modèle.")
    )
    reopened = assistant.get_conversation(conversation_id)

    assert calls == [("Que produit la mitochondrie ?", 3), ("Sur quelle page ?", 2)]
    assert first["conversation_id"] == conversation_id
    assert first["question"]["role"] == "user"
    assert first["question"]["content"] == "Que produit la mitochondrie ?"
    assert first["question"]["sources"] == []
    assert first["answer"]["role"] == "assistant"
    assert first["answer"]["content"] == "Réponse à : Que produit la mitochondrie ?"
    assert first["answer"]["sources"] == [_source()]
    assert second["answer"]["sources"] == [_source()]
    assert [message["content"] for message in reopened["messages"]] == [
        "Que produit la mitochondrie ?",
        "Réponse à : Que produit la mitochondrie ?",
        "Sur quelle page ?",
        "Réponse à : Sur quelle page ?",
    ]
    assert reopened["messages"][1]["sources"] == [_source()]
    assert reopened["messages"][3]["sources"] == [_source()]


def test_reading_a_conversation_does_not_call_the_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assistant, _created = _assistant(tmp_path, monkeypatch)
    conversation_id = assistant.create_conversation("Déjà enregistrée")
    assistant.ask = lambda *args, **kwargs: _answer_for("Question enregistrée")
    assistant.send_message(conversation_id, "Question enregistrée")
    assistant.ask = lambda *args, **kwargs: (_ for _ in ()).throw(
        AssertionError("La consultation ne doit pas interroger le modèle.")
    )

    loaded = assistant.get_conversation(conversation_id)
    listed = assistant.list_conversations()

    assert loaded["messages"][0]["content"] == "Question enregistrée"
    assert loaded["messages"][1]["content"] == "Réponse à : Question enregistrée"
    assert listed[0]["id"] == conversation_id


def test_missing_conversation_does_not_call_the_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assistant, _created = _assistant(tmp_path, monkeypatch)
    calls: list[str] = []

    def spy_ask(question: str, top_k: int = 3) -> dict[str, str | list]:
        calls.append(question)
        return _answer_for(question, top_k)

    assistant.ask = spy_ask

    with pytest.raises(ValueError, match="introuvable"):
        assistant.send_message(4, "Question sans conversation")

    assert calls == []
    assert assistant.list_conversations() == []


def test_failed_answer_is_not_saved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assistant, _created = _assistant(tmp_path, monkeypatch)
    conversation_id = assistant.create_conversation("Échec")
    calls: list[str] = []

    def fail_ask(question: str, top_k: int = 3) -> dict[str, str | list]:
        calls.append(question)
        raise RuntimeError("La génération de la réponse a échoué.")

    assistant.ask = fail_ask

    with pytest.raises(RuntimeError, match="génération"):
        assistant.send_message(conversation_id, "Question interrompue")

    stored = assistant.get_conversation(conversation_id)
    assert calls == ["Question interrompue"]
    assert [message["role"] for message in stored["messages"]] == ["user"]
    assert stored["messages"][0]["content"] == "Question interrompue"


def test_conversation_remains_after_a_new_assistant(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assistant, _created = _assistant(tmp_path, monkeypatch)
    conversation_id = assistant.create_conversation("Persistante")
    assistant.ask = lambda *args, **kwargs: _answer_for("Question conservée")
    assistant.send_message(conversation_id, "Question conservée")
    saved = assistant.get_conversation(conversation_id)

    restarted = assistant_module.EduRAGAssistant(
        tmp_path / "documents",
        tmp_path / "index",
    )
    restarted.ask = lambda *args, **kwargs: (_ for _ in ()).throw(
        AssertionError("La réouverture ne doit pas interroger le modèle.")
    )

    assert restarted.list_conversations() == assistant.list_conversations()
    assert restarted.get_conversation(conversation_id) == saved
    assert restarted.llm.prompts == []
