"""Tests de la bibliothèque de PDF.

Les PDF sont créés dans un dossier temporaire.
Les embeddings et FAISS sont simulés : aucun modèle n'est téléchargé.
"""

import hashlib
from pathlib import Path

import pytest
from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

from src.ingestion.library import DocumentLibrary


class FakeEmbeddingModel:
    """Remplace EmbeddingModel.embed_chunks sans charger de modèle."""

    def __init__(self) -> None:
        self.calls = 0

    def embed_chunks(self, chunks: list[dict]) -> list[list[float]]:
        self.calls += 1
        return [[1.0, float(index)] for index, _chunk in enumerate(chunks)]


class FakeVectorStore:
    """Remplace VectorStore.create et VectorStore.save."""

    def __init__(self) -> None:
        self.chunks: list[dict] = []
        self.saved_directories: list[Path] = []
        self.fail_on_save = False

    def create(self, chunks: list[dict], embeddings: list[list[float]]) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError("Chaque chunk doit avoir un embedding.")
        self.chunks = [dict(chunk) for chunk in chunks]

    def save(self, directory: str | Path) -> None:
        if self.fail_on_save:
            raise RuntimeError("sauvegarde impossible")
        folder = Path(directory)
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "index.faiss").write_text(str(len(self.chunks)), encoding="utf-8")
        (folder / "edurag_manifest.json").write_text("{}", encoding="utf-8")
        self.saved_directories.append(folder)


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


def _write_blank_pdf(path: Path) -> Path:
    """Crée un PDF valide sans texte extractible."""
    writer = PdfWriter()
    writer.add_blank_page(width=612, height=792)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        writer.write(handle)
    return path


def _library(tmp_path: Path) -> tuple[DocumentLibrary, FakeEmbeddingModel, FakeVectorStore]:
    embeddings = FakeEmbeddingModel()
    store = FakeVectorStore()
    library = DocumentLibrary(
        tmp_path / "documents",
        tmp_path / "index",
        embeddings,
        store,
    )
    return library, embeddings, store


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_add_pdf_registers_the_document(tmp_path: Path) -> None:
    library, embeddings, store = _library(tmp_path)
    source = _write_text_pdf(tmp_path / "cours.pdf", ["Cours de Python"])

    added = library.add_pdf(source)
    documents = library.list_documents()

    assert documents == [added]
    assert added["filename"] == "cours.pdf"
    assert added["sha256"] == _sha256(source)
    assert len(added["document_id"]) == 32
    assert Path(added["stored_path"]).is_file()
    assert Path(added["stored_path"]).name == f"{added['document_id']}.pdf"
    assert embeddings.calls == 1
    assert store.chunks
    assert {chunk["document_id"] for chunk in store.chunks} == {added["document_id"]}
    assert (tmp_path / "index" / "index.faiss").is_file()


def test_add_pdfs_registers_each_document(tmp_path: Path) -> None:
    library, _embeddings, store = _library(tmp_path)
    first = _write_text_pdf(tmp_path / "python.pdf", ["Cours de Python"])
    second = _write_text_pdf(tmp_path / "bio.pdf", ["Cours de biologie"])

    added = library.add_pdfs([first, second])

    assert [item["filename"] for item in library.list_documents()] == ["python.pdf", "bio.pdf"]
    assert [item["document_id"] for item in added] == [
        item["document_id"] for item in library.list_documents()
    ]
    identifiers = {chunk["document_id"] for chunk in store.chunks}
    assert identifiers == {item["document_id"] for item in added}


def test_list_documents_is_empty_at_start(tmp_path: Path) -> None:
    library, _embeddings, _store = _library(tmp_path)

    assert library.list_documents() == []
    assert not (tmp_path / "index" / "index.faiss").exists()


def test_duplicate_content_is_refused(tmp_path: Path) -> None:
    library, embeddings, _store = _library(tmp_path)
    source = _write_text_pdf(tmp_path / "cours.pdf", ["Cours de Python"])
    copy = tmp_path / "autre-nom.pdf"
    copy.write_bytes(source.read_bytes())
    library.add_pdf(source)

    with pytest.raises(ValueError, match="déjà"):
        library.add_pdf(copy)

    assert len(library.list_documents()) == 1
    assert embeddings.calls == 1
    assert list((tmp_path / "documents").glob("*.pdf")) == [
        Path(library.list_documents()[0]["stored_path"])
    ]


def test_same_filename_with_different_content_is_kept(tmp_path: Path) -> None:
    library, _embeddings, store = _library(tmp_path)
    first = _write_text_pdf(tmp_path / "un" / "cours.pdf", ["Cours de Python"])
    second = _write_text_pdf(tmp_path / "deux" / "cours.pdf", ["Cours de biologie"])

    added = library.add_pdfs([first, second])

    assert added[0]["filename"] == added[1]["filename"] == "cours.pdf"
    assert added[0]["sha256"] != added[1]["sha256"]
    assert added[0]["document_id"] != added[1]["document_id"]
    assert added[0]["stored_path"] != added[1]["stored_path"]
    by_text = {chunk["text"]: chunk["document_id"] for chunk in store.chunks}
    assert by_text["Cours de Python"] == added[0]["document_id"]
    assert by_text["Cours de biologie"] == added[1]["document_id"]
    assert {chunk["filename"] for chunk in store.chunks} == {"cours.pdf"}


def test_remove_document_rebuilds_the_index_from_the_remaining_pdf(tmp_path: Path) -> None:
    library, _embeddings, store = _library(tmp_path)
    first = _write_text_pdf(tmp_path / "python.pdf", ["Cours de Python"])
    second = _write_text_pdf(tmp_path / "bio.pdf", ["Cours de biologie"])
    added = library.add_pdfs([first, second])
    removed = added[0]

    library.remove_document(removed["document_id"])
    documents = library.list_documents()

    assert [item["document_id"] for item in documents] == [added[1]["document_id"]]
    assert not Path(removed["stored_path"]).exists()
    assert Path(documents[0]["stored_path"]).is_file()
    assert {chunk["document_id"] for chunk in store.chunks} == {added[1]["document_id"]}
    assert (tmp_path / "index" / "index.faiss").read_text(encoding="utf-8") == "1"


def test_rebuild_index_uses_every_remaining_document(tmp_path: Path) -> None:
    library, embeddings, store = _library(tmp_path)
    first = _write_text_pdf(tmp_path / "python.pdf", ["Cours de Python"])
    second = _write_text_pdf(tmp_path / "bio.pdf", ["Cours de biologie"])
    added = library.add_pdfs([first, second])
    calls_before = embeddings.calls

    library.rebuild_index()

    assert embeddings.calls == calls_before + 1
    assert {chunk["document_id"] for chunk in store.chunks} == {
        added[0]["document_id"],
        added[1]["document_id"],
    }


def test_failed_add_keeps_the_existing_document(tmp_path: Path) -> None:
    library, _embeddings, store = _library(tmp_path)
    first = _write_text_pdf(tmp_path / "python.pdf", ["Cours de Python"])
    library.add_pdf(first)
    kept = library.list_documents()[0]
    marker = (tmp_path / "index" / "index.faiss").read_text(encoding="utf-8")
    store.fail_on_save = True
    second = _write_text_pdf(tmp_path / "bio.pdf", ["Cours de biologie"])

    with pytest.raises(RuntimeError, match="sauvegarde"):
        library.add_pdf(second)

    assert library.list_documents() == [kept]
    assert Path(kept["stored_path"]).is_file()
    assert (tmp_path / "index" / "index.faiss").read_text(encoding="utf-8") == marker
    assert len(list((tmp_path / "documents").glob("*.pdf"))) == 1


def test_failed_removal_keeps_the_document_and_its_index(tmp_path: Path) -> None:
    library, _embeddings, _store = _library(tmp_path)
    source = _write_text_pdf(tmp_path / "python.pdf", ["Cours de Python"])
    added = library.add_pdf(source)
    marker = (tmp_path / "index" / "index.faiss").read_text(encoding="utf-8")

    def fail_catalog(_entries: list[dict]) -> None:
        raise RuntimeError("catalogue illisible")

    library._write_catalog = fail_catalog  # type: ignore[method-assign]

    with pytest.raises(RuntimeError, match="catalogue"):
        library.remove_document(added["document_id"])

    assert library.list_documents() == [added]
    assert Path(added["stored_path"]).is_file()
    assert (tmp_path / "index" / "index.faiss").read_text(encoding="utf-8") == marker


def test_removing_the_last_document_deletes_the_index(tmp_path: Path) -> None:
    library, _embeddings, _store = _library(tmp_path)
    source = _write_text_pdf(tmp_path / "python.pdf", ["Cours de Python"])
    added = library.add_pdf(source)

    library.remove_document(added["document_id"])

    assert library.list_documents() == []
    assert not Path(added["stored_path"]).exists()
    assert not (tmp_path / "index").exists()


def test_invalid_files_are_refused(tmp_path: Path) -> None:
    library, embeddings, _store = _library(tmp_path)
    missing = tmp_path / "absent.pdf"
    invalid = tmp_path / "notes.txt"
    invalid.write_text("ceci n'est pas un pdf", encoding="utf-8")
    blank = _write_blank_pdf(tmp_path / "vide.pdf")

    with pytest.raises(FileNotFoundError):
        library.add_pdf(missing)
    with pytest.raises(ValueError):
        library.add_pdf(invalid)
    with pytest.raises(ValueError):
        library.add_pdf(blank)

    assert library.list_documents() == []
    assert embeddings.calls == 0
    assert list((tmp_path / "documents").glob("*.pdf")) == []


def test_unknown_document_cannot_be_removed(tmp_path: Path) -> None:
    library, _embeddings, _store = _library(tmp_path)
    source = _write_text_pdf(tmp_path / "python.pdf", ["Cours de Python"])
    library.add_pdf(source)

    with pytest.raises(ValueError, match="introuvable"):
        library.remove_document("a" * 32)

    assert len(library.list_documents()) == 1
