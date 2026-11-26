"""Tests du CourseReader Smart Revision V2."""

from __future__ import annotations

from pathlib import Path

import pytest
from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

from src.ingestion.library import DocumentLibrary
from src.revision.engine.llm_guard import MAX_ANALYSIS_WINDOWS, MAX_COURSE_TOKENS_PER_WINDOW
from src.revision.engine.models import CoursePage
from src.revision.engine.reader import CourseReader, _clean_page_text


class FakeEmbeddingModel:
    def embed_chunks(self, chunks: list[dict]) -> list[list[float]]:
        return [[1.0, float(index)] for index, _chunk in enumerate(chunks)]


class FakeVectorStore:
    def create(self, chunks: list[dict], embeddings: list[list[float]]) -> None:
        return None

    def save(self, directory: str | Path) -> None:
        folder = Path(directory)
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "index.faiss").write_text("ok", encoding="utf-8")


def _write_text_pdf(path: Path, pages: list[str]) -> Path:
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


def _library_with_pdf(tmp_path: Path, pages: list[str], filename: str = "cours.pdf"):
    library = DocumentLibrary(
        tmp_path / "documents",
        tmp_path / "index",
        FakeEmbeddingModel(),
        FakeVectorStore(),
    )
    source = _write_text_pdf(tmp_path / filename, pages)
    added = library.add_pdf(source)
    return library, added


def test_read_returns_whole_document_as_course_pages(tmp_path: Path) -> None:
    library, added = _library_with_pdf(
        tmp_path,
        ["Page un Python", "Page deux variables", "Page trois fonctions"],
        filename="cours_python.pdf",
    )
    reader = CourseReader(library)

    result = reader.read(added["document_id"])

    assert result.filename == "cours_python.pdf"
    assert result.page_start == 1
    assert result.page_end == 3
    assert result.pages == [
        CoursePage(page_number=1, text="Page un Python", filename="cours_python.pdf"),
        CoursePage(page_number=2, text="Page deux variables", filename="cours_python.pdf"),
        CoursePage(page_number=3, text="Page trois fonctions", filename="cours_python.pdf"),
    ]


def test_read_returns_page_interval(tmp_path: Path) -> None:
    library, added = _library_with_pdf(
        tmp_path,
        ["Page un", "Page deux", "Page trois"],
    )
    reader = CourseReader(library)

    result = reader.read(added["document_id"], page_start=2, page_end=3)

    assert [page.page_number for page in result.pages] == [2, 3]
    assert [page.text for page in result.pages] == ["Page deux", "Page trois"]
    assert result.page_start == 2
    assert result.page_end == 3


def test_read_unknown_document_is_refused(tmp_path: Path) -> None:
    library, _added = _library_with_pdf(tmp_path, ["Page un"])
    reader = CourseReader(library)

    with pytest.raises(ValueError, match="introuvable"):
        reader.read("document-absent")


def test_windows_respect_token_budget(tmp_path: Path) -> None:
    library, added = _library_with_pdf(
        tmp_path,
        ["AAAA", "BBBB", "CCCC", "DDDD"],
    )
    # Chaque page = 800 tokens → deux pages max par fenêtre (800+800 ≤ 1800).
    reader = CourseReader(library, count_tokens=lambda text: 800)

    result = reader.read(added["document_id"])

    assert len(result.windows) == 2
    assert [page.text for page in result.windows[0]] == ["AAAA", "BBBB"]
    assert [page.text for page in result.windows[1]] == ["CCCC", "DDDD"]
    for window in result.windows:
        assert sum(800 for _ in window) <= MAX_COURSE_TOKENS_PER_WINDOW


def test_too_many_windows_raises_explicit_error(tmp_path: Path) -> None:
    pages = [f"Page{i}" for i in range(1, MAX_ANALYSIS_WINDOWS + 2)]
    library, added = _library_with_pdf(tmp_path, pages)
    # Une page = une fenêtre pleine → besoin de MAX+1 fenêtres > plafond.
    reader = CourseReader(
        library,
        count_tokens=lambda text: MAX_COURSE_TOKENS_PER_WINDOW,
    )

    with pytest.raises(ValueError, match="capacité d'analyse|fenêtres"):
        reader.read(added["document_id"])


def test_single_page_over_window_budget_raises(tmp_path: Path) -> None:
    library, added = _library_with_pdf(tmp_path, ["HUGE"])
    reader = CourseReader(
        library,
        count_tokens=lambda text: MAX_COURSE_TOKENS_PER_WINDOW + 1,
    )

    with pytest.raises(ValueError, match="moins de pages"):
        reader.read(added["document_id"])


def test_clean_page_text_preserves_paragraphs_and_newlines() -> None:
    raw = (
        "Titre du cours\n"
        "  Ligne   avec   espaces\tredondants  \n"
        "\n"
        "\n"
        "Paragraphe suivant.\n"
        "Encore une ligne.\n"
        "\n"
    )
    cleaned = _clean_page_text(raw)
    assert "\n" in cleaned
    assert "Titre du cours\nLigne avec espaces redondants\n\nParagraphe suivant.\nEncore une ligne." == cleaned
    assert "Titre du cours" in cleaned
    assert "Paragraphe suivant." in cleaned
    assert "  " not in cleaned.replace("\n\n", "")


def test_clean_page_text_does_not_drop_content() -> None:
    raw = "Alpha\nBeta\nGamma"
    cleaned = _clean_page_text(raw)
    assert "Alpha" in cleaned and "Beta" in cleaned and "Gamma" in cleaned
    assert cleaned.count("\n") == 2


def test_count_tokens_must_be_callable(tmp_path: Path) -> None:
    library, _added = _library_with_pdf(tmp_path, ["Page un"])
    with pytest.raises(TypeError, match="count_tokens|appelable"):
        CourseReader(library, count_tokens="not-callable")  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "bad_value",
    [0, -3, True, False, "12", 3.5, None],
)
def test_count_tokens_rejects_non_positive_int_results(
    tmp_path: Path, bad_value: object
) -> None:
    library, added = _library_with_pdf(tmp_path, ["Page un"])
    reader = CourseReader(library, count_tokens=lambda _text: bad_value)  # type: ignore[arg-type, return-value]
    with pytest.raises(ValueError, match="entier strictement positif|count_tokens"):
        reader.read(added["document_id"])


def test_count_tokens_accepts_strict_positive_int(tmp_path: Path) -> None:
    library, added = _library_with_pdf(tmp_path, ["AAAA", "BBBB"])
    reader = CourseReader(library, count_tokens=lambda _text: 1)
    result = reader.read(added["document_id"])
    assert len(result.pages) == 2
    assert len(result.windows) == 1
