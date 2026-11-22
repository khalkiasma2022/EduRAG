"""Tests du découpage des pages en chunks.

Les pages ont le même format que load_pdf :
text, page_number et filename.
"""

import pytest

from src.ingestion.text_splitter import split_documents


def _page(text: str, page_number: int = 1, filename: str = "cours.pdf") -> dict:
    return {"text": text, "page_number": page_number, "filename": filename}


def test_split_long_text() -> None:
    documents = [_page("abcdefghij" * 80)]

    chunks = split_documents(documents)

    assert len(chunks) > 1
    assert "".join(chunk["text"] for chunk in chunks).count("a") > 0


def test_chunk_size_is_respected() -> None:
    documents = [_page("abcdefghij" * 40)]

    chunks = split_documents(documents, chunk_size=50, chunk_overlap=10)

    assert chunks
    assert all(len(chunk["text"]) <= 50 for chunk in chunks)


def test_chunks_overlap() -> None:
    documents = [_page("abcdefghij" * 12)]

    chunks = split_documents(documents, chunk_size=20, chunk_overlap=5)

    assert len(chunks) > 1
    for current, following in zip(chunks, chunks[1:]):
        assert current["text"][-5:] == following["text"][:5]


def test_split_keeps_filename() -> None:
    documents = [_page("Introduction au cours", filename="lecon.pdf")]

    chunks = split_documents(documents)

    assert chunks
    assert all(chunk["filename"] == "lecon.pdf" for chunk in chunks)


def test_split_keeps_page_number() -> None:
    documents = [
        _page("Premiere page du cours", page_number=2),
        _page("Deuxieme page du cours", page_number=5),
    ]

    chunks = split_documents(documents)

    assert [chunk["page_number"] for chunk in chunks] == [2, 5]


def test_chunk_ids_are_sequential() -> None:
    documents = [
        _page("abcdefghij" * 12, page_number=1, filename="cours.pdf"),
        _page("Conclusion du cours", page_number=3, filename="cours.pdf"),
    ]

    chunks = split_documents(documents, chunk_size=20, chunk_overlap=5)
    chunk_ids = [chunk["chunk_id"] for chunk in chunks]

    assert chunk_ids == list(range(1, len(chunk_ids) + 1))
    assert chunks[-1]["page_number"] == 3


def test_empty_document_list() -> None:
    assert split_documents([]) == []


@pytest.mark.parametrize(
    "documents",
    [
        None,
        {"text": "bonjour", "page_number": 1, "filename": "cours.pdf"},
        ["pas un dictionnaire"],
        [{"page_number": 1, "filename": "cours.pdf"}],
        [{"text": "bonjour", "filename": "cours.pdf"}],
        [{"text": "bonjour", "page_number": "1", "filename": "cours.pdf"}],
        [{"text": 12, "page_number": 1, "filename": "cours.pdf"}],
        [{"text": "bonjour", "page_number": 1, "filename": ""}],
    ],
)
def test_invalid_documents_raise(documents) -> None:
    with pytest.raises((TypeError, ValueError)):
        split_documents(documents)


def test_blank_pages_are_ignored() -> None:
    documents = [
        _page("   \n", page_number=1),
        _page("Contenu utile", page_number=2),
        _page("", page_number=3),
    ]

    chunks = split_documents(documents)

    assert len(chunks) == 1
    assert chunks[0]["text"] == "Contenu utile"
    assert chunks[0]["page_number"] == 2
    assert chunks[0]["filename"] == "cours.pdf"
    assert chunks[0]["chunk_id"] == 1
