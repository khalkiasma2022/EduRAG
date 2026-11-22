"""Tests de la base vectorielle FAISS.

Les embeddings sont simulés : le modèle Sentence Transformers n'est pas chargé.
"""

from pathlib import Path

import pytest

from src.retrieval.vector_store import VectorStore


class FakeEmbeddingModel:
    """Remplace EmbeddingModel : aucun téléchargement, vecteurs fixes."""

    def __init__(self) -> None:
        self.queries: list[str] = []

    def embed_query(self, question: str) -> list[float]:
        self.queries.append(question)
        if not isinstance(question, str) or question.strip() == "":
            raise ValueError("La question ne peut pas être vide.")
        if "python" in question.lower():
            return [1.0, 0.0, 0.0, 0.0]
        return [0.0, 1.0, 0.0, 0.0]


def _chunk(
    text: str,
    chunk_id: int,
    filename: str = "cours.pdf",
    page_number: int = 1,
) -> dict:
    return {
        "text": text,
        "filename": filename,
        "page_number": page_number,
        "chunk_id": chunk_id,
    }


def _sample_data() -> tuple[list[dict], list[list[float]]]:
    chunks = [
        _chunk("Le langage Python sert à programmer.", 1, "python.pdf", 1),
        _chunk("La photosynthèse produit de l'oxygène.", 2, "bio.pdf", 4),
        _chunk("La Révolution française commence en 1789.", 3, "histoire.pdf", 2),
        _chunk("Un triangle a trois côtés.", 4, "maths.pdf", 3),
    ]
    embeddings = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
    return chunks, embeddings


def _ready_store() -> VectorStore:
    chunks, embeddings = _sample_data()
    store = VectorStore(FakeEmbeddingModel())
    store.create(chunks, embeddings)
    return store


def test_create_vector_store() -> None:
    store = _ready_store()

    results = store.search("Cours de Python", top_k=1)

    assert results[0]["text"] == "Le langage Python sert à programmer."


def test_search_returns_nearest_chunk() -> None:
    store = _ready_store()

    results = store.search("Explique Python", top_k=2)

    assert results[0]["chunk_id"] == 1
    assert results[1]["chunk_id"] != 1


def test_search_keeps_metadata() -> None:
    store = _ready_store()

    result = store.search("Python", top_k=1)[0]

    assert result["filename"] == "python.pdf"
    assert result["page_number"] == 1
    assert result["chunk_id"] == 1
    assert result["text"] == "Le langage Python sert à programmer."


def test_save_creates_local_files(tmp_path: Path) -> None:
    store = _ready_store()
    destination = tmp_path / "index"

    store.save(destination)

    assert destination.is_dir()
    assert any(destination.iterdir())


def test_load_saved_store(tmp_path: Path) -> None:
    original = _ready_store()
    destination = tmp_path / "index"
    original.save(destination)

    loaded = VectorStore(FakeEmbeddingModel())
    loaded.load(destination)

    result = loaded.search("Python", top_k=1)[0]
    assert result["text"] == "Le langage Python sert à programmer."
    assert result["filename"] == "python.pdf"
    assert result["page_number"] == 1
    assert result["chunk_id"] == 1


def test_search_after_load_matches_original(tmp_path: Path) -> None:
    original = _ready_store()
    destination = tmp_path / "index"
    original.save(destination)
    loaded = VectorStore(FakeEmbeddingModel())
    loaded.load(destination)

    before = original.search("La biologie des plantes", top_k=1)
    after = loaded.search("La biologie des plantes", top_k=1)

    assert after == before
    assert after[0]["filename"] == "bio.pdf"


def test_top_k_limits_results() -> None:
    store = _ready_store()

    assert len(store.search("Python")) == 3
    assert len(store.search("Python", top_k=1)) == 1
    assert len(store.search("Python", top_k=4)) == 4


@pytest.mark.parametrize(
    "call",
    [
        lambda store: store.create([], []),
        lambda store: store.search("Python"),
        lambda store: store.save("index"),
        lambda store: store.create(
            [_chunk("Python", 1)],
            [[1.0, 0.0], [0.0, 1.0]],
        ),
        lambda store: store.create(
            [{"text": "Python"}],
            [[1.0, 0.0, 0.0, 0.0]],
        ),
        lambda store: store.create("pas une liste", [[1.0, 0.0, 0.0, 0.0]]),
        lambda store: store.create(
            [_chunk("Python", 1)],
            "pas des vecteurs",
        ),
        lambda store: _ready_store().search(""),
        lambda store: _ready_store().search(None),
        lambda store: _ready_store().search("Python", top_k=0),
        lambda store: _ready_store().search("Python", top_k=-1),
        lambda store: VectorStore("pas un modèle"),
    ],
)
def test_invalid_inputs_raise(call) -> None:
    store = VectorStore(FakeEmbeddingModel())

    with pytest.raises((TypeError, ValueError)):
        call(store)


def test_save_rejects_invalid_path(tmp_path: Path) -> None:
    store = _ready_store()
    missing_parent = tmp_path / "absent" / "index"
    existing_file = tmp_path / "fichier.txt"
    existing_file.write_text("occupe", encoding="utf-8")

    with pytest.raises(ValueError):
        store.save(missing_parent)
    with pytest.raises(ValueError):
        store.save(existing_file)
    with pytest.raises((TypeError, ValueError)):
        store.save("")


def test_load_missing_store(tmp_path: Path) -> None:
    store = VectorStore(FakeEmbeddingModel())

    with pytest.raises(FileNotFoundError):
        store.load(tmp_path / "inexistant")


def test_search_embeds_the_question_once() -> None:
    model = FakeEmbeddingModel()
    chunks, embeddings = _sample_data()
    store = VectorStore(model)
    store.create(chunks, embeddings)

    result = store.search("Cours de Python", top_k=1)

    assert model.queries == ["Cours de Python"]
    assert result[0]["chunk_id"] == 1
    assert result[0]["filename"] == "python.pdf"


def test_search_by_vector_passes_the_vector_to_faiss() -> None:
    model = FakeEmbeddingModel()
    chunks, embeddings = _sample_data()
    store = VectorStore(model)
    store.create(chunks, embeddings)
    received: dict[str, object] = {}
    original = store._store.similarity_search_with_score_by_vector

    def spy(embedding, k=4, **kwargs):
        received["embedding"] = list(embedding)
        received["k"] = k
        return original(embedding, k=k, **kwargs)

    store._store.similarity_search_with_score_by_vector = spy
    vector = [0.0, 1.0, 0.0, 0.0]

    results = store.search_by_vector(vector, top_k=1)

    assert model.queries == []
    assert received["embedding"] == vector
    assert received["k"] == 1
    assert results[0]["text"] == "La photosynthèse produit de l'oxygène."
    assert results[0]["filename"] == "bio.pdf"
    assert results[0]["page_number"] == 4
    assert results[0]["chunk_id"] == 2


def test_search_by_vector_rejects_wrong_dimension() -> None:
    store = _ready_store()

    with pytest.raises(ValueError, match="dimension"):
        store.search_by_vector([1.0, 0.0], top_k=1)


def test_search_by_vector_rejects_empty_store_and_invalid_vector() -> None:
    store = VectorStore(FakeEmbeddingModel())

    with pytest.raises(ValueError, match="vide"):
        store.search_by_vector([1.0, 0.0, 0.0, 0.0])

    ready = _ready_store()
    with pytest.raises(TypeError):
        ready.search_by_vector(["pas", "un", "vecteur", "numérique"])


def test_load_rejects_untrusted_directory(tmp_path: Path) -> None:
    foreign = tmp_path / "telechargement"
    foreign.mkdir()
    (foreign / "index.faiss").write_bytes(b"contenu non fiable")

    store = VectorStore(FakeEmbeddingModel())

    with pytest.raises(ValueError, match="fiable"):
        store.load(foreign)
