"""Tests unitaires des embeddings.

Le vrai modèle n'est pas téléchargé : SentenceTransformer est simulé.
Le test réel reste ignoré tant que EDURAG_RUN_REAL_EMBEDDINGS n'est pas défini.
"""

import os
from unittest.mock import MagicMock, patch

import pytest

from src.retrieval.embeddings import DEFAULT_MODEL_NAME, EmbeddingModel

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def _chunk(text: str, chunk_id: int = 1) -> dict:
    """Reprend le format produit par split_documents."""
    return {
        "text": text,
        "filename": "cours.pdf",
        "page_number": 1,
        "chunk_id": chunk_id,
    }


@pytest.fixture
def fake_encoder() -> MagicMock:
    encoder = MagicMock()
    encoder.get_sentence_embedding_dimension.return_value = 4
    return encoder


@pytest.fixture
def embedding_model(fake_encoder: MagicMock):
    with (
        patch("src.retrieval.embeddings.SentenceTransformer", return_value=fake_encoder) as constructor,
        patch("src.retrieval.embeddings.torch.cuda.is_available", return_value=False),
    ):
        model = EmbeddingModel()
    return model, constructor, fake_encoder


def test_default_model_name() -> None:
    assert DEFAULT_MODEL_NAME == MODEL_NAME


def test_load_model(embedding_model) -> None:
    model, constructor, fake_encoder = embedding_model

    constructor.assert_called_once_with(MODEL_NAME, device="cpu")
    assert model.model is fake_encoder
    assert model.device == "cpu"
    assert model.dimension == 4


def test_embed_query_returns_vector(embedding_model) -> None:
    model, _constructor, fake_encoder = embedding_model
    fake_encoder.encode.return_value = [0.1, 0.2, 0.3, 0.4]

    vector = model.embed_query("Quelle est la définition ?")

    assert vector == pytest.approx([0.1, 0.2, 0.3, 0.4])
    fake_encoder.encode.assert_called_once()
    assert fake_encoder.encode.call_args.args[0] == "Quelle est la définition ?"


def test_embed_several_chunks(embedding_model) -> None:
    model, _constructor, fake_encoder = embedding_model
    fake_encoder.encode.return_value = [
        [0.1, 0.0, 0.0, 0.0],
        [0.0, 0.2, 0.0, 0.0],
        [0.0, 0.0, 0.3, 0.0],
    ]
    chunks = [
        _chunk("premier morceau", 1),
        _chunk("deuxième morceau", 2),
        _chunk("troisième morceau", 3),
    ]

    vectors = model.embed_chunks(chunks)

    assert len(vectors) == 3
    assert vectors[1] == pytest.approx([0.0, 0.2, 0.0, 0.0])
    assert fake_encoder.encode.call_args.args[0] == [
        "premier morceau",
        "deuxième morceau",
        "troisième morceau",
    ]


def test_vector_dimension(embedding_model) -> None:
    model, _constructor, fake_encoder = embedding_model

    assert model.dimension == fake_encoder.get_sentence_embedding_dimension()


def test_query_and_chunks_share_dimension(embedding_model) -> None:
    model, _constructor, fake_encoder = embedding_model
    fake_encoder.encode.side_effect = [
        [0.4, 0.3, 0.2, 0.1],
        [[0.1, 0.2, 0.3, 0.4], [0.2, 0.2, 0.2, 0.2]],
    ]

    query_vector = model.embed_query("Question de cours")
    chunk_vectors = model.embed_chunks([_chunk("Réponse possible"), _chunk("Autre passage", 2)])

    assert len(query_vector) == model.dimension
    assert all(len(vector) == model.dimension for vector in chunk_vectors)
    assert len({len(query_vector), *(len(vector) for vector in chunk_vectors)}) == 1


def test_model_is_not_reloaded_between_calls(embedding_model) -> None:
    model, constructor, fake_encoder = embedding_model
    fake_encoder.encode.return_value = [0.1, 0.2, 0.3, 0.4]

    model.embed_query("Première question")
    fake_encoder.encode.return_value = [[0.1, 0.2, 0.3, 0.4]]
    model.embed_chunks([_chunk("Un passage")])

    assert constructor.call_count == 1


def test_gpu_is_used_when_available(fake_encoder: MagicMock) -> None:
    with (
        patch("src.retrieval.embeddings.SentenceTransformer", return_value=fake_encoder) as constructor,
        patch("src.retrieval.embeddings.torch.cuda.is_available", return_value=True),
    ):
        model = EmbeddingModel()

    constructor.assert_called_once_with(MODEL_NAME, device="cuda")
    assert model.device == "cuda"


@pytest.mark.parametrize(
    ("call", "message"),
    [
        (lambda model: model.embed_chunks([]), "vide"),
        (lambda model: model.embed_query(""), "vide"),
        (lambda model: model.embed_query("   "), "vide"),
        (lambda model: model.embed_chunks([_chunk("   ")]), "vide"),
    ],
)
def test_empty_inputs_raise(embedding_model, call, message: str) -> None:
    model, _constructor, _fake_encoder = embedding_model

    with pytest.raises(ValueError, match=message):
        call(model)


@pytest.mark.parametrize(
    "call",
    [
        lambda model: model.embed_chunks(None),
        lambda model: model.embed_chunks("un texte"),
        lambda model: model.embed_chunks([{"filename": "cours.pdf"}]),
        lambda model: model.embed_chunks([{"text": 12}]),
        lambda model: model.embed_chunks(["juste une chaîne"]),
        lambda model: model.embed_query(None),
        lambda model: model.embed_query(42),
    ],
)
def test_invalid_inputs_raise(embedding_model, call) -> None:
    model, _constructor, _fake_encoder = embedding_model

    with pytest.raises((TypeError, ValueError)):
        call(model)


@pytest.mark.skipif(
    os.environ.get("EDURAG_RUN_REAL_EMBEDDINGS") != "1",
    reason=(
        "Test réel facultatif. Le lancer avec "
        "EDURAG_RUN_REAL_EMBEDDINGS=1 une fois le modèle téléchargé."
    ),
)
def test_real_model_when_available() -> None:
    model = EmbeddingModel()
    vector = model.embed_query("Bonjour les étudiants")
    chunk_vectors = model.embed_chunks([_chunk("Les cours sont en français")])

    assert model.model_name == MODEL_NAME
    assert model.dimension == 384
    assert len(vector) == model.dimension
    assert len(chunk_vectors) == 1
    assert len(chunk_vectors[0]) == model.dimension
