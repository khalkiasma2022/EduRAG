"""Base vectorielle locale avec FAISS.

Les chunks viennent de text_splitter et les vecteurs de EmbeddingModel.
Le modèle est reçu déjà chargé : cette classe ne le recharge pas.
"""

import json
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings

# Fichier texte qui identifie une base créée par EduRAG.
# Le chargement FAISS utilise pickle : on n'ouvre pas un dossier sans ce témoin.
MANIFEST_NAME = "edurag_manifest.json"
_CREATOR = "EduRAG"
_FORMAT_VERSION = 1


class _EmbeddingAdapter(Embeddings):
    """Présente EmbeddingModel au format attendu par LangChain."""

    def __init__(self, embedding_model: object) -> None:
        self._embedding_model = embedding_model

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_query(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embedding_model.embed_query(text)


class VectorStore:
    """Index FAISS des chunks, avec leurs métadonnées."""

    def __init__(self, embedding_model: object) -> None:
        """Conserve le modèle déjà créé par l'appelant."""
        if not callable(getattr(embedding_model, "embed_query", None)):
            raise TypeError("embedding_model doit fournir une méthode embed_query.")

        self.embedding_model = embedding_model
        self._adapter = _EmbeddingAdapter(embedding_model)
        self._store: FAISS | None = None

    def create(
        self,
        chunks: list[dict[str, str | int]],
        embeddings: list[list[float]],
    ) -> None:
        """Construit l'index à partir des chunks et des vecteurs déjà calculés."""
        pairs, metadatas = _validated_records(chunks, embeddings)
        # Les vecteurs sont fournis : FAISS ne relance pas le modèle.
        self._store = FAISS.from_embeddings(
            pairs,
            self._adapter,
            metadatas=metadatas,
        )

    def save(self, directory: str | Path) -> None:
        """Enregistre l'index dans un dossier local."""
        if self._store is None:
            raise ValueError("Impossible de sauvegarder une base vide.")

        folder = _directory_to_save(directory)
        self._store.save_local(str(folder))
        manifest = {
            "creator": _CREATOR,
            "format_version": _FORMAT_VERSION,
        }
        (folder / MANIFEST_NAME).write_text(
            json.dumps(manifest),
            encoding="utf-8",
        )

    def load(self, directory: str | Path) -> None:
        """Charge une base créée par EduRAG.

        LangChain lit un fichier pickle. Ce fichier peut exécuter du code
        s'il a été modifié. On vérifie d'abord le témoin EduRAG, puis on
        autorise explicitement la désérialisation.
        """
        folder = _directory_to_load(directory)
        _require_trusted_manifest(folder)
        self._store = FAISS.load_local(
            str(folder),
            self._adapter,
            allow_dangerous_deserialization=True,
        )

    def search(self, question: str, top_k: int = 3) -> list[dict[str, str | int]]:
        """Vectorise la question une fois, puis cherche les chunks proches."""
        self._ensure_store()
        if not isinstance(question, str):
            raise TypeError("La question doit être une chaîne de caractères.")
        if question.strip() == "":
            raise ValueError("La question ne peut pas être vide.")
        _check_top_k(top_k)

        vector = self.embedding_model.embed_query(question)
        return self.search_by_vector(vector, top_k=top_k)

    def search_by_vector(
        self,
        vector: list[float],
        top_k: int = 3,
    ) -> list[dict[str, str | int]]:
        """Cherche les chunks à partir d'un vecteur déjà calculé.

        FAISS reçoit ce vecteur directement : le modèle n'est pas rappelé.
        """
        self._ensure_store()
        _check_top_k(top_k)
        prepared = _read_vector(vector)
        index_dimension = int(self._store.index.d)
        if len(prepared) != index_dimension:
            raise ValueError(
                "Le vecteur a "
                f"{len(prepared)} dimensions, l'index en attend {index_dimension}."
            )

        found = self._store.similarity_search_with_score_by_vector(prepared, k=top_k)
        return _passages_from_matches(found)

    def _ensure_store(self) -> None:
        if self._store is None:
            raise ValueError("Impossible de rechercher dans une base vide.")


def _validated_records(
    chunks: object,
    embeddings: object,
) -> tuple[list[tuple[str, list[float]]], list[dict[str, str | int]]]:
    """Contrôle les chunks et aligne chaque texte sur son vecteur."""
    if not isinstance(chunks, list):
        raise TypeError("chunks doit être une liste.")
    if not isinstance(embeddings, list):
        raise TypeError("embeddings doit être une liste de vecteurs.")
    if len(chunks) == 0 or len(embeddings) == 0:
        raise ValueError("La liste de chunks ne peut pas être vide.")
    if len(chunks) != len(embeddings):
        raise ValueError("Chaque chunk doit avoir exactement un embedding.")

    pairs: list[tuple[str, list[float]]] = []
    metadatas: list[dict[str, str | int]] = []
    expected_size: int | None = None

    for chunk, embedding in zip(chunks, embeddings, strict=True):
        text, metadata = _read_chunk(chunk)
        vector = _read_vector(embedding)
        if expected_size is None:
            expected_size = len(vector)
        elif len(vector) != expected_size:
            raise ValueError("Tous les embeddings doivent avoir la même dimension.")
        pairs.append((text, vector))
        metadatas.append(metadata)

    return pairs, metadatas


def _read_chunk(chunk: object) -> tuple[str, dict[str, str | int]]:
    """Lit le texte et les métadonnées demandées."""
    if not isinstance(chunk, dict):
        raise TypeError("Chaque chunk doit être un dictionnaire.")

    required = ("text", "filename", "page_number", "chunk_id")
    missing = [key for key in required if key not in chunk]
    if missing:
        raise ValueError(f"Chunk incomplet, champ manquant : {', '.join(missing)}.")

    text = chunk["text"]
    filename = chunk["filename"]
    page_number = chunk["page_number"]
    chunk_id = chunk["chunk_id"]

    if not isinstance(text, str) or text.strip() == "":
        raise ValueError("Le texte du chunk ne peut pas être vide.")
    if not isinstance(filename, str) or filename.strip() == "":
        raise ValueError("Le nom de fichier du chunk ne peut pas être vide.")
    if isinstance(page_number, bool) or not isinstance(page_number, int):
        raise TypeError("page_number doit être un entier.")
    if isinstance(chunk_id, bool) or not isinstance(chunk_id, int):
        raise TypeError("chunk_id doit être un entier.")

    metadata: dict[str, str | int] = {
        "filename": filename,
        "page_number": page_number,
        "chunk_id": chunk_id,
    }
    # Les anciens chunks n'ont pas ce champ : on ne le rend pas obligatoire.
    if "document_id" in chunk:
        document_id = chunk["document_id"]
        if not isinstance(document_id, str) or document_id.strip() == "":
            raise ValueError("document_id doit être une chaîne de caractères non vide.")
        metadata["document_id"] = document_id.strip()
    return text, metadata


def _check_top_k(top_k: object) -> None:
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
        raise ValueError("top_k doit être un entier positif.")


def _passages_from_matches(found: list) -> list[dict[str, str | int]]:
    """Reprend le texte et les métadonnées de chaque passage FAISS."""
    passages: list[dict[str, str | int]] = []
    for document, _distance in found:
        metadata = document.metadata
        passage: dict[str, str | int] = {
            "text": document.page_content,
            "filename": metadata["filename"],
            "page_number": metadata["page_number"],
            "chunk_id": metadata["chunk_id"],
        }
        document_id = metadata.get("document_id")
        if isinstance(document_id, str) and document_id.strip() != "":
            passage["document_id"] = document_id
        passages.append(passage)
    return passages


def _read_vector(embedding: object) -> list[float]:
    """Vérifie qu'un embedding est une liste de nombres."""
    if not isinstance(embedding, list) or len(embedding) == 0:
        raise TypeError("Chaque embedding doit être une liste de nombres.")

    vector: list[float] = []
    for value in embedding:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError("Un embedding ne doit contenir que des nombres.")
        vector.append(float(value))
    return vector


def _directory_to_save(directory: object) -> Path:
    """Refuse un chemin vide, un fichier, ou un dossier parent absent."""
    if not isinstance(directory, (str, Path)):
        raise TypeError("Le chemin de sauvegarde doit être un chemin.")
    if isinstance(directory, str) and directory.strip() == "":
        raise ValueError("Le chemin de sauvegarde est invalide.")

    path = Path(directory)
    if path.exists() and not path.is_dir():
        raise ValueError("Le chemin de sauvegarde est invalide.")
    if not path.parent.exists() or not path.parent.is_dir():
        raise ValueError("Le chemin de sauvegarde est invalide.")
    return path


def _directory_to_load(directory: object) -> Path:
    """Vérifie que le dossier de base existe."""
    if not isinstance(directory, (str, Path)):
        raise TypeError("Le chemin de chargement doit être un chemin.")

    path = Path(directory)
    if not path.exists():
        raise FileNotFoundError(f"Base introuvable : {path}")
    if not path.is_dir():
        raise ValueError("Le chemin de chargement est invalide.")
    return path


def _require_trusted_manifest(folder: Path) -> None:
    """N'accepte que les bases écrites par VectorStore.save."""
    manifest_path = folder / MANIFEST_NAME
    if not manifest_path.is_file():
        raise ValueError(
            "Chargement refusé : cette base ne provient pas d'une source fiable."
        )

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(
            "Chargement refusé : cette base ne provient pas d'une source fiable."
        ) from error

    if (
        not isinstance(manifest, dict)
        or manifest.get("creator") != _CREATOR
        or manifest.get("format_version") != _FORMAT_VERSION
    ):
        raise ValueError(
            "Chargement refusé : cette base ne provient pas d'une source fiable."
        )
