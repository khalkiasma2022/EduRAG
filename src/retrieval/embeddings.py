"""Transformation des chunks et des questions en vecteurs.

Le modèle est chargé une seule fois sur GPU CUDA.
device='cpu' reste autorisé uniquement pour les tests unitaires.
"""

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

# Modèle multilingue léger, adapté à des cours en français.
DEFAULT_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


class EmbeddingModel:
    """Calcule les embeddings des morceaux de cours et des questions."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        device: str | None = None,
    ) -> None:
        """Charge le modèle une fois sur CUDA (sauf device explicite pour les tests)."""
        if not isinstance(model_name, str) or model_name.strip() == "":
            raise ValueError("Le nom du modèle ne peut pas être vide.")

        self.model_name = model_name
        self.device = _select_device(device)
        # Le modèle reste en mémoire : les appels suivants ne le rechargent pas.
        self.model = SentenceTransformer(self.model_name, device=self.device)

    @property
    def dimension(self) -> int:
        """Nombre de valeurs contenues dans chaque vecteur."""
        size = self.model.get_sentence_embedding_dimension()
        if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
            raise ValueError("Le modèle ne fournit pas une dimension valide.")
        return size

    def embed_chunks(self, chunks: list[dict[str, str | int]]) -> list[list[float]]:
        """Transforme les chunks de split_documents en vecteurs.

        L'ordre des vecteurs suit l'ordre des chunks.
        """
        texts = _chunk_texts(chunks)
        vectors = self._encode(texts)
        self._check_same_dimension(vectors)
        return vectors

    def embed_query(self, question: str) -> list[float]:
        """Transforme une question en un seul vecteur."""
        if not isinstance(question, str):
            raise TypeError("La question doit être une chaîne de caractères.")
        if question.strip() == "":
            raise ValueError("La question ne peut pas être vide.")

        encoded = self.model.encode(question, convert_to_numpy=True)
        vector = np.asarray(encoded, dtype=float).reshape(-1).tolist()
        if len(vector) != self.dimension:
            raise ValueError("La dimension du vecteur ne correspond pas au modèle.")
        return vector

    def _encode(self, texts: list[str]) -> list[list[float]]:
        """Encode un lot de textes avec le modèle déjà chargé."""
        encoded = self.model.encode(texts, convert_to_numpy=True)
        array = np.asarray(encoded, dtype=float)
        if array.ndim == 1:
            array = array.reshape(1, -1)
        if array.ndim != 2:
            raise ValueError("Le modèle a renvoyé un format de vecteurs inattendu.")
        return array.tolist()

    def _check_same_dimension(self, vectors: list[list[float]]) -> None:
        """Vérifie que chaque vecteur a la taille annoncée par le modèle."""
        expected = self.dimension
        if any(len(vector) != expected for vector in vectors):
            raise ValueError("Les vecteurs n'ont pas tous la dimension du modèle.")


def _select_device(device: str | None) -> str:
    """Exige CUDA par défaut. device='cpu' reste autorisé pour les tests unitaires."""
    if device is None:
        device = "cuda"
    if not isinstance(device, str) or device not in {"cpu", "cuda"}:
        raise ValueError("Le processeur demandé doit être 'cpu' ou 'cuda'.")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA est requis pour EduRAG, mais aucun GPU n'est disponible. "
            "Installez un PyTorch CUDA (pas le build CPU)."
        )
    return device


def _chunk_texts(chunks: object) -> list[str]:
    """Extrait le texte des chunks et refuse les entrées vides ou mal formées."""
    if not isinstance(chunks, list):
        raise TypeError("chunks doit être une liste de dictionnaires.")
    if len(chunks) == 0:
        raise ValueError("La liste de chunks ne peut pas être vide.")

    texts: list[str] = []
    for chunk in chunks:
        if not isinstance(chunk, dict):
            raise TypeError("Chaque chunk doit être un dictionnaire.")
        if "text" not in chunk:
            raise ValueError("Champ text manquant dans un chunk.")
        text = chunk["text"]
        if not isinstance(text, str):
            raise TypeError("Le champ text doit être une chaîne de caractères.")
        if text.strip() == "":
            raise ValueError("Un chunk ne peut pas avoir un texte vide.")
        texts.append(text)
    return texts
