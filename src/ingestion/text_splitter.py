"""Découpage du texte des pages en chunks.

Chaque entrée suit le format de load_pdf :
text, page_number et filename.
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter


def split_documents(
    documents: list[dict[str, str | int]],
    chunk_size: int = 500,
    chunk_overlap: int = 100,
) -> list[dict[str, str | int]]:
    """Découpe le texte de chaque page en morceaux.

    Args:
        documents: Pages produites par load_pdf.
        chunk_size: Taille maximale visée d'un chunk, en caractères.
        chunk_overlap: Nombre de caractères répétés entre deux chunks voisins.

    Returns:
        Une liste de chunks. Chaque chunk contient :
        - text : le morceau de texte
        - filename : le nom du PDF d'origine
        - page_number : le numéro de la page d'origine
        - chunk_id : l'identifiant du morceau, à partir de 1

    Raises:
        TypeError: la liste ou un champ n'a pas le type attendu.
        ValueError: une page est incomplète, ou les tailles sont incohérentes.
    """
    if not isinstance(documents, list):
        raise TypeError("documents doit être une liste de pages.")

    _check_chunk_parameters(chunk_size, chunk_overlap)

    # Le découpeur essaie d'abord les paragraphes, puis les lignes, les mots et les caractères.
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    chunks: list[dict[str, str | int]] = []
    for document in documents:
        page = _read_page(document)
        # Une page blanche n'apporte rien à la recherche : on la laisse de côté.
        if page["text"].strip() == "":
            continue

        for piece in splitter.split_text(page["text"]):
            if piece.strip() == "":
                continue
            chunks.append(
                {
                    "text": piece,
                    "filename": page["filename"],
                    "page_number": page["page_number"],
                    "chunk_id": len(chunks) + 1,
                }
            )

    return chunks


def _check_chunk_parameters(chunk_size: int, chunk_overlap: int) -> None:
    """Vérifie que la taille et le chevauchement peuvent être utilisés ensemble."""
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size <= 0:
        raise ValueError("chunk_size doit être un entier positif.")

    if isinstance(chunk_overlap, bool) or not isinstance(chunk_overlap, int) or chunk_overlap < 0:
        raise ValueError("chunk_overlap doit être un entier positif ou nul.")

    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap doit être plus petit que chunk_size.")


def _read_page(document: object) -> dict[str, str | int]:
    """Relit une page et signale clairement une entrée inutilisable."""
    if not isinstance(document, dict):
        raise TypeError("Chaque page doit être un dictionnaire.")

    required_keys = ("text", "page_number", "filename")
    missing = [key for key in required_keys if key not in document]
    if missing:
        raise ValueError(f"Page incomplète, champ manquant : {', '.join(missing)}.")

    text = document["text"]
    page_number = document["page_number"]
    filename = document["filename"]

    if not isinstance(text, str):
        raise TypeError("Le champ text doit être une chaîne de caractères.")

    # bool est un int en Python, mais un numéro de page ne peut pas être un booléen.
    if isinstance(page_number, bool) or not isinstance(page_number, int):
        raise TypeError("Le champ page_number doit être un entier.")

    if not isinstance(filename, str):
        raise TypeError("Le champ filename doit être une chaîne de caractères.")

    if filename.strip() == "":
        raise ValueError("Le champ filename ne peut pas être vide.")

    return {"text": text, "page_number": page_number, "filename": filename}
