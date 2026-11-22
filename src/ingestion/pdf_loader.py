"""Extraction du texte des fichiers PDF.

Ce module lit un PDF page par page.
Il ne découpe pas encore le texte en chunks.
"""

from pathlib import Path

from pypdf import PdfReader
from pypdf.errors import PyPdfError


def load_pdf(file_path: str | Path) -> list[dict[str, str | int]]:
    """Extrait le texte de chaque page d'un PDF.

    Args:
        file_path: Chemin du fichier PDF.

    Returns:
        Une liste de pages. Chaque page est un dictionnaire :
        - text : texte extrait de la page
        - page_number : numéro de page, en commençant à 1
        - filename : nom du fichier, sans le dossier

    Raises:
        FileNotFoundError: le fichier n'existe pas.
        ValueError: le fichier n'est pas un PDF lisible,
            ou aucune page ne contient de texte.
    """
    path = Path(file_path)

    # Un chemin absent n'est pas la même erreur qu'un fichier illisible.
    if not path.exists():
        raise FileNotFoundError(f"Fichier introuvable : {path}")

    if not path.is_file():
        raise ValueError(f"Le chemin ne désigne pas un fichier : {path}")

    try:
        reader = PdfReader(path)
        pages: list[dict[str, str | int]] = []
        for index, page in enumerate(reader.pages, start=1):
            # extract_text() peut renvoyer None quand la page n'a pas de texte.
            extracted = page.extract_text() or ""
            pages.append(
                {
                    "text": extracted,
                    "page_number": index,
                    "filename": path.name,
                }
            )
    except PyPdfError as error:
        raise ValueError(f"Fichier PDF invalide : {path.name}") from error

    # Un PDF blanc, ou dont le texte ne peut pas être lu, n'est pas exploitable.
    if not any(str(page["text"]).strip() for page in pages):
        raise ValueError(f"Le PDF ne contient aucun texte extractible : {path.name}")

    return pages
