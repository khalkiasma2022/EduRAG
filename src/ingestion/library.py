"""Bibliothèque locale de documents PDF.

Un cours est copié, inscrit dans un catalogue JSON, puis ajouté à l'index.
Pour retirer un cours, l'index est d'abord reconstruit sans lui.
Le fichier n'est supprimé qu'ensuite.
"""

import hashlib
import json
import shutil
import uuid
from pathlib import Path

from src.ingestion.pdf_loader import load_pdf
from src.ingestion.text_splitter import split_documents

# Le catalogue reste à côté des PDF, dans le dossier de documents.
CATALOG_NAME = "catalog.json"


class DocumentLibrary:
    """Range les PDF et reconstruit l'index FAISS à partir de ceux qui restent."""

    def __init__(
        self,
        documents_directory: str | Path,
        index_directory: str | Path,
        embedding_model: object,
        vector_store: object,
    ) -> None:
        """Prépare les dossiers. Aucun modèle n'est chargé ici."""
        if not callable(getattr(embedding_model, "embed_chunks", None)):
            raise TypeError("embedding_model doit fournir embed_chunks.")
        if not callable(getattr(vector_store, "create", None)):
            raise TypeError("vector_store doit fournir create.")
        if not callable(getattr(vector_store, "save", None)):
            raise TypeError("vector_store doit fournir save.")

        self.documents_directory = Path(documents_directory).resolve()
        self.index_directory = Path(index_directory).resolve()
        if self.documents_directory == self.index_directory:
            raise ValueError("Les PDF et l'index doivent être dans deux dossiers différents.")

        self.embedding_model = embedding_model
        self.vector_store = vector_store
        self.documents_directory.mkdir(parents=True, exist_ok=True)
        self.index_directory.parent.mkdir(parents=True, exist_ok=True)

    def add_pdf(self, file_path: str | Path) -> dict[str, str]:
        """Copie un PDF valide et reconstruit l'index avec lui.

        Un même contenu, reconnu par son empreinte SHA-256, est refusé.
        Deux fichiers de même nom mais de contenu différent sont conservés.
        """
        source = _source_file(file_path)
        # On vérifie le PDF avant de toucher au catalogue ou à l'index.
        load_pdf(source)
        digest = _sha256(source)
        entries = self._read_catalog()
        if any(entry["sha256"] == digest for entry in entries):
            raise ValueError("Ce document est déjà dans la bibliothèque.")

        document_id = uuid.uuid4().hex
        destination = self.documents_directory / f"{document_id}.pdf"
        shutil.copyfile(source, destination)
        if _sha256(destination) != digest:
            destination.unlink(missing_ok=True)
            raise RuntimeError("La copie du PDF est incomplète.")

        entry = {
            "document_id": document_id,
            "filename": source.name,
            "stored_path": str(destination),
            "sha256": digest,
        }
        try:
            self._commit(entries + [entry])
        except Exception:
            destination.unlink(missing_ok=True)
            raise
        return dict(entry)

    def add_pdfs(self, file_paths: list[str | Path]) -> list[dict[str, str]]:
        """Ajoute chaque PDF. Un échec n'efface pas les documents déjà enregistrés."""
        if not isinstance(file_paths, list):
            raise TypeError("La liste de PDF doit être une liste.")
        if len(file_paths) == 0:
            raise ValueError("La liste de PDF ne peut pas être vide.")

        added: list[dict[str, str]] = []
        for file_path in file_paths:
            added.append(self.add_pdf(file_path))
        return added

    def list_documents(self) -> list[dict[str, str]]:
        """Retourne une copie du catalogue, du plus ancien au plus récent."""
        return self._read_catalog()

    def remove_document(self, document_id: str) -> None:
        """Retire un document sans effacer les autres.

        L'index des documents restants est enregistré avant le catalogue,
        et le fichier n'est supprimé qu'après ces deux écritures.
        """
        if not isinstance(document_id, str) or document_id.strip() == "":
            raise ValueError("L'identifiant du document est invalide.")

        entries = self._read_catalog()
        selected = [entry for entry in entries if entry["document_id"] == document_id]
        if not selected:
            raise ValueError("Document introuvable dans la bibliothèque.")

        remaining = [entry for entry in entries if entry["document_id"] != document_id]
        stored = Path(selected[0]["stored_path"])
        delete_path = stored if _is_inside(stored, self.documents_directory) else None
        self._commit(remaining, delete_path=delete_path)

    def rebuild_index(self) -> None:
        """Reconstruit l'index à partir des PDF encore inscrits au catalogue."""
        self._commit(self._read_catalog())

    def _commit(self, entries: list[dict[str, str]], delete_path: Path | None = None) -> None:
        """Publie un nouvel index, puis le catalogue, puis supprime l'ancien PDF."""
        backup = self._prepare_new_index(entries)
        try:
            self._write_catalog(entries)
        except Exception:
            self._restore_index_backup(backup)
            raise
        _discard_backup(backup)
        if delete_path is not None and delete_path.is_file():
            delete_path.unlink()

    def _prepare_new_index(self, entries: list[dict[str, str]]) -> Path | None:
        """Construit le nouvel index à côté de l'ancien, puis le met en place.

        L'ancien index est conservé sous un nom de sauvegarde jusqu'à la
        réussite du catalogue.
        """
        backup = self.index_directory.with_name(self.index_directory.name + "_sauvegarde")
        if backup.exists():
            shutil.rmtree(backup)

        if not entries:
            if self.index_directory.exists():
                self.index_directory.rename(backup)
            return backup if backup.exists() else None

        temporary = self.index_directory.with_name(self.index_directory.name + "_nouveau")
        if temporary.exists():
            shutil.rmtree(temporary)
        try:
            chunks = _chunks_for(entries)
            embeddings = self.embedding_model.embed_chunks(chunks)
            self.vector_store.create(chunks, embeddings)
            self.vector_store.save(temporary)
        except Exception:
            shutil.rmtree(temporary, ignore_errors=True)
            self._reload_previous_index()
            raise

        try:
            if self.index_directory.exists():
                self.index_directory.rename(backup)
            temporary.rename(self.index_directory)
        except Exception:
            shutil.rmtree(temporary, ignore_errors=True)
            if backup.exists() and not self.index_directory.exists():
                backup.rename(self.index_directory)
            self._reload_previous_index()
            raise
        return backup if backup.exists() else None

    def _restore_index_backup(self, backup: Path | None) -> None:
        """Remet l'index précédent si le catalogue n'a pas pu être écrit."""
        if self.index_directory.exists():
            shutil.rmtree(self.index_directory)
        if backup is not None and backup.exists():
            backup.rename(self.index_directory)
        self._reload_previous_index()

    def _reload_previous_index(self) -> None:
        """Recharge l'index du disque lorsque VectorStore sait le faire."""
        load = getattr(self.vector_store, "load", None)
        manifest = self.index_directory / "edurag_manifest.json"
        if not callable(load) or not manifest.is_file():
            return
        try:
            load(self.index_directory)
        except Exception:
            return

    def _read_catalog(self) -> list[dict[str, str]]:
        """Lit le catalogue. Un fichier absent signifie une bibliothèque vide."""
        path = self.documents_directory / CATALOG_NAME
        if not path.exists():
            return []
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError("Le catalogue de la bibliothèque est illisible.") from error
        if not isinstance(raw, list):
            raise ValueError("Le catalogue de la bibliothèque est illisible.")
        return [_check_entry(item) for item in raw]

    def _write_catalog(self, entries: list[dict[str, str]]) -> None:
        """Écrit le catalogue dans un fichier temporaire, puis le remplace."""
        path = self.documents_directory / CATALOG_NAME
        temporary = path.with_suffix(".json.tmp")
        content = json.dumps(entries, ensure_ascii=False, indent=2)
        temporary.write_text(content + "\n", encoding="utf-8")
        temporary.replace(path)


def _chunks_for(entries: list[dict[str, str]]) -> list[dict[str, str | int]]:
    """Relit les PDF du catalogue et ajoute l'identifiant à chaque chunk."""
    chunks: list[dict[str, str | int]] = []
    for entry in entries:
        pages = load_pdf(entry["stored_path"])
        for page in pages:
            # Le fichier stocké porte l'identifiant. On garde le nom d'origine.
            page["filename"] = entry["filename"]
        for chunk in split_documents(pages):
            chunk["document_id"] = entry["document_id"]
            chunks.append(chunk)

    for number, chunk in enumerate(chunks, start=1):
        chunk["chunk_id"] = number
    if not chunks:
        raise ValueError("Les documents de la bibliothèque ne contiennent aucun texte.")
    return chunks


def _source_file(file_path: object) -> Path:
    """Vérifie que le chemin reçu désigne un fichier."""
    if not isinstance(file_path, (str, Path)):
        raise TypeError("Le chemin du PDF doit être un chemin.")
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Fichier introuvable : {path}")
    if not path.is_file():
        raise ValueError(f"Le chemin ne désigne pas un fichier : {path}")
    return path


def _sha256(path: Path) -> str:
    """Calcule l'empreinte SHA-256 du fichier complet."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _check_entry(item: object) -> dict[str, str]:
    """Relit une ligne du catalogue."""
    if not isinstance(item, dict):
        raise ValueError("Le catalogue de la bibliothèque est illisible.")
    keys = ("document_id", "filename", "stored_path", "sha256")
    entry: dict[str, str] = {}
    for key in keys:
        value = item.get(key)
        if not isinstance(value, str) or value.strip() == "":
            raise ValueError("Le catalogue de la bibliothèque est illisible.")
        entry[key] = value
    return entry


def _is_inside(path: Path, root: Path) -> bool:
    """Indique si un chemin reste dans le dossier des documents."""
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def _discard_backup(backup: Path | None) -> None:
    """Efface la sauvegarde de l'ancien index une fois le catalogue écrit."""
    if backup is not None and backup.exists():
        shutil.rmtree(backup, ignore_errors=True)
