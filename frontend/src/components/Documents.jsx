import { useCallback, useEffect, useRef, useState } from "react";
import { ApiError } from "../api/client.js";
import { deleteDocument, listDocuments, uploadDocument } from "../api/documents.js";
import "./Documents.css";

/** Aligné sur DEFAULT_MAX_UPLOAD_BYTES du backend (20 Mio). */
const MAX_UPLOAD_BYTES = 20 * 1024 * 1024;
const ALLOWED_PDF_TYPES = new Set([
  "",
  "application/pdf",
  "application/octet-stream",
  "application/x-pdf",
]);

function formatBytes(bytes) {
  if (bytes < 1024) {
    return `${bytes} o`;
  }
  if (bytes < 1024 * 1024) {
    return `${(bytes / 1024).toFixed(1)} Kio`;
  }
  return `${(bytes / (1024 * 1024)).toFixed(1)} Mio`;
}

function apiErrorMessage(error) {
  if (error instanceof ApiError) {
    return error.message;
  }
  if (error instanceof Error && error.message) {
    return error.message;
  }
  return "Une erreur inattendue est survenue.";
}

function validatePdfFile(file) {
  if (!(file instanceof File)) {
    return "Aucun fichier sélectionné.";
  }

  const name = file.name.split(/[/\\]/).pop() || "";
  if (name === "" || name === "." || name === "..") {
    return "Le nom du fichier est invalide.";
  }
  if (!name.toLowerCase().endsWith(".pdf")) {
    return "Le fichier sélectionné doit être un PDF.";
  }

  const media = (file.type || "").split(";", 1)[0].trim().toLowerCase();
  if (!ALLOWED_PDF_TYPES.has(media)) {
    return "Le fichier sélectionné doit être un PDF.";
  }

  if (file.size <= 0) {
    return "Le fichier PDF est vide.";
  }
  if (file.size > MAX_UPLOAD_BYTES) {
    return `Le PDF dépasse la taille autorisée (${formatBytes(MAX_UPLOAD_BYTES)}).`;
  }

  return null;
}

function Documents() {
  const [documents, setDocuments] = useState([]);
  const [loading, setLoading] = useState(true);
  const [busy, setBusy] = useState(false);
  const [pendingDeleteId, setPendingDeleteId] = useState(null);
  const [error, setError] = useState("");
  const [success, setSuccess] = useState("");
  const [selectedName, setSelectedName] = useState("");
  const fileInputRef = useRef(null);

  const refreshDocuments = useCallback(async ({ quiet = false } = {}) => {
    if (!quiet) {
      setLoading(true);
    }
    setError("");

    try {
      const items = await listDocuments();
      setDocuments(Array.isArray(items) ? items : []);
    } catch (err) {
      setError(apiErrorMessage(err));
      throw err;
    } finally {
      if (!quiet) {
        setLoading(false);
      }
    }
  }, []);

  useEffect(() => {
    refreshDocuments().catch(() => {
      /* message déjà posé dans refreshDocuments */
    });
  }, [refreshDocuments]);

  async function handleUpload(event) {
    event.preventDefault();
    const input = fileInputRef.current;
    const file = input?.files?.[0];
    const validationError = validatePdfFile(file);
    if (validationError) {
      setSuccess("");
      setError(validationError);
      return;
    }

    setBusy(true);
    setError("");
    setSuccess("");

    try {
      const uploaded = await uploadDocument(file, file.name);
      await refreshDocuments({ quiet: true });
      setSuccess(`Document importé : ${uploaded.filename}`);
      setSelectedName("");
      if (input) {
        input.value = "";
      }
    } catch (err) {
      setError(apiErrorMessage(err));
    } finally {
      setBusy(false);
    }
  }

  async function handleDelete(documentId, filename) {
    const confirmed = window.confirm(
      `Supprimer le document « ${filename} » de la bibliothèque ?`,
    );
    if (!confirmed) {
      return;
    }

    setBusy(true);
    setPendingDeleteId(documentId);
    setError("");
    setSuccess("");

    try {
      await deleteDocument(documentId);
      await refreshDocuments({ quiet: true });
      setSuccess(`Document supprimé : ${filename}`);
    } catch (err) {
      setError(apiErrorMessage(err));
    } finally {
      setPendingDeleteId(null);
      setBusy(false);
    }
  }

  function handleFileChange(event) {
    const file = event.target.files?.[0];
    setSelectedName(file ? file.name : "");
    setError("");
    setSuccess("");
  }

  const actionsDisabled = loading || busy;

  return (
    <section className="documents" aria-labelledby="documents-title">
      <div className="documents__header">
        <p className="documents__eyebrow">Bibliothèque</p>
        <h1 id="documents-title" className="documents__title">
          Documents
        </h1>
        <p className="documents__lead">
          Importez vos cours PDF et gérez la bibliothèque utilisée par EduRAG.
          Taille maximale : {formatBytes(MAX_UPLOAD_BYTES)}.
        </p>
      </div>

      <form className="documents__upload" onSubmit={handleUpload}>
        <div className="documents__upload-row">
          <label className="documents__file-label">
            <span className="documents__file-button">Choisir un PDF</span>
            <input
              ref={fileInputRef}
              type="file"
              accept=".pdf,application/pdf"
              onChange={handleFileChange}
              disabled={actionsDisabled}
            />
          </label>
          <button
            type="submit"
            className="documents__primary"
            disabled={actionsDisabled || selectedName === ""}
          >
            {busy && pendingDeleteId === null ? "Import…" : "Importer"}
          </button>
          <button
            type="button"
            className="documents__secondary"
            onClick={() => {
              refreshDocuments().catch(() => {});
            }}
            disabled={actionsDisabled}
          >
            Actualiser
          </button>
        </div>
        <p className="documents__selected">
          {selectedName ? `Fichier sélectionné : ${selectedName}` : "Aucun fichier sélectionné."}
        </p>
      </form>

      {error ? (
        <p className="documents__message documents__message--error" role="alert">
          {error}
        </p>
      ) : null}
      {success ? (
        <p className="documents__message documents__message--success" role="status">
          {success}
        </p>
      ) : null}

      <div className="documents__list-panel">
        <div className="documents__list-header">
          <h2>Cours enregistrés</h2>
          <p>
            {loading
              ? "Chargement…"
              : `${documents.length} document${documents.length > 1 ? "s" : ""}`}
          </p>
        </div>

        {loading ? (
          <p className="documents__empty">Chargement de la bibliothèque…</p>
        ) : documents.length === 0 ? (
          <p className="documents__empty">
            Aucun document pour le moment. Importez un PDF pour commencer.
          </p>
        ) : (
          <ul className="documents__list">
            {documents.map((doc) => {
              const isDeleting = pendingDeleteId === doc.document_id;

              return (
                <li key={doc.document_id} className="documents__item">
                  <div className="documents__item-info">
                    <p className="documents__item-name">{doc.filename}</p>
                    <p className="documents__item-id">{doc.document_id}</p>
                  </div>
                  <button
                    type="button"
                    className="documents__danger"
                    onClick={() => handleDelete(doc.document_id, doc.filename)}
                    disabled={actionsDisabled}
                  >
                    {isDeleting ? "Suppression…" : "Supprimer"}
                  </button>
                </li>
              );
            })}
          </ul>
        )}
      </div>
    </section>
  );
}

export default Documents;
