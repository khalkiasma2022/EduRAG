import { useCallback, useEffect, useState } from "react";
import { ApiError } from "../api/client.js";
import { listDocuments } from "../api/documents.js";
import {
  createRevision,
  downloadRevisionMarkdown,
  downloadRevisionPdf,
  getRevision,
  listRevisions,
} from "../api/revisions.js";
import "./Revision.css";

const LANGUAGES = [
  { value: "fr", label: "Français" },
  { value: "en", label: "English" },
  { value: "ar", label: "العربية" },
];

const LEVELS = [
  { value: "beginner", label: "Débutant" },
  { value: "intermediate", label: "Intermédiaire" },
  { value: "advanced", label: "Avancé" },
];

const SHEET_TYPES = [
  { value: "short", label: "Fiche courte" },
  { value: "complete", label: "Fiche complète" },
];

const LANGUAGE_LABELS = Object.fromEntries(LANGUAGES.map((item) => [item.value, item.label]));
const LEVEL_LABELS = Object.fromEntries(LEVELS.map((item) => [item.value, item.label]));
const TYPE_LABELS = Object.fromEntries(SHEET_TYPES.map((item) => [item.value, item.label]));

function apiErrorMessage(error) {
  if (error instanceof ApiError) {
    return error.message;
  }
  if (error instanceof Error && error.message) {
    return error.message;
  }
  return "Une erreur inattendue est survenue.";
}

function formatWhen(value) {
  if (!value) {
    return "";
  }
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) {
    return String(value);
  }
  return date.toLocaleString("fr-FR", {
    dateStyle: "short",
    timeStyle: "short",
  });
}

function labelOf(map, code) {
  return map[code] || code;
}

async function triggerBrowserDownload(fetcher, revisionId) {
  const { blob, filename } = await fetcher(revisionId);
  const objectUrl = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = objectUrl;
  link.download = filename || `fiche-${revisionId}`;
  document.body.appendChild(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(objectUrl);
}

function Revision() {
  const [documents, setDocuments] = useState([]);
  const [revisions, setRevisions] = useState([]);
  const [selectedDocumentId, setSelectedDocumentId] = useState("");
  const [language, setLanguage] = useState("fr");
  const [level, setLevel] = useState("beginner");
  const [sheetType, setSheetType] = useState("complete");
  const [wholeDocument, setWholeDocument] = useState(true);
  const [pageStart, setPageStart] = useState("1");
  const [pageEnd, setPageEnd] = useState("1");
  const [activeSheet, setActiveSheet] = useState(null);
  const [loadingDocs, setLoadingDocs] = useState(true);
  const [loadingHistory, setLoadingHistory] = useState(true);
  const [generating, setGenerating] = useState(false);
  const [openingId, setOpeningId] = useState(null);
  const [exporting, setExporting] = useState("");
  const [error, setError] = useState("");
  const [success, setSuccess] = useState("");

  const formLocked = generating || openingId !== null || exporting !== "";

  const refreshDocuments = useCallback(async () => {
    setLoadingDocs(true);
    try {
      const items = await listDocuments();
      const list = Array.isArray(items) ? items : [];
      setDocuments(list);
      setSelectedDocumentId((current) => {
        if (current && list.some((doc) => doc.document_id === current)) {
          return current;
        }
        return list[0]?.document_id || "";
      });
    } catch (err) {
      setError(apiErrorMessage(err));
    } finally {
      setLoadingDocs(false);
    }
  }, []);

  const refreshRevisions = useCallback(async ({ quiet = false } = {}) => {
    if (!quiet) {
      setLoadingHistory(true);
    }
    try {
      const items = await listRevisions();
      setRevisions(Array.isArray(items) ? items : []);
    } catch (err) {
      setError(apiErrorMessage(err));
    } finally {
      if (!quiet) {
        setLoadingHistory(false);
      }
    }
  }, []);

  useEffect(() => {
    refreshDocuments();
    refreshRevisions();
  }, [refreshDocuments, refreshRevisions]);

  function validateForm() {
    if (!selectedDocumentId) {
      return "Sélectionnez un document PDF.";
    }
    if (wholeDocument) {
      return null;
    }
    const start = Number(pageStart);
    const end = Number(pageEnd);
    if (!Number.isInteger(start) || !Number.isInteger(end)) {
      return "L'intervalle de pages est invalide.";
    }
    if (start < 1 || end < start) {
      return "L'intervalle de pages est invalide.";
    }
    return null;
  }

  async function handleGenerate(event) {
    event.preventDefault();
    const validationError = validateForm();
    if (validationError) {
      setSuccess("");
      setError(validationError);
      return;
    }

    setGenerating(true);
    setError("");
    setSuccess("");

    try {
      /** @type {{ document_id: string, language: string, level: string, sheet_type: string, page_start?: number, page_end?: number }} */
      const payload = {
        document_id: selectedDocumentId,
        language,
        level,
        sheet_type: sheetType,
      };

      if (!wholeDocument) {
        payload.page_start = Number(pageStart);
        payload.page_end = Number(pageEnd);
      }

      const created = await createRevision(payload);
      setActiveSheet(created);
      setSuccess("Fiche générée et enregistrée.");
      await refreshRevisions({ quiet: true });
    } catch (err) {
      setError(apiErrorMessage(err));
    } finally {
      setGenerating(false);
    }
  }

  async function handleOpenRevision(revisionId) {
    if (openingId !== null || generating) {
      return;
    }
    setOpeningId(revisionId);
    setError("");
    setSuccess("");
    try {
      const sheet = await getRevision(revisionId);
      setActiveSheet(sheet);
    } catch (err) {
      setError(apiErrorMessage(err));
    } finally {
      setOpeningId(null);
    }
  }

  async function handleExport(kind) {
    if (!activeSheet?.id) {
      return;
    }
    setExporting(kind);
    setError("");
    try {
      if (kind === "markdown") {
        await triggerBrowserDownload(downloadRevisionMarkdown, activeSheet.id);
      } else {
        await triggerBrowserDownload(downloadRevisionPdf, activeSheet.id);
      }
      setSuccess(
        kind === "markdown"
          ? "Téléchargement Markdown démarré."
          : "Téléchargement PDF démarré.",
      );
    } catch (err) {
      setError(apiErrorMessage(err));
    } finally {
      setExporting("");
    }
  }

  const contentDir = activeSheet?.language === "ar" ? "rtl" : "ltr";

  return (
    <section className="revision" aria-labelledby="revision-title">
      <div className="revision__header">
        <p className="revision__eyebrow">Fiches</p>
        <h1 id="revision-title" className="revision__title">
          Smart Revision
        </h1>
        <p className="revision__lead">
          Générez une fiche de révision à partir d’un cours PDF, puis exportez-la
          en Markdown ou en PDF sans régénérer le contenu.
        </p>
      </div>

      {error ? (
        <p className="revision__banner revision__banner--error" role="alert">
          {error}
        </p>
      ) : null}
      {success ? (
        <p className="revision__banner revision__banner--success" role="status">
          {success}
        </p>
      ) : null}

      <div className="revision__layout">
        <form className="revision__form" onSubmit={handleGenerate}>
          <h2>Nouvelle fiche</h2>

          <label className="revision__field">
            <span>Document PDF</span>
            <select
              value={selectedDocumentId}
              onChange={(event) => setSelectedDocumentId(event.target.value)}
              disabled={formLocked || loadingDocs || documents.length === 0}
              required
            >
              {documents.length === 0 ? (
                <option value="">Aucun document disponible</option>
              ) : (
                documents.map((doc) => (
                  <option key={doc.document_id} value={doc.document_id}>
                    {doc.filename}
                  </option>
                ))
              )}
            </select>
          </label>

          <fieldset className="revision__fieldset" disabled={formLocked}>
            <legend>Langue</legend>
            <div className="revision__options">
              {LANGUAGES.map((item) => (
                <label key={item.value} className="revision__option">
                  <input
                    type="radio"
                    name="revision-language"
                    value={item.value}
                    checked={language === item.value}
                    onChange={() => setLanguage(item.value)}
                  />
                  <span>{item.label}</span>
                </label>
              ))}
            </div>
          </fieldset>

          <fieldset className="revision__fieldset" disabled={formLocked}>
            <legend>Niveau</legend>
            <div className="revision__options">
              {LEVELS.map((item) => (
                <label key={item.value} className="revision__option">
                  <input
                    type="radio"
                    name="revision-level"
                    value={item.value}
                    checked={level === item.value}
                    onChange={() => setLevel(item.value)}
                  />
                  <span>{item.label}</span>
                </label>
              ))}
            </div>
          </fieldset>

          <fieldset className="revision__fieldset" disabled={formLocked}>
            <legend>Format</legend>
            <div className="revision__options">
              {SHEET_TYPES.map((item) => (
                <label key={item.value} className="revision__option">
                  <input
                    type="radio"
                    name="revision-type"
                    value={item.value}
                    checked={sheetType === item.value}
                    onChange={() => setSheetType(item.value)}
                  />
                  <span>{item.label}</span>
                </label>
              ))}
            </div>
          </fieldset>

          <fieldset className="revision__fieldset" disabled={formLocked}>
            <legend>Pages</legend>
            <label className="revision__option">
              <input
                type="radio"
                name="revision-pages"
                checked={wholeDocument}
                onChange={() => setWholeDocument(true)}
              />
              <span>Tout le document</span>
            </label>
            <label className="revision__option">
              <input
                type="radio"
                name="revision-pages"
                checked={!wholeDocument}
                onChange={() => setWholeDocument(false)}
              />
              <span>Plage de pages</span>
            </label>

            {!wholeDocument ? (
              <div className="revision__page-range">
                <label className="revision__field">
                  <span>Page de début</span>
                  <input
                    type="number"
                    min={1}
                    step={1}
                    value={pageStart}
                    onChange={(event) => setPageStart(event.target.value)}
                    disabled={formLocked}
                    required
                  />
                </label>
                <label className="revision__field">
                  <span>Page de fin</span>
                  <input
                    type="number"
                    min={1}
                    step={1}
                    value={pageEnd}
                    onChange={(event) => setPageEnd(event.target.value)}
                    disabled={formLocked}
                    required
                  />
                </label>
              </div>
            ) : null}
          </fieldset>

          <button
            type="submit"
            className="revision__primary"
            disabled={formLocked || loadingDocs || documents.length === 0}
          >
            {generating ? "Génération en cours…" : "Générer ma fiche"}
          </button>

          {generating ? (
            <p className="revision__hint" role="status">
              Qwen prépare la fiche. Cette étape peut prendre un moment sur le GPU.
            </p>
          ) : null}
        </form>

        <aside className="revision__history" aria-label="Historique des fiches">
          <div className="revision__history-header">
            <h2>Historique</h2>
            <button
              type="button"
              className="revision__secondary"
              onClick={() => refreshRevisions()}
              disabled={formLocked || loadingHistory}
            >
              Actualiser
            </button>
          </div>

          {loadingHistory ? (
            <p className="revision__hint">Chargement des fiches…</p>
          ) : revisions.length === 0 ? (
            <p className="revision__hint">Aucune fiche enregistrée pour le moment.</p>
          ) : (
            <ul className="revision__history-list">
              {revisions.map((item) => {
                const isActive = activeSheet?.id === item.id;
                const isOpening = openingId === item.id;

                return (
                  <li key={item.id}>
                    <button
                      type="button"
                      className={`revision__history-item${isActive ? " revision__history-item--active" : ""}`}
                      onClick={() => handleOpenRevision(item.id)}
                      disabled={formLocked}
                    >
                      <span className="revision__history-name">{item.filename}</span>
                      <span className="revision__history-meta">
                        {formatWhen(item.created_at)}
                        {" · "}
                        {labelOf(LANGUAGE_LABELS, item.language)}
                        {" · "}
                        {labelOf(TYPE_LABELS, item.sheet_type)}
                      </span>
                      {isOpening ? (
                        <span className="revision__history-status">Ouverture…</span>
                      ) : null}
                    </button>
                  </li>
                );
              })}
            </ul>
          )}
        </aside>
      </div>

      <section className="revision__sheet" aria-label="Fiche sélectionnée">
        <div className="revision__sheet-header">
          <h2>Fiche affichée</h2>
          {activeSheet ? (
            <div className="revision__exports">
              <button
                type="button"
                className="revision__secondary"
                onClick={() => handleExport("markdown")}
                disabled={formLocked}
              >
                {exporting === "markdown" ? "Markdown…" : "Télécharger Markdown"}
              </button>
              <button
                type="button"
                className="revision__secondary"
                onClick={() => handleExport("pdf")}
                disabled={formLocked}
              >
                {exporting === "pdf" ? "PDF…" : "Télécharger PDF"}
              </button>
            </div>
          ) : null}
        </div>

        {!activeSheet ? (
          <p className="revision__hint">
            Générez une fiche ou ouvrez-en une depuis l’historique. L’ouverture ne
            relance jamais le modèle.
          </p>
        ) : (
          <>
            <dl className="revision__meta">
              <div>
                <dt>Document</dt>
                <dd>{activeSheet.filename}</dd>
              </div>
              <div>
                <dt>Pages</dt>
                <dd>
                  {activeSheet.page_start === activeSheet.page_end
                    ? `page ${activeSheet.page_start}`
                    : `pages ${activeSheet.page_start}–${activeSheet.page_end}`}
                </dd>
              </div>
              <div>
                <dt>Langue</dt>
                <dd>{labelOf(LANGUAGE_LABELS, activeSheet.language)}</dd>
              </div>
              <div>
                <dt>Niveau</dt>
                <dd>{labelOf(LEVEL_LABELS, activeSheet.level)}</dd>
              </div>
              <div>
                <dt>Format</dt>
                <dd>{labelOf(TYPE_LABELS, activeSheet.sheet_type)}</dd>
              </div>
              <div>
                <dt>Créée le</dt>
                <dd>{formatWhen(activeSheet.created_at)}</dd>
              </div>
            </dl>

            <pre
              className="revision__content"
              dir={contentDir}
              lang={activeSheet.language}
            >
              {activeSheet.content}
            </pre>
          </>
        )}
      </section>
    </section>
  );
}

export default Revision;
