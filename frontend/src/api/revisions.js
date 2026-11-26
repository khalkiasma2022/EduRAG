import { apiRequest } from "./client.js";

/**
 * Génère et enregistre une fiche de révision.
 * Les nouveaux clients n'envoient que document_id et language.
 * @param {{
 *   document_id: string,
 *   language?: string,
 * }} payload
 * @returns {Promise<object>}
 */
export function createRevision(payload) {
  return apiRequest("/revisions", {
    method: "POST",
    json: {
      document_id: payload.document_id,
      language: payload.language ?? "fr",
    },
  });
}

/**
 * Liste les fiches enregistrées.
 * @returns {Promise<object[]>}
 */
export function listRevisions() {
  return apiRequest("/revisions", {
    method: "GET",
  });
}

/**
 * Relit une fiche enregistrée.
 * @param {number|string} revisionId
 * @returns {Promise<object>}
 */
export function getRevision(revisionId) {
  return apiRequest(`/revisions/${encodeURIComponent(String(revisionId))}`, {
    method: "GET",
  });
}

/**
 * Télécharge une fiche au format Markdown (Blob).
 * @param {number|string} revisionId
 * @returns {Promise<{ blob: Blob, filename: string | null, contentType: string | null }>}
 */
export function downloadRevisionMarkdown(revisionId) {
  return apiRequest(
    `/revisions/${encodeURIComponent(String(revisionId))}/markdown`,
    {
      method: "GET",
      responseType: "blob",
    },
  );
}

/**
 * Télécharge une fiche au format PDF (Blob).
 * @param {number|string} revisionId
 * @returns {Promise<{ blob: Blob, filename: string | null, contentType: string | null }>}
 */
export function downloadRevisionPdf(revisionId) {
  return apiRequest(
    `/revisions/${encodeURIComponent(String(revisionId))}/pdf`,
    {
      method: "GET",
      responseType: "blob",
    },
  );
}
