import { apiRequest } from "./client.js";

/**
 * Génère et enregistre une fiche de révision.
 * @param {{
 *   document_id: string,
 *   language?: string,
 *   level?: string,
 *   sheet_type?: string,
 *   page_start?: number | null,
 *   page_end?: number | null,
 * }} payload
 * @returns {Promise<object>}
 */
export function createRevision(payload) {
  return apiRequest("/revisions", {
    method: "POST",
    json: {
      document_id: payload.document_id,
      language: payload.language ?? "fr",
      level: payload.level ?? "beginner",
      sheet_type: payload.sheet_type ?? "complete",
      page_start: payload.page_start ?? null,
      page_end: payload.page_end ?? null,
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
