import { apiRequest } from "./client.js";

/**
 * Liste les cours PDF enregistrés.
 * @returns {Promise<Array<{ document_id: string, filename: string }>>}
 */
export function listDocuments() {
  return apiRequest("/documents", {
    method: "GET",
  });
}

/**
 * Importe un cours PDF (multipart, champ `file`).
 * @param {File | Blob} file
 * @param {string} [filename]
 * @returns {Promise<{ document_id: string, filename: string }>}
 */
export function uploadDocument(file, filename) {
  const formData = new FormData();
  if (filename) {
    formData.append("file", file, filename);
  } else {
    formData.append("file", file);
  }

  return apiRequest("/documents", {
    method: "POST",
    formData,
  });
}

/**
 * Retire un cours par son identifiant.
 * @param {string} documentId
 * @returns {Promise<null>}
 */
export function deleteDocument(documentId) {
  return apiRequest(`/documents/${encodeURIComponent(documentId)}`, {
    method: "DELETE",
    responseType: "void",
  });
}
