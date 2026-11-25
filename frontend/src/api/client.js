const DEFAULT_API_BASE_URL = "http://localhost:8000";

/**
 * URL de base de l'API FastAPI.
 * Priorité : VITE_API_BASE_URL, sinon localhost:8000.
 */
export function getApiBaseUrl() {
  const configured = import.meta.env.VITE_API_BASE_URL;
  if (typeof configured === "string" && configured.trim() !== "") {
    return configured.replace(/\/+$/, "");
  }
  return DEFAULT_API_BASE_URL;
}

/**
 * Erreur HTTP normalisée pour le frontend.
 * Le message privilégie le champ FastAPI `detail` lorsqu'il est présent.
 */
export class ApiError extends Error {
  /**
   * @param {string} message
   * @param {number} status
   * @param {unknown} [detail]
   */
  constructor(message, status, detail) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.detail = detail;
  }
}

/**
 * Transforme le champ FastAPI `detail` en message lisible.
 * @param {unknown} detail
 * @param {number} status
 * @returns {string}
 */
export function formatApiDetail(detail, status) {
  if (typeof detail === "string" && detail.trim() !== "") {
    return detail;
  }

  if (Array.isArray(detail) && detail.length > 0) {
    const parts = detail.map((item) => {
      if (typeof item === "string") {
        return item;
      }
      if (item && typeof item === "object") {
        if (typeof item.msg === "string" && item.msg.trim() !== "") {
          return item.msg;
        }
        if (typeof item.message === "string" && item.message.trim() !== "") {
          return item.message;
        }
      }
      return null;
    }).filter(Boolean);

    if (parts.length > 0) {
      return parts.join(" ; ");
    }
  }

  if (detail && typeof detail === "object" && typeof detail.message === "string") {
    return detail.message;
  }

  return `Erreur HTTP ${status}`;
}

/**
 * Extrait un nom de fichier depuis Content-Disposition, sans chemin local.
 * @param {string | null} header
 * @returns {string | null}
 */
export function parseContentDispositionFilename(header) {
  if (typeof header !== "string" || header.trim() === "") {
    return null;
  }

  const utf8Match = /filename\*=UTF-8''([^;]+)/i.exec(header);
  if (utf8Match?.[1]) {
    try {
      return decodeURIComponent(utf8Match[1].trim().replace(/^"|"$/g, ""));
    } catch {
      return utf8Match[1].trim().replace(/^"|"$/g, "");
    }
  }

  const plainMatch = /filename="([^"]+)"/i.exec(header) || /filename=([^;]+)/i.exec(header);
  if (!plainMatch?.[1]) {
    return null;
  }

  return plainMatch[1].trim().replace(/^"|"$/g, "");
}

/**
 * @param {Response} response
 * @returns {Promise<unknown>}
 */
async function readErrorDetail(response) {
  const contentType = response.headers.get("Content-Type") || "";

  if (contentType.includes("application/json")) {
    try {
      const payload = await response.json();
      if (payload && typeof payload === "object" && "detail" in payload) {
        return payload.detail;
      }
      return payload;
    } catch {
      return null;
    }
  }

  try {
    const text = await response.text();
    return text.trim() === "" ? null : text;
  } catch {
    return null;
  }
}

/**
 * @param {Response} response
 * @returns {Promise<never>}
 */
async function throwApiError(response) {
  const detail = await readErrorDetail(response);
  throw new ApiError(formatApiDetail(detail, response.status), response.status, detail);
}

/**
 * Construit l'URL absolue d'un chemin API.
 * @param {string} path
 * @returns {string}
 */
export function buildApiUrl(path) {
  const normalized = path.startsWith("/") ? path : `/${path}`;
  return `${getApiBaseUrl()}${normalized}`;
}

/**
 * Client HTTP centralisé basé sur fetch.
 *
 * @param {string} path
 * @param {RequestInit & {
 *   json?: unknown,
 *   formData?: FormData,
 *   responseType?: "json" | "blob" | "void",
 * }} [options]
 * @returns {Promise<any>}
 */
export async function apiRequest(path, options = {}) {
  const {
    json,
    formData,
    responseType = "json",
    headers: customHeaders,
    body: explicitBody,
    ...rest
  } = options;

  /** @type {HeadersInit} */
  const headers = { ...(customHeaders || {}) };

  let body = explicitBody;

  if (formData instanceof FormData) {
    body = formData;
    // Ne pas fixer Content-Type : le navigateur ajoute la boundary multipart.
  } else if (json !== undefined) {
    body = JSON.stringify(json);
    if (!hasHeader(headers, "Content-Type")) {
      headers["Content-Type"] = "application/json";
    }
  }

  const response = await fetch(buildApiUrl(path), {
    ...rest,
    headers,
    body,
  });

  if (!response.ok) {
    await throwApiError(response);
  }

  if (response.status === 204 || responseType === "void") {
    return null;
  }

  if (responseType === "blob") {
    const blob = await response.blob();
    const filename = parseContentDispositionFilename(
      response.headers.get("Content-Disposition"),
    );
    return { blob, filename, contentType: response.headers.get("Content-Type") };
  }

  const contentType = response.headers.get("Content-Type") || "";
  if (!contentType.includes("application/json")) {
    const text = await response.text();
    return text === "" ? null : text;
  }

  return response.json();
}

/**
 * @param {HeadersInit} headers
 * @param {string} name
 * @returns {boolean}
 */
function hasHeader(headers, name) {
  const needle = name.toLowerCase();

  if (typeof Headers !== "undefined" && headers instanceof Headers) {
    return headers.has(name);
  }

  if (Array.isArray(headers)) {
    return headers.some(([key]) => String(key).toLowerCase() === needle);
  }

  return Object.keys(headers).some((key) => key.toLowerCase() === needle);
}
