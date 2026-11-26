/**
 * Helpers d'affichage des sources PDF / Web du chatbot.
 * Pas de dépendances externes : testable avec `node --test`.
 */

/**
 * @param {unknown} value
 * @returns {boolean}
 */
export function isSafeHttpUrl(value) {
  if (typeof value !== "string") {
    return false;
  }
  const trimmed = value.trim();
  if (!trimmed) {
    return false;
  }
  try {
    const parsed = new URL(trimmed);
    return parsed.protocol === "http:" || parsed.protocol === "https:";
  } catch {
    return false;
  }
}

/**
 * @param {string} url
 * @returns {string}
 */
export function hostnameFromUrl(url) {
  try {
    return new URL(url).hostname.replace(/^www\./, "");
  } catch {
    return url;
  }
}

/**
 * @param {unknown} source
 * @returns {boolean}
 */
export function isWebSource(source) {
  if (!source || typeof source !== "object") {
    return false;
  }
  const item = /** @type {Record<string, unknown>} */ (source);
  if (item.type === "web") {
    return true;
  }
  return (
    typeof item.url === "string"
    && item.filename === undefined
    && item.page_number === undefined
  );
}

/**
 * @param {unknown} source
 * @returns {boolean}
 */
export function isPdfSource(source) {
  if (!source || typeof source !== "object") {
    return false;
  }
  const item = /** @type {Record<string, unknown>} */ (source);
  return (
    typeof item.filename === "string"
    && item.filename.trim() !== ""
    && typeof item.page_number === "number"
  );
}

/**
 * @param {unknown} sources
 * @returns {{ pdf: object[], web: object[] }}
 */
export function partitionSources(sources) {
  const pdf = [];
  const web = [];
  if (!Array.isArray(sources)) {
    return { pdf, web };
  }
  for (const source of sources) {
    if (isWebSource(source) && isSafeHttpUrl(source.url)) {
      web.push(source);
      continue;
    }
    if (isPdfSource(source)) {
      pdf.push(source);
    }
  }
  return { pdf, web };
}

/**
 * Tronque un extrait texte (jamais interprété comme HTML côté React).
 * @param {unknown} value
 * @param {number} [maxLength]
 * @returns {string}
 */
export function shortSnippet(value, maxLength = 180) {
  if (typeof value !== "string") {
    return "";
  }
  const cleaned = value.trim().replace(/\s+/g, " ");
  if (cleaned.length <= maxLength) {
    return cleaned;
  }
  return `${cleaned.slice(0, Math.max(0, maxLength - 1))}…`;
}
