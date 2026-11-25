import { apiRequest } from "./client.js";

/**
 * Vérifie que l'API FastAPI répond, sans charger l'assistant.
 * @returns {Promise<{ status: string }>}
 */
export function checkHealth() {
  return apiRequest("/health", { method: "GET" });
}
