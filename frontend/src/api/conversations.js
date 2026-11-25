import { apiRequest } from "./client.js";

/**
 * Crée une conversation vide.
 * @param {string} title
 * @returns {Promise<{ id: number, title: string, created_at: string }>}
 */
export function createConversation(title) {
  return apiRequest("/conversations", {
    method: "POST",
    json: { title },
  });
}

/**
 * Liste les conversations.
 * @returns {Promise<Array<{ id: number, title: string, created_at: string }>>}
 */
export function listConversations() {
  return apiRequest("/conversations", {
    method: "GET",
  });
}

/**
 * Relit une conversation et ses messages.
 * @param {number|string} conversationId
 * @returns {Promise<object>}
 */
export function getConversation(conversationId) {
  return apiRequest(`/conversations/${encodeURIComponent(String(conversationId))}`, {
    method: "GET",
  });
}

/**
 * Supprime une conversation.
 * @param {number|string} conversationId
 * @returns {Promise<null>}
 */
export function deleteConversation(conversationId) {
  return apiRequest(`/conversations/${encodeURIComponent(String(conversationId))}`, {
    method: "DELETE",
    responseType: "void",
  });
}

/**
 * Envoie une question dans une conversation existante.
 * @param {number|string} conversationId
 * @param {string} question
 * @returns {Promise<object>}
 */
export function sendMessage(conversationId, question) {
  return apiRequest(
    `/conversations/${encodeURIComponent(String(conversationId))}/messages`,
    {
      method: "POST",
      json: { question },
    },
  );
}
