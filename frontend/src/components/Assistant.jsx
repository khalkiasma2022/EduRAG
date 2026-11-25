import { useCallback, useEffect, useRef, useState } from "react";
import { ApiError } from "../api/client.js";
import {
  createConversation,
  deleteConversation,
  getConversation,
  listConversations,
  sendMessage,
} from "../api/conversations.js";
import "./Assistant.css";

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

function titleFromQuestion(question) {
  const cleaned = question.trim().replace(/\s+/g, " ");
  if (cleaned.length <= 60) {
    return cleaned;
  }
  return `${cleaned.slice(0, 57)}...`;
}

function MessageSources({ sources }) {
  if (!Array.isArray(sources) || sources.length === 0) {
    return null;
  }

  return (
    <div className="assistant__sources">
      <p className="assistant__sources-title">Sources</p>
      <ul className="assistant__sources-list">
        {sources.map((source, index) => (
          <li
            key={`${source.chunk_id ?? index}-${source.filename}-${source.page_number}`}
            className="assistant__source"
          >
            <p className="assistant__source-meta">
              <span className="assistant__source-file">{source.filename}</span>
              <span className="assistant__source-page">
                page {source.page_number}
              </span>
            </p>
            {source.text ? (
              <p className="assistant__source-text">{source.text}</p>
            ) : null}
          </li>
        ))}
      </ul>
    </div>
  );
}

function Assistant() {
  const [conversations, setConversations] = useState([]);
  const [activeId, setActiveId] = useState(null);
  const [messages, setMessages] = useState([]);
  const [activeTitle, setActiveTitle] = useState("");
  const [draft, setDraft] = useState("");
  const [listLoading, setListLoading] = useState(true);
  const [threadLoading, setThreadLoading] = useState(false);
  const [sending, setSending] = useState(false);
  const [creating, setCreating] = useState(false);
  const [deletingId, setDeletingId] = useState(null);
  const [error, setError] = useState("");
  const messagesEndRef = useRef(null);
  const inputRef = useRef(null);

  const busy = listLoading || threadLoading || sending || creating || deletingId !== null;

  const refreshConversations = useCallback(async ({ quiet = false } = {}) => {
    if (!quiet) {
      setListLoading(true);
    }
    try {
      const items = await listConversations();
      setConversations(Array.isArray(items) ? items : []);
      return Array.isArray(items) ? items : [];
    } catch (err) {
      setError(apiErrorMessage(err));
      throw err;
    } finally {
      if (!quiet) {
        setListLoading(false);
      }
    }
  }, []);

  const openConversation = useCallback(async (conversationId) => {
    setThreadLoading(true);
    setError("");
    try {
      const detail = await getConversation(conversationId);
      setActiveId(detail.id);
      setActiveTitle(detail.title || "");
      setMessages(Array.isArray(detail.messages) ? detail.messages : []);
    } catch (err) {
      setError(apiErrorMessage(err));
      throw err;
    } finally {
      setThreadLoading(false);
    }
  }, []);

  useEffect(() => {
    refreshConversations()
      .then((items) => {
        if (items.length > 0) {
          return openConversation(items[0].id);
        }
        setActiveId(null);
        setActiveTitle("");
        setMessages([]);
        return null;
      })
      .catch(() => {});
  }, [refreshConversations, openConversation]);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [messages, sending, threadLoading]);

  async function handleCreateConversation() {
    const suggested = `Conversation ${new Date().toLocaleString("fr-FR", {
      dateStyle: "short",
      timeStyle: "short",
    })}`;
    const entered = window.prompt("Titre de la nouvelle conversation", suggested);
    if (entered === null) {
      return;
    }
    const title = entered.trim();
    if (title === "") {
      setError("Le titre de la conversation est invalide.");
      return;
    }

    setCreating(true);
    setError("");
    try {
      const created = await createConversation(title);
      await refreshConversations({ quiet: true });
      await openConversation(created.id);
      inputRef.current?.focus();
    } catch (err) {
      setError(apiErrorMessage(err));
    } finally {
      setCreating(false);
    }
  }

  async function handleSelectConversation(conversationId) {
    if (conversationId === activeId || busy) {
      return;
    }
    try {
      await openConversation(conversationId);
    } catch {
      /* message déjà affiché */
    }
  }

  async function handleDeleteConversation(event, conversation) {
    event.stopPropagation();
    const confirmed = window.confirm(
      `Supprimer la conversation « ${conversation.title} » et tous ses messages ?`,
    );
    if (!confirmed) {
      return;
    }

    setDeletingId(conversation.id);
    setError("");
    try {
      await deleteConversation(conversation.id);
      const remaining = await refreshConversations({ quiet: true });
      if (activeId === conversation.id) {
        if (remaining.length > 0) {
          await openConversation(remaining[0].id);
        } else {
          setActiveId(null);
          setActiveTitle("");
          setMessages([]);
        }
      }
    } catch (err) {
      setError(apiErrorMessage(err));
    } finally {
      setDeletingId(null);
    }
  }

  async function handleSend(event) {
    event.preventDefault();
    const question = draft.trim();
    if (question === "" || sending) {
      return;
    }

    setSending(true);
    setError("");

    try {
      let conversationId = activeId;

      if (conversationId === null) {
        const created = await createConversation(titleFromQuestion(question));
        conversationId = created.id;
        await refreshConversations({ quiet: true });
        setActiveId(created.id);
        setActiveTitle(created.title);
        setMessages([]);
      }

      setDraft("");
      const exchange = await sendMessage(conversationId, question);
      const detail = await getConversation(conversationId);
      setMessages(Array.isArray(detail.messages) ? detail.messages : []);
      setActiveTitle(detail.title || activeTitle);

      if (!exchange?.answer) {
        setError("La réponse du modèle est incomplète.");
      }

      await refreshConversations({ quiet: true });
    } catch (err) {
      setError(apiErrorMessage(err));
    } finally {
      setSending(false);
      inputRef.current?.focus();
    }
  }

  return (
    <section className="assistant" aria-labelledby="assistant-title">
      <div className="assistant__header">
        <p className="assistant__eyebrow">Tutorat</p>
        <h1 id="assistant-title" className="assistant__title">
          Assistant conversationnel
        </h1>
        <p className="assistant__lead">
          Posez vos questions sur vos cours. Les réponses s’appuient sur les documents
          indexés et citent leurs sources.
        </p>
      </div>

      {error ? (
        <p className="assistant__banner assistant__banner--error" role="alert">
          {error}
        </p>
      ) : null}

      <div className="assistant__workspace">
        <aside className="assistant__sidebar" aria-label="Conversations">
          <div className="assistant__sidebar-top">
            <h2>Conversations</h2>
            <button
              type="button"
              className="assistant__new"
              onClick={handleCreateConversation}
              disabled={busy}
            >
              {creating ? "Création…" : "+ Nouvelle conversation"}
            </button>
          </div>

          {listLoading ? (
            <p className="assistant__hint">Chargement des conversations…</p>
          ) : conversations.length === 0 ? (
            <p className="assistant__hint">
              Aucune conversation. Créez-en une ou envoyez directement une question.
            </p>
          ) : (
            <ul className="assistant__conversation-list">
              {conversations.map((conversation) => {
                const isActive = conversation.id === activeId;
                const isDeleting = deletingId === conversation.id;

                return (
                  <li key={conversation.id}>
                    <button
                      type="button"
                      className={`assistant__conversation${isActive ? " assistant__conversation--active" : ""}`}
                      onClick={() => handleSelectConversation(conversation.id)}
                      disabled={busy && !isActive}
                    >
                      <span className="assistant__conversation-title">
                        {conversation.title}
                      </span>
                      <span className="assistant__conversation-date">
                        {formatWhen(conversation.created_at)}
                      </span>
                    </button>
                    <button
                      type="button"
                      className="assistant__delete"
                      onClick={(event) => handleDeleteConversation(event, conversation)}
                      disabled={busy}
                      aria-label={`Supprimer ${conversation.title}`}
                    >
                      {isDeleting ? "…" : "Suppr."}
                    </button>
                  </li>
                );
              })}
            </ul>
          )}
        </aside>

        <div className="assistant__chat">
          <div className="assistant__chat-header">
            <h2>{activeTitle || "Aucune conversation sélectionnée"}</h2>
            {threadLoading ? <span className="assistant__status">Chargement…</span> : null}
            {sending ? <span className="assistant__status">Qwen réfléchit…</span> : null}
          </div>

          <div className="assistant__messages" aria-live="polite">
            {!activeId && !threadLoading ? (
              <p className="assistant__empty">
                Sélectionnez une conversation ou créez-en une pour commencer.
                Vous pouvez aussi écrire une question : une conversation sera créée
                automatiquement.
              </p>
            ) : null}

            {activeId && !threadLoading && messages.length === 0 && !sending ? (
              <p className="assistant__empty">
                Conversation vide. Posez votre première question ci-dessous.
              </p>
            ) : null}

            {messages.map((message) => {
              const isUser = message.role === "user";
              return (
                <article
                  key={message.id}
                  className={`assistant__bubble assistant__bubble--${isUser ? "user" : "assistant"}`}
                >
                  <header className="assistant__bubble-meta">
                    <span>{isUser ? "Vous" : "Assistant"}</span>
                    <time dateTime={message.created_at}>{formatWhen(message.created_at)}</time>
                  </header>
                  <p className="assistant__bubble-content">{message.content}</p>
                  {!isUser ? <MessageSources sources={message.sources} /> : null}
                </article>
              );
            })}

            {sending ? (
              <article className="assistant__bubble assistant__bubble--assistant assistant__bubble--pending">
                <p className="assistant__bubble-content">Génération de la réponse…</p>
              </article>
            ) : null}

            <div ref={messagesEndRef} />
          </div>

          <form className="assistant__composer" onSubmit={handleSend}>
            <label className="assistant__composer-label" htmlFor="assistant-question">
              Votre question
            </label>
            <div className="assistant__composer-row">
              <textarea
                id="assistant-question"
                ref={inputRef}
                className="assistant__input"
                rows={3}
                value={draft}
                onChange={(event) => setDraft(event.target.value)}
                placeholder="Posez une question sur vos documents…"
                disabled={sending || creating || deletingId !== null || threadLoading}
                onKeyDown={(event) => {
                  if (event.key === "Enter" && !event.shiftKey) {
                    event.preventDefault();
                    event.currentTarget.form?.requestSubmit();
                  }
                }}
              />
              <button
                type="submit"
                className="assistant__send"
                disabled={
                  sending
                  || creating
                  || deletingId !== null
                  || threadLoading
                  || draft.trim() === ""
                }
              >
                {sending ? "Envoi…" : "Envoyer"}
              </button>
            </div>
          </form>
        </div>
      </div>
    </section>
  );
}

export default Assistant;
