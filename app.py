"""Interface Streamlit d'EduRAG.

Une seule instance d'EduRAGAssistant est créée, puis réutilisée.
La page centrale est la conversation. Les cours et les fiches restent accessibles à part.
"""

import tempfile
from pathlib import Path, PureWindowsPath

import streamlit as st

from src.assistant import EduRAGAssistant

_ROOT = Path(__file__).resolve().parent
_DOCUMENTS = _ROOT / "data" / "documents"
_INDEX = _ROOT / "data" / "vector_store"
_REVISION_ID = "revision_id"
_OPENED_REVISION_ID = "opened_revision_id"
_CONVERSATION_ID = "conversation_id"
_SCREEN = "screen"
_CHAT_ERROR = "chat_error"
_CHAT_SCREEN = "chat"
_COURSES_SCREEN = "courses"
_TITLE_LIMIT = 80
_LANGUAGES = (("Français", "fr"), ("Anglais", "en"), ("Arabe", "ar"))
_LEVELS = (
    ("Débutant", "beginner"),
    ("Intermédiaire", "intermediate"),
    ("Avancé", "advanced"),
)
_SHEET_TYPES = (("Fiche courte", "short"), ("Fiche complète", "complete"))
_LANGUAGE_LABELS = dict((code, label) for label, code in _LANGUAGES)
_LEVEL_LABELS = dict((code, label) for label, code in _LEVELS)
_TYPE_LABELS = dict((code, label) for label, code in _SHEET_TYPES)


@st.cache_resource
def get_assistant() -> EduRAGAssistant:
    """Crée l'assistant une fois pour ces dossiers, puis le conserve."""
    return EduRAGAssistant(_DOCUMENTS, _INDEX)


def _store_uploaded_pdf(assistant: EduRAGAssistant, uploaded: object) -> dict[str, str]:
    """Copie le PDF hors de la bibliothèque, puis le fait enregistrer.

    Le fichier temporaire est supprimé même si l'ajout échoue.
    Le nom est réduit au fichier : « / » et « \\ » sont retirés sous Windows et Linux.
    """
    name = PureWindowsPath(str(getattr(uploaded, "name", ""))).name
    if name in {"", ".", ".."} or not name.lower().endswith(".pdf"):
        raise ValueError("Le fichier sélectionné doit être un PDF.")

    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / name
        path.write_bytes(uploaded.getvalue())
        added = assistant.add_pdf(path)
    return added


def _revision_pages(
    whole_document: bool,
    page_start: object,
    page_end: object,
) -> tuple[int | None, int | None]:
    """Choisit tout le document, ou un intervalle de deux entiers."""
    if whole_document:
        return None, None
    if isinstance(page_start, bool) or not isinstance(page_start, int):
        raise ValueError("La page de début doit être un entier.")
    if isinstance(page_end, bool) or not isinstance(page_end, int):
        raise ValueError("La page de fin doit être un entier.")
    if page_start < 1 or page_end < 1 or page_start > page_end:
        raise ValueError("L'intervalle de pages est invalide.")
    return page_start, page_end


def _create_selected_revision(
    assistant: EduRAGAssistant,
    document_id: str,
    sheet_type: str,
    language: str,
    level: str,
    whole_document: bool,
    page_start: object,
    page_end: object,
) -> dict[str, str | int]:
    """Demande une seule fiche avec les choix du formulaire."""
    start, end = _revision_pages(whole_document, page_start, page_end)
    return assistant.create_revision(
        document_id,
        page_start=start,
        page_end=end,
        language=language,
        level=level,
        sheet_type=sheet_type,
    )


def _remember_revision(state: dict[str, object], saved: dict[str, str | int]) -> None:
    """Garde l'identifiant de la fiche qui vient d'être générée."""
    state[_REVISION_ID] = saved["id"]


def _remember_opened_revision(state: dict[str, object], sheet_id: object) -> None:
    """Garde l'identifiant d'une fiche rouverte, sans toucher à la fiche nouvelle."""
    state[_OPENED_REVISION_ID] = sheet_id


def _sheet_summary(sheet: dict[str, str | int]) -> str:
    """Décrit une fiche enregistrée sans exposer de chemin interne."""
    language = _LANGUAGE_LABELS.get(str(sheet["language"]), str(sheet["language"]))
    level = _LEVEL_LABELS.get(str(sheet["level"]), str(sheet["level"]))
    sheet_type = _TYPE_LABELS.get(str(sheet["sheet_type"]), str(sheet["sheet_type"]))
    return f"{sheet['filename']} — {sheet['created_at']} — {sheet_type} — {language} — {level}"


def _course_is_missing(sheet: dict[str, str | int], documents: list[dict[str, str]]) -> bool:
    """Indique si le PDF d'origine n'est plus dans la bibliothèque."""
    known = {str(document.get("document_id")) for document in documents}
    return str(sheet.get("document_id")) not in known


def _open_saved_revision(assistant: EduRAGAssistant, sheet_id: object) -> dict[str, str | int]:
    """Relit une fiche enregistrée, sans la générer de nouveau."""
    return assistant.get_revision(sheet_id)


def _export_revision_bytes(assistant: EduRAGAssistant, sheet_id: object, kind: str) -> bytes:
    """Exporte une fiche hors de la bibliothèque, puis supprime le fichier."""
    suffix = ".md" if kind == "markdown" else ".pdf"
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / f"fiche_revision_{sheet_id}{suffix}"
        if kind == "markdown":
            written = assistant.export_revision_markdown(sheet_id, path)
        else:
            written = assistant.export_revision_pdf(sheet_id, path)
        return Path(written).read_bytes()


def _cached_export(
    state: dict[str, object],
    assistant: EduRAGAssistant,
    sheet_id: object,
    kind: str,
) -> bytes:
    """Réutilise les octets déjà exportés pour cet identifiant."""
    key = f"export_{kind}_{sheet_id}"
    cached = state.get(key)
    if isinstance(cached, bytes):
        return cached
    data = _export_revision_bytes(assistant, sheet_id, kind)
    state[key] = data
    return data


def _format_pages(page_start: int, page_end: int) -> str:
    """Décrit une page seule ou un intervalle."""
    if page_start == page_end:
        return str(page_start)
    return f"{page_start} à {page_end}"


def _show_revision(sheet: dict[str, str | int], documents: list[dict[str, str]]) -> None:
    """Affiche les informations enregistrées et le Markdown de la fiche."""
    st.subheader(str(sheet["filename"]))
    st.write(f"Pages : {_format_pages(int(sheet['page_start']), int(sheet['page_end']))}")
    st.write(f"Langue : {_LANGUAGE_LABELS.get(str(sheet['language']), sheet['language'])}")
    st.write(f"Niveau : {_LEVEL_LABELS.get(str(sheet['level']), sheet['level'])}")
    st.write(f"Type : {_TYPE_LABELS.get(str(sheet['sheet_type']), sheet['sheet_type'])}")
    st.write(f"Date de création : {sheet['created_at']}")
    if _course_is_missing(sheet, documents):
        st.info(
            "Le cours PDF d'origine n'est plus dans la bibliothèque, "
            "mais cette fiche reste consultable."
        )
    st.markdown(str(sheet["content"]))


def _offer_downloads(assistant: EduRAGAssistant, sheet_id: object, section: str) -> None:
    """Propose le Markdown et le PDF de la fiche affichée dans cette section."""
    try:
        markdown = _cached_export(st.session_state, assistant, sheet_id, "markdown")
        pdf = _cached_export(st.session_state, assistant, sheet_id, "pdf")
    except (OSError, ValueError, RuntimeError) as error:
        st.error(str(error))
        return
    st.download_button(
        "Télécharger le Markdown",
        data=markdown,
        file_name=f"fiche_revision_{sheet_id}.md",
        mime="text/markdown",
        key=f"{section}_markdown",
    )
    st.download_button(
        "Télécharger le PDF",
        data=pdf,
        file_name=f"fiche_revision_{sheet_id}.pdf",
        mime="application/pdf",
        key=f"{section}_pdf",
    )


def _begin_new_conversation(state: dict[str, object]) -> None:
    """Ouvre une conversation vide. La base n'est créée qu'à la première question."""
    state[_CONVERSATION_ID] = None
    state[_SCREEN] = _CHAT_SCREEN


def _select_conversation(state: dict[str, object], conversation_id: object) -> None:
    """Retient l'identifiant SQLite de la conversation à afficher."""
    state[_CONVERSATION_ID] = conversation_id
    state[_SCREEN] = _CHAT_SCREEN


def _show_courses(state: dict[str, object]) -> None:
    """Affiche les cours et les fiches, sans fermer la conversation retenue."""
    state[_SCREEN] = _COURSES_SCREEN


def _conversation_title(question: str) -> str:
    """Résume la première question pour le titre enregistré."""
    if not isinstance(question, str):
        raise ValueError("La question ne peut pas être vide.")
    title = " ".join(question.split())
    if title == "":
        raise ValueError("La question ne peut pas être vide.")
    if len(title) <= _TITLE_LIMIT:
        return title
    return title[: _TITLE_LIMIT - 1].rstrip() + "…"


def _submit_chat(
    assistant: EduRAGAssistant,
    state: dict[str, object],
    question: str,
) -> dict[str, str | int | dict]:
    """Envoie une question. Une conversation nouvelle n'appelle send_message qu'une fois."""
    conversation_id = state.get(_CONVERSATION_ID)
    if conversation_id is None:
        conversation_id = assistant.create_conversation(_conversation_title(question))
        state[_CONVERSATION_ID] = conversation_id
    return assistant.send_message(conversation_id, question)


def _load_conversation(
    assistant: EduRAGAssistant,
    conversation_id: object,
) -> dict[str, str | int | list]:
    """Relit une conversation enregistrée, sans produire de nouvelle réponse."""
    return assistant.get_conversation(conversation_id)


def _delete_conversation(
    assistant: EduRAGAssistant,
    state: dict[str, object],
    conversation_id: object,
) -> None:
    """Supprime une conversation. L'écran redevient vide si c'était la conversation ouverte."""
    assistant.delete_conversation(conversation_id)
    if state.get(_CONVERSATION_ID) == conversation_id:
        state[_CONVERSATION_ID] = None


def _conversation_entries(
    conversations: list[dict[str, str | int]],
    current_id: object,
) -> list[dict[str, str | int | bool]]:
    """Garde l'ordre reçu et marque la conversation ouverte par son identifiant."""
    return [
        {
            "id": conversation["id"],
            "title": str(conversation["title"]),
            "current": conversation["id"] == current_id,
        }
        for conversation in conversations
    ]


def _format_message_sources(sources: object) -> str:
    """Cite le fichier et la page, sans le texte du passage ni son identifiant interne."""
    if not isinstance(sources, list):
        return ""
    parts: list[str] = []
    for source in sources:
        if not isinstance(source, dict):
            continue
        filename = source.get("filename")
        page_number = source.get("page_number")
        if not isinstance(filename, str) or filename.strip() == "":
            continue
        if isinstance(page_number, bool) or not isinstance(page_number, int):
            continue
        parts.append(f"{filename.strip()}, page {page_number}")
    return " · ".join(parts)


def _remove_course(assistant: EduRAGAssistant, document_id: str) -> None:
    """Retire un cours de la bibliothèque déjà ouverte."""
    assistant.remove_document(document_id)


def _apply_layout() -> None:
    """Limite la largeur de lecture sans rétrécir la barre latérale."""
    st.markdown(
        """
        <style>
        section.main > div.block-container {
            max-width: 860px;
            padding-top: 1.5rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _render_sidebar(assistant: EduRAGAssistant) -> None:
    """Affiche les conversations et l'accès aux cours."""
    st.sidebar.header("EduRAG")
    if st.sidebar.button("+ Nouvelle conversation", use_container_width=True):
        _begin_new_conversation(st.session_state)
        st.rerun()

    conversations = assistant.list_conversations()
    current_id = st.session_state.get(_CONVERSATION_ID)
    if not conversations:
        st.sidebar.caption("Aucune conversation pour le moment.")
    for entry in _conversation_entries(conversations, current_id):
        title_column, delete_column = st.sidebar.columns([5, 1], vertical_alignment="center")
        button_type = "primary" if entry["current"] else "secondary"
        if title_column.button(
            str(entry["title"]),
            key=f"open_{entry['id']}",
            type=button_type,
            use_container_width=True,
        ):
            _select_conversation(st.session_state, entry["id"])
            st.rerun()
        if delete_column.button(
            "✕",
            key=f"delete_{entry['id']}",
            help="Supprimer cette conversation",
        ):
            try:
                _delete_conversation(assistant, st.session_state, entry["id"])
            except (ValueError, RuntimeError) as error:
                st.sidebar.error(str(error))
            else:
                st.rerun()

    st.sidebar.divider()
    if st.sidebar.button("Cours et fiches", use_container_width=True):
        _show_courses(st.session_state)
        st.rerun()


def _render_messages(messages: list[dict[str, str | int | list]]) -> None:
    """Affiche les messages déjà enregistrés."""
    for message in messages:
        role = str(message.get("role"))
        if role not in {"user", "assistant"}:
            role = "assistant"
        with st.chat_message(role):
            st.markdown(str(message.get("content", "")))
            sources = _format_message_sources(message.get("sources"))
            if sources:
                st.caption(f"Sources : {sources}")


def _render_chat(assistant: EduRAGAssistant) -> None:
    """Affiche la conversation ouverte, puis envoie une nouvelle question."""
    conversation_id = st.session_state.get(_CONVERSATION_ID)
    if conversation_id is None:
        st.title("EduRAG")
        st.markdown(
            "Posez une question sur vos cours PDF. "
            "La réponse peut citer le fichier et la page. "
            "Français, anglais et arabe sont pris en charge."
        )
    else:
        try:
            conversation = _load_conversation(assistant, conversation_id)
        except (ValueError, RuntimeError) as error:
            st.session_state[_CONVERSATION_ID] = None
            st.error(str(error))
        else:
            st.subheader(str(conversation["title"]))
            messages = conversation.get("messages")
            if isinstance(messages, list):
                _render_messages(messages)

    error = st.session_state.pop(_CHAT_ERROR, None)
    if isinstance(error, str) and error.strip() != "":
        st.error(error)

    question = st.chat_input("Posez une question sur vos cours")
    if not question:
        return
    try:
        with st.spinner("Génération de la réponse…"):
            _submit_chat(assistant, st.session_state, question)
    except (OSError, ValueError, RuntimeError) as error:
        st.session_state[_CHAT_ERROR] = str(error)
    st.rerun()


def _render_courses(assistant: EduRAGAssistant) -> None:
    """Affiche la bibliothèque et Smart Revision."""
    st.title("Cours et fiches")

    st.header("Bibliothèque")
    uploaded = st.file_uploader("Importer un cours", type=["pdf"])
    if st.button("Ajouter le cours"):
        if uploaded is None:
            st.warning("Sélectionnez un fichier PDF avant de l'ajouter.")
        else:
            try:
                added = _store_uploaded_pdf(assistant, uploaded)
            except (OSError, ValueError, RuntimeError) as error:
                st.error(str(error))
            else:
                st.success(f"Le cours « {added['filename']} » a été ajouté.")
                st.rerun()

    documents = assistant.list_documents()
    if not documents:
        st.info("Aucun cours n'est enregistré pour le moment.")
    else:
        for document in documents:
            name_column, remove_column = st.columns([5, 1], vertical_alignment="center")
            name_column.write(document["filename"])
            if remove_column.button("Retirer", key=f"remove_{document['document_id']}"):
                try:
                    _remove_course(assistant, str(document["document_id"]))
                except (OSError, ValueError, RuntimeError) as error:
                    st.error(str(error))
                else:
                    st.rerun()

    st.header("Nouvelle fiche de révision")
    if not documents:
        st.info("Ajoutez d'abord un cours PDF pour créer une fiche.")
    else:
        selected = st.selectbox(
            "Cours",
            documents,
            format_func=lambda document: document["filename"],
        )
        sheet_type = _choice("Type de fiche", _SHEET_TYPES)
        language = _choice("Langue", _LANGUAGES)
        level = _choice("Niveau", _LEVELS)
        whole_document = st.radio(
            "Périmètre",
            ["Document entier", "Intervalle de pages"],
        ) == "Document entier"
        page_start: int | None = None
        page_end: int | None = None
        if not whole_document:
            page_start = int(st.number_input("Page de début", min_value=1, value=1, step=1))
            page_end = int(st.number_input("Page de fin", min_value=1, value=1, step=1))
        if st.button("Générer la fiche"):
            try:
                with st.spinner("Génération de la fiche…"):
                    saved = _create_selected_revision(
                        assistant,
                        str(selected["document_id"]),
                        sheet_type,
                        language,
                        level,
                        whole_document,
                        page_start,
                        page_end,
                    )
            except (ValueError, RuntimeError) as error:
                st.error(str(error))
            else:
                _remember_revision(st.session_state, saved)
                st.success("La fiche a été enregistrée.")

    revision_id = st.session_state.get(_REVISION_ID)
    if revision_id is not None:
        try:
            _show_revision(assistant.get_revision(revision_id), documents)
        except (ValueError, RuntimeError) as error:
            st.error(str(error))
        else:
            _offer_downloads(assistant, revision_id, "new")

    st.header("Mes fiches enregistrées")
    saved_sheets = assistant.list_revisions()
    if not saved_sheets:
        st.info("Aucune fiche n'est enregistrée pour le moment.")
    else:
        for sheet in saved_sheets:
            st.write(_sheet_summary(sheet))
        chosen = st.selectbox(
            "Fiche enregistrée",
            saved_sheets,
            format_func=_sheet_summary,
        )
        if st.button("Ouvrir la fiche"):
            try:
                opened = _open_saved_revision(assistant, chosen["id"])
            except (ValueError, RuntimeError) as error:
                st.error(str(error))
            else:
                _remember_opened_revision(st.session_state, opened["id"])

    opened_id = st.session_state.get(_OPENED_REVISION_ID)
    if opened_id is not None:
        try:
            _show_revision(_open_saved_revision(assistant, opened_id), documents)
        except (ValueError, RuntimeError) as error:
            st.error(str(error))
        else:
            _offer_downloads(assistant, opened_id, "opened")


def _choice(label: str, options: tuple[tuple[str, str], ...]) -> str:
    """Affiche un libellé et renvoie la valeur interne correspondante."""
    labels = [item[0] for item in options]
    selected = st.selectbox(label, labels)
    return dict(options)[selected]


def main() -> None:
    """Affiche la conversation, ou les cours et les fiches."""
    st.set_page_config(
        page_title="EduRAG",
        page_icon="📚",
        layout="wide",
    )
    _apply_layout()
    with st.spinner("Ouverture de l'assistant…"):
        assistant = get_assistant()
    _render_sidebar(assistant)
    if st.session_state.get(_SCREEN) == _COURSES_SCREEN:
        _render_courses(assistant)
        return
    _render_chat(assistant)


if __name__ == "__main__":
    main()
