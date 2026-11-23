"""Accueil Streamlit d'EduRAG.

Une seule instance d'EduRAGAssistant est créée, puis réutilisée.
Cette page présente le projet, les cours, l'ajout d'un PDF et les fiches enregistrées.
"""

import tempfile
from pathlib import Path

import streamlit as st

from src.assistant import EduRAGAssistant

_ROOT = Path(__file__).resolve().parent
_DOCUMENTS = _ROOT / "data" / "documents"
_INDEX = _ROOT / "data" / "vector_store"
_REVISION_ID = "revision_id"
_OPENED_REVISION_ID = "opened_revision_id"
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
    """
    name = Path(str(getattr(uploaded, "name", ""))).name
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


def _choice(label: str, options: tuple[tuple[str, str], ...]) -> str:
    """Affiche un libellé et renvoie la valeur interne correspondante."""
    labels = [item[0] for item in options]
    selected = st.selectbox(label, labels)
    return dict(options)[selected]


def main() -> None:
    """Affiche l'accueil, les cours et le formulaire de fiche."""
    st.set_page_config(
        page_title="EduRAG",
        page_icon="📚",
        layout="centered",
    )
    st.title("EduRAG")
    st.subheader("Assistant éducatif à partir de vos cours PDF")
    st.markdown(
        """
        EduRAG s'appuie sur les PDF que vous enregistrez.
        Les réponses et les fiches restent liées à ces cours.

        **Smart Revision** crée une fiche de révision à partir d'un cours.

        **Smart Practice** est prévu pour s'entraîner sur les cours.
        Il n'est pas encore disponible.

        **AI Tutor** peut déjà répondre à une question sur un cours
        dans le moteur EduRAG. Cette page n'a pas encore de zone de dialogue.

        Langues prises en charge : français, anglais et arabe.
        """
    )

    with st.spinner("Ouverture de l'assistant…"):
        assistant = get_assistant()

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

    documents = assistant.list_documents()
    if not documents:
        st.info("Aucun cours n'est enregistré pour le moment.")
    else:
        for document in documents:
            st.write(document["filename"])

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


if __name__ == "__main__":
    main()
