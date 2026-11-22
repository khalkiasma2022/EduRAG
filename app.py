"""Point d'entrée de l'application EduRAG.

Cette première version affiche seulement l'accueil.
Le pipeline RAG sera ajouté dans une étape suivante.
"""

import streamlit as st


def main() -> None:
    """Affiche la page d'accueil, sans interroger de documents ni de modèle."""
    st.set_page_config(
        page_title="EduRAG",
        page_icon="📚",
        layout="centered",
    )
    st.title("EduRAG")
    st.subheader("Assistant intelligent pour les étudiants")
    st.info(
        "L'environnement du projet est prêt. "
        "L'ingestion, la recherche et la génération ne sont pas encore développées."
    )
    st.markdown(
        """
        Les modules prévus sont :

        - `src/ingestion` : lecture des cours
        - `src/retrieval` : recherche des passages utiles
        - `src/generation` : rédaction de la réponse
        - `src/evaluation` : mesure de la qualité
        """
    )


if __name__ == "__main__":
    main()
