# EduRAG

EduRAG est un assistant destiné aux étudiants. Il s'appuiera sur un grand modèle de langage et sur une architecture RAG (*Retrieval-Augmented Generation*) : les réponses devront s'appuyer sur les cours fournis, et pas seulement sur la mémoire du modèle.

Cette première étape prépare uniquement l'environnement. L'ingestion, la recherche, la génération et l'évaluation ne sont pas encore écrites.

## Arborescence

```text
EduRAG/
├── app.py                 # Accueil Streamlit
├── requirements.txt       # Bibliothèques, versions figées
├── .env.example           # Modèle sans clé réelle
├── .gitignore             # Secrets et données locales exclus de Git
├── src/                   # Code de l'application
│   ├── ingestion/         # Lecture des cours (à venir)
│   ├── retrieval/         # Recherche des passages (à venir)
│   ├── generation/        # Réponse du modèle (à venir)
│   └── evaluation/        # Mesure de la qualité (à venir)
├── data/
│   ├── documents/         # Cours déposés localement
│   └── vector_store/      # Index vectoriel local
├── tests/                 # Tests de l'environnement
├── notebooks/             # Explorations
└── pages/                 # Pages Streamlit futures
```

Les dossiers créés sont `src`, `src/ingestion`, `src/retrieval`, `src/generation`, `src/evaluation`, `data/documents`, `data/vector_store`, `tests`, `notebooks` et `pages`.

## Installation

Python 3.12 est utilisé pour ce projet.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Les bibliothèques prévues sont LangChain, Hugging Face (`transformers` et `huggingface-hub`), FAISS (`faiss-cpu`), PyPDF (`pypdf`), Streamlit, `sentence-transformers`, `python-dotenv` et pytest.

## Clés API

Les clés ne vont jamais dans le code ni dans Git.

1. Copiez le modèle : `Copy-Item .env.example .env`
2. Ouvrez `.env` et renseignez `HF_TOKEN` (jeton Hugging Face).
3. Laissez `OPENAI_API_KEY` vide tant qu'aucun modèle OpenAI n'est utilisé.

`.env`, `.streamlit/secrets.toml`, les fichiers de certificats et les fichiers de type `credentials.json` sont ignorés par Git. `.env.example` reste versionné, avec des valeurs vides.

Les PDF déposés dans `data/documents/` et l'index dans `data/vector_store/` restent aussi sur la machine.

## Lancer l'accueil

```powershell
streamlit run app.py
```

La page confirme que l'environnement est prêt. Elle ne répond pas encore aux questions.

## Tests

```powershell
python -m pytest
```

Les tests vérifient les dossiers, les fichiers, la syntaxe de `requirements.txt` et le fait que Git ignore les fichiers sensibles. Git pour Windows doit être disponible dans le PATH pour ce dernier contrôle : [git-scm.com/download/win](https://git-scm.com/download/win).

## Hors périmètre pour l'instant

- chargement et découpage des documents
- création de l'index FAISS
- recherche sémantique
- appel à un modèle
- évaluation des réponses
