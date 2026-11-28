"""Lance l'API FastAPI avec EduRAGAssistant déjà enregistré."""

from pathlib import Path

import uvicorn
from dotenv import load_dotenv

from src.api.app import create_app
from src.assistant import EduRAGAssistant

_ROOT = Path(__file__).resolve().parent
_DOCUMENTS = _ROOT / "data" / "documents"
_INDEX = _ROOT / "data" / "vector_store"


def main() -> None:
    load_dotenv(_ROOT / ".env")
    assistant = EduRAGAssistant(_DOCUMENTS, _INDEX)
    app = create_app(assistant)
    uvicorn.run(app, host="127.0.0.1", port=8000)


if __name__ == "__main__":
    main()
