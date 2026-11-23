"""Tests de l'environnement EduRAG.

Ils contrôlent la structure, requirements.txt et l'ignorance Git des secrets.
Ils ne lancent pas de pipeline RAG.
"""

from __future__ import annotations

import ast
import os
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

REQUIRED_DIRECTORIES = (
    "src",
    "src/ingestion",
    "src/retrieval",
    "src/generation",
    "src/evaluation",
    "data/documents",
    "data/vector_store",
    "tests",
    "notebooks",
    "pages",
)

REQUIRED_FILES = (
    "app.py",
    "requirements.txt",
    ".env.example",
    ".gitignore",
    "README.md",
    "src/__init__.py",
    "src/ingestion/__init__.py",
    "src/retrieval/__init__.py",
    "src/generation/__init__.py",
    "src/evaluation/__init__.py",
    "data/documents/.gitkeep",
    "data/vector_store/.gitkeep",
    "notebooks/.gitkeep",
    "pages/.gitkeep",
)

EXPECTED_PACKAGES = {
    "langchain",
    "langchain-community",
    "langchain-text-splitters",
    "transformers",
    "huggingface-hub",
    "faiss-cpu",
    "pypdf",
    "fpdf2",
    "uharfbuzz",
    "streamlit",
    "sentence-transformers",
    "python-dotenv",
    "pytest",
}

IGNORED_BY_GIT = (
    ".env",
    ".env.local",
    ".streamlit/secrets.toml",
    "credentials.json",
    "secrets.json",
    "private.pem",
    "api.key",
    "data/documents/cours.pdf",
    "data/vector_store/index.faiss",
    "__pycache__/module.pyc",
)

TRACKED_BY_GIT = (
    ".env.example",
    "README.md",
    "requirements.txt",
    "app.py",
    "data/documents/.gitkeep",
    "data/vector_store/.gitkeep",
)


def _requirement_class():
    try:
        from packaging.requirements import Requirement
    except ImportError:
        from pip._vendor.packaging.requirements import Requirement
    return Requirement


def _iter_requirements():
    requirement = _requirement_class()
    parsed = []
    for line_number, raw_line in enumerate(
        (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        if line.startswith(("-r", "-c", "--", "http://", "https://", "-e")):
            raise AssertionError(
                f"Ligne {line_number} de requirements.txt n'est pas une dépendance simple : {raw_line}"
            )
        parsed.append(requirement(line))
    return parsed


def _git_executable() -> str:
    found = shutil.which("git")
    if found:
        return found

    candidates = [
        Path(r"C:\Program Files\Git\cmd\git.exe"),
        Path(r"C:\Program Files\Git\bin\git.exe"),
        Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "Git" / "cmd" / "git.exe",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)

    pytest.fail(
        "Git est introuvable. Installez Git, puis relancez pytest "
        "pour vérifier que les fichiers sensibles sont ignorés."
    )


def test_required_directories_exist() -> None:
    missing = [path for path in REQUIRED_DIRECTORIES if not (ROOT / path).is_dir()]
    assert missing == [], f"Dossiers manquants : {missing}"


def test_required_files_exist() -> None:
    missing = [path for path in REQUIRED_FILES if not (ROOT / path).is_file()]
    assert missing == [], f"Fichiers manquants : {missing}"


def test_python_modules_have_valid_syntax() -> None:
    python_files = [path for path in REQUIRED_FILES if path.endswith(".py")]
    assert python_files, "Aucun module Python à contrôler."
    for relative in python_files:
        source = (ROOT / relative).read_text(encoding="utf-8")
        ast.parse(source, filename=relative)


def test_requirements_file_is_valid() -> None:
    requirements = _iter_requirements()
    names = [item.name.lower().replace("_", "-") for item in requirements]

    assert names, "requirements.txt ne contient aucune dépendance."
    assert len(names) == len(set(names)), f"Dépendances en double : {names}"
    assert set(names) == EXPECTED_PACKAGES, (
        f"Paquets attendus {sorted(EXPECTED_PACKAGES)}, obtenus {sorted(names)}"
    )

    for item in requirements:
        assert len(item.specifier) == 1, f"{item.name} doit avoir une version figée."
        specifier = next(iter(item.specifier))
        assert specifier.operator == "==", f"{item.name} doit être fixé avec ==."
        assert specifier.version, f"{item.name} n'a pas de numéro de version."


def test_env_example_has_empty_secrets() -> None:
    text = (ROOT / ".env.example").read_text(encoding="utf-8")
    values = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        key, separator, value = line.partition("=")
        assert separator == "=", f"Ligne invalide dans .env.example : {raw_line}"
        values[key.strip()] = value.strip()

    assert values["HF_TOKEN"] == ""
    assert values["OPENAI_API_KEY"] == ""
    assert values["EDURAG_LLM_MODEL"] == "Qwen/Qwen2.5-1.5B-Instruct"


def test_source_code_does_not_embed_secrets() -> None:
    forbidden_markers = ("sk-", "hf_", "api_key =", "api_key=")
    python_files = [ROOT / "app.py", *ROOT.joinpath("src").rglob("*.py")]
    for path in python_files:
        source = path.read_text(encoding="utf-8").lower()
        for marker in forbidden_markers:
            assert marker not in source, f"Secret potentiel dans {path.name} : {marker}"


def test_readme_explains_the_project() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "EduRAG" in readme
    assert ".env" in readme
    assert "requirements.txt" in readme
    assert "pytest" in readme
    for folder in REQUIRED_DIRECTORIES:
        assert folder in readme, f"Le README ne présente pas {folder}."


def test_sensitive_files_are_ignored_by_git(tmp_path: Path) -> None:
    git = _git_executable()
    repository = tmp_path / "repo"
    repository.mkdir()
    shutil.copyfile(ROOT / ".gitignore", repository / ".gitignore")

    init = subprocess.run(
        [git, "init"],
        cwd=repository,
        capture_output=True,
        text=True,
        check=False,
    )
    assert init.returncode == 0, init.stderr

    for relative in IGNORED_BY_GIT:
        result = subprocess.run(
            [git, "check-ignore", "-q", "--", relative],
            cwd=repository,
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, f"Git n'ignore pas {relative}. {result.stderr}"

    for relative in TRACKED_BY_GIT:
        result = subprocess.run(
            [git, "check-ignore", "-q", "--", relative],
            cwd=repository,
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 1, (
            f"Git ignore {relative}, alors que ce fichier doit rester versionné. {result.stderr}"
        )
