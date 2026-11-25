"""ContentFormatter : IR → Markdown canonique et parse inverse pour le PDF."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from src.revision.engine.models import (
    ChapterDraft,
    DefinitionBlock,
    DiagramBlock,
    ExampleBlock,
    ParagraphBlock,
    PedagogicalDocument,
    QuestionBlock,
    RetainBlock,
    SynthesisBlock,
    TableBlock,
)

_CALLOUT_MARKERS = {
    "definition": "**Définition.**",
    "retain": "**À retenir.**",
    "example_course": "**Exemple (cours).**",
    "example_added": "**Exemple (ajout pédagogique).**",
}
_CALLOUT_PREFIX = re.compile(
    r"^>\s*\*\*(Définition|À retenir|Exemple \(cours\)|Exemple \(ajout pédagogique\))\.\*\*\s*(.*)$"
)
_HEADING = re.compile(r"^(#{1,3})\s+(\S.*)$")
_BULLET = re.compile(r"^\s*[-*]\s+(.*\S)\s*$")
_NUMBERED = re.compile(r"^\s*\d+[.)]\s+(.*\S)\s*$")
_TABLE_SEPARATOR = re.compile(r"^:?-{3,}:?$")
_DIAGRAM_PREFIX = "**Schéma :**"
_SYNTHESIS_TITLES = frozenset({"synthèse", "synthese"})
_QUESTIONS_TITLES = frozenset({"questions de révision", "questions de revision"})


@dataclass(frozen=True)
class MdHeading:
    level: int
    text: str


@dataclass(frozen=True)
class MdParagraph:
    text: str


@dataclass(frozen=True)
class MdCallout:
    kind: Literal["definition", "retain", "example_course", "example_added"]
    text: str


@dataclass(frozen=True)
class MdTable:
    headers: list[str]
    rows: list[list[str]]


@dataclass(frozen=True)
class MdDiagram:
    text: str


@dataclass(frozen=True)
class MdSynthesis:
    level: int
    items: list[str]


@dataclass(frozen=True)
class MdQuestions:
    level: int
    items: list[str]


MarkdownBlock = (
    MdHeading
    | MdParagraph
    | MdCallout
    | MdTable
    | MdDiagram
    | MdSynthesis
    | MdQuestions
)


class ContentFormatter:
    """Convertit un PedagogicalDocument en Markdown structuré déterministe."""

    def to_markdown(self, document: PedagogicalDocument) -> str:
        lines: list[str] = [
            f"# Fiche de révision — {document.filename}",
            "",
        ]
        for chapter in document.chapters:
            lines.extend(self._format_chapter(chapter))
        while lines and lines[-1] == "":
            lines.pop()
        return "\n".join(lines) + "\n"

    def _format_chapter(self, chapter: ChapterDraft) -> list[str]:
        lines: list[str] = [f"## {chapter.title}", ""]
        index = 0
        blocks = chapter.blocks
        while index < len(blocks):
            block = blocks[index]
            if isinstance(block, QuestionBlock):
                questions: list[str] = []
                while index < len(blocks) and isinstance(blocks[index], QuestionBlock):
                    questions.append(blocks[index].text.strip())
                    index += 1
                lines.append("## Questions de révision")
                lines.append("")
                for number, text in enumerate(questions, start=1):
                    lines.append(f"{number}. {text}")
                lines.append("")
                continue

            if isinstance(block, ParagraphBlock):
                lines.append(block.text.strip())
                lines.append("")
            elif isinstance(block, DefinitionBlock):
                lines.append(f"> {_CALLOUT_MARKERS['definition']} {block.text.strip()}")
                lines.append("")
            elif isinstance(block, RetainBlock):
                lines.append(f"> {_CALLOUT_MARKERS['retain']} {block.text.strip()}")
                lines.append("")
            elif isinstance(block, ExampleBlock):
                marker = (
                    _CALLOUT_MARKERS["example_course"]
                    if block.origin == "from_course"
                    else _CALLOUT_MARKERS["example_added"]
                )
                lines.append(f"> {marker} {block.text.strip()}")
                lines.append("")
            elif isinstance(block, TableBlock):
                lines.extend(_format_table(block.headers, block.rows))
                lines.append("")
            elif isinstance(block, DiagramBlock):
                lines.append(f"{_DIAGRAM_PREFIX} {block.text.strip()}")
                lines.append("")
            elif isinstance(block, SynthesisBlock):
                lines.append("### Synthèse")
                lines.append("")
                for item in _synthesis_items(block.text):
                    lines.append(f"- {item}")
                lines.append("")
            else:
                raise TypeError(f"Type de bloc non supporté : {type(block)!r}")
            index += 1
        return lines


def _synthesis_items(text: str) -> list[str]:
    cleaned = text.strip()
    if cleaned == "":
        return []
    parts = [part.strip() for part in cleaned.split("\n") if part.strip()]
    return parts if parts else [cleaned]


def _format_table(headers: list[str], rows: list[list[str]]) -> list[str]:
    if not headers:
        raise ValueError("Un tableau Markdown doit avoir des en-têtes.")
    width = len(headers)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        cells = list(row) + [""] * max(0, width - len(row))
        lines.append("| " + " | ".join(cells[:width]) + " |")
    return lines


def parse_markdown(markdown: str) -> list[MarkdownBlock]:
    """Reconstruit une structure logique depuis le seul Markdown sauvegardé."""
    if not isinstance(markdown, str):
        raise TypeError("Le Markdown doit être une chaîne de caractères.")
    text = markdown.replace("\r\n", "\n").replace("\r", "\n")
    lines = text.split("\n")
    blocks: list[MarkdownBlock] = []
    index = 0
    while index < len(lines):
        line = lines[index]
        stripped = line.strip()
        if stripped == "":
            index += 1
            continue

        heading = _HEADING.match(stripped)
        if heading:
            level = len(heading.group(1))
            title = heading.group(2).strip()
            title_key = _normalize_title(title)
            if title_key in _SYNTHESIS_TITLES:
                items, index = _collect_list_items(lines, index + 1)
                blocks.append(MdSynthesis(level=level, items=items))
                continue
            if title_key in _QUESTIONS_TITLES:
                items, index = _collect_list_items(lines, index + 1)
                blocks.append(MdQuestions(level=level, items=items))
                continue
            blocks.append(MdHeading(level=level, text=title))
            index += 1
            continue

        callout = _parse_callout(stripped)
        if callout is not None:
            kind, body = callout
            text_parts = [body] if body else []
            index += 1
            while index < len(lines):
                cont = lines[index].strip()
                if cont.startswith(">") and not _CALLOUT_PREFIX.match(cont):
                    text_parts.append(cont.lstrip(">").strip())
                    index += 1
                    continue
                break
            blocks.append(MdCallout(kind=kind, text=" ".join(p for p in text_parts if p)))
            continue

        if stripped.startswith(_DIAGRAM_PREFIX):
            blocks.append(MdDiagram(text=stripped[len(_DIAGRAM_PREFIX) :].strip()))
            index += 1
            continue

        table_cells = _table_cells(stripped)
        if table_cells is not None:
            rows: list[list[str]] = []
            while index < len(lines):
                cells = _table_cells(lines[index].strip())
                if cells is None:
                    break
                if not _is_table_separator(cells):
                    rows.append(cells)
                index += 1
            if not rows:
                continue
            headers = rows[0]
            data = rows[1:]
            blocks.append(MdTable(headers=headers, rows=data))
            continue

        blocks.append(MdParagraph(text=stripped))
        index += 1

    return blocks


def _normalize_title(title: str) -> str:
    return title.strip().casefold()


def _parse_callout(
    stripped: str,
) -> tuple[Literal["definition", "retain", "example_course", "example_added"], str] | None:
    match = _CALLOUT_PREFIX.match(stripped)
    if not match:
        return None
    label = match.group(1)
    body = match.group(2).strip()
    mapping: dict[str, Literal["definition", "retain", "example_course", "example_added"]] = {
        "Définition": "definition",
        "À retenir": "retain",
        "Exemple (cours)": "example_course",
        "Exemple (ajout pédagogique)": "example_added",
    }
    return mapping[label], body


def _collect_list_items(lines: list[str], start: int) -> tuple[list[str], int]:
    items: list[str] = []
    index = start
    while index < len(lines):
        stripped = lines[index].strip()
        if stripped == "":
            if items:
                # Autoriser une ligne vide interne puis reprendre ; stopper si bloc suivant.
                peek = index + 1
                while peek < len(lines) and lines[peek].strip() == "":
                    peek += 1
                if peek >= len(lines):
                    break
                nxt = lines[peek].strip()
                if (
                    _HEADING.match(nxt)
                    or nxt.startswith(">")
                    or nxt.startswith(_DIAGRAM_PREFIX)
                    or _table_cells(nxt) is not None
                ):
                    break
                index = peek
                continue
            index += 1
            continue
        numbered = _NUMBERED.match(lines[index])
        if numbered:
            items.append(numbered.group(1).strip())
            index += 1
            continue
        bullet = _BULLET.match(lines[index])
        if bullet:
            items.append(bullet.group(1).strip())
            index += 1
            continue
        if _HEADING.match(stripped) or stripped.startswith(">") or stripped.startswith(
            _DIAGRAM_PREFIX
        ):
            break
        # Paragraphe sous synthèse / questions
        items.append(stripped)
        index += 1
        break
    return items, index


def _table_cells(line: str) -> list[str] | None:
    stripped = line.strip()
    if stripped.count("|") < 2:
        return None
    if stripped.startswith("|"):
        stripped = stripped[1:]
    if stripped.endswith("|"):
        stripped = stripped[:-1]
    return [cell.strip() for cell in stripped.split("|")]


def _is_table_separator(cells: list[str]) -> bool:
    return bool(cells) and all(_TABLE_SEPARATOR.fullmatch(cell) for cell in cells)
