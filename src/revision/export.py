"""Exporte une fiche de révision enregistrée en Markdown ou en PDF.

Le contenu généré est recopié dans son ordre logique.
Le PDF embarque Noto Naskh Arabic et façonne l'arabe avec HarfBuzz.
"""

import re
from pathlib import Path

from fpdf import FPDF
from fpdf.fonts import FontFace

_LANGUAGES = frozenset({"fr", "en", "ar"})
_LEVELS = frozenset({"beginner", "intermediate", "advanced"})
_SHEET_TYPES = frozenset({"short", "complete"})

_LANGUAGE_LABELS = {"fr": "Français", "en": "Anglais", "ar": "Arabe"}
_LEVEL_LABELS = {
    "beginner": "Débutant",
    "intermediate": "Intermédiaire",
    "advanced": "Avancé",
}
_TYPE_LABELS = {"short": "Fiche courte", "complete": "Fiche complète"}
_FONT_PATH = Path(__file__).resolve().parents[2] / "assets" / "fonts" / "NotoNaskhArabic-Regular.ttf"
_FONT_FAMILY = "NotoNaskhArabic"
_TEXT_DIRECTIONS = {
    "fr": {"direction": "ltr", "script": "latn", "language": "fra", "align": "L"},
    "en": {"direction": "ltr", "script": "latn", "language": "eng", "align": "L"},
    "ar": {"direction": "rtl", "script": "arab", "language": "ara", "align": "R"},
}
_HEADING = re.compile(r"^(#{1,3})\s+(\S.*)$")
_BULLET = re.compile(r"^\s*[-*]\s+(\S.*)$")
_NUMBERED = re.compile(r"^\s*(\d+[.)])\s+(\S.*)$")
_TABLE_SEPARATOR = re.compile(r"^:?-{3,}:?$")


class RevisionExport:
    """Prépare le Markdown d'une fiche, puis l'écrit dans un fichier."""

    def to_markdown(self, sheet: object) -> str:
        """Transforme le dictionnaire d'une fiche enregistrée en Markdown."""
        fields = _validate_sheet(sheet)
        pages = _format_pages(int(fields["page_start"]), int(fields["page_end"]))
        return "\n".join(
            [
                "# Fiche de révision",
                "",
                f"- Cours : {fields['filename']}",
                f"- Pages : {pages}",
                f"- Langue : {_LANGUAGE_LABELS[str(fields['language'])]}",
                f"- Niveau : {_LEVEL_LABELS[str(fields['level'])]}",
                f"- Type : {_TYPE_LABELS[str(fields['sheet_type'])]}",
                f"- Date de création : {fields['created_at']}",
                "",
                str(fields["content"]),
                "",
            ]
        )

    def export_markdown(self, sheet: object, output_path: str | Path) -> Path:
        """Écrit la fiche en UTF-8 et renvoie le chemin du fichier créé."""
        markdown = self.to_markdown(sheet)
        path = _check_output_path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(markdown, encoding="utf-8", newline="\n")
        return path

    def export_pdf(self, sheet: object, output_path: str | Path) -> Path:
        """Écrit la fiche dans un PDF et renvoie le chemin du fichier créé."""
        fields = _validate_sheet(sheet)
        path = _check_output_path(output_path, suffix=".pdf")
        path.parent.mkdir(parents=True, exist_ok=True)
        _write_pdf(fields, path)
        return path


def _validate_sheet(sheet: object) -> dict[str, str | int]:
    """Vérifie une fiche telle que RevisionStore.get() la renvoie."""
    if not isinstance(sheet, dict):
        raise ValueError("La fiche à exporter est invalide.")

    _check_sheet_id(sheet.get("id"))
    document_id = _required_line(sheet.get("document_id"), "L'identifiant du document est invalide.")
    filename = _required_line(sheet.get("filename"), "Le nom du fichier est invalide.")
    page_start = sheet.get("page_start")
    page_end = sheet.get("page_end")
    if not _is_page_number(page_start) or not _is_page_number(page_end) or page_end < page_start:
        raise ValueError("L'intervalle de pages est invalide.")
    language = _required_choice(sheet.get("language"), _LANGUAGES, "La langue demandée n'est pas prise en charge.")
    level = _required_choice(sheet.get("level"), _LEVELS, "Le niveau demandé n'est pas pris en charge.")
    sheet_type = _required_choice(
        sheet.get("sheet_type"),
        _SHEET_TYPES,
        "Le type de fiche demandé n'est pas pris en charge.",
    )
    created_at = _required_line(sheet.get("created_at"), "La date de création est invalide.")
    content = _required_text(sheet.get("content"), "Le contenu de la fiche est invalide.")
    return {
        "id": sheet["id"],
        "document_id": document_id,
        "filename": filename,
        "page_start": page_start,
        "page_end": page_end,
        "language": language,
        "level": level,
        "sheet_type": sheet_type,
        "created_at": created_at,
        "content": content.strip(),
    }


def _required_line(value: object, message: str) -> str:
    """Exige un texte d'une seule ligne, sans espaces superflus."""
    text = _required_text(value, message)
    if "\n" in text or "\r" in text:
        raise ValueError(message)
    return text


def _required_text(value: object, message: str) -> str:
    """Exige un texte qui contient autre chose que des espaces."""
    if not isinstance(value, str) or value.strip() == "":
        raise ValueError(message)
    return value.strip()


def _required_choice(value: object, accepted: frozenset[str], message: str) -> str:
    """Exige une valeur prévue, sans la corriger."""
    if not isinstance(value, str) or value not in accepted:
        raise ValueError(message)
    return value


def _is_page_number(value: object) -> bool:
    """Reconnaît un numéro de page. Un booléen n'en est pas un."""
    return not isinstance(value, bool) and isinstance(value, int) and value >= 1


def _check_sheet_id(sheet_id: object) -> None:
    """Refuse un identifiant qui ne vient pas de la base."""
    if isinstance(sheet_id, bool) or not isinstance(sheet_id, int) or sheet_id <= 0:
        raise ValueError("L'identifiant de la fiche est invalide.")


def _format_pages(page_start: int, page_end: int) -> str:
    """Décrit une page seule ou un intervalle."""
    if page_start == page_end:
        return str(page_start)
    return f"{page_start} à {page_end}"


def _check_output_path(output_path: object, suffix: str = ".md") -> Path:
    """Refuse un chemin vide ou dont l'extension n'est pas celle attendue."""
    if isinstance(output_path, str):
        if output_path.strip() == "":
            raise ValueError("Le chemin de sortie ne peut pas être vide.")
        path = Path(output_path)
    elif isinstance(output_path, Path):
        if str(output_path).strip() == "":
            raise ValueError("Le chemin de sortie ne peut pas être vide.")
        path = output_path
    else:
        raise TypeError("Le chemin de sortie doit être un chemin de fichier.")

    if path.suffix.lower() != suffix:
        if suffix == ".pdf":
            raise ValueError("Le fichier de sortie doit être un fichier PDF (.pdf).")
        raise ValueError("Le fichier de sortie doit être un fichier Markdown (.md).")
    if path.exists() and path.is_dir():
        raise ValueError("Le chemin de sortie doit désigner un fichier.")
    return path


def _write_pdf(fields: dict[str, str | int], path: Path) -> None:
    """Dessine la fiche. Le texte n'est jamais renversé à la main."""
    if not _FONT_PATH.is_file():
        raise RuntimeError("La police Noto Naskh Arabic est introuvable.")

    direction = _TEXT_DIRECTIONS[str(fields["language"])]
    align = str(direction["align"])
    pdf = FPDF(format="A4")
    pdf.set_auto_page_break(auto=True, margin=18)
    pdf.add_page()
    pdf.add_font(_FONT_FAMILY, fname=str(_FONT_PATH))
    pdf.set_font(_FONT_FAMILY, size=18)
    pdf.set_text_shaping(
        use_shaping_engine=True,
        direction=str(direction["direction"]),
        script=str(direction["script"]),
        language=str(direction["language"]),
    )
    _write_line(pdf, "Fiche de révision", align)
    pdf.ln(2)
    pdf.set_font(_FONT_FAMILY, size=11)
    for line in _metadata_lines(fields):
        _write_line(pdf, line, align)
    pdf.ln(4)
    pdf.set_font(_FONT_FAMILY, size=12)
    for block in _content_blocks(str(fields["content"])):
        _write_block(pdf, block, align)
    pdf.output(str(path))


def _metadata_lines(fields: dict[str, str | int]) -> list[str]:
    """Reprend les informations affichées dans l'export Markdown."""
    return [
        f"Cours : {fields['filename']}",
        f"Pages : {_format_pages(int(fields['page_start']), int(fields['page_end']))}",
        f"Langue : {_LANGUAGE_LABELS[str(fields['language'])]}",
        f"Niveau : {_LEVEL_LABELS[str(fields['level'])]}",
        f"Type : {_TYPE_LABELS[str(fields['sheet_type'])]}",
        f"Date de création : {fields['created_at']}",
    ]


def _write_line(pdf: FPDF, text: str, align: str, size: int | None = None) -> None:
    """Écrit un paragraphe qui passe à la ligne, puis à la page suivante si besoin."""
    if size is not None:
        pdf.set_font(_FONT_FAMILY, size=size)
    pdf.multi_cell(0, text=text, align=align, new_x="LMARGIN", new_y="NEXT")
    if size is not None:
        pdf.set_font(_FONT_FAMILY, size=12)


def _write_block(pdf: FPDF, block: tuple, align: str) -> None:
    """Écrit un titre, un paragraphe, une puce, une entrée numérotée ou un tableau."""
    kind = block[0]
    if kind == "heading":
        size = {1: 16, 2: 14, 3: 13}[block[1]]
        _write_line(pdf, str(block[2]), align, size=size)
        pdf.ln(1)
        return
    if kind == "item":
        _write_line(pdf, f"• {block[1]}", align)
        return
    if kind == "numbered":
        _write_line(pdf, f"{block[1]} {block[2]}", align)
        return
    if kind == "table":
        _write_table(pdf, block[1], align)
        pdf.ln(2)
        return
    _write_line(pdf, str(block[1]), align)
    pdf.ln(1)


def _write_table(pdf: FPDF, rows: list[list[str]], align: str) -> None:
    """Dessine un tableau simple, y compris lorsqu'il change de page."""
    width = max(len(row) for row in rows)
    with pdf.table(
        text_align=align,
        width=pdf.epw,
        first_row_as_headings=True,
        headings_style=FontFace(emphasis=""),
        line_height=7,
    ) as table:
        for row in rows:
            cells = row + [""] * (width - len(row))
            table.row(cells)


def _content_blocks(content: str) -> list[tuple]:
    """Découpe le Markdown simple produit pour une fiche."""
    blocks: list[tuple] = []
    table_rows: list[list[str]] = []

    def flush_table() -> None:
        if table_rows:
            blocks.append(("table", [list(row) for row in table_rows]))
            table_rows.clear()

    for line in content.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        cells = _table_cells(line)
        if cells is not None:
            if not _is_table_separator(cells):
                table_rows.append(cells)
            continue
        flush_table()
        stripped = line.strip()
        if stripped == "":
            continue
        heading = _HEADING.match(stripped)
        if heading:
            blocks.append(("heading", len(heading.group(1)), heading.group(2).strip()))
            continue
        numbered = _NUMBERED.match(line)
        if numbered:
            blocks.append(("numbered", numbered.group(1), numbered.group(2).strip()))
            continue
        bullet = _BULLET.match(line)
        if bullet:
            blocks.append(("item", bullet.group(1).strip()))
            continue
        blocks.append(("paragraph", stripped))
    flush_table()
    return blocks


def _table_cells(line: str) -> list[str] | None:
    """Reconnaît une ligne de tableau Markdown."""
    stripped = line.strip()
    if stripped.count("|") < 2:
        return None
    if stripped.startswith("|"):
        stripped = stripped[1:]
    if stripped.endswith("|"):
        stripped = stripped[:-1]
    return [cell.strip() for cell in stripped.split("|")]


def _is_table_separator(cells: list[str]) -> bool:
    """Ignore la ligne qui sépare l'en-tête des données."""
    return bool(cells) and all(_TABLE_SEPARATOR.fullmatch(cell) for cell in cells)
