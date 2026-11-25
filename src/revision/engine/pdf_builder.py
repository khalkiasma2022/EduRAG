"""ProfessionalPDFBuilder : PDF pédagogique depuis le Markdown canonique V2."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fpdf import FPDF
from fpdf.enums import XPos, YPos
from fpdf.fonts import FontFace

from src.revision.engine.formatter import (
    MdCallout,
    MdDiagram,
    MdHeading,
    MdParagraph,
    MdQuestions,
    MdSynthesis,
    MdTable,
    MarkdownBlock,
    parse_markdown,
)

# Palette EduRAG — ardoise / teal
COLOR_SLATE = (47, 62, 70)
COLOR_TEAL = (13, 148, 136)
COLOR_SLATE_SOFT = (232, 236, 238)
COLOR_TEAL_SOFT = (224, 242, 240)
COLOR_AMBER_SOFT = (255, 247, 230)
COLOR_VIOLET_SOFT = (243, 240, 255)
COLOR_INK = (33, 37, 41)
COLOR_MUTED = (100, 116, 139)

AR_FONT_PATH = (
    Path(__file__).resolve().parents[3] / "assets" / "fonts" / "NotoNaskhArabic-Regular.ttf"
)
AR_FONT_FAMILY = "NotoNaskhArabic"

_LANGUAGE_LABELS = {"fr": "Français", "en": "Anglais", "ar": "Arabe"}
_LEVEL_LABELS = {
    "beginner": "Débutant",
    "intermediate": "Intermédiaire",
    "advanced": "Avancé",
}
_TYPE_LABELS = {"short": "Fiche courte", "complete": "Fiche complète"}
_TEXT_DIRECTIONS = {
    "fr": {"direction": "ltr", "script": "latn", "language": "fra", "align": "L"},
    "en": {"direction": "ltr", "script": "latn", "language": "eng", "align": "L"},
    "ar": {"direction": "rtl", "script": "arab", "language": "ara", "align": "R"},
}

_CALLOUT_STYLES = {
    "definition": {
        "label": "Définition",
        "fill": COLOR_TEAL_SOFT,
        "border": COLOR_TEAL,
    },
    "retain": {
        "label": "À retenir",
        "fill": COLOR_AMBER_SOFT,
        "border": (217, 119, 6),
    },
    "example_course": {
        "label": "Exemple (cours)",
        "fill": COLOR_SLATE_SOFT,
        "border": COLOR_SLATE,
    },
    "example_added": {
        "label": "Exemple (ajout pédagogique)",
        "fill": COLOR_VIOLET_SOFT,
        "border": (109, 40, 217),
    },
}


class ProfessionalPDFBuilder:
    """Construit un PDF professionnel à partir du Markdown stocké uniquement."""

    def build_from_markdown(
        self,
        markdown: str,
        output_path: str | Path,
        meta: dict[str, Any] | None = None,
    ) -> Path:
        if not isinstance(markdown, str) or markdown.strip() == "":
            raise ValueError("Le Markdown de la fiche est invalide.")
        path = Path(output_path)
        if path.suffix.lower() != ".pdf":
            raise ValueError("Le fichier de sortie doit être un fichier PDF (.pdf).")
        path.parent.mkdir(parents=True, exist_ok=True)

        info = _normalize_meta(meta)
        blocks = parse_markdown(markdown)
        pdf = _RevisionPDF(language=str(info["language"]))
        pdf.build(blocks, info)
        pdf.output(str(path))
        return path


def _normalize_meta(meta: dict[str, Any] | None) -> dict[str, Any]:
    data = dict(meta or {})
    return {
        "filename": str(data.get("filename") or "cours.pdf"),
        "page_start": int(data.get("page_start") or 1),
        "page_end": int(data.get("page_end") or data.get("page_start") or 1),
        "language": str(data.get("language") or "fr"),
        "level": str(data.get("level") or "beginner"),
        "sheet_type": str(data.get("sheet_type") or "complete"),
    }


class _RevisionPDF(FPDF):
    def __init__(self, language: str) -> None:
        super().__init__(format="A4")
        self.language = language if language in _TEXT_DIRECTIONS else "fr"
        self._course_label = ""
        self.set_auto_page_break(auto=True, margin=22)
        self.set_margins(18, 20, 18)
        if not AR_FONT_PATH.is_file():
            raise RuntimeError("La police Noto Naskh Arabic est introuvable.")
        self.add_font(AR_FONT_FAMILY, fname=str(AR_FONT_PATH))
        self._font = AR_FONT_FAMILY
        direction = _TEXT_DIRECTIONS[self.language]
        self.set_text_shaping(
            use_shaping_engine=True,
            direction=str(direction["direction"]),
            script=str(direction["script"]),
            language=str(direction["language"]),
        )
        self.align = str(direction["align"])

    def header(self) -> None:
        if self.page_no() == 1:
            return
        self.set_font(self._font, size=9)
        self.set_text_color(*COLOR_MUTED)
        self.cell(0, 8, text="EduRAG · Smart Revision", align=self.align)
        self.set_draw_color(*COLOR_TEAL)
        self.set_line_width(0.4)
        y = self.get_y() + 1
        self.line(self.l_margin, y, self.w - self.r_margin, y)
        self.ln(6)
        self.set_text_color(*COLOR_INK)

    def footer(self) -> None:
        self.set_y(-16)
        self.set_font(self._font, size=9)
        self.set_text_color(*COLOR_MUTED)
        label = self._course_label
        left = label if label else "EduRAG"
        self.cell(self.epw / 2, 8, text=left, align="L")
        self.cell(self.epw / 2, 8, text=str(self.page_no()), align="R")
        self.set_text_color(*COLOR_INK)

    def build(self, blocks: list[MarkdownBlock], meta: dict[str, Any]) -> None:
        self._course_label = str(meta["filename"])
        self._write_cover(meta)
        self.add_page()
        for block in blocks:
            self._write_block(block)

    def _write_cover(self, meta: dict[str, Any]) -> None:
        self.add_page()
        self.set_fill_color(*COLOR_SLATE)
        self.rect(0, 0, self.w, 48, style="F")
        self.set_fill_color(*COLOR_TEAL)
        self.rect(0, 48, self.w, 4, style="F")

        self.set_y(18)
        self.set_font(self._font, size=14)
        self.set_text_color(255, 255, 255)
        self.cell(0, 10, text="EduRAG", align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.set_font(self._font, size=11)
        self.cell(
            0,
            8,
            text="Smart Revision",
            align="C",
            new_x=XPos.LMARGIN,
            new_y=YPos.NEXT,
        )

        self.set_y(70)
        self.set_text_color(*COLOR_SLATE)
        self.set_font(self._font, size=22)
        self.multi_cell(
            0,
            12,
            text="Fiche de révision",
            align="C",
            new_x=XPos.LMARGIN,
            new_y=YPos.NEXT,
        )
        self.ln(4)
        self.set_draw_color(*COLOR_TEAL)
        self.set_line_width(0.8)
        mid = self.w / 2
        self.line(mid - 28, self.get_y(), mid + 28, self.get_y())
        self.ln(10)

        self.set_font(self._font, size=12)
        self.set_text_color(*COLOR_INK)
        pages = _format_pages(int(meta["page_start"]), int(meta["page_end"]))
        for line in (
            f"Cours : {meta['filename']}",
            f"Pages : {pages}",
            f"Langue : {_LANGUAGE_LABELS.get(str(meta['language']), str(meta['language']))}",
            f"Niveau : {_LEVEL_LABELS.get(str(meta['level']), str(meta['level']))}",
            f"Type : {_TYPE_LABELS.get(str(meta['sheet_type']), str(meta['sheet_type']))}",
        ):
            self.cell(0, 9, text=line, align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)

        self.set_y(self.h - 40)
        self.set_font(self._font, size=10)
        self.set_text_color(*COLOR_MUTED)
        self.cell(
            0,
            8,
            text="Document généré pour révision — contenu fidèle au cours",
            align="C",
            new_x=XPos.LMARGIN,
            new_y=YPos.NEXT,
        )
        self.set_text_color(*COLOR_INK)

    def _write_block(self, block: MarkdownBlock) -> None:
        if isinstance(block, MdHeading):
            self._write_heading(block.level, block.text)
            return
        if isinstance(block, MdParagraph):
            self._write_paragraph(block.text)
            return
        if isinstance(block, MdCallout):
            self._write_callout(block.kind, block.text)
            return
        if isinstance(block, MdTable):
            self._write_table(block.headers, block.rows)
            return
        if isinstance(block, MdDiagram):
            self._write_diagram(block.text)
            return
        if isinstance(block, MdSynthesis):
            self._write_heading(block.level, "Synthèse")
            for item in block.items:
                self._write_bullet(item)
            self.ln(2)
            return
        if isinstance(block, MdQuestions):
            self._write_heading(block.level, "Questions de révision")
            for number, item in enumerate(block.items, start=1):
                self._write_paragraph(f"{number}. {item}", size=11)
            self.ln(2)

    def _write_heading(self, level: int, text: str) -> None:
        sizes = {1: 18, 2: 15, 3: 13}
        size = sizes.get(level, 12)
        self.ln(2 if level > 1 else 1)
        self.set_font(self._font, size=size)
        self.set_text_color(*COLOR_SLATE if level == 1 else COLOR_TEAL if level == 2 else COLOR_SLATE)
        self.multi_cell(
            0,
            size * 0.55,
            text=text,
            align=self.align,
            new_x=XPos.LMARGIN,
            new_y=YPos.NEXT,
        )
        if level <= 2:
            self.set_draw_color(*(COLOR_TEAL if level == 2 else COLOR_SLATE))
            self.set_line_width(0.35)
            y = self.get_y() + 0.5
            self.line(self.l_margin, y, self.w - self.r_margin, y)
            self.ln(3)
        else:
            self.ln(2)
        self.set_text_color(*COLOR_INK)
        self.set_font(self._font, size=11)

    def _write_paragraph(self, text: str, size: int = 11) -> None:
        self.set_font(self._font, size=size)
        self.set_text_color(*COLOR_INK)
        self.multi_cell(
            0,
            6,
            text=text,
            align=self.align,
            new_x=XPos.LMARGIN,
            new_y=YPos.NEXT,
        )
        self.ln(1.5)

    def _write_bullet(self, text: str) -> None:
        self._write_paragraph(f"• {text}", size=11)

    def _write_callout(self, kind: str, text: str) -> None:
        style = _CALLOUT_STYLES[kind]
        label = str(style["label"])
        fill = style["fill"]
        border = style["border"]
        body = f"{label}. {text}" if not text.startswith(label) else text

        self.set_font(self._font, size=11)
        # Estimation de hauteur pour le fond
        line_height = 6
        usable = self.epw - 8
        # Découpe approximative pour réserver la boîte
        words = body.split()
        lines_est = 1
        current = ""
        for word in words:
            trial = f"{current} {word}".strip()
            if self.get_string_width(trial) <= usable:
                current = trial
            else:
                lines_est += 1
                current = word
        box_h = max(14.0, lines_est * line_height + 8)

        if self.get_y() + box_h > self.page_break_trigger:
            self.add_page()

        x = self.l_margin
        y = self.get_y()
        self.set_fill_color(*fill)
        self.set_draw_color(*border)
        self.set_line_width(0.6)
        self.rect(x, y, self.epw, box_h, style="DF")
        self.set_xy(x + 4, y + 3)
        self.set_text_color(*COLOR_SLATE)
        self.multi_cell(
            self.epw - 8,
            line_height,
            text=body,
            align=self.align,
            new_x=XPos.LMARGIN,
            new_y=YPos.NEXT,
        )
        self.set_y(y + box_h + 3)
        self.set_text_color(*COLOR_INK)

    def _write_diagram(self, text: str) -> None:
        self.set_fill_color(*COLOR_SLATE_SOFT)
        self.set_draw_color(*COLOR_SLATE)
        self.set_line_width(0.4)
        body = f"Schéma : {text}"
        self.set_font(self._font, size=11)
        height = 16
        if self.get_y() + height > self.page_break_trigger:
            self.add_page()
        x, y = self.l_margin, self.get_y()
        self.rect(x, y, self.epw, height, style="DF")
        self.set_xy(x + 4, y + 4)
        self.set_text_color(*COLOR_SLATE)
        self.multi_cell(
            self.epw - 8,
            6,
            text=body,
            align=self.align,
            new_x=XPos.LMARGIN,
            new_y=YPos.NEXT,
        )
        self.set_y(y + height + 3)
        self.set_text_color(*COLOR_INK)

    def _write_table(self, headers: list[str], rows: list[list[str]]) -> None:
        table_rows = [list(headers)] + [list(row) for row in rows]
        width = max(len(row) for row in table_rows)
        self.set_font(self._font, size=10)
        with self.table(
            text_align=self.align,
            width=self.epw,
            first_row_as_headings=True,
            headings_style=FontFace(emphasis="", color=COLOR_SLATE, fill_color=COLOR_TEAL_SOFT),
            line_height=7,
        ) as table:
            for row in table_rows:
                cells = row + [""] * (width - len(row))
                table.row(cells)
        self.ln(3)


def _format_pages(page_start: int, page_end: int) -> str:
    if page_start == page_end:
        return str(page_start)
    return f"{page_start} à {page_end}"
