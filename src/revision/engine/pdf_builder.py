"""ProfessionalPDFBuilder : PDF pédagogique depuis le Markdown canonique V2."""

from __future__ import annotations

import re
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

# ---------------------------------------------------------------------------
# Palette EduRAG — ardoise / teal (académique)
# ---------------------------------------------------------------------------
COLOR_SLATE = (31, 51, 64)  # #1F3340
COLOR_TEAL = (15, 118, 110)  # #0F766E
COLOR_SLATE_SOFT = (232, 236, 238)
COLOR_TEAL_SOFT = (224, 242, 240)
COLOR_AMBER_SOFT = (255, 247, 230)
COLOR_VIOLET_SOFT = (243, 240, 255)
COLOR_INK = (30, 41, 59)  # #1E293B
COLOR_MUTED = (100, 116, 139)  # #64748B
COLOR_RULE = (203, 213, 225)
COLOR_WHITE = (255, 255, 255)

# Espacement (mm)
MARGIN_LEFT = 20.0
MARGIN_TOP = 22.0
MARGIN_RIGHT = 20.0
MARGIN_BOTTOM = 24.0
GAP_AFTER_H1 = 8.0
GAP_AFTER_H2 = 5.0
GAP_AFTER_H3 = 3.0
GAP_PARAGRAPH = 3.0
GAP_CALLOUT = 4.0
ACCENT_BAR_W = 2.5
BODY_LINE_H = 6.2
BODY_SIZE = 11
CAPTION_SIZE = 9
META_SIZE = 10

# Échelle typographique
SIZE_H1 = 20
SIZE_H2 = 15
SIZE_H3 = 12.5

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
    created = data.get("created_at")
    return {
        "filename": str(data.get("filename") or "cours.pdf"),
        "page_start": int(data.get("page_start") or 1),
        "page_end": int(data.get("page_end") or data.get("page_start") or 1),
        "language": str(data.get("language") or "fr"),
        "level": str(data.get("level") or "beginner"),
        "sheet_type": str(data.get("sheet_type") or "complete"),
        "created_at": str(created).strip() if isinstance(created, str) and created.strip() else "",
    }


class _RevisionPDF(FPDF):
    def __init__(self, language: str) -> None:
        super().__init__(format="A4")
        self.language = language if language in _TEXT_DIRECTIONS else "fr"
        self._course_label = ""
        self.set_auto_page_break(auto=True, margin=MARGIN_BOTTOM)
        self.set_margins(MARGIN_LEFT, MARGIN_TOP, MARGIN_RIGHT)
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
        self.set_font(self._font, size=CAPTION_SIZE)
        self.set_text_color(*COLOR_MUTED)
        self.cell(0, 7, text="EduRAG · Smart Revision", align=self.align)
        self.set_draw_color(*COLOR_TEAL)
        self.set_line_width(0.45)
        y = self.get_y() + 1.5
        self.line(self.l_margin, y, self.w - self.r_margin, y)
        self.ln(8)
        self.set_text_color(*COLOR_INK)

    def footer(self) -> None:
        self.set_y(-18)
        self.set_draw_color(*COLOR_RULE)
        self.set_line_width(0.3)
        self.line(self.l_margin, self.get_y(), self.w - self.r_margin, self.get_y())
        self.ln(2)
        self.set_font(self._font, size=CAPTION_SIZE)
        self.set_text_color(*COLOR_MUTED)
        label = self._course_label
        left = label if label else "EduRAG"
        self.cell(self.epw / 2, 7, text=left, align="L")
        self.cell(self.epw / 2, 7, text=str(self.page_no()), align="R")
        self.set_text_color(*COLOR_INK)

    def build(self, blocks: list[MarkdownBlock], meta: dict[str, Any]) -> None:
        self._course_label = str(meta["filename"])
        self._write_cover(meta)
        self.add_page()
        for block in blocks:
            self._write_block(block)

    def _write_cover(self, meta: dict[str, Any]) -> None:
        self.add_page()
        band_h = 58.0
        self.set_fill_color(*COLOR_SLATE)
        self.rect(0, 0, self.w, band_h, style="F")
        self.set_fill_color(*COLOR_TEAL)
        self.rect(0, band_h, self.w, 3.5, style="F")

        self.set_y(20)
        self.set_font(self._font, size=16)
        self.set_text_color(*COLOR_WHITE)
        self.cell(0, 10, text="EduRAG", align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.set_font(self._font, size=11)
        self.set_text_color(204, 251, 241)
        self.cell(
            0,
            8,
            text="Smart Revision",
            align="C",
            new_x=XPos.LMARGIN,
            new_y=YPos.NEXT,
        )

        self.set_y(band_h + 22)
        self.set_text_color(*COLOR_SLATE)
        self.set_font(self._font, size=24)
        self.multi_cell(
            0,
            12,
            text="Fiche de révision",
            align="C",
            new_x=XPos.LMARGIN,
            new_y=YPos.NEXT,
        )
        self.ln(6)
        self.set_draw_color(*COLOR_TEAL)
        self.set_line_width(1.0)
        mid = self.w / 2
        self.line(mid - 22, self.get_y(), mid + 22, self.get_y())
        self.ln(14)

        pages = _format_pages(int(meta["page_start"]), int(meta["page_end"]))
        meta_lines: list[str] = [
            f"Cours : {meta['filename']}",
            f"Pages : {pages}",
            f"Langue : {_LANGUAGE_LABELS.get(str(meta['language']), str(meta['language']))}",
            f"Niveau : {_LEVEL_LABELS.get(str(meta['level']), str(meta['level']))}",
            f"Type : {_TYPE_LABELS.get(str(meta['sheet_type']), str(meta['sheet_type']))}",
        ]
        created_at = str(meta.get("created_at") or "").strip()
        if created_at:
            meta_lines.append(f"Date de création : {created_at}")
        self._write_meta_card(tuple(meta_lines))

        self.set_y(self.h - 40)
        self.set_font(self._font, size=META_SIZE)
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

    def _write_meta_card(self, lines: tuple[str, ...]) -> None:
        line_h = 8.0
        pad_y = 8.0
        pad_x = 10.0
        box_h = pad_y * 2 + line_h * len(lines)
        box_w = min(self.epw, 140.0)
        x = (self.w - box_w) / 2
        y = self.get_y()

        self.set_fill_color(*COLOR_SLATE_SOFT)
        self.set_draw_color(*COLOR_RULE)
        self.set_line_width(0.35)
        self.rect(x, y, box_w, box_h, style="DF")
        self.set_fill_color(*COLOR_TEAL)
        self.rect(x, y, ACCENT_BAR_W, box_h, style="F")

        self.set_xy(x + pad_x, y + pad_y)
        self.set_font(self._font, size=META_SIZE + 1)
        self.set_text_color(*COLOR_INK)
        for line in lines:
            self.set_x(x + pad_x)
            self.cell(
                box_w - pad_x * 2,
                line_h,
                text=line,
                align="C",
                new_x=XPos.LMARGIN,
                new_y=YPos.NEXT,
            )
        self.set_y(y + box_h + 6)

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
            self.ln(GAP_PARAGRAPH)
            return
        if isinstance(block, MdQuestions):
            self._write_heading(block.level, "Questions de révision")
            for number, item in enumerate(block.items, start=1):
                self._write_paragraph(f"{number}. {item}", size=BODY_SIZE)
            self.ln(GAP_PARAGRAPH)

    def _write_heading(self, level: int, text: str) -> None:
        sizes = {1: SIZE_H1, 2: SIZE_H2, 3: SIZE_H3}
        size = sizes.get(level, BODY_SIZE)
        gaps = {1: GAP_AFTER_H1, 2: GAP_AFTER_H2, 3: GAP_AFTER_H3}
        after = gaps.get(level, GAP_AFTER_H3)

        self.ln(5 if level == 1 else 3 if level == 2 else 2)

        if level == 1:
            self._write_h1(text, size)
        elif level == 2:
            self._write_h2(text, size)
        else:
            self.set_font(self._font, size=size)
            self.set_text_color(*COLOR_MUTED if level >= 3 else COLOR_SLATE)
            self.multi_cell(
                0,
                size * 0.55,
                text=text,
                align=self.align,
                new_x=XPos.LMARGIN,
                new_y=YPos.NEXT,
            )

        self.ln(after)
        self.set_text_color(*COLOR_INK)
        self.set_font(self._font, size=BODY_SIZE)

    def _write_h1(self, text: str, size: float) -> None:
        bar = ACCENT_BAR_W
        indent = bar + 3.0
        y0 = self.get_y()
        self.set_font(self._font, size=size)
        self.set_text_color(*COLOR_SLATE)
        self.set_x(self.l_margin + indent)
        self.multi_cell(
            self.epw - indent,
            size * 0.55,
            text=text,
            align=self.align,
            new_x=XPos.LMARGIN,
            new_y=YPos.NEXT,
        )
        y1 = self.get_y()
        self.set_fill_color(*COLOR_TEAL)
        self.rect(self.l_margin, y0, bar, max(6.0, y1 - y0), style="F")
        self.set_draw_color(*COLOR_RULE)
        self.set_line_width(0.4)
        underline_y = y1 + 1.2
        self.line(self.l_margin, underline_y, self.w - self.r_margin, underline_y)
        self.set_y(underline_y + 1.5)

    def _write_h2(self, text: str, size: float) -> None:
        self.set_font(self._font, size=size)
        self.set_text_color(*COLOR_TEAL)
        self.multi_cell(
            0,
            size * 0.55,
            text=text,
            align=self.align,
            new_x=XPos.LMARGIN,
            new_y=YPos.NEXT,
        )
        self.set_draw_color(*COLOR_TEAL)
        self.set_line_width(0.7)
        y = self.get_y() + 0.8
        underline_w = self.epw * 0.4
        if self.align == "R":
            x0 = self.w - self.r_margin - underline_w
            x1 = self.w - self.r_margin
        else:
            x0 = self.l_margin
            x1 = self.l_margin + underline_w
        self.line(x0, y, x1, y)
        self.set_y(y + 1.2)

    def _write_paragraph(self, text: str, size: float = BODY_SIZE) -> None:
        bullet = re.match(r"^\s*[-*]\s+(\S.*)$", text)
        if bullet:
            self._write_bullet(bullet.group(1).strip())
            return
        numbered = re.match(r"^\s*(\d+[.)])\s+(\S.*)$", text)
        if numbered:
            self.set_font(self._font, size=size)
            self.set_text_color(*COLOR_INK)
            indent = 4.0
            self.set_x(self.l_margin + indent)
            self.multi_cell(
                self.epw - indent,
                BODY_LINE_H,
                text=f"{numbered.group(1)} {numbered.group(2).strip()}",
                align=self.align,
                new_x=XPos.LMARGIN,
                new_y=YPos.NEXT,
            )
            self.ln(1.5)
            return

        self.set_font(self._font, size=size)
        self.set_text_color(*COLOR_INK)
        self.multi_cell(
            0,
            BODY_LINE_H,
            text=text,
            align=self.align,
            new_x=XPos.LMARGIN,
            new_y=YPos.NEXT,
        )
        self.ln(GAP_PARAGRAPH)

    def _write_bullet(self, text: str) -> None:
        indent = 4.0
        self.set_font(self._font, size=BODY_SIZE)
        self.set_text_color(*COLOR_INK)
        self.set_x(self.l_margin + indent)
        self.multi_cell(
            self.epw - indent,
            BODY_LINE_H,
            text=f"• {text}",
            align=self.align,
            new_x=XPos.LMARGIN,
            new_y=YPos.NEXT,
        )
        self.ln(1.5)

    def _write_accent_panel(
        self,
        *,
        body: str,
        fill: tuple[int, int, int],
        border: tuple[int, int, int],
        min_h: float = 14.0,
    ) -> None:
        self.set_font(self._font, size=BODY_SIZE)
        pad_x = 6.0
        pad_y = 4.5
        line_height = BODY_LINE_H
        usable = self.epw - pad_x * 2 - ACCENT_BAR_W
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
        box_h = max(min_h, lines_est * line_height + pad_y * 2)

        if self.get_y() + box_h > self.page_break_trigger:
            self.add_page()

        x = self.l_margin
        y = self.get_y()
        self.set_fill_color(*fill)
        self.set_draw_color(*COLOR_RULE)
        self.set_line_width(0.35)
        self.rect(x, y, self.epw, box_h, style="DF")
        self.set_fill_color(*border)
        self.rect(x, y, ACCENT_BAR_W, box_h, style="F")

        self.set_xy(x + ACCENT_BAR_W + pad_x - 1, y + pad_y)
        self.set_text_color(*COLOR_SLATE)
        self.multi_cell(
            self.epw - ACCENT_BAR_W - pad_x * 2 + 1,
            line_height,
            text=body,
            align=self.align,
            new_x=XPos.LMARGIN,
            new_y=YPos.NEXT,
        )
        self.set_y(y + box_h + GAP_CALLOUT)
        self.set_text_color(*COLOR_INK)

    def _write_callout(self, kind: str, text: str) -> None:
        style = _CALLOUT_STYLES[kind]
        label = str(style["label"])
        fill = style["fill"]
        border = style["border"]
        body = f"{label}. {text}" if not text.startswith(label) else text
        self._write_accent_panel(body=body, fill=fill, border=border)

    def _write_diagram(self, text: str) -> None:
        body = f"Schéma : {text}"
        self._write_accent_panel(
            body=body,
            fill=COLOR_SLATE_SOFT,
            border=COLOR_SLATE,
            min_h=16.0,
        )

    def _write_table(self, headers: list[str], rows: list[list[str]]) -> None:
        table_rows = [list(headers)] + [list(row) for row in rows]
        width = max(len(row) for row in table_rows)
        self.set_font(self._font, size=META_SIZE)
        with self.table(
            text_align=self.align,
            width=self.epw,
            first_row_as_headings=True,
            headings_style=FontFace(
                emphasis="",
                color=COLOR_SLATE,
                fill_color=COLOR_TEAL_SOFT,
            ),
            line_height=8,
        ) as table:
            for row in table_rows:
                cells = row + [""] * (width - len(row))
                table.row(cells)
        self.ln(GAP_CALLOUT)


def _format_pages(page_start: int, page_end: int) -> str:
    if page_start == page_end:
        return str(page_start)
    return f"{page_start} à {page_end}"
