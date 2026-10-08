#!/usr/bin/env python3
"""Convertit docs/FICHE_RESUME_ENCADRANTE.md en PDF (fpdf2)."""

from __future__ import annotations

import re
import sys
from pathlib import Path

from fpdf import FPDF

ROOT = Path(__file__).resolve().parents[1]
MD_PATH = ROOT / "docs" / "FICHE_RESUME_ENCADRANTE.md"
PDF_PATH = ROOT / "docs" / "FICHE_RESUME_ENCADRANTE.pdf"

# Polices Windows avec accents français
FONT_REG = "C:/Windows/Fonts/arial.ttf"
FONT_BOLD = "C:/Windows/Fonts/arialbd.ttf"
FONT_MONO = "C:/Windows/Fonts/consola.ttf"


class FichePDF(FPDF):
    def header(self):
        self.set_font("Arial", "I", 8)
        self.set_text_color(100, 100, 100)
        self.cell(0, 6, "EmoLearn — Fiche de synthèse", align="R")
        self.ln(8)

    def footer(self):
        self.set_y(-12)
        self.set_font("Arial", "I", 8)
        self.set_text_color(120, 120, 120)
        self.cell(0, 8, f"{self.page_no()}/{{nb}}", align="C")


def strip_md(text: str) -> str:
    text = text.replace("**", "")
    text = text.replace("`", "")
    text = text.replace("*", "")
    return text.strip()


def parse_table(lines: list[str], start: int) -> tuple[list[list[str]], int]:
    rows = []
    i = start
    while i < len(lines) and lines[i].strip().startswith("|"):
        raw = lines[i].strip()
        if re.match(r"^\|[\s\-:|]+\|$", raw):
            i += 1
            continue
        cells = [strip_md(c) for c in raw.strip("|").split("|")]
        rows.append(cells)
        i += 1
    return rows, i


def render_table(pdf: FichePDF, rows: list[list[str]]):
    if not rows:
        return
    n = max(len(r) for r in rows)
    for r in rows:
        while len(r) < n:
            r.append("")
    usable = pdf.epw
    # Largeur proportionnelle (première colonne un peu plus large si 2 cols)
    if n == 2:
        widths = [usable * 0.32, usable * 0.68]
    elif n == 3:
        widths = [usable * 0.40, usable * 0.30, usable * 0.30]
    elif n == 4:
        widths = [usable / 4] * 4
    elif n == 5:
        widths = [usable * 0.34] + [usable * 0.165] * 4
    elif n >= 6:
        widths = [usable * 0.14] + [(usable * 0.86) / (n - 1)] * (n - 1)
    else:
        widths = [usable / n] * n

    pdf.set_font("Arial", "B", 8)
    pdf.set_fill_color(235, 240, 245)
    for ri, row in enumerate(rows):
        if ri == 0:
            pdf.set_font("Arial", "B", 8)
            pdf.set_fill_color(230, 236, 242)
        else:
            pdf.set_font("Arial", "", 8)
            g = 248 if ri % 2 == 0 else 255
            pdf.set_fill_color(g, g, g)

        line_h = 4.2
        max_lines = 1
        for cell, w in zip(row, widths):
            cell_lines = pdf.multi_cell(w, line_h, cell, dry_run=True, output="LINES")
            max_lines = max(max_lines, len(cell_lines))
        row_h = max_lines * line_h + 1.5

        if pdf.get_y() + row_h > pdf.h - 15:
            pdf.add_page()

        x0 = pdf.l_margin
        y0 = pdf.get_y()
        for cell, w in zip(row, widths):
            pdf.set_xy(x0, y0)
            pdf.rect(x0, y0, w, row_h, style="DF")
            pdf.set_xy(x0 + 0.8, y0 + 0.6)
            pdf.multi_cell(w - 1.6, line_h, cell, border=0)
            x0 += w
        pdf.set_xy(pdf.l_margin, y0 + row_h)
    pdf.ln(2)


def _starts_block(line: str) -> bool:
    """True si la ligne commence un nouveau bloc (pas la suite d'un paragraphe)."""
    s = line.strip()
    if not s:
        return True
    if s.startswith("```") or s == "---":
        return True
    if line.startswith("# ") or line.startswith("## ") or line.startswith("### "):
        return True
    if s.startswith("|") or s.startswith("- "):
        return True
    return False


def convert(md_path: Path, pdf_path: Path):
    text = md_path.read_text(encoding="utf-8")
    lines = text.splitlines()

    pdf = FichePDF(format="A4")
    pdf.alias_nb_pages()
    pdf.set_auto_page_break(auto=True, margin=14)
    pdf.add_font("Arial", "", FONT_REG)
    pdf.add_font("Arial", "B", FONT_BOLD)
    pdf.add_font("Arial", "I", FONT_REG)
    pdf.add_font("Mono", "", FONT_MONO)
    pdf.add_page()
    pdf.set_text_color(30, 30, 30)

    i = 0
    in_code = False
    code_buf: list[str] = []

    while i < len(lines):
        line = lines[i]

        if line.strip().startswith("```"):
            if not in_code:
                in_code = True
                code_buf = []
            else:
                in_code = False
                pdf.set_font("Mono", "", 7.5)
                pdf.set_fill_color(245, 245, 245)
                block = "\n".join(code_buf)
                pdf.multi_cell(0, 3.8, block, fill=True)
                pdf.ln(2)
            i += 1
            continue

        if in_code:
            code_buf.append(line.replace("│", "|").replace("▼", "v").replace("→", "->"))
            i += 1
            continue

        if line.strip() == "---":
            pdf.ln(1)
            pdf.set_draw_color(180, 180, 180)
            pdf.line(pdf.l_margin, pdf.get_y(), pdf.w - pdf.r_margin, pdf.get_y())
            pdf.ln(3)
            i += 1
            continue

        if line.startswith("# "):
            pdf.set_font("Arial", "B", 16)
            pdf.multi_cell(0, 8, strip_md(line[2:]))
            pdf.ln(2)
            i += 1
            continue

        if line.startswith("## "):
            pdf.ln(2)
            pdf.set_font("Arial", "B", 12)
            pdf.set_text_color(20, 60, 100)
            pdf.multi_cell(0, 7, strip_md(line[3:]))
            pdf.set_text_color(30, 30, 30)
            pdf.ln(1)
            i += 1
            continue

        if line.startswith("### "):
            pdf.ln(1.5)
            pdf.set_font("Arial", "B", 10.5)
            pdf.set_text_color(40, 80, 120)
            pdf.multi_cell(0, 6, strip_md(line[4:]))
            pdf.set_text_color(30, 30, 30)
            pdf.ln(0.5)
            i += 1
            continue

        if line.strip().startswith("|"):
            rows, i = parse_table(lines, i)
            render_table(pdf, rows)
            continue

        if line.strip().startswith("- "):
            pdf.set_font("Arial", "", 9.5)
            pdf.set_x(pdf.l_margin + 3)
            pdf.multi_cell(pdf.epw - 3, 5, "- " + strip_md(line.strip()[2:]))
            pdf.set_x(pdf.l_margin)
            i += 1
            continue

        if not line.strip():
            pdf.ln(1.5)
            i += 1
            continue

        # Paragraphe Markdown = plusieurs lignes jusqu'à une ligne vide / un bloc.
        # Sans ça, chaque retour à la ligne du .md devient un paragraphe trop court dans le PDF.
        parts = [strip_md(line)]
        i += 1
        while i < len(lines) and not _starts_block(lines[i]):
            parts.append(strip_md(lines[i]))
            i += 1
        para = " ".join(p for p in parts if p)
        pdf.set_font("Arial", "", 9.5)
        pdf.set_x(pdf.l_margin)
        pdf.multi_cell(0, 5, para)

    pdf.output(str(pdf_path))
    print(f"PDF écrit : {pdf_path}")


if __name__ == "__main__":
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else MD_PATH
    dst = Path(sys.argv[2]) if len(sys.argv) > 2 else PDF_PATH
    convert(src, dst)
