"""Convert a manuscript Markdown file to .docx for review circulation.

Written for CAFE_Hybrid_Manuscript.md but general enough for the other manuscripts in
this repository. Handles the constructs those files actually use -- headings, pipe
tables, bold/italic/code spans, blockquote figure callouts, horizontal rules and lists --
and **embeds the referenced figure images** rather than leaving a path string, so the
reviewer sees the plots inline.

    .venv/Scripts/python.exe scripts/md_to_docx.py CAFE_Hybrid_Manuscript.md

Deliberately not a general Markdown implementation: anything it does not recognise is
emitted as plain text rather than silently dropped, so nothing disappears from a document
someone is about to review.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

REPO = Path(__file__).resolve().parents[1]

BODY_FONT = "Times New Roman"
MONO_FONT = "Consolas"
BODY_PT = 11
ACCENT = RGBColor(0x1F, 0x3B, 0x63)

# **bold**, *italic*, `code`, and [text](url) -- captured so the split keeps delimiters.
INLINE = re.compile(r"(\*\*.+?\*\*|(?<!\*)\*[^*]+?\*(?!\*)|`[^`]+`|\[[^\]]+\]\([^)]+\))")

# Backslash-escaped punctuation is parked on a sentinel before the inline pass, so an
# escaped asterisk cannot be mistaken for emphasis, then restored as a literal character.
_ESC = re.compile(r"\\([*_`|\[\]\\])")
_SENT = ""


def _park(text: str) -> str:
    return _ESC.sub(lambda m: _SENT + str(ord(m.group(1))) + _SENT, text)


def _unpark(text: str) -> str:
    return re.sub(_SENT + r"(\d+)" + _SENT, lambda m: chr(int(m.group(1))), text)


def add_runs(par, text: str) -> None:
    """Emit `text` into `par`, honouring inline bold / italic / code / link markup."""
    for tok in INLINE.split(_park(text)):
        if not tok:
            continue
        if tok.startswith("**") and tok.endswith("**") and len(tok) > 4:
            par.add_run(_unpark(tok[2:-2])).bold = True
        elif tok.startswith("`") and tok.endswith("`") and len(tok) > 2:
            r = par.add_run(_unpark(tok[1:-1]))
            r.font.name = MONO_FONT
            r.font.size = Pt(BODY_PT - 1.5)
        elif tok.startswith("[") and "](" in tok:
            par.add_run(_unpark(tok[1: tok.index("](")])).underline = True
        elif tok.startswith("*") and tok.endswith("*") and len(tok) > 2:
            par.add_run(_unpark(tok[1:-1])).italic = True
        else:
            par.add_run(_unpark(tok))


def split_row(line: str) -> list[str]:
    """Split a Markdown table row on unescaped pipes."""
    parts, cur, esc = [], "", False
    for ch in line.strip().strip("|"):
        if esc:
            cur += ch
            esc = False
        elif ch == "\\":
            esc = True
        elif ch == "|":
            parts.append(cur)
            cur = ""
        else:
            cur += ch
    parts.append(cur)
    return [p.strip() for p in parts]


def shade(cell, hex_fill: str) -> None:
    el = OxmlElement("w:shd")
    el.set(qn("w:val"), "clear")
    el.set(qn("w:fill"), hex_fill)
    cell._tc.get_or_add_tcPr().append(el)


def add_table(doc: Document, rows: list[str]) -> None:
    header = split_row(rows[0])
    body = [split_row(r) for r in rows[2:]]          # rows[1] is the |---|---| rule
    ncol = len(header)
    t = doc.add_table(rows=1, cols=ncol)
    t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, h in enumerate(header):
        cell = t.rows[0].cells[i]
        cell.text = ""
        p = cell.paragraphs[0]
        p.paragraph_format.space_before = Pt(2)
        p.paragraph_format.space_after = Pt(2)
        add_runs(p, h)
        for r in p.runs:
            r.bold = True
            r.font.size = Pt(BODY_PT - 1.5)
        shade(cell, "DCE6F1")
    for brow in body:
        cells = t.add_row().cells
        for i in range(ncol):
            val = brow[i] if i < len(brow) else ""
            cells[i].text = ""
            p = cells[i].paragraphs[0]
            p.paragraph_format.space_before = Pt(1)
            p.paragraph_format.space_after = Pt(1)
            add_runs(p, val)
            for r in p.runs:
                r.font.size = Pt(BODY_PT - 1.5)
    doc.add_paragraph()


def add_quote(doc: Document, text: str) -> None:
    """A blockquote that is NOT a figure callout -- e.g. the decision rule in §3.7.

    Rendered as an indented, lightly-shaded pull-quote. Centring and italicising this the
    way a figure caption is styled would misrepresent a normative statement as a caption.
    """
    p = doc.add_paragraph()
    p.paragraph_format.left_indent = Inches(0.4)
    p.paragraph_format.right_indent = Inches(0.4)
    p.paragraph_format.space_before = Pt(6)
    p.paragraph_format.space_after = Pt(10)
    add_runs(p, text)
    pr = p._p.get_or_add_pPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:fill"), "F2F5F9")
    pr.append(shd)


def add_figure(doc: Document, text: str) -> None:
    """Blockquote figure callout: embed the image if the referenced file exists."""
    m = re.search(r"`([^`]+\.(?:png|jpg|jpeg))`", text)
    if m:
        path = REPO / m.group(1)
        if path.exists():
            doc.add_picture(str(path), width=Inches(6.0))
            doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    cap = doc.add_paragraph()
    cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    cap.paragraph_format.space_after = Pt(12)
    # Drop the path token from the caption -- the image is now inline above it.
    clean = re.sub(r"\s*`[^`]+\.(?:png|jpg|jpeg)`\s*(--|—)?\s*", " ", text).strip()
    add_runs(cap, clean)
    for r in cap.runs:
        r.font.size = Pt(BODY_PT - 2)
        r.italic = True


def convert(md_path: Path, out_path: Path) -> None:
    lines = md_path.read_text(encoding="utf-8").splitlines()

    doc = Document()
    st = doc.styles["Normal"]
    st.font.name = BODY_FONT
    st.font.size = Pt(BODY_PT)
    st.element.rPr.rFonts.set(qn("w:eastAsia"), BODY_FONT)
    st.paragraph_format.space_after = Pt(7)
    st.paragraph_format.line_spacing = 1.15
    for s in doc.sections:
        s.left_margin = s.right_margin = Inches(1.0)
        s.top_margin = s.bottom_margin = Inches(0.9)

    i = 0
    while i < len(lines):
        raw = lines[i]
        line = raw.rstrip()
        stripped = line.strip()

        if not stripped:
            i += 1
            continue

        # ---- horizontal rule -> spacing only (avoids a page full of grey bars)
        if re.fullmatch(r"-{3,}|\*{3,}", stripped):
            doc.add_paragraph()
            i += 1
            continue

        # ---- headings
        m = re.match(r"^(#{1,6})\s+(.*)$", stripped)
        if m:
            lvl, text = len(m.group(1)), m.group(2)
            if lvl == 1:
                p = doc.add_paragraph()
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                add_runs(p, text)
                for r in p.runs:
                    r.bold = True
                    r.font.size = Pt(17)
                    r.font.color.rgb = ACCENT
                p.paragraph_format.space_after = Pt(14)
            else:
                h = doc.add_heading(level=min(lvl, 4))
                h.text = ""
                add_runs(h, text)
                for r in h.runs:
                    r.font.color.rgb = ACCENT
                    r.font.name = BODY_FONT
                    r.font.size = Pt({2: 14, 3: 12, 4: 11}.get(lvl, 11))
                h.paragraph_format.space_before = Pt(13 if lvl == 2 else 10)
                h.paragraph_format.space_after = Pt(5)
            i += 1
            continue

        # ---- table: a pipe row followed by a |---| separator
        if (stripped.startswith("|") and i + 1 < len(lines)
                and re.match(r"^\s*\|[\s:|-]+\|\s*$", lines[i + 1])):
            block = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                block.append(lines[i].strip())
                i += 1
            add_table(doc, block)
            continue

        # ---- blockquote (figure callouts in these manuscripts)
        if stripped.startswith(">"):
            block = []
            while i < len(lines) and lines[i].strip().startswith(">"):
                block.append(lines[i].strip().lstrip(">").strip())
                i += 1
            joined = " ".join(block)
            if re.search(r"\*\*Figure\b", joined) or re.search(
                    r"`[^`]+\.(?:png|jpg|jpeg)`", joined):
                add_figure(doc, joined)
            else:
                add_quote(doc, joined)
            continue

        # ---- list item (may wrap onto continuation lines)
        m = re.match(r"^(\s*)([-*]|\d+\.)\s+(.*)$", raw)
        if m:
            indent, marker, text = m.group(1), m.group(2), m.group(3)
            i += 1
            while (i < len(lines) and lines[i].strip()
                   and not re.match(r"^\s*([-*]|\d+\.)\s+", lines[i])
                   and not lines[i].strip().startswith(("#", "|", ">"))
                   and lines[i].startswith((" ", "\t"))):
                text += " " + lines[i].strip()
                i += 1
            style = "List Number" if marker[0].isdigit() else "List Bullet"
            p = doc.add_paragraph(style=style)
            if len(indent) >= 2:
                p.paragraph_format.left_indent = Inches(0.65)
            add_runs(p, text)
            continue

        # ---- paragraph: join wrapped lines until a blank or a block starter
        buf = [stripped]
        i += 1
        while i < len(lines):
            nxt = lines[i].strip()
            if (not nxt or nxt.startswith(("#", "|", ">"))
                    or re.fullmatch(r"-{3,}|\*{3,}", nxt)
                    or re.match(r"^\s*([-*]|\d+\.)\s+", lines[i])):
                break
            buf.append(nxt)
            i += 1
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        add_runs(p, " ".join(buf))

    doc.save(str(out_path))


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    md = Path(sys.argv[1])
    if not md.is_absolute():
        md = REPO / md
    if not md.exists():
        print(f"not found: {md}")
        return 1
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else md.with_suffix(".docx")
    if not out.is_absolute():
        out = REPO / out
    convert(md, out)
    kb = out.stat().st_size / 1024
    print(f"wrote {out.relative_to(REPO)}  ({kb:.0f} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
