"""Utilitários de formatação do relatório em .docx (estilo ABNT)."""

from __future__ import annotations

from pathlib import Path

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

FONT = "Times New Roman"


def new_document() -> Document:
    doc = Document()
    section = doc.sections[0]
    section.page_height = Cm(29.7)
    section.page_width = Cm(21.0)
    section.top_margin = Cm(3.0)
    section.bottom_margin = Cm(2.0)
    section.left_margin = Cm(3.0)
    section.right_margin = Cm(2.0)
    section.different_first_page_header_footer = True

    normal = doc.styles["Normal"]
    normal.font.name = FONT
    normal.font.size = Pt(12)
    normal.element.rPr.rFonts.set(qn("w:eastAsia"), FONT)
    lang = OxmlElement("w:lang")
    lang.set(qn("w:val"), "pt-BR")
    lang.set(qn("w:eastAsia"), "pt-BR")
    normal.element.rPr.append(lang)
    normal.paragraph_format.line_spacing = 1.5
    normal.paragraph_format.space_after = Pt(6)
    normal.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY

    doc.core_properties.title = "Q-learning em ambientes Toy Text do Gymnasium"
    doc.core_properties.author = "Expedito Hebert"
    doc.core_properties.subject = "Relatório de experimentos – Inteligência Artificial para Jogos I"
    doc.core_properties.language = "pt-BR"
    doc.core_properties.keywords = "Q-learning; aprendizado por reforço; Gymnasium; Toy Text"

    for name, size, before, after in [
        ("Heading 1", 14, 18, 10),
        ("Heading 2", 12, 14, 8),
        ("Heading 3", 12, 12, 6),
    ]:
        style = doc.styles[name]
        style.font.name = FONT
        style.font.size = Pt(size)
        style.font.bold = True
        style.font.color.rgb = RGBColor(0, 0, 0)
        style.element.rPr.rFonts.set(qn("w:eastAsia"), FONT)
        style.paragraph_format.space_before = Pt(before)
        style.paragraph_format.space_after = Pt(after)
        style.paragraph_format.keep_with_next = True

    # Numeração de páginas no rodapé.
    footer = section.footer.paragraphs[0]
    footer.alignment = WD_ALIGN_PARAGRAPH.CENTER
    footer.add_run("Página ")
    _add_field(footer, "PAGE")

    # Cabeçalho discreto.
    header = section.header.paragraphs[0]
    header.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    run = header.add_run("Inteligência Artificial para Jogos I")
    run.font.size = Pt(9)
    run.font.name = FONT
    run.font.color.rgb = RGBColor(90, 90, 90)

    # O sumário é um campo TOC. Ele não é marcado para atualização automática,
    # pois a atualização automática na abertura bloqueia a exportação para PDF
    # no Microsoft Word. A atualização é feita no passo de conversão.
    return doc


def _add_field(paragraph, field: str) -> None:
    run = paragraph.add_run()
    begin = OxmlElement("w:fldChar")
    begin.set(qn("w:fldCharType"), "begin")
    instruction = OxmlElement("w:instrText")
    instruction.set(qn("xml:space"), "preserve")
    instruction.text = field
    end = OxmlElement("w:fldChar")
    end.set(qn("w:fldCharType"), "end")
    run._r.append(begin)
    run._r.append(instruction)
    run._r.append(end)


def add_toc(doc: Document) -> None:
    paragraph = doc.add_paragraph()
    run = paragraph.add_run()
    begin = OxmlElement("w:fldChar")
    begin.set(qn("w:fldCharType"), "begin")
    run._r.append(begin)

    instruction_run = paragraph.add_run()
    instruction = OxmlElement("w:instrText")
    instruction.set(qn("xml:space"), "preserve")
    instruction.text = 'TOC \\o "1-2" \\z \\u'
    instruction_run._r.append(instruction)

    separator_run = paragraph.add_run()
    separator = OxmlElement("w:fldChar")
    separator.set(qn("w:fldCharType"), "separate")
    separator_run._r.append(separator)
    separator_run.add_text("Atualize o sumário no Word (Ctrl+A e F9).")

    end_run = paragraph.add_run()
    end = OxmlElement("w:fldChar")
    end.set(qn("w:fldCharType"), "end")
    end_run._r.append(end)


def heading(doc: Document, text: str, level: int = 1, page_break: bool = False) -> None:
    para = doc.add_heading(text, level=level)
    if page_break:
        para.paragraph_format.page_break_before = True


def paragraph(
    doc: Document,
    text: str,
    bold: bool = False,
    italic: bool = False,
    indent: bool = True,
    align=WD_ALIGN_PARAGRAPH.JUSTIFY,
    space_after: int = 6,
    size: int | None = None,
) -> None:
    para = doc.add_paragraph()
    para.alignment = align
    if indent:
        para.paragraph_format.first_line_indent = Cm(1.25)
    para.paragraph_format.space_after = Pt(space_after)
    run = para.add_run(text)
    run.bold = bold
    run.italic = italic
    if size:
        run.font.size = Pt(size)


def bullet(doc: Document, text: str) -> None:
    para = doc.add_paragraph(style="List Bullet")
    para.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    para.paragraph_format.line_spacing = 1.5
    para.paragraph_format.space_after = Pt(3)
    para.add_run(text)


def equation(doc: Document, text: str) -> None:
    para = doc.add_paragraph()
    para.alignment = WD_ALIGN_PARAGRAPH.CENTER
    para.paragraph_format.space_before = Pt(6)
    para.paragraph_format.space_after = Pt(6)
    run = para.add_run(text)
    run.italic = True
    run.font.name = FONT
    run.font.size = Pt(12)


def caption(doc: Document, text: str) -> None:
    para = doc.add_paragraph()
    para.alignment = WD_ALIGN_PARAGRAPH.LEFT
    para.paragraph_format.space_before = Pt(8)
    para.paragraph_format.space_after = Pt(2)
    para.paragraph_format.keep_with_next = True
    label, _, rest = text.partition(" – ")
    bold_run = para.add_run(label + " – ")
    bold_run.bold = True
    bold_run.font.size = Pt(10)
    normal_run = para.add_run(rest)
    normal_run.font.size = Pt(10)
    return para


def source_note(doc: Document, text: str = "Fonte: elaboração própria (2026).") -> None:
    para = doc.add_paragraph()
    para.alignment = WD_ALIGN_PARAGRAPH.LEFT
    para.paragraph_format.space_before = Pt(2)
    para.paragraph_format.space_after = Pt(10)
    run = para.add_run(text)
    run.font.size = Pt(9)
    run.italic = True


def figure(doc: Document, path: Path, text: str, width_cm: float = 16.0) -> None:
    caption(doc, text)
    para = doc.add_paragraph()
    para.alignment = WD_ALIGN_PARAGRAPH.CENTER
    para.paragraph_format.space_after = Pt(2)
    para.add_run().add_picture(str(path), width=Cm(width_cm))
    source_note(doc)


def table(
    doc: Document,
    headers: list[str],
    rows: list[list[str]],
    text: str,
    widths: list[float] | None = None,
    font_size: int = 9,
) -> None:
    caption(doc, text)
    table_obj = doc.add_table(rows=1, cols=len(headers))
    table_obj.style = "Table Grid"
    table_obj.alignment = WD_TABLE_ALIGNMENT.CENTER
    table_obj.autofit = False

    header_cells = table_obj.rows[0].cells
    for index, header_text in enumerate(headers):
        cell = header_cells[index]
        cell.text = ""
        para = cell.paragraphs[0]
        para.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = para.add_run(header_text)
        run.bold = True
        run.font.size = Pt(font_size)
        run.font.name = FONT
        _shade(cell, "D9E2F3")

    for row_values in rows:
        cells = table_obj.add_row().cells
        for index, value in enumerate(row_values):
            cell = cells[index]
            cell.text = ""
            para = cell.paragraphs[0]
            para.alignment = (
                WD_ALIGN_PARAGRAPH.LEFT if index == 0 else WD_ALIGN_PARAGRAPH.CENTER
            )
            run = para.add_run(value)
            run.font.size = Pt(font_size)
            run.font.name = FONT

    if widths:
        for row in table_obj.rows:
            for index, width in enumerate(widths):
                row.cells[index].width = Cm(width)

    # Evita que uma linha seja dividida entre páginas e repete o cabeçalho
    # quando a tabela continua na página seguinte.
    for row_index, row in enumerate(table_obj.rows):
        tr_pr = row._tr.get_or_add_trPr()
        cant_split = OxmlElement("w:cantSplit")
        tr_pr.append(cant_split)
        if row_index == 0:
            tr_pr.append(OxmlElement("w:tblHeader"))

    source_note(doc)


def _shade(cell, color: str) -> None:
    shading = OxmlElement("w:shd")
    shading.set(qn("w:fill"), color)
    cell._tc.get_or_add_tcPr().append(shading)


def page_break(doc: Document) -> None:
    doc.add_page_break()


def references(doc: Document, items: list[str]) -> None:
    heading(doc, "Referências", level=1)
    for item in items:
        para = doc.add_paragraph()
        para.alignment = WD_ALIGN_PARAGRAPH.LEFT
        para.paragraph_format.left_indent = Cm(0)
        para.paragraph_format.first_line_indent = Cm(0)
        para.paragraph_format.space_after = Pt(10)
        run = para.add_run(item)
        run.font.size = Pt(11)
