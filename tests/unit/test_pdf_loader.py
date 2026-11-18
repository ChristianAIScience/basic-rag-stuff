from fpdf import FPDF

from rag_lab.ingestion.loader import load_documents
from rag_lab.ingestion.pdf_loader import (
    TextLine,
    extract_lines,
    find_body_size,
    find_heading_sizes,
    load_pdf_sections,
    split_into_sections,
)

LINES = [
    TextLine("Intro text before any heading.", 11.0, 1),
    TextLine("1 Chapter One", 18.0, 1),
    TextLine("1.1 Section A", 14.0, 1),
    TextLine("Body of section A, long enough to dominate the character count.", 11.0, 1),
    TextLine("1.2 Section B", 14.0, 1),
    TextLine("Body of section B, also long enough to dominate the count.", 11.0, 1),
    TextLine("2 Chapter Two", 18.0, 2),
    TextLine("Body of chapter two without any sections at all.", 11.0, 2),
]


def test_body_size_is_the_size_with_most_characters():
    assert find_body_size(LINES) == 11.0


def test_heading_sizes_are_larger_sizes_sorted_largest_first():
    assert find_heading_sizes(LINES, body_size=11.0) == [18.0, 14.0]


def test_slightly_larger_text_is_not_a_heading():
    lines = [TextLine("body " * 20, 11.0, 1), TextLine("footnote-ish", 12.0, 1)]
    assert find_heading_sizes(lines, body_size=11.0, min_ratio=1.15) == []


def test_split_at_chapters_and_sections():
    sections = split_into_sections(LINES, [18.0, 14.0], max_heading_level=2)

    assert [(s["heading"], s["heading_level"], s["chapter"], s["page"]) for s in sections] == [
        ("", 0, "", 1),
        ("1.1 Section A", 2, "1 Chapter One", 1),
        ("1.2 Section B", 2, "1 Chapter One", 1),
        ("2 Chapter Two", 1, "2 Chapter Two", 2),
    ]


def test_chapter_heading_without_body_is_carried_into_first_section():
    first_section = split_into_sections(LINES, [18.0, 14.0])[1]
    assert first_section["lines"][:2] == ["1 Chapter One", "1.1 Section A"]


def test_max_heading_level_1_keeps_sections_inside_their_chapter():
    sections = split_into_sections(LINES, [18.0, 14.0], max_heading_level=1)

    assert [s["heading"] for s in sections] == ["", "1 Chapter One", "2 Chapter Two"]
    assert "1.2 Section B" in sections[1]["lines"]


def test_heading_wrapped_over_two_lines_is_one_heading():
    lines = [
        TextLine("1 A very long chapter title that", 18.0, 1),
        TextLine("wraps onto a second line", 18.0, 1),
        TextLine("Body text of the chapter.", 11.0, 1),
    ]
    (section,) = split_into_sections(lines, [18.0])
    assert section["heading"] == "1 A very long chapter title that wraps onto a second line"
    assert section["chapter"] == section["heading"]


def make_pdf(path, items):
    pdf = FPDF()
    pdf.add_page()
    for size, text in items:
        pdf.set_font("Helvetica", size=size)
        pdf.multi_cell(0, size * 0.55, text, new_x="LMARGIN", new_y="NEXT")
    pdf.output(str(path))


def test_real_pdf_is_split_into_sections_with_metadata(tmp_path):
    pdf_path = tmp_path / "guide.pdf"
    body = "This is ordinary body text that appears in the document. " * 3
    make_pdf(pdf_path, [(18, "1 Basics"), (11, body), (18, "2 Advanced"), (11, body)])

    assert {line.size for line in extract_lines(pdf_path)} == {18.0, 11.0}

    sections = load_pdf_sections(pdf_path, {"source": "guide.pdf"})
    assert [s.metadata["heading"] for s in sections] == ["1 Basics", "2 Advanced"]
    assert sections[0].metadata["source"] == "guide.pdf"
    assert sections[0].page_content.startswith("1 Basics\nThis is ordinary body text")


def test_loader_includes_pdf_sections(tmp_path):
    (tmp_path / "manuals").mkdir()
    make_pdf(tmp_path / "manuals" / "guide.pdf", [(18, "1 Basics"), (11, "Body text. " * 10)])

    (document,) = load_documents(tmp_path)

    assert document.metadata["source"] == "manuals/guide.pdf"
    assert document.metadata["category"] == "manuals"
    assert document.metadata["heading"] == "1 Basics"
