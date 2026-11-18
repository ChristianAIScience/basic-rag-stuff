"""Load a PDF and split it into sections at headings, detected by FONT SIZE.

Why? A PDF has no Markdown "#" markers - but headings are usually printed larger
than the body text. Splitting at headings keeps each chapter/section together,
which gives more meaningful chunks than cutting every N characters.

How it works (pdfplumber gives us every character with its font size):

  1. Read all text lines and the font size of each line.
  2. Body size   = the font size used for the MOST characters.
  3. Heading sizes = all sizes clearly larger than the body (>= body * min_ratio),
                   largest first:  20pt -> level 1 (chapter), 14pt -> level 2 (section)
  4. Start a new section at every heading up to `max_heading_level`.
     Smaller headings stay inside their parent section as normal text.

Each section becomes one Document. Very long sections are split further by the
normal text splitter later in the pipeline - the heading metadata is kept.

Limitations: font size is a heuristic. Bold headings in body size, large page
headers/footers or multi-column layouts can confuse it. Scanned PDFs have no text.
"""

from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import pdfplumber
from langchain_core.documents import Document


@dataclass
class TextLine:
    text: str
    size: float  # font size in points
    page: int    # 1-based page number


def extract_lines(pdf_path: Path) -> list[TextLine]:
    """Every text line of the PDF with the font size used by most of its characters."""
    lines = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            for line in page.extract_text_lines():
                sizes = Counter(round(char["size"], 1) for char in line["chars"] if char["text"].strip())
                if sizes:
                    lines.append(TextLine(line["text"].strip(), sizes.most_common(1)[0][0], page.page_number))
    return lines


def find_body_size(lines: list[TextLine]) -> float:
    """The font size that covers the most characters is the body text."""
    characters_per_size = Counter()
    for line in lines:
        characters_per_size[line.size] += len(line.text)
    return characters_per_size.most_common(1)[0][0]


def find_heading_sizes(lines: list[TextLine], body_size: float, min_ratio: float = 1.15) -> list[float]:
    """All font sizes clearly larger than the body, largest first. Index 0 = level 1."""
    return sorted({line.size for line in lines if line.size >= body_size * min_ratio}, reverse=True)


def split_into_sections(
    lines: list[TextLine], heading_sizes: list[float], max_heading_level: int = 2
) -> list[dict]:
    """Group lines into sections. A section starts at each heading of level <= max_heading_level.

    Each section dict has: heading, heading_level, chapter (latest level-1 heading), page, lines.
    """
    split_sizes = heading_sizes[:max_heading_level]
    sections: list[dict] = []
    chapter = ""
    # Text before the first heading (e.g. an intro on the title page) gets heading level 0.
    current = {"heading": "", "heading_level": 0, "chapter": "", "page": lines[0].page if lines else 1,
               "lines": [], "has_body": False}

    for line in lines:
        if line.size not in split_sizes:
            current["lines"].append(line.text)
            current["has_body"] = True
            continue

        level = split_sizes.index(line.size) + 1

        # A heading that wraps onto several lines: same size and no body text yet -> same heading.
        if current["heading_level"] == level and not current["has_body"]:
            current["heading"] += " " + line.text
            current["lines"].append(line.text)
            if level == 1:
                chapter = current["chapter"] = current["heading"]
            continue

        if level == 1:
            chapter = line.text

        # A heading without body text (e.g. "2 Chapter" directly followed by "2.1 Section")
        # is not emitted alone - its lines are carried into the next section as context.
        carried_lines = [] if current["has_body"] else current["lines"]
        if current["has_body"]:
            sections.append(current)

        current = {"heading": line.text, "heading_level": level, "chapter": chapter, "page": line.page,
                   "lines": [*carried_lines, line.text], "has_body": False}

    if current["lines"]:
        sections.append(current)
    return sections


def load_pdf_sections(
    pdf_path: Path,
    metadata: dict,
    min_ratio: float = 1.15,
    max_heading_level: int = 2,
) -> list[Document]:
    """Load one PDF as a list of section Documents (one per chapter/section)."""
    lines = extract_lines(pdf_path)
    if not lines:
        return []  # e.g. a scanned PDF without a text layer

    heading_sizes = find_heading_sizes(lines, find_body_size(lines), min_ratio)
    return [
        Document(
            page_content="\n".join(section["lines"]),
            metadata={
                **metadata,
                "chapter": section["chapter"],
                "heading": section["heading"],
                "heading_level": section["heading_level"],
                "page": section["page"],
            },
        )
        for section in split_into_sections(lines, heading_sizes, max_heading_level)
    ]
