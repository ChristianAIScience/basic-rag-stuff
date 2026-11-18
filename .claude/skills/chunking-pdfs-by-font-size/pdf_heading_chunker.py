"""Split PDFs into chapter/section Documents by font size - self-contained template.

Dependencies:  pip install pdfplumber langchain-core

Usage:
    sections = load_pdf_sections(Path("manual.pdf"), {"source": "manual.pdf"}, max_heading_level=2)
    # each Document: page_content = heading + text, metadata = chapter, heading, heading_level, page
    # long sections: split further with RecursiveCharacterTextSplitter (metadata is copied to every chunk)

Inspect a PDF:  python pdf_heading_chunker.py manual.pdf [max_heading_level]
"""

import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from langchain_core.documents import Document

# Sizes >= body size * MIN_RATIO count as headings: ignores e.g. 11.5 pt emphasis next to 11 pt text,
# catches the usual 14 pt+ headings.
MIN_RATIO = 1.15


@dataclass
class TextLine:
    text: str
    size: float  # points
    page: int    # 1-based


def extract_lines(pdf_path: Path) -> list[TextLine]:
    """Each line with the font size used by most of its characters (one large initial is not a heading)."""
    import pdfplumber

    lines = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            for line in page.extract_text_lines():
                sizes = Counter(round(char["size"], 1) for char in line["chars"] if char["text"].strip())
                if sizes:
                    lines.append(TextLine(line["text"].strip(), sizes.most_common(1)[0][0], page.page_number))
    return lines


def find_body_size(lines: list[TextLine]) -> float:
    """The size covering the most CHARACTERS (headings are many short lines)."""
    characters = Counter()
    for line in lines:
        characters[line.size] += len(line.text)
    return characters.most_common(1)[0][0]


def find_heading_sizes(lines: list[TextLine], body_size: float, min_ratio: float = MIN_RATIO) -> list[float]:
    """Largest first: index 0 = level 1 (chapter). Levels are relative - a big cover title becomes level 1."""
    return sorted({line.size for line in lines if line.size >= body_size * min_ratio}, reverse=True)


def split_into_sections(lines: list[TextLine], heading_sizes: list[float], max_heading_level: int = 2) -> list[dict]:
    split_sizes = heading_sizes[:max_heading_level]
    sections: list[dict] = []
    chapter = ""
    current = {"heading": "", "heading_level": 0, "chapter": "", "page": lines[0].page if lines else 1,
               "lines": [], "has_body": False}   # text before the first heading -> level 0

    for line in lines:
        if line.size not in split_sizes:
            current["lines"].append(line.text)
            current["has_body"] = True
            continue
        level = split_sizes.index(line.size) + 1
        if current["heading_level"] == level and not current["has_body"]:   # heading wrapped over lines
            current["heading"] += " " + line.text
            current["lines"].append(line.text)
            if level == 1:
                chapter = current["chapter"] = current["heading"]
            continue
        if level == 1:
            chapter = line.text
        carried = [] if current["has_body"] else current["lines"]   # heading without body -> carry into next
        if current["has_body"]:
            sections.append(current)
        current = {"heading": line.text, "heading_level": level, "chapter": chapter, "page": line.page,
                   "lines": [*carried, line.text], "has_body": False}

    if current["lines"]:
        sections.append(current)
    return sections


def load_pdf_sections(pdf_path: Path, metadata: dict, min_ratio: float = MIN_RATIO,
                      max_heading_level: int = 2) -> list[Document]:
    lines = extract_lines(pdf_path)
    if not lines:
        return []  # scanned PDF without a text layer -> needs OCR
    heading_sizes = find_heading_sizes(lines, find_body_size(lines), min_ratio)
    return [
        Document(page_content="\n".join(s["lines"]),
                 metadata={**metadata, "chapter": s["chapter"], "heading": s["heading"],
                           "heading_level": s["heading_level"], "page": s["page"]})
        for s in split_into_sections(lines, heading_sizes, max_heading_level)
    ]


if __name__ == "__main__":
    path = Path(sys.argv[1])
    level = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    for section in load_pdf_sections(path, {"source": path.name}, max_heading_level=level):
        meta = section.metadata
        print(f"p.{meta['page']}  L{meta['heading_level']}  {len(section.page_content):>5} chars  {meta['heading'] or '(intro)'}")
