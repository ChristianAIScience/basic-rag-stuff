"""Show how a PDF is split into chapters/sections by font size. No database or API key needed.

Run:  uv run python -m examples.pdf_chunking
      uv run python -m examples.pdf_chunking path/to/your.pdf 1     (split at level-1 headings only)
"""

import sys
from collections import Counter
from pathlib import Path

from rag_lab.ingestion.pdf_loader import extract_lines, find_body_size, find_heading_sizes, load_pdf_sections
from rag_lab.ingestion.splitter import split_documents

DEFAULT_PDF = Path(__file__).resolve().parent.parent / "data" / "handbook" / "employee_handbook.pdf"
CHUNK_SIZE = 500  # same default as CHUNK_SIZE in .env


def main() -> None:
    pdf_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_PDF
    max_heading_level = int(sys.argv[2]) if len(sys.argv) > 2 else 2

    # Step 1-3: font size statistics -> body size and heading levels
    lines = extract_lines(pdf_path)
    body_size = find_body_size(lines)
    heading_sizes = find_heading_sizes(lines, body_size)

    print(f"PDF: {pdf_path.name}  ({len(lines)} text lines)\n")
    print("Font size   characters   role")
    characters_per_size = Counter()
    for line in lines:
        characters_per_size[line.size] += len(line.text)
    for size, count in sorted(characters_per_size.items(), reverse=True):
        if size == body_size:
            role = "body text"
        elif size in heading_sizes:
            level = heading_sizes.index(size) + 1
            role = f"heading level {level}" + ("  (split here)" if level <= max_heading_level else "  (kept as text)")
        else:
            role = "other"
        print(f"{size:>7.1f} pt  {count:>10}   {role}")

    # Step 4: sections
    sections = load_pdf_sections(pdf_path, {"source": pdf_path.name}, max_heading_level=max_heading_level)
    print(f"\n{len(sections)} sections:")
    for section in sections:
        meta = section.metadata
        heading = meta["heading"] or "(text before first heading)"
        print(f"  p.{meta['page']}  L{meta['heading_level']}  {len(section.page_content):>4} chars  {heading}")

    # Step 5: long sections are split further by the normal splitter - metadata is kept
    chunks = split_documents(sections, chunk_size=CHUNK_SIZE, chunk_overlap=50)
    print(f"\nAfter the text splitter (chunk_size={CHUNK_SIZE}): {len(chunks)} chunks")
    for chunk in chunks:
        preview = chunk.page_content.replace("\n", " ")[:60]
        print(f"  [{chunk.metadata['heading'][:28]:<28}] {preview}...")


if __name__ == "__main__":
    main()
