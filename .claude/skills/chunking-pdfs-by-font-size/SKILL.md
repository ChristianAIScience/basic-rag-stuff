---
name: chunking-pdfs-by-font-size
description: Splits PDFs into chapter and section chunks by detecting headings from font sizes with pdfplumber. Use when ingesting PDFs, tuning heading detection, debugging wrong PDF sections, or writing tests with generated PDFs.
---

# Chunking PDFs by font size

Reference implementation: `rag_lab/ingestion/pdf_loader.py`, called by `loader.py` for every `.pdf`.
Demo without database or API key: `uv run python -m examples.pdf_chunking [path.pdf] [max_heading_level]`.
Sample PDF: `uv run python -m examples.create_sample_pdf`. Learner-facing explanation: `docs/08_pdf_chunking.md`.

## Algorithm

1. `extract_lines()`: `page.extract_text_lines()`; a line's size is the most common rounded size of its non-space characters (one large initial does not make a heading).
2. `find_body_size()`: the size covering the most **characters**, not lines - headings are many short lines.
3. `find_heading_sizes()`: sizes `>= body * min_ratio`, largest first, so index 0 is level 1. `min_ratio = 1.15` ignores near-body sizes such as 11.5 pt emphasis next to 11 pt text while catching the usual 14 pt+ headings.
4. `split_into_sections()`: a new section at every heading with level `<= max_heading_level` (default 2 = chapters and sections).
5. The normal text splitter cuts only sections longer than `CHUNK_SIZE`; `chapter`, `heading`, `heading_level`, `page` are copied to each chunk.

Edge cases that must keep working:
- Text before the first heading becomes its own section with `heading_level = 0`.
- A heading directly followed by a lower-level heading is carried into the next section instead of becoming a near-empty chunk.
- A heading wrapped over several lines (same size, no body text yet) is joined into one heading.
- Levels are relative: a large cover title becomes level 1 and pushes chapters to level 2.

## Change the heuristic

Copy this checklist:

```
- [ ] 1. Add a failing case to tests/unit/test_pdf_loader.py
- [ ] 2. Change pdf_loader.py
- [ ] 3. uv run python -m pytest   -> fix and repeat until green
- [ ] 4. examples.pdf_chunking on the sample PDF: 7 sections at level 2, 4 at level 1 (3 chapters + intro)
```

If step 4 yields other numbers, either fix the change or state why the new numbers are correct.

Testing patterns:
- Logic tests build `TextLine(text, size, page)` lists directly - no PDF needed.
- End-to-end tests generate PDFs with fpdf2 (dev dependency). Always call `multi_cell(0, height, text, new_x="LMARGIN", new_y="NEXT")`; without `new_x`/`new_y` fpdf2 raises "Not enough horizontal space".

Known limits: bold-only headings at body size, large headers/footers, multi-column layouts, scanned PDFs (no text layer → empty result).
