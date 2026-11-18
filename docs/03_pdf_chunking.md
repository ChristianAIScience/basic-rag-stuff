# 3. PDF Chunking by Font Size (pdfplumber)

## What

`rag_lab/ingestion/pdf_loader.py` splits a PDF at its headings, e.g. into chapters and
sections. Headings are detected by **font size**: PDFs have no Markdown `#` markers,
but headings are almost always printed larger than the body text.

## Why

The default splitter cuts text every ~500 characters, regardless of content. A chunk can then
start in the middle of one chapter and end in the next. Splitting at headings keeps topics together,
and every chunk knows which chapter it belongs to (metadata `chapter`, `heading`, `page`).

## How it works

pdfplumber returns every character together with its font size. From that:

```
1. extract_lines()        every text line + its font size (size used by most of its characters)
2. find_body_size()       size with the MOST characters            -> 11 pt = body text
3. find_heading_sizes()   sizes >= body * 1.15, largest first       -> [18 pt, 14 pt]
                          18 pt = level 1 (chapter), 14 pt = level 2 (section)
4. split_into_sections()  new section at each heading up to max_heading_level
5. split_documents()      (existing splitter) only sections longer than CHUNK_SIZE are cut further
```

Details handled in `split_into_sections()`:
- **Text before the first heading** becomes its own section (`heading_level = 0`).
- **A chapter heading directly followed by a section heading** is not emitted as a near-empty
  chunk; it is carried into the first section ("2 Working at Lumora / 2.1 Onboarding / …").
- **A heading that wraps onto two lines** (same size, no body text in between) is joined into one heading.
- `max_heading_level=1` splits only at chapters; sections then stay inside their chapter as text.

Each section Document carries this metadata:

```python
{"source": "handbook/employee_handbook.pdf", "category": "handbook",
 "chapter": "2 Working at Lumora", "heading": "2.2 Business Travel",
 "heading_level": 2, "page": 1}
```

## How it connects

`loader.py` calls `load_pdf_sections()` for every `.pdf` in `data/`. Everything after that
(cleaning, splitting, ids, embeddings, storage) is unchanged, so
`uv run python -m examples.ingest_documents` now ingests the handbook PDF too.
`source` and `category` are SQL columns; `chapter`, `heading` and `page` end up in the JSON
metadata column. To *filter* by chapter, add `"chapter"` to `METADATA_COLUMNS` in
`db/vector_store.py` and recreate the table.

## Run

```bash
uv run python -m examples.create_sample_pdf          # (re)creates data/handbook/employee_handbook.pdf
uv run python -m examples.pdf_chunking               # split at chapters + sections
uv run python -m examples.pdf_chunking data/handbook/employee_handbook.pdf 1   # chapters only
uv run python -m examples.pdf_chunking path/to/your.pdf                        # try your own PDF
```

No database and no API key needed.

## Expected output

```
Font size   characters   role
   18.0 pt         103   heading level 1  (split here)
   14.0 pt          89   heading level 2  (split here)
   11.0 pt        1870   body text

7 sections:
  p.1  L0   107 chars  (text before first heading)
  p.1  L2   269 chars  1.1 Our History
  p.1  L2   245 chars  1.2 Our Values
  p.1  L2   270 chars  2.1 Onboarding
  p.1  L2   748 chars  2.2 Business Travel
  p.2  L2   244 chars  3.1 Visitors
  p.2  L2   204 chars  3.2 Fire Safety

After the text splitter (chunk_size=500): 8 chunks     <- only "2.2 Business Travel" was cut
```

## Limitations

- Font size is a heuristic. Headings that are only **bold** at body size are not detected
  (you could additionally check `fontname` for "Bold").
- Large page headers/footers or a big title on the cover page also count as headings;
  heading levels are *relative* (the largest size is always level 1).
- Multi-column layouts can mix lines from different columns.
- Scanned PDFs have no text layer; they need OCR first.
