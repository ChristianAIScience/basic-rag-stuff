# 8. PDF-Chunking nach Schriftgröße (pdfplumber)

## Was

`rag_lab/ingestion/pdf_loader.py` teilt ein PDF an seinen Überschriften auf, z. B. in Kapitel und Abschnitte.
Überschriften werden an der **Schriftgröße** erkannt: PDFs haben keine Markdown-Zeichen wie `#`,
aber Überschriften sind fast immer größer gedruckt als der Fließtext.

## Warum

Der Standard-Splitter schneidet den Text alle ~500 Zeichen, egal was dort steht. Ein Chunk kann dann mitten in
einem Kapitel beginnen und im nächsten enden. Wird an Überschriften geteilt, bleiben Themen zusammen, und jeder
Chunk weiß, zu welchem Kapitel er gehört (Metadaten `chapter`, `heading`, `page`).

## Wie es funktioniert

pdfplumber liefert jedes Zeichen zusammen mit seiner Schriftgröße. Daraus:

```
1. extract_lines()        every text line + its font size (size used by most of its characters)
2. find_body_size()       size with the MOST characters            -> 11 pt = body text
3. find_heading_sizes()   sizes >= body * 1.15, largest first       -> [18 pt, 14 pt]
                          18 pt = level 1 (chapter), 14 pt = level 2 (section)
4. split_into_sections()  new section at each heading up to max_heading_level
5. split_documents()      (existing splitter) only sections longer than CHUNK_SIZE are cut further
```

Sonderfälle in `split_into_sections()`:
- **Text vor der ersten Überschrift** wird ein eigener Abschnitt (`heading_level = 0`).
- **Eine Kapitelüberschrift direkt vor einer Abschnittsüberschrift** wird kein fast leerer Chunk;
  sie wird dem ersten Abschnitt vorangestellt („2 Working at Lumora / 2.1 Onboarding / …").
- **Eine Überschrift über zwei Zeilen** (gleiche Größe, kein Fließtext dazwischen) wird zu einer Überschrift zusammengefügt.
- `max_heading_level=1` teilt nur an Kapiteln; Abschnitte bleiben dann als Text in ihrem Kapitel.

Jedes Abschnitts-Document trägt diese Metadaten:

```python
{"source": "handbook/employee_handbook.pdf", "category": "handbook",
 "chapter": "2 Working at Lumora", "heading": "2.2 Business Travel",
 "heading_level": 2, "page": 1}
```

## Wie es zusammenhängt

`loader.py` ruft `load_pdf_sections()` für jedes `.pdf` in `data/` auf. Alles danach
(Bereinigen, Aufteilen, IDs, Embeddings, Speichern) bleibt gleich, deshalb liest
`uv run python -m examples.ingest_documents` jetzt auch das Handbuch-PDF ein.
`source` und `category` sind SQL-Spalten; `chapter`, `heading` und `page` landen in der JSON-Metadatenspalte.
Um nach Kapiteln zu *filtern*, `"chapter"` zu `METADATA_COLUMNS` in `db/vector_store.py` hinzufügen und die Tabelle neu anlegen.

## Ausführen

```bash
uv run python -m examples.create_sample_pdf          # (re)creates data/handbook/employee_handbook.pdf
uv run python -m examples.pdf_chunking               # split at chapters + sections
uv run python -m examples.pdf_chunking data/handbook/employee_handbook.pdf 1   # chapters only
uv run python -m examples.pdf_chunking path/to/your.pdf                        # try your own PDF
```

Weder Datenbank noch API-Key nötig.

## Erwartete Ausgabe

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

## Grenzen

- Die Schriftgröße ist eine Heuristik. Überschriften, die nur **fett** in Fließtextgröße sind, werden nicht erkannt
  (zusätzlich könnte man `fontname` auf „Bold" prüfen).
- Große Kopf- oder Fußzeilen oder ein großer Titel auf dem Deckblatt zählen ebenfalls als Überschrift;
  die Ebenen sind *relativ* (die größte Schrift ist immer Ebene 1).
- Mehrspaltige Layouts können Zeilen aus verschiedenen Spalten vermischen.
- Gescannte PDFs haben keine Textebene; sie brauchen vorher eine Texterkennung (OCR).
