# 4. Retrieval-Strategien

Jede Strategie liegt in einem eigenen Modul in `rag_lab/retrieval/` und hat dieselbe Form:
`strategie(vector_store, query, ...) -> list[(Document, score)]`.
`build_strategies()` in `retrieval/comparison.py` führt alle fünf mit ihren Einstellungen auf; der Vergleich
unten und die Hit@k/MRR-Auswertung aus Kapitel 11 nutzen beide diese Liste.

| # | Modul | Idee | Zurückgegebener Score |
|---|---|---|---|
| 1 | `similarity.py` | die k nächsten Vektoren | **Kosinus-Distanz** – kleiner ist besser (0 … 2) |
| 2 | `mmr.py` | relevant **und** vielfältig | keiner – MMR sortiert um |
| 3 | `score_threshold.py` | Chunks unter einem Mindestwert verwerfen | **Relevanz** = 1 − Distanz – größer ist besser |
| 4 | `metadata_filter.py` | SQL `WHERE category = …` vor dem Ranking | Kosinus-Distanz |
| 5 | `hybrid.py` | Stichwortsuche + Vektorsuche, zusammengeführt | **RRF-Score** – größer ist besser |

> Die Scores bedeuten Verschiedenes. Eine Distanz von 0,4 und ein RRF-Score von 0,03 lassen sich nicht vergleichen.

## 1. Ähnlichkeitssuche (Similarity Search)
Bettet die Frage ein und liefert die k Chunks mit der kleinsten Kosinus-Distanz.
Liefert immer k Ergebnisse – auch bei Fragen, die die Dokumente gar nicht beantworten können.

## 2. Maximum Marginal Relevance (MMR)
Holt `fetch_k` (20) Kandidaten und wählt daraus nacheinander k aus. Jede Auswahl wägt die Ähnlichkeit
zur Frage gegen die Ähnlichkeit zu den bereits gewählten Chunks ab
(`lambda_mult`: 1 = nur Relevanz, 0 = nur Vielfalt). So werden fast identische Chunks vermieden.

## 3. Score-Schwelle (Similarity Score Threshold)
Wie die Ähnlichkeitssuche, verwirft aber Chunks, deren Relevanz-Score unter der Schwelle liegt.
**Grenzen:** Der Score ist keine Wahrscheinlichkeit, und die typischen Werte hängen vom Embedding-Modell ab
(bei text-embedding-3-small erreichen verwandte Texte oft nur 0,3–0,6). Die Schwelle muss an den eigenen Daten eingestellt werden.

## 4. Metadaten-Filter
`filter={"category": {"$eq": "hr"}}` wird zu einer SQL-`WHERE`-Bedingung auf der echten Spalte `category`,
es werden also nur HR-Chunks bewertet. Nützlich, wenn Nutzer oder Anwendung den Themenbereich kennen.

## 5. Hybride Suche (Volltext + Vektor, RRF)
Die Vektorsuche versteht Bedeutung, übersieht aber exakte Begriffe wie `TK-407`; die Stichwortsuche
ist das Gegenteil. Beide laufen in Postgres:

- Vektor: `ORDER BY embedding <=> :q`
- Stichwort: `WHERE to_tsvector('english', content) @@ plainto_tsquery('english', :q) ORDER BY ts_rank_cd(...)`

Reciprocal Rank Fusion führt beide Ranglisten nur anhand der Positionen zusammen:
`score = Σ 1 / (Rang + 60)`. Ein Chunk, der in beiden Listen weit oben steht, gewinnt. Weil RRF die
Rohwerte ignoriert, kann es eine Distanz und einen Text-Rang gefahrlos kombinieren.
Grenze: `plainto_tsquery` verlangt, dass **alle** Wörter vorkommen. Die Stichwortsuche funktioniert daher am besten
mit kurzen, stichwortartigen Fragen ("What is error TK-407?" trifft; "…TK-407 mean?" nicht).

## Ausführen

```bash
uv run python -m examples.compare_retrievers
uv run python -m examples.compare_retrievers "How many vacation days do I get?" hr
```

## Erwartete Ausgabe (Form)

```
=== similarity  (4 chunks, 350 ms)
    score: cosine distance (lower = more similar)
     0.412  [it_support/vpn_setup.md]  ## Common Problems  If the connection drops ...
...
=== hybrid (RRF)  (4 chunks, 120 ms)
    score: RRF score (rank-based, higher = better)
     0.033  [it_support/vpn_setup.md]  ## Common Problems ...
```

Die Zeit der ersten Strategie enthält den Verbindungsaufbau; jede Strategie bettet die Frage einmal über
die OpenAI-API ein, und das dominiert die gemessene Zeit.
