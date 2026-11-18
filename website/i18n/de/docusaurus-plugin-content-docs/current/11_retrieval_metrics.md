# 11. Retrieval-Metriken: Hit@k und MRR

## Was

Die LLM-Judges (Kapitel 10) bewerten *Antworten*. Hit@k und MRR messen den Schritt davor: Setzt der Retriever
den richtigen Chunk **nach oben**? Dafür braucht man gelabelte Daten – Fragen, deren richtiger Chunk bekannt ist.
Das Projekt erzeugt diese Labels automatisch:

```
1. sample 10 random chunks from the database           (fixed seed -> always the same chunks)
2. an LLM writes 2 questions per chunk                  -> 20 questions
3. label: the chunk a question was written from is its "gold chunk"
4. every retrieval strategy searches for every question -> ranked chunk ids
5. compare the ranking with the gold chunk               -> Hit@1, Hit@3, Hit@5, MRR@5
```

## Die Metriken

Für eine Frage liefert der Retriever eine Rangliste, z. B. `[c7, c2, c9, c4, c1]`, und der Gold-Chunk ist `c2`:

| Metrik | Definition | Hier |
|---|---|---|
| **Hit@k** | 1, wenn der Gold-Chunk unter den ersten *k* Ergebnissen ist, sonst 0 | Hit@1 = 0, Hit@3 = 1, Hit@5 = 1 |
| **Reziproker Rang** | 1 / Position des Gold-Chunks (0, wenn er nicht innerhalb der Grenze liegt) | 1/2 = 0,5 |
| **MRR@k** | mittlerer reziproker Rang über alle Fragen | – |

Über alle Fragen gemittelt bedeutet Hit@5 = 0,85: Bei 85 % der Fragen war der Gold-Chunk unter den ersten 5.
MRR belohnt zusätzlich, *wie weit oben* er steht: Rang 1 zählt 1,0, Rang 2 nur 0,5, Rang 3 0,33.
Beides wird mit einfachem Python berechnet – exakt, wiederholbar und kostenlos (kein Judge beteiligt).

Warum das wichtig ist: Die RAG-Chain schickt die besten 4 Chunks an das LLM. Ist Hit@4 niedrig, kann die Antwort
nicht gut sein, egal wie gut das LLM ist.

## Warum die Fragen mit einem LLM erzeugen?

Für viele Chunks Fragen von Hand zu schreiben dauert lange. Ein Generator-LLM schreibt sie in Sekunden, und weil
jede Frage aus einem bestimmten Chunk stammt, ist das Label gratis bekannt. Der Prompt verlangt natürliche, konkrete
Fragen in eigenen Worten, damit die Stichwortsuche keinen unfairen Vorteil durch kopierte Formulierungen bekommt.

**Grenzen – vor dem Vertrauen in die Zahlen lesen:**
- Nur ein Chunk zählt als richtig. Beantwortet ein benachbarter (überlappender) Chunk die Frage ebenfalls, zählt die
  Metrik einen Fehltreffer – die Zahlen sind eine **Untergrenze**.
- Erzeugte Fragen können zu leicht oder zu vage sein. **`evaluation_data/retrieval_dataset.json` durchsehen** und
  schlechte Fragen löschen oder korrigieren – wenige gute Fragen sind mehr wert als viele schlechte.
- 20 Fragen sind ein Schnelltest: Eine einzige Frage verändert Hit@k um 0,05.
- Labels können veralten. Chunk-IDs sind `uuid5(Quelle#Position)`, ein erneutes Einlesen mit anderer `CHUNK_SIZE`
  vergibt also **dieselben IDs für anderen Text**. Jedes Label speichert deshalb zusätzlich einen Hash seines
  Chunk-Texts; `check_dataset()` vergleicht ihn vor jeder Auswertung und bricht bei geänderten oder fehlenden
  Chunks, einem leeren Datensatz oder doppelten Fragen ab. Dann den Datensatz neu erzeugen.
- Wird dieselbe Frage für zwei Chunks erzeugt, bleibt sie nur einmal erhalten – sonst hätte sie zwei richtige Antworten.

## Der Code

```python
def hit_at_k(ranked_ids: list[str], gold_id: str, k: int) -> float:
    return 1.0 if gold_id in ranked_ids[:k] else 0.0


def reciprocal_rank(ranked_ids: list[str], gold_id: str, k: int) -> float:
    top_k = ranked_ids[:k]
    return 1.0 / (top_k.index(gold_id) + 1) if gold_id in top_k else 0.0
```

| Datei | Aufgabe |
|---|---|
| `db/chunks.py` | `fetch_all_chunks()` – alle Chunks mit ihren IDs (eine SQL-Abfrage) |
| `evaluation/question_generation.py` | `sample_chunks()`, `generate_questions()`, `check_dataset()`, Datensatz speichern/laden |
| `evaluation/retrieval_metrics.py` | `hit_at_k()`, `reciprocal_rank()`, `evaluate_retriever()` |
| `examples/generate_retrieval_dataset.py` | erzeugt `evaluation_data/retrieval_dataset.json` |
| `examples/evaluate_retrieval.py` | gibt Hit@1/3/5 und MRR@5 pro Retrieval-Strategie aus |

## Wie es zusammenhängt

Bewertet werden die fünf Strategien aus Kapitel 3 mit denselben gelabelten Fragen. Der Metadaten-Filter bekommt die
Kategorie des Gold-Chunks – ein „Orakel", das den bestmöglichen Fall des Filterns zeigt, markiert mit `*`.
Die Antwort-Judges aus Kapitel 10 und diese Retrieval-Metriken ergänzen sich: Niedriges Hit@k deutet auf den
Retriever, gutes Hit@k bei schlechten Antworten auf den Prompt oder das LLM.

## Ausführen

Braucht einen echten `LLM_API_KEY` und Daten, die **mit echten Embeddings** eingelesen wurden (Fake-Embeddings machen
die Rangfolge zufällig).

```bash
uv run python -m examples.generate_retrieval_dataset            # 10 LLM calls -> 20 labelled questions
# review evaluation_data/retrieval_dataset.json
uv run python -m examples.evaluate_retrieval
uv run python -m examples.evaluate_retrieval --details          # rank of the gold chunk per question
```

Optionen: `--chunks 10 --questions-per-chunk 2 --seed 42`. Die Datensatzdatei in Git behalten, damit jede
Auswertung dieselben Fragen nutzt.

## Erwartete Ausgabe (Form – die Zahlen hängen von deinen Daten und Modellen ab)

```
20 questions from 10 labelled chunks (generated by gpt-5-mini, seed 42)

strategy                    Hit@1    Hit@3    Hit@5    MRR@5
similarity                   0.70     0.90     0.95     0.80
mmr                          0.70     0.85     0.90     0.78
score_threshold (0.4)        0.65     0.80     0.80     0.72
metadata_filter*             0.80     0.95     1.00     0.87
hybrid (RRF)                 0.75     0.90     0.95     0.83

* filter uses the known category of the gold chunk (upper bound for metadata filtering).
```

So liest man das: Ist Hit@5 deutlich höher als Hit@1, wird der richtige Chunk gefunden, aber zu weit unten
eingeordnet – ein Reranker oder andere MMR-Einstellungen können helfen. Verliert die Score-Schwelle Treffer,
ist die Schwelle zu streng.
