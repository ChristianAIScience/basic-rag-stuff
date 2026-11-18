# 9. Tests

## Was

Der Ordner `tests/` prüft jeden Teil des Projekts automatisch mit [pytest](https://docs.pytest.org/).
Ein Test ist eine kleine Python-Funktion, die ein Stück Code ausführt und mit **assert** festhält, wie das Ergebnis sein muss.
Ist ein `assert` falsch, meldet pytest den Test als fehlgeschlagen und zeigt, was erwartet wurde und was herauskam.

```
tests/
  unit/          fast, offline: no database, no API key, no costs
  integration/   against the real Docker database (fake embeddings)
                 + two end-to-end tests that call OpenAI (skipped without a real key)
```

## Warum

- **Sicherheitsnetz**: Du änderst ein Modul, und die Tests sagen dir innerhalb einer Sekunde, ob woanders etwas kaputtgegangen ist.
- **Dokumentation, die nicht veraltet**: Ein Test zeigt genau, wie eine Funktion benutzt werden soll.
- **Keine Kosten**: LLM-Aufrufe werden durch Fakes ersetzt, der normale Testlauf ruft also nie eine kostenpflichtige API auf.

## Ausführen

```bash
uv run python -m pytest                       # all unit tests (default)
uv run python -m pytest -m integration        # integration tests - needs: docker compose up -d
uv run python -m pytest -v                    # one line per test with its name
uv run python -m pytest tests/unit/test_routing.py      # one file
uv run python -m pytest -k hybrid             # only tests whose name contains "hybrid"
uv run python -m pytest -x                    # stop at the first failure
```

Integrationstests sind standardmäßig ausgeschlossen, weil `pyproject.toml` `addopts = "-m 'not integration'"` setzt.

## Die Ausgabe lesen

```
..........................................................               [100%]
58 passed, 10 deselected in 1.25s
```

| Zeichen | Bedeutung |
|---|---|
| `.` | Test bestanden |
| `F` | Test **fehlgeschlagen**: ein `assert` war falsch |
| `E` | **Fehler**: der Test konnte gar nicht laufen (z. B. Datenbank nicht erreichbar) |
| `s` | absichtlich **übersprungen** (z. B. kein echter OpenAI-Key) |
| `deselected` | per `-m` herausgefiltert (hier: die Integrationstests) |

Ein Fehlschlag zeigt die fehlerhafte Zeile (`>`), die verglichenen Werte (`E`) und die Unterschiede. Hier hat der
erwartete Text eine Leerzeile weniger als das tatsächliche Ergebnis:

```
    def test_clean_text_normalizes_whitespace():
>       assert clean_text("  a  \r\nb\n\n\n\nc  ") == "a\nb\nc"
E       AssertionError: assert 'a\nb\n\nc' == 'a\nb\nc'
E           a
E           b
E         +
E           c
tests/unit/test_ingestion.py:24: AssertionError
```

So liest man das: *Die Funktion lieferte `'a\nb\n\nc'`, der Test erwartete `'a\nb\nc'`.* Dann entscheiden, was
falsch ist – der Code oder die Erwartung.

## Was jede Testdatei prüft

| Datei | Prüft |
|---|---|
| `unit/test_config.py` | `.env`-Werte werden gelesen, Standardwerte greifen, fehlende Passwörter/Keys ergeben einen klaren Fehler |
| `unit/test_ingestion.py` | Laden von `.txt`/`.md`, Bereinigen von Leerraum, Chunk-Größe und Überlappung, stabile Chunk-IDs |
| `unit/test_pdf_loader.py` | Überschriftenerkennung nach Schriftgröße, Aufteilung in Kapitel/Abschnitte, umbrochene Überschriften, echte erzeugte PDFs |
| `unit/test_vector_store_errors.py` | Tabellenanlage ist wiederholbar, andere Datenbankfehler werden nicht verschluckt, Passwort-Escaping in der URL |
| `unit/test_retrieval_config.py` | jede Retrieval-Strategie ruft den Vector Store mit den richtigen Parametern auf |
| `unit/test_langchain_chain.py` | die Chain liefert Antwort + Quellen und setzt den Kontext in den Prompt |
| `unit/test_langgraph_nodes.py` | jeder Graph-Node einzeln und die Node-Reihenfolge des einfachen Graphen |
| `unit/test_routing.py` | die Routing-Regeln des agentischen Graphen, inklusive Wiederholungsgrenze |
| `unit/test_agentic_nodes.py` | jeder agentische Node: Entscheidungen, Bewertung, Umformulierung, Fallback-Antwort |
| `unit/test_agentic_graph.py` | komplette agentische Durchläufe: direkte Antwort, Antwort mit Suche, Wiederholungen bis zum Fallback |
| `unit/test_evaluation.py` | LLM-Judges mit gefälschtem Judge: Prompts, Urteile, „nicht anwendbar", Bestehensquoten, Ziel-Adapter |
| `unit/test_retrieval_metrics.py` | Hit@k, reziproker Rang, MRR, wiederholbare Chunk-Stichprobe, Labeln der Fragen |
| `unit/test_tracking.py` | MLflow an/aus/nicht erreichbar, geloggte Parameter und Metriknamen (MLflow gemockt) |
| `integration/test_vector_store.py` | echte Tabellenanlage, Einlesen ohne Duplikate, Metadaten-Filter in SQL |
| `integration/test_retrieval_strategies.py` | alle fünf Strategien gegen echtes PostgreSQL |
| `integration/test_retrieval_evaluation.py` | alle Chunks mit IDs lesen; Hit@1 = 1, wenn die Frage dem Chunk-Text entspricht |
| `integration/test_mlflow_tracking.py` | ein Run mit Parametern, Metriken, Artefakt und verknüpftem LangChain-Trace auf dem echten MLflow-Server |
| `integration/test_end_to_end.py` | echte OpenAI-Antworten und Judge-Kalibrierung (ohne Key übersprungen) |

## Testtechniken mit kurzen Beispielen

Die Beispiele sind gekürzte Fassungen echter Tests in `tests/` – nach dem Funktionsnamen suchen, um den vollständigen Test zu sehen.

### 1. Ein einfaches assert

Die einfachste Form: Funktion aufrufen, Ergebnis vergleichen.

```python
def test_clean_text_normalizes_whitespace():
    assert clean_text("  a  \r\nb\n\n\n\nc  ") == "a\nb\n\nc"
```

### 2. Ein Test, viele Eingaben: `parametrize`

Führt denselben Test für jeden Wert aus – hier wird die Wiederholungsgrenze knapp darunter geprüft.

```python
@pytest.mark.parametrize("retry_count", [0, 1])
def test_grading_routes_to_rewrite_while_retries_are_left(retry_count):
    state = {"relevant_documents": [], "retry_count": retry_count, "max_retries": 2}
    assert route_after_grading(state) == "rewrite_question"
```

### 3. Einen Fehler erwarten: `pytest.raises`

Der Test besteht nur, wenn der Code den Fehler auslöst – hier, wenn das Passwort fehlt.

```python
def test_missing_required_variable_raises(minimal_env, monkeypatch):
    monkeypatch.delenv("POSTGRES_PASSWORD")
    with pytest.raises(ValueError, match="POSTGRES_PASSWORD"):
        load_settings()
```

### 4. Fixtures: `monkeypatch` und `tmp_path`

Fixtures sind Argumente, die pytest automatisch füllt. `monkeypatch` ändert Umgebungsvariablen nur für diesen
Test; `tmp_path` ist ein frischer, leerer Ordner, der danach gelöscht wird.

```python
def test_loader_reads_txt_and_md_and_sets_metadata(tmp_path):
    (tmp_path / "hr").mkdir()
    (tmp_path / "hr" / "policy.md").write_text("# Policy", encoding="utf-8")
    documents = load_documents(tmp_path)
    assert documents[0].metadata == {"source": "hr/policy.md", "category": "hr"}
```

### 5. Mocks: die Datenbank ersetzen

Ein `MagicMock` nimmt jeden Aufruf an und merkt ihn sich. Der Test prüft dann, *wie* der Code ihn aufgerufen hat –
ohne Datenbank.

```python
def test_metadata_filter_builds_category_filter():
    store = MagicMock()
    metadata_filter_search(store, "q", category="hr")
    store.similarity_search_with_score.assert_called_once_with(
        "q", k=4, filter={"category": {"$eq": "hr"}})
```

### 6. Fake-LLMs: testen ohne OpenAI

`GenericFakeChatModel` antwortet mit vorbereiteten Nachrichten. Die Chain läuft genau wie im echten Betrieb, nur das
Modell ist gefälscht – kostenlos und immer mit demselben Ergebnis.

```python
def test_chain_returns_answer_and_sources():
    llm = GenericFakeChatModel(messages=iter([AIMessage(content="You get 27 days.")]))
    chain = build_rag_chain(RunnableLambda(fake_retriever), llm)
    result = chain.invoke("How many vacation days?")
    assert result["answer"] == "You get 27 days."
```

Für den agentischen Graphen stellt `tests/unit/fakes.py` `fake_llm(...)` bereit, das auch die strukturierten
Entscheidungen (`needs_retrieval`, `is_relevant`) vorgibt.

### 7. Einen ganzen Graphen testen: der Weg, den er nimmt

Statt nur die Endantwort zu prüfen, zeichnet der Test auf, welche Nodes liefen – das beweist, dass die
Wiederholungsgrenze die Schleife wirklich beendet.

```python
def test_irrelevant_documents_trigger_rewrites_until_max_retries_then_fallback():
    graph = build_agentic_graph(fake_retriever(), fake_llm(relevant=False, text="rewritten query"))
    visited, state = run(graph, "dogs in the office?", max_retries=2)
    assert visited.count("rewrite_question") == 2
    assert state["answer"] == NO_RELEVANT_DOCUMENTS_ANSWER
```

### 8. Integrationstests: die echte Datenbank

Gekennzeichnet mit `pytestmark = pytest.mark.integration`. Sie nutzen das echte PostgreSQL, aber eine eigene
Tabelle (`test_documents`) und Fake-Embeddings – sie kosten also nichts und berühren deine Daten nie.

```python
def test_ingestion_stores_chunks_and_rerun_does_not_duplicate(empty_vector_store, embeddings):
    chunks = ingest_directory(DATA_DIR, empty_vector_store, embeddings, 500, 50)
    assert count_rows(empty_vector_store) == len(chunks) > 0

    ingest_directory(DATA_DIR, empty_vector_store, embeddings, 500, 50)
    assert count_rows(empty_vector_store) == len(chunks)     # no duplicates
```

Die gemeinsamen Fixtures (`settings`, `engine`, `embeddings`, `empty_vector_store`) liegen in
`tests/integration/conftest.py`.

## Einen eigenen Test schreiben

1. Die Testdatei zum geänderten Modul wählen (`rag_lab/retrieval/mmr.py` → `tests/unit/test_retrieval_config.py`).
2. Den Test nach dem Verhalten benennen: `test_<was>_<erwartetes_ergebnis>`.
3. Eingabe vorbereiten, Code aufrufen, Ergebnis mit `assert` prüfen – ein Verhalten pro Test.
4. Alles Langsame oder Kostenpflichtige (Datenbank, OpenAI) in Unit-Tests durch Mock oder Fake ersetzen.
5. `uv run python -m pytest` ausführen, bis alles grün ist.

## Wie es zusammenhängt

Zum Code jedes Kapitels gibt es passende Tests, und die Testsuite läuft offline in etwa einer Sekunde. Führe sie
nach jeder Änderung aus – so erfährst du am schnellsten, ob das Projekt noch funktioniert.
