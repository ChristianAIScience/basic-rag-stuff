# 9. Testing

## What

The `tests/` folder checks every part of the project automatically with [pytest](https://docs.pytest.org/).
A test is a small Python function that runs a piece of code and **asserts** what the result must be.
If an `assert` is false, pytest reports the test as failed and shows what it expected and what it got.

```
tests/
  unit/          fast, offline: no database, no API key, no costs
  integration/   against the real Docker database (fake embeddings)
                 + two end-to-end tests that call OpenAI (skipped without a real key)
```

## Why

- **Safety net**: change one module and the tests tell you within a second whether something else broke.
- **Documentation that cannot go stale**: a test shows exactly how a function is meant to be used.
- **No costs**: LLM calls are replaced by fakes, so the normal test run never calls a paid API.

## Run

```bash
uv run python -m pytest                       # all unit tests (default)
uv run python -m pytest -m integration        # integration tests - needs: docker compose up -d
uv run python -m pytest -v                    # one line per test with its name
uv run python -m pytest tests/unit/test_routing.py      # one file
uv run python -m pytest -k hybrid             # only tests whose name contains "hybrid"
uv run python -m pytest -x                    # stop at the first failure
```

Integration tests are excluded by default because `pyproject.toml` sets `addopts = "-m 'not integration'"`.

## Reading the output

```
..........................................................               [100%]
58 passed, 10 deselected in 1.25s
```

| Symbol | Meaning |
|---|---|
| `.` | test passed |
| `F` | test **failed**: an `assert` was false |
| `E` | **error**: the test could not even run (e.g. the database is not reachable) |
| `s` | **skipped** on purpose (e.g. no real OpenAI key) |
| `deselected` | filtered out by `-m` (here: the integration tests) |

A failure shows the failing line (`>`), the compared values (`E`) and a diff. Here the expected
string has one blank line fewer than the real result:

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

Read it as: *the function returned `'a\nb\n\nc'`, the test expected `'a\nb\nc'`.* Then decide which
one is wrong - the code or the expectation.

## What each test file checks

| File | Checks |
|---|---|
| `unit/test_config.py` | `.env` values are read, defaults apply, missing passwords/keys raise a clear error |
| `unit/test_ingestion.py` | loading `.txt`/`.md`, whitespace cleaning, chunk size and overlap, stable chunk ids |
| `unit/test_pdf_loader.py` | font-size heading detection, chapter/section splitting, wrapped headings, real generated PDFs |
| `unit/test_vector_store_errors.py` | table creation is idempotent, other database errors are not swallowed, password escaping in the URL |
| `unit/test_retrieval_config.py` | each retrieval strategy calls the vector store with the right parameters |
| `unit/test_langchain_chain.py` | the chain returns answer + sources and puts the context into the prompt |
| `unit/test_langgraph_nodes.py` | each graph node on its own, and the node order of the basic graph |
| `unit/test_routing.py` | the routing rules of the agentic graph, including the retry limit |
| `unit/test_agentic_nodes.py` | each agentic node: decisions, grading, rewriting, fallback answer |
| `unit/test_agentic_graph.py` | complete agentic runs: direct answer, retrieval answer, retries until fallback |
| `unit/test_evaluation.py` | LLM judges with a fake judge: prompts, verdicts, "not applicable", pass rates, target adapters |
| `unit/test_retrieval_metrics.py` | Hit@k, reciprocal rank, MRR, reproducible chunk sampling, question labelling |
| `unit/test_tracking.py` | MLflow on/off/unreachable, logged params and metric names (MLflow mocked) |
| `integration/test_vector_store.py` | real table creation, ingestion without duplicates, metadata filters in SQL |
| `integration/test_retrieval_strategies.py` | all five strategies against real PostgreSQL |
| `integration/test_retrieval_evaluation.py` | reading all chunks with ids; Hit@1 = 1 when the question equals the chunk text |
| `integration/test_mlflow_tracking.py` | a run with params, metrics, artifact and a linked LangChain trace on the real MLflow server |
| `integration/test_end_to_end.py` | real OpenAI answers and judge calibration (skipped without a key) |

## Test techniques, with short examples

The examples are shortened versions of real tests in `tests/` - search for the function name to see the full test.

### 1. A plain assert

The simplest form: call a function, compare the result.

```python
def test_clean_text_normalizes_whitespace():
    assert clean_text("  a  \r\nb\n\n\n\nc  ") == "a\nb\n\nc"
```

### 2. One test, many inputs: `parametrize`

Runs the same test for every value - here the retry limit is checked just below and exactly at the limit.

```python
@pytest.mark.parametrize("retry_count", [0, 1])
def test_grading_routes_to_rewrite_while_retries_are_left(retry_count):
    state = {"relevant_documents": [], "retry_count": retry_count, "max_retries": 2}
    assert route_after_grading(state) == "rewrite_question"
```

### 3. Expecting an error: `pytest.raises`

The test passes only if the code raises the error - here when the password is missing.

```python
def test_missing_required_variable_raises(minimal_env, monkeypatch):
    monkeypatch.delenv("POSTGRES_PASSWORD")
    with pytest.raises(ValueError, match="POSTGRES_PASSWORD"):
        load_settings()
```

### 4. Fixtures: `monkeypatch` and `tmp_path`

Fixtures are arguments that pytest fills in automatically. `monkeypatch` changes environment variables
only for this test; `tmp_path` is a fresh empty folder that is deleted afterwards.

```python
def test_loader_reads_txt_and_md_and_sets_metadata(tmp_path):
    (tmp_path / "hr").mkdir()
    (tmp_path / "hr" / "policy.md").write_text("# Policy", encoding="utf-8")
    documents = load_documents(tmp_path)
    assert documents[0].metadata == {"source": "hr/policy.md", "category": "hr"}
```

### 5. Mocks: replace the database

A `MagicMock` accepts every call and remembers it. The test then checks *how* the code called it -
no database needed.

```python
def test_metadata_filter_builds_category_filter():
    store = MagicMock()
    metadata_filter_search(store, "q", category="hr")
    store.similarity_search_with_score.assert_called_once_with(
        "q", k=4, filter={"category": {"$eq": "hr"}})
```

### 6. Fake LLMs: test without OpenAI

`GenericFakeChatModel` answers with prepared messages. The chain runs exactly as in production, only the
model is fake - free and deterministic.

```python
def test_chain_returns_answer_and_sources():
    llm = GenericFakeChatModel(messages=iter([AIMessage(content="You get 27 days.")]))
    chain = build_rag_chain(RunnableLambda(fake_retriever), llm)
    result = chain.invoke("How many vacation days?")
    assert result["answer"] == "You get 27 days."
```

For the agentic graph, `tests/unit/fakes.py` provides `fake_llm(...)`, which also scripts the
structured decisions (`needs_retrieval`, `is_relevant`).

### 7. Testing a whole graph: the path it takes

Instead of only checking the final answer, the test records which nodes ran - this proves the retry
limit really stops the loop.

```python
def test_irrelevant_documents_trigger_rewrites_until_max_retries_then_fallback():
    graph = build_agentic_graph(fake_retriever(), fake_llm(relevant=False, text="rewritten query"))
    visited, state = run(graph, "dogs in the office?", max_retries=2)
    assert visited.count("rewrite_question") == 2
    assert state["answer"] == NO_RELEVANT_DOCUMENTS_ANSWER
```

### 8. Integration tests: the real database

Marked with `pytestmark = pytest.mark.integration`. They use the real PostgreSQL, but their own
table (`test_documents`) and fake embeddings - so they cost nothing and never touch your data.

```python
def test_ingestion_stores_chunks_and_rerun_does_not_duplicate(empty_vector_store, embeddings):
    chunks = ingest_directory(DATA_DIR, empty_vector_store, embeddings, 500, 50)
    assert count_rows(empty_vector_store) == len(chunks) > 0

    ingest_directory(DATA_DIR, empty_vector_store, embeddings, 500, 50)
    assert count_rows(empty_vector_store) == len(chunks)     # no duplicates
```

The shared fixtures (`settings`, `engine`, `embeddings`, `empty_vector_store`) live in
`tests/integration/conftest.py`.

## Writing your own test

1. Pick the file of the module you changed (`rag_lab/retrieval/mmr.py` → `tests/unit/test_retrieval_config.py`).
2. Name the test after the behaviour: `test_<what>_<expected result>`.
3. Arrange the input, call the code, `assert` the result - one behaviour per test.
4. Replace anything slow or paid (database, OpenAI) with a mock or fake in unit tests.
5. Run `uv run python -m pytest` until everything is green.

## How it connects

Every chapter's code has matching tests, and the test suite runs offline in about a second. Run it
after every change - it is the fastest way to know the project still works.
