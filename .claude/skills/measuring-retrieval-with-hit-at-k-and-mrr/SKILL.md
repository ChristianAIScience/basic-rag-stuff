---
name: measuring-retrieval-with-hit-at-k-and-mrr
description: Measures retrieval quality with Hit@k and Mean Reciprocal Rank (MRR) on chunks labelled with LLM-generated questions - sampling random chunks, generating and reviewing questions, and scoring every retrieval strategy. Use when computing Hit@k, MRR or recall@k, building a labelled retrieval dataset, comparing retrieval strategies by ranking quality, or when labelled chunk ids no longer match the database.
---

# Measuring retrieval with Hit@k and MRR

Reference implementation: `rag_lab/db/chunks.py`, `rag_lab/evaluation/question_generation.py`, `rag_lab/evaluation/retrieval_metrics.py`;
examples `examples/generate_retrieval_dataset.py` → `evaluation_data/retrieval_dataset.json` → `examples/evaluate_retrieval.py`.
Learner-facing explanation: `docs/11_retrieval_metrics.md`.
Copyable, self-contained version for other projects: [retrieval_metrics_template.py](retrieval_metrics_template.py) (dependencies in its header).

## Definitions

For one question: ranked chunk ids from the retriever, exactly one gold chunk.

| Metric | Value | Example: ranked `[c7, c2, c9]`, gold `c2` |
|---|---|---|
| Hit@k | 1 if gold in the first k, else 0 | Hit@1 = 0, Hit@3 = 1 |
| Reciprocal rank | 1 / position of gold, 0 beyond the cutoff | 0.5 |
| MRR@k | mean reciprocal rank over all questions | – |

Both are plain Python: exact, repeatable, free. No LLM judge is involved.

## Build the labelled dataset and evaluate

Copy this checklist:

```
- [ ] 1. Ingest with REAL embeddings (fake embeddings make every ranking random)
- [ ] 2. uv run python -m examples.generate_retrieval_dataset   (--chunks 10 --questions-per-chunk 2 --seed 42)
- [ ] 3. Review evaluation_data/retrieval_dataset.json: delete vague questions, fix copied phrasing
- [ ] 4. uv run python -m examples.evaluate_retrieval --details
- [ ] 5. Read the per-question ranks: misses with a neighbouring chunk ranked first are label problems, not retriever problems
```

## Rules

- **Labels**: `sample_chunks()` uses a fixed seed and skips chunks under `MIN_CHUNK_CHARS` (150; shorter chunks are mostly headings). The chunk a question was generated from is the single gold label.
- **Scores are a lower bound**: an overlapping chunk that also answers counts as a miss.
- **Question prompt**: natural employee wording in the generator's own words, no "the text", documents treated as data - copied phrases would favour keyword search.
- **Labels go stale silently by id**: chunk ids are `uuid5(source#position)`, so re-ingesting with another chunking reuses the SAME ids for DIFFERENT text. Each label stores `chunk_hash` (sha256 of the chunk text); `check_dataset()` runs before every evaluation and stops on changed/missing chunks, an empty dataset or duplicate questions. Regenerate the dataset (same seed) instead of editing ids or hashes. Keep the reviewed file in git.
- **Duplicates**: the same question generated for two chunks is kept once - two gold chunks would make the label ambiguous (and break the per-question category lookup of the oracle filter).
- **Cost**: `evaluate_retrieval` wraps the embeddings in `QueryCachedEmbeddings` (`rag_lab/utils/cached_embeddings.py`), so each question is embedded once for all five strategies.
- **Metadata filter** receives the gold chunk's category - an oracle upper bound, explained in the note under the results table.
- **MLflow** metric names: `hit_at_1`, `hit_at_3`, `hit_at_5`, `mrr_at_5` (`RetrievalScores.as_metrics()`), logged as one nested run per strategy (`tracking-with-mlflow`).

## Add a metric

Run the unit tests after every edit:

```
- [ ] 1. Pure function in retrieval_metrics.py
- [ ] 2. Unit test with a hand-computed ranked list in tests/unit/test_retrieval_metrics.py
- [ ] 3. Extend RetrievalScores, as_metrics() and the table in examples/evaluate_retrieval.py
- [ ] 4. uv run python -m pytest -> fix and repeat until green
- [ ] 5. Document it in docs/11_retrieval_metrics.md and the German copy (maintaining-docs-website)
```

Integration check without API calls: fake embeddings map identical text to identical vectors, so "question = chunk text" must give Hit@1 = 1.0 (`tests/integration/test_retrieval_evaluation.py`).
