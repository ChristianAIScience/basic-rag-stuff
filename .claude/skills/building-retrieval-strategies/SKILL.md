---
name: building-retrieval-strategies
description: Implements and compares pgvector retrieval strategies - similarity search, MMR, score threshold, metadata filter and hybrid full-text plus vector search with RRF. Use when adding or tuning a retrieval strategy, interpreting its scores, or extending the retriever comparison.
---

# Building retrieval strategies

Reference implementation: one module per strategy in `rag_lab/retrieval/`. Each exports one function
`(vector_store, query, ...) -> list[tuple[Document, float | None]]` and a `SCORE_LABEL` string.
Strategy list: `build_strategies()` in `rag_lab/retrieval/comparison.py`, the single list that `examples/compare_retrievers.py` and `examples/evaluate_retrieval.py` both run. Learner-facing explanation: `docs/04_retrieval.md`.
Copyable, self-contained version for other projects: [retrieval_strategies_template.py](retrieval_strategies_template.py) (dependencies in its header).

## Score semantics

Every printed score carries its label; scores of different strategies are never compared with each other.

| Strategy | Vector store call | Score |
|---|---|---|
| similarity | `similarity_search_with_score` | cosine distance, lower = better (0..2) |
| mmr | `max_marginal_relevance_search` | none - MMR re-ranks |
| score_threshold | `similarity_search_with_relevance_scores(score_threshold=)` | relevance = 1 - distance, higher = better |
| metadata_filter | `similarity_search_with_score(filter={"category": {"$eq": ...}})` | cosine distance |
| hybrid | `similarity_search_with_score(hybrid_search_config=...)` | RRF score, higher = better |

## Hybrid search (langchain-postgres `HybridSearchConfig`)

- Create a **fresh** `HybridSearchConfig(fusion_function=reciprocal_rank_fusion, ...)` per call: the library writes the query into `config.fts_query` and would reuse it.
- Without a tsvector column the library computes `to_tsvector('english', content)` per query - acceptable for small tables and keeps one simple table.
- `plainto_tsquery` requires **every** word: "What is error TK-407?" matches, "What does TK-407 mean?" does not. Keep example questions keyword-like.
- RRF uses `1 / (rank + 60)`; 60 is the standard constant from the RRF paper and damps the influence of top ranks.
- With fake embeddings, RRF ties when the keyword hit is missing from the vector candidates; tests set `candidates_per_search` above the chunk count.

## Add a strategy

Run the unit tests after every edit. Copy this checklist:

```
- [ ] 1. New module in rag_lab/retrieval/ with SCORE_LABEL + one function; docstring: how it works, limits
- [ ] 2. Add one Strategy entry to build_strategies() in rag_lab/retrieval/comparison.py (comparison and Hit@k/MRR evaluation both pick it up)
- [ ] 3. Unit test with a MagicMock store asserting the exact call; extend the build_strategies tests (tests/unit/test_retrieval_config.py)
- [ ] 4. Integration test with fake embeddings (tests/integration/test_retrieval_strategies.py)
- [ ] 5. uv run python -m pytest and -m integration -> fix and repeat until green
- [ ] 6. Add rows to docs/04_retrieval.md, docs/08_comparison.md and their German copies (maintaining-docs-website)
```
