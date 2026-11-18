# 4. Retrieval Strategies

Each strategy lives in its own module in `rag_lab/retrieval/` and has the same shape:
`strategy(vector_store, query, ...) -> list[(Document, score)]`.
`build_strategies()` in `retrieval/comparison.py` lists all five with their settings; the comparison
below and the Hit@k/MRR evaluation of chapter 11 both run that list.

| # | Module | Idea | Score returned |
|---|---|---|---|
| 1 | `similarity.py` | k nearest vectors | **cosine distance** – lower is better (0 … 2) |
| 2 | `mmr.py` | relevant **and** diverse | none – MMR re-ranks |
| 3 | `score_threshold.py` | drop chunks below a minimum | **relevance** = 1 − distance – higher is better |
| 4 | `metadata_filter.py` | SQL `WHERE category = …` before ranking | cosine distance |
| 5 | `hybrid.py` | keyword search + vector search, merged | **RRF score** – higher is better |

> The scores mean different things. A distance of 0.4 and an RRF score of 0.03 cannot be compared.

## 1. Similarity search
Embeds the question, returns the k chunks with the smallest cosine distance.
Always returns k results – even for questions the documents can't answer.

## 2. Maximum Marginal Relevance (MMR)
Fetches `fetch_k` (20) candidates, then picks k one by one, each time balancing
similarity to the question against similarity to chunks already picked
(`lambda_mult`: 1 = only relevance, 0 = only diversity). Avoids near-duplicate chunks.

## 3. Similarity score threshold
Like similarity search, but drops chunks whose relevance score is below the threshold.
**Limitations:** the score is not a probability, and typical ranges depend on the embedding
model (with text-embedding-3-small, related texts often score only 0.3–0.6). Tune it on your data.

## 4. Metadata filtering
`filter={"category": {"$eq": "hr"}}` becomes a SQL `WHERE` clause on the real `category`
column, so only HR chunks are ranked. Useful when the user (or the app) knows the domain.

## 5. Hybrid search (full-text + vector, RRF)
Vector search understands meaning but can miss exact tokens such as `TK-407`; keyword search
is the opposite. Both run inside Postgres:

- vector: `ORDER BY embedding <=> :q`
- keyword: `WHERE to_tsvector('english', content) @@ plainto_tsquery('english', :q) ORDER BY ts_rank_cd(...)`

Reciprocal Rank Fusion merges the two ranked lists using only positions:
`score = Σ 1 / (rank + 60)`. A chunk ranked high in both lists wins. Because RRF ignores the
raw scores, it can safely combine a distance with a text rank.
Limitation: `plainto_tsquery` requires **all** words to match, so keyword search works best
with short, keyword-like questions ("What is error TK-407?" matches; "…TK-407 mean?" does not).

## Run

```bash
uv run python -m examples.compare_retrievers
uv run python -m examples.compare_retrievers "How many vacation days do I get?" hr
```

## Expected output (shape)

```
=== similarity  (4 chunks, 350 ms)
    score: cosine distance (lower = more similar)
     0.412  [it_support/vpn_setup.md]  ## Common Problems  If the connection drops ...
...
=== hybrid (RRF)  (4 chunks, 120 ms)
    score: RRF score (rank-based, higher = better)
     0.033  [it_support/vpn_setup.md]  ## Common Problems ...
```

The first strategy's time includes warming up the connection; every strategy embeds the question
once via the OpenAI API, which dominates the timing.
