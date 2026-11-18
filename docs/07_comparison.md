# 7. Comparison

## LangChain chain vs. LangGraph workflow

| | LangChain (LCEL chain) | LangGraph (StateGraph) |
|---|---|---|
| Building block | Runnables connected with `\|` | nodes (functions) + edges |
| Data flow | output of one step = input of the next | shared, typed state; nodes return updates |
| Control flow | linear (branches possible but awkward) | edges, conditional edges, loops |
| Inspecting steps | callbacks / tracing | `stream(stream_mode="updates")` shows each node's update |
| Best for | fixed pipelines: retrieve → prompt → answer | workflows with decisions, retries, multiple tools |
| In this project | `langchain_rag/chain.py` (≈ 15 lines) | `langgraph_rag/graph.py` + 2 node files |

For the basic RAG task both produce the same answer – they share the retriever and prompt.
LangGraph pays off once you need the agentic features below.

## Deterministic RAG vs. agentic RAG

| | Deterministic (chain / basic graph) | Agentic (`agentic_rag/`) |
|---|---|---|
| Retrieval | always | only if the LLM says it is needed |
| Bad retrieval | answers anyway (prompt says "don't know") | grades chunks, rewrites the query, retries |
| No relevant documents | LLM decides what to say | fixed fallback, LLM not called |
| LLM calls per question | 1 | 2 – ~15 |
| Latency / cost | low, predictable | higher, varies |
| Testability | straightforward | routing is pure Python; LLM decisions are mocked |

Rule of thumb: start deterministic. Add agentic steps when you observe concrete failures
(irrelevant retrievals, questions that need no retrieval) that justify the extra calls.

## Retrieval strategies

| Strategy | Strength | Weakness |
|---|---|---|
| Similarity | simple, fast | near-duplicates; always returns k |
| MMR | diverse context | may drop the 2nd-best relevant chunk |
| Score threshold | can return "nothing" | threshold depends on the model; scores aren't probabilities |
| Metadata filter | precise scoping | needs known metadata at query time |
| Hybrid (RRF) | exact terms (codes, names) + meaning | keyword side needs all words to match (`plainto_tsquery`) |

See [03_retrieval.md](03_retrieval.md) for details and `examples.compare_retrievers` to see them side by side.
