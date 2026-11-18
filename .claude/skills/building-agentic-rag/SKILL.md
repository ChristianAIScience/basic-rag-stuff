---
name: building-agentic-rag
description: Builds and tests the agentic RAG graph in LangGraph - question analysis, document grading, query rewriting, conditional routing, retry limit and a no-answer fallback. Use when adding an LLM decision, node or route to the agentic graph, or testing its paths offline.
---

# Building agentic RAG (LangGraph)

Reference implementation: `rag_lab/agentic_rag/` - `state.py`, `prompts.py`, `nodes/` (five files), `routing.py`, `graph.py`;
example `examples/agentic_rag.py`. Learner-facing explanation: `docs/06_agentic_rag.md`. Node conventions: `building-langgraph-workflows`.
Copyable, self-contained version for other projects: [agentic_rag_template.py](agentic_rag_template.py) - state, schemas, nodes, routing and graph in one file.

## Design rules

- **The LLM judges, Python routes.** LLM decisions are stored in state via `llm.with_structured_output(PydanticModel)` (`RetrievalDecision`, `RelevanceGrade`). Routing functions in `routing.py` only read state and return the next node name (`Literal[...]` return type).
- **Termination is guaranteed by Python**: `rewrite_question` increments `retry_count`; `route_after_grading` sends the run to `generate_answer` once `retry_count >= max_retries`. `DEFAULT_MAX_RETRIES = 2` allows two reformulations while capping the worst case at 15 LLM calls per question (k = 4).
- **Fallback without the LLM**: retrieval needed but no relevant documents → `generate_answer` returns `NO_RELEVANT_DOCUMENTS_ANSWER` and never calls the model, so nothing can be invented.
- **Grade against the original `question`**, search with `search_query` (the rewritten one).
- `analyze_question` initialises `search_query`, `retry_count`, `relevant_documents`; the state is `TypedDict, total=False`.
- Each node docstring is labelled `[LLM decision]`, `[LLM generates text]` or `[deterministic]`.
- Worst-case LLM calls: 1 analysis + k gradings × (max_retries + 1) retrievals + max_retries rewrites + 1 answer. Update `docs/06_agentic_rag.md` when this changes.

## Test doubles (offline)

`GenericFakeChatModel` has no `with_structured_output`, so `tests/unit/fakes.py` provides:
- `fake_llm(needs_retrieval, relevant, text)`: a `MagicMock` whose `with_structured_output(schema)` returns a scripted answer per schema and whose `invoke` returns `AIMessage(text)`.
- `fake_retriever(documents)`: returns fixed documents.

Test levels: `tests/unit/test_routing.py` (pure functions, include the boundary `retry_count == max_retries`), `test_agentic_nodes.py` (one node each), `test_agentic_graph.py` (exact visited node sequence and final answer).

## Add a decision or node

Run the unit tests after every edit. Copy this checklist:

```
- [ ] 1. Pydantic schema + prompt in prompts.py
- [ ] 2. Node file in nodes/ (labelled), new keys in state.py
- [ ] 3. Pure routing function in routing.py, wired with add_conditional_edges in graph.py
- [ ] 4. Tests at all three levels
- [ ] 5. Extend describe() in examples/agentic_rag.py
- [ ] 6. uv run python -m pytest -> fix and repeat until green
```

Done when every new route has a routing test, the graph test covers the new path, and every loop still ends at `max_retries`.
