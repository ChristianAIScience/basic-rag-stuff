---
name: building-langgraph-workflows
description: Builds LangGraph StateGraph workflows with a typed state, one file per node and a separate graph module, and shows how state moves between nodes. Use when adding or changing a LangGraph node, edge or state key, or testing a graph offline.
---

# Building LangGraph workflows

Reference implementation: `rag_lab/langgraph_rag/state.py`, `nodes/retrieve_documents.py`, `nodes/generate_answer.py`, `graph.py`;
example `examples/basic_langgraph_rag.py`. Learner-facing explanation: `docs/06_langgraph_rag.md`.
The agentic graph follows the same conventions (`building-agentic-rag`).
Copyable, self-contained version for other projects: [langgraph_rag_template.py](langgraph_rag_template.py) (dependencies in its header).

## Conventions

- **State**: a `TypedDict`; a node returns only the keys it changes.
- **One node per file**: a plain function `node(state, <dependency>) -> dict` whose docstring lists `Reads:` and `Writes:` state keys.
- **Dependencies** are bound in `graph.py` with `functools.partial(node, llm=llm)`. LangGraph calls `node(state)`; tests call the function directly with a mock.
- **Explicit model calls in nodes**: `messages = PROMPT.invoke({...})`, then `llm.invoke(messages).text` (a `str` subclass). Explicit calls read well for learners and work with `MagicMock` LLMs.
- **graph.py only wires**: `StateGraph(State)`, `add_node`, `add_edge(START, ...)`, `compile()`.
- **Show the state flow** with `graph.stream(input, stream_mode="updates")` (one `{node_name: update}` per step) and `graph.get_graph().draw_mermaid()`.
- Reuse `RAG_PROMPT` and `format_docs` so results match the LangChain chain.

## Add or change a node

Run the unit tests after every edit. Copy this checklist:

```
- [ ] 1. New file in nodes/ with Reads:/Writes: docstring
- [ ] 2. New state keys in state.py
- [ ] 3. Wire the node in graph.py
- [ ] 4. Node test: call it with a dict state + MagicMock dependency (tests/unit/test_langgraph_nodes.py)
- [ ] 5. Graph test: assert the visited node order from stream(..., stream_mode="updates")
- [ ] 6. uv run python -m pytest -> fix and repeat until green
```

Done when both tests pass and `examples.basic_langgraph_rag` prints one "updated state keys" line per node.
