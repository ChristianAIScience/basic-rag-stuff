---
name: building-langchain-rag-chains
description: Builds and tests the LCEL RAG chain (retriever, prompt, LLM) that returns question, sources and answer. Use when changing the LangChain RAG pipeline, its grounding prompt or retriever, or testing it offline with fake models.
---

# Building LangChain RAG chains (LCEL)

Reference implementation: `rag_lab/langchain_rag/retriever.py`, `prompt.py`, `chain.py`; example `examples/basic_langchain_rag.py`.
Learner-facing explanation: `docs/05_langchain_rag.md`.
Copyable, self-contained version for other projects: [rag_chain_template.py](rag_chain_template.py) (includes an offline test snippet).

## Shape

```python
RunnableParallel(sources=retriever, question=RunnablePassthrough()).assign(answer=generate_answer)
# invoke("question") -> {"question": ..., "sources": [Document, ...], "answer": "..."}
```

The output keeps `sources`: users must be able to verify every answer.

## Rules

- **Grounding** lives in `RAG_PROMPT` ("use ONLY the context, otherwise say you don't know"). Both LangGraph graphs reuse it, so answers stay comparable; change it in this one place.
- **LCEL coercion**: two plain callables cannot be piped (`itemgetter("sources") | format_docs` raises `TypeError`). Inside the input dict use `lambda inputs: format_docs(inputs["sources"])`.
- **Retriever**: `vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})`; alternatives are in `building-retrieval-strategies`.
- **Models** come from `rag_lab/models.py`; gpt-5 models reject a custom `temperature`, so none is set.
- The chain always retrieves, even for "Hello" - the reason the `building-agentic-rag` skill exists.

## Change the chain

```
- [ ] 1. Edit chain.py / prompt.py
- [ ] 2. Test offline in tests/unit/test_langchain_chain.py
- [ ] 3. uv run python -m pytest -> fix and repeat until green
- [ ] 4. With a real LLM_API_KEY: uv run python -m examples.basic_langchain_rag
```

Offline test doubles: `RunnableLambda(fake_retriever)` as retriever; `GenericFakeChatModel(messages=iter([AIMessage(content="...")]))` for answers; a recording `RunnableLambda` LLM (`prompt_value.to_string()`) to assert prompt contents.
Done when the tests pass and the output dict still has exactly `question`, `sources`, `answer`.
