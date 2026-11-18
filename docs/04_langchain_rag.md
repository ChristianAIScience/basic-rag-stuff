# 4. LangChain RAG Chain

## What

`rag_lab/langchain_rag/` builds the classic pipeline:

**Question → Retriever → Relevant documents → Prompt → LLM → Answer**

| File | Role |
|---|---|
| `retriever.py` | `vector_store.as_retriever(search_kwargs={"k": 4})` – question in, documents out |
| `prompt.py` | `RAG_PROMPT`: "use ONLY the context; say you don't know otherwise" |
| `chain.py` | wires everything with LCEL and returns answer **and** sources |

## Why

The prompt is what makes an answer *grounded*: the model sees the retrieved text and is told
to use nothing else. Returning the sources lets the user check the answer.

## The code

```python
generate_answer = (
    {"context": lambda inputs: format_docs(inputs["sources"]),
     "question": itemgetter("question")}
    | RAG_PROMPT | llm | StrOutputParser()
)
chain = RunnableParallel(sources=retriever, question=RunnablePassthrough()).assign(answer=generate_answer)

chain.invoke("How many vacation days do I get?")
# {"question": "...", "sources": [Document, ...], "answer": "..."}
```

`|` connects Runnables: each output becomes the next input. `RunnableParallel` runs the
retriever and passes the question through; `.assign()` adds the answer computed from both.

## How it connects

It uses the same vector store as everything else (`db/vector_store.py`), and its prompt is
reused by the LangGraph implementations so the answers are comparable.

## Run

```bash
uv run python -m examples.basic_langchain_rag
uv run python -m examples.basic_langchain_rag "How long is the Glowbeam X2 warranty?"
```

## Expected output

```
Question: How many vacation days do I get, and how many can I carry over?

Answer:
You get 27 paid vacation days per year; up to 5 unused days can be carried over
until March 31st (hr/vacation_policy.md).

Sources (4):
  - [hr/vacation_policy.md] # Lumora Systems Vacation Policy  ## Annual Vacation Allowance ...
```

Try a question that is not in the documents ("What is the CEO's name?") – the model should
say it doesn't know. Note that the chain *always* retrieves, even for "Hello".
