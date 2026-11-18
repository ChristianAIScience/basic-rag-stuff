# 5. LangChain-RAG-Chain

## Was

`rag_lab/langchain_rag/` baut die klassische Pipeline:

**Frage → Retriever → relevante Dokumente → Prompt → LLM → Antwort**

| Datei | Aufgabe |
|---|---|
| `retriever.py` | `vector_store.as_retriever(search_kwargs={"k": 4})` – Frage rein, Dokumente raus |
| `prompt.py` | `RAG_PROMPT`: „Nutze NUR den Kontext; sag sonst, dass du es nicht weißt" |
| `chain.py` | verbindet alles mit LCEL und liefert Antwort **und** Quellen |

## Warum

Der Prompt macht eine Antwort *fundiert* (grounded): Das Modell sieht den gefundenen Text und soll
nichts anderes verwenden. Weil die Quellen mitgeliefert werden, kann der Nutzer die Antwort überprüfen.

## Der Code

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

`|` verbindet Runnables: Die Ausgabe eines Schritts wird die Eingabe des nächsten. `RunnableParallel` führt den
Retriever aus und reicht die Frage durch; `.assign()` ergänzt die Antwort, die aus beidem berechnet wird.

## Wie es zusammenhängt

Die Chain nutzt denselben Vector Store wie alles andere (`db/vector_store.py`), und ihr Prompt wird von
den LangGraph-Varianten wiederverwendet, damit die Antworten vergleichbar bleiben.

## Ausführen

```bash
uv run python -m examples.basic_langchain_rag
uv run python -m examples.basic_langchain_rag "How long is the Glowbeam X2 warranty?"
```

## Erwartete Ausgabe

```
Question: How many vacation days do I get, and how many can I carry over?

Answer:
You get 27 paid vacation days per year; up to 5 unused days can be carried over
until March 31st (hr/vacation_policy.md).

Sources (4):
  - [hr/vacation_policy.md] # Lumora Systems Vacation Policy  ## Annual Vacation Allowance ...
```

Probiere eine Frage, die nicht in den Dokumenten steht ("What is the CEO's name?") – das Modell sollte sagen,
dass es die Antwort nicht kennt. Die Chain sucht übrigens *immer* in der Datenbank, selbst bei „Hallo".
