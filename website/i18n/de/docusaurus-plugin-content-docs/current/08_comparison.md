# 8. Vergleich

## LangChain-Chain vs. LangGraph-Workflow

| | LangChain (LCEL-Chain) | LangGraph (StateGraph) |
|---|---|---|
| Baustein | Runnables, verbunden mit `\|` | Nodes (Funktionen) + Edges |
| Datenfluss | Ausgabe eines Schritts = Eingabe des nächsten | gemeinsamer, typisierter State; Nodes liefern Updates |
| Ablaufsteuerung | linear (Verzweigungen möglich, aber umständlich) | Edges, bedingte Edges, Schleifen |
| Schritte nachvollziehen | Callbacks / Tracing | `stream(stream_mode="updates")` zeigt das Update jedes Nodes |
| Am besten für | feste Pipelines: suchen → Prompt → antworten | Abläufe mit Entscheidungen, Wiederholungen, mehreren Werkzeugen |
| In diesem Projekt | `langchain_rag/chain.py` (≈ 15 Zeilen) | `langgraph_rag/graph.py` + 2 Node-Dateien |

Für die einfache RAG-Aufgabe liefern beide dieselbe Antwort – sie teilen sich Retriever und Prompt.
LangGraph lohnt sich, sobald man die agentischen Funktionen unten braucht.

## Deterministisches vs. agentisches RAG

| | Deterministisch (Chain / einfacher Graph) | Agentisch (`agentic_rag/`) |
|---|---|---|
| Suche | immer | nur wenn das LLM sie für nötig hält |
| Schlechte Suchergebnisse | antwortet trotzdem (Prompt sagt „weiß ich nicht") | bewertet Chunks, formuliert die Anfrage um, versucht es erneut |
| Keine relevanten Dokumente | das LLM entscheidet, was es sagt | fester Fallback, LLM wird nicht aufgerufen |
| LLM-Aufrufe pro Frage | 1 | 2 – ~15 |
| Latenz / Kosten | gering, vorhersehbar | höher, schwankend |
| Testbarkeit | unkompliziert | Routing ist reines Python; LLM-Entscheidungen werden gemockt |

Faustregel: deterministisch anfangen. Agentische Schritte erst hinzufügen, wenn konkrete Fehler auftreten
(irrelevante Suchergebnisse, Fragen ohne Suchbedarf), die die zusätzlichen Aufrufe rechtfertigen.

## Retrieval-Strategien

| Strategie | Stärke | Schwäche |
|---|---|---|
| Ähnlichkeitssuche | einfach, schnell | fast identische Treffer; liefert immer k |
| MMR | vielfältiger Kontext | kann den zweitbesten relevanten Chunk weglassen |
| Score-Schwelle | kann „nichts gefunden" liefern | Schwelle hängt vom Modell ab; Scores sind keine Wahrscheinlichkeiten |
| Metadaten-Filter | genaue Eingrenzung | braucht bekannte Metadaten zum Zeitpunkt der Anfrage |
| Hybrid (RRF) | exakte Begriffe (Codes, Namen) + Bedeutung | Stichwortsuche braucht alle Wörter (`plainto_tsquery`) |

Details in [04_retrieval.md](04_retrieval.md); `examples.compare_retrievers` zeigt die Strategien nebeneinander.
