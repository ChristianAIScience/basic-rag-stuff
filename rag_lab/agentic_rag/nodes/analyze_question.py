"""Node: decide whether the question needs the knowledge base.   [LLM decision]

Reads:  question
Writes: needs_retrieval, search_query (= question), retry_count (= 0), relevant_documents (= [])

The routing function route_after_analysis() then reads `needs_retrieval`.
"""

from langchain_core.language_models import BaseChatModel

from rag_lab.agentic_rag.prompts import ANALYZE_PROMPT, RetrievalDecision
from rag_lab.agentic_rag.state import AgenticRAGState


def analyze_question(state: AgenticRAGState, llm: BaseChatModel) -> dict:
    messages = ANALYZE_PROMPT.invoke({"question": state["question"]})
    decision: RetrievalDecision = llm.with_structured_output(RetrievalDecision).invoke(messages)
    return {
        "needs_retrieval": decision.needs_retrieval,
        "search_query": state["question"],
        "retry_count": 0,
        "relevant_documents": [],
    }
