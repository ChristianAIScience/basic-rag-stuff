"""Node: write a better search query after a failed retrieval.   [LLM generates text]

Reads:  question, search_query, retry_count
Writes: search_query (new), retry_count (+1)

Incrementing retry_count here is what lets route_after_grading() stop the loop.
"""

from langchain_core.language_models import BaseChatModel

from rag_lab.agentic_rag.prompts import REWRITE_PROMPT
from rag_lab.agentic_rag.state import AgenticRAGState


def rewrite_question(state: AgenticRAGState, llm: BaseChatModel) -> dict:
    messages = REWRITE_PROMPT.invoke(
        {"question": state["question"], "search_query": state["search_query"]}
    )
    new_query = llm.invoke(messages).text.strip()
    return {"search_query": new_query, "retry_count": state["retry_count"] + 1}
