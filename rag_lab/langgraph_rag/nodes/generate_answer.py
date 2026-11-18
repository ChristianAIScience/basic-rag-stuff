"""Node 2: generate an answer from the question and the retrieved chunks.

Reads:  state["question"], state["documents"]
Writes: state["answer"]

It reuses the prompt of the LangChain chain, so both implementations produce
comparable answers - only the orchestration differs.
"""

from langchain_core.language_models import BaseChatModel

from rag_lab.langchain_rag.prompt import RAG_PROMPT
from rag_lab.langgraph_rag.state import RAGState
from rag_lab.utils.formatting import format_docs


def generate_answer(state: RAGState, llm: BaseChatModel) -> dict:
    messages = RAG_PROMPT.invoke(
        {"context": format_docs(state["documents"]), "question": state["question"]}
    )
    response = llm.invoke(messages)
    return {"answer": response.text}
