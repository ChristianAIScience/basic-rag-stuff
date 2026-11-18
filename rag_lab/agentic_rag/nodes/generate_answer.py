"""Node: produce the final answer. Three cases:

1. No retrieval needed            -> LLM answers directly                [LLM]
2. Retrieval needed, but nothing  -> fixed fallback message, NO LLM call [deterministic rule]
   relevant was found                (so the model cannot invent an answer)
3. Relevant documents found       -> LLM answers from those documents    [LLM]

Reads:  question, needs_retrieval, relevant_documents
Writes: answer
"""

from langchain_core.language_models import BaseChatModel

from rag_lab.agentic_rag.prompts import DIRECT_ANSWER_PROMPT, NO_RELEVANT_DOCUMENTS_ANSWER
from rag_lab.agentic_rag.state import AgenticRAGState
from rag_lab.langchain_rag.prompt import RAG_PROMPT
from rag_lab.utils.formatting import format_docs


def generate_answer(state: AgenticRAGState, llm: BaseChatModel) -> dict:
    if not state["needs_retrieval"]:
        messages = DIRECT_ANSWER_PROMPT.invoke({"question": state["question"]})
        return {"answer": llm.invoke(messages).text}

    if not state["relevant_documents"]:
        return {"answer": NO_RELEVANT_DOCUMENTS_ANSWER}

    messages = RAG_PROMPT.invoke(
        {"context": format_docs(state["relevant_documents"]), "question": state["question"]}
    )
    return {"answer": llm.invoke(messages).text}
