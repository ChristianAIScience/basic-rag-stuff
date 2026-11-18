"""Scripted fake LLM and retriever for agentic RAG tests (no API calls)."""

from unittest.mock import MagicMock

from langchain_core.documents import Document
from langchain_core.messages import AIMessage

from rag_lab.agentic_rag.prompts import RelevanceGrade, RetrievalDecision

RELEVANT_DOC = Document(page_content="Error TK-407: renew the device certificate.", metadata={"source": "it/vpn.md"})
OTHER_DOC = Document(page_content="Glowbeam X2 costs 129 euros.", metadata={"source": "products/pricing.md"})


def fake_llm(needs_retrieval: bool = True, relevant: bool = True, text: str = "LLM text") -> MagicMock:
    """A fake chat model.

    - with_structured_output(RetrievalDecision) -> returns needs_retrieval
    - with_structured_output(RelevanceGrade)    -> grades every document as `relevant`
    - invoke(...)                               -> AIMessage(text) (rewrites and answers)
    """
    llm = MagicMock()

    def structured(schema):
        runnable = MagicMock()
        if schema is RetrievalDecision:
            runnable.invoke.return_value = RetrievalDecision(needs_retrieval=needs_retrieval, reason="test")
        else:
            runnable.invoke.return_value = RelevanceGrade(is_relevant=relevant)
        return runnable

    llm.with_structured_output.side_effect = structured
    llm.invoke.return_value = AIMessage(content=text)
    return llm


def fake_retriever(documents: list[Document] | None = None) -> MagicMock:
    retriever = MagicMock()
    retriever.invoke.return_value = documents if documents is not None else [RELEVANT_DOC, OTHER_DOC]
    return retriever
