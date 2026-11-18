from langchain_core.documents import Document
from langchain_core.language_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from rag_lab.langchain_rag.chain import build_rag_chain
from rag_lab.langchain_rag.prompt import RAG_PROMPT
from rag_lab.utils.formatting import format_docs

DOCS = [
    Document(page_content="Employees get 27 vacation days.", metadata={"source": "hr/vacation.md"}),
    Document(page_content="Up to 5 days carry over.", metadata={"source": "hr/vacation.md"}),
]


def fake_retriever(question: str) -> list[Document]:
    return DOCS


def test_chain_returns_answer_and_sources():
    llm = GenericFakeChatModel(messages=iter([AIMessage(content="You get 27 days.")]))
    chain = build_rag_chain(RunnableLambda(fake_retriever), llm)

    result = chain.invoke("How many vacation days?")

    assert result == {"question": "How many vacation days?", "sources": DOCS, "answer": "You get 27 days."}


def test_chain_puts_context_and_question_into_prompt():
    seen_prompts = []

    def recording_llm(prompt_value):
        seen_prompts.append(prompt_value.to_string())
        return "answer"

    chain = build_rag_chain(RunnableLambda(fake_retriever), RunnableLambda(recording_llm))
    chain.invoke("How many vacation days?")

    assert "27 vacation days" in seen_prompts[0]
    assert "[Source: hr/vacation.md]" in seen_prompts[0]
    assert "How many vacation days?" in seen_prompts[0]


def test_format_docs_tags_each_chunk_with_its_source():
    assert format_docs(DOCS[:1]) == "[Source: hr/vacation.md]\nEmployees get 27 vacation days."


def test_prompt_has_context_and_question_variables():
    assert set(RAG_PROMPT.input_variables) == {"context", "question"}
