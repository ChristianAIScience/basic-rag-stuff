from unittest.mock import MagicMock

from langchain_core.documents import Document
from langchain_core.language_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

from rag_lab.langgraph_rag.graph import build_rag_graph
from rag_lab.langgraph_rag.nodes.generate_answer import generate_answer
from rag_lab.langgraph_rag.nodes.retrieve_documents import retrieve_documents

DOCS = [Document(page_content="Employees get 27 vacation days.", metadata={"source": "hr/vacation.md"})]


def fake_retriever() -> MagicMock:
    retriever = MagicMock()
    retriever.invoke.return_value = DOCS
    return retriever


def test_retrieve_node_writes_documents_for_the_question():
    retriever = fake_retriever()
    update = retrieve_documents({"question": "vacation?"}, retriever=retriever)
    assert update == {"documents": DOCS}
    retriever.invoke.assert_called_once_with("vacation?")


def test_generate_node_sends_context_to_llm_and_writes_answer():
    llm = MagicMock()
    llm.invoke.return_value = AIMessage(content="27 days.")

    update = generate_answer({"question": "vacation?", "documents": DOCS}, llm=llm)

    assert update == {"answer": "27 days."}
    prompt_text = llm.invoke.call_args.args[0].to_string()
    assert "27 vacation days" in prompt_text and "vacation?" in prompt_text


def test_graph_runs_retrieve_then_generate():
    llm = GenericFakeChatModel(messages=iter([AIMessage(content="27 days.")]))
    graph = build_rag_graph(fake_retriever(), llm)

    steps = list(graph.stream({"question": "vacation?"}, stream_mode="updates"))

    assert [next(iter(step)) for step in steps] == ["retrieve_documents", "generate_answer"]


def test_graph_final_state_contains_question_documents_and_answer():
    llm = GenericFakeChatModel(messages=iter([AIMessage(content="27 days.")]))
    final_state = build_rag_graph(fake_retriever(), llm).invoke({"question": "vacation?"})
    assert final_state == {"question": "vacation?", "documents": DOCS, "answer": "27 days."}
