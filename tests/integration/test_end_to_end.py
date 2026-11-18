"""End-to-end with REAL OpenAI models (costs a few cents). Skipped without a real API key."""

import dataclasses

import pytest

from rag_lab.agentic_rag.graph import build_agentic_graph
from rag_lab.agentic_rag.prompts import NO_RELEVANT_DOCUMENTS_ANSWER
from rag_lab.db.vector_store import METADATA_COLUMNS, get_vector_store
from rag_lab.ingestion.pipeline import ingest_directory
from rag_lab.langchain_rag.chain import build_rag_chain
from rag_lab.langchain_rag.retriever import build_retriever
from rag_lab.langgraph_rag.graph import build_rag_graph
from rag_lab.models import get_chat_model, get_embeddings
from langchain_postgres import Column
from tests.integration.conftest import DATA_DIR

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def openai_store(engine, settings):
    if settings.llm_api_key.startswith("your_"):
        pytest.skip("LLM_API_KEY in .env is still the placeholder")
    real_settings = dataclasses.replace(settings, vector_size=1536, vector_table="test_documents_openai")
    engine.init_vectorstore_table(
        table_name=real_settings.vector_table,
        vector_size=real_settings.vector_size,
        metadata_columns=[Column(name, "TEXT") for name in METADATA_COLUMNS],
        overwrite_existing=True,
    )
    embeddings = get_embeddings(real_settings)
    store = get_vector_store(engine, embeddings, real_settings)
    ingest_directory(DATA_DIR, store, embeddings, 500, 50)
    return store, real_settings


def test_langchain_and_langgraph_answer_from_the_documents(openai_store):
    store, settings = openai_store
    retriever, llm = build_retriever(store), get_chat_model(settings)

    chain_result = build_rag_chain(retriever, llm).invoke("How many paid vacation days do employees get?")
    graph_result = build_rag_graph(retriever, llm).invoke({"question": "How many paid vacation days do employees get?"})

    assert "27" in chain_result["answer"]
    assert "27" in graph_result["answer"]
    assert any(doc.metadata["category"] == "hr" for doc in chain_result["sources"])


def test_judges_agree_with_hand_labelled_calibration_cases(settings):
    if settings.llm_api_key.startswith("your_"):
        pytest.skip("LLM_API_KEY in .env is still the placeholder")
    from examples.calibrate_judges import run_case
    from rag_lab.evaluation.dataset import CALIBRATION_CASES
    from rag_lab.models import get_judge_model

    llm = get_judge_model(settings)
    agreements = sum(run_case(llm, case).passed == case.expected for case in CALIBRATION_CASES)

    # The cases are obvious; allow one disagreement because LLM verdicts are not fully deterministic.
    assert agreements >= len(CALIBRATION_CASES) - 1


def test_agentic_rag_falls_back_for_unknown_topics(openai_store):
    store, settings = openai_store
    graph = build_agentic_graph(build_retriever(store), get_chat_model(settings))

    state = graph.invoke({"question": "What is Lumora's policy on bringing dogs to the office?", "max_retries": 1})

    assert state["answer"] == NO_RELEVANT_DOCUMENTS_ANSWER
