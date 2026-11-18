"""The copyable template scripts in .claude/skills/ must import and work - no database, no API calls."""

import importlib.util
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from langchain_core.documents import Document
from langchain_core.language_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

SKILLS = Path(__file__).resolve().parents[2] / ".claude" / "skills"
TEMPLATES = {
    "pgvector_docker_tools": "managing-pgvector-docker/pgvector_docker_tools.py",
    "ingestion_template": "ingesting-documents/ingestion_template.py",
    "pdf_heading_chunker": "chunking-pdfs-by-font-size/pdf_heading_chunker.py",
    "retrieval_strategies_template": "building-retrieval-strategies/retrieval_strategies_template.py",
    "rag_chain_template": "building-langchain-rag-chains/rag_chain_template.py",
    "langgraph_rag_template": "building-langgraph-workflows/langgraph_rag_template.py",
    "agentic_rag_template": "building-agentic-rag/agentic_rag_template.py",
    "llm_judges_template": "evaluating-rag-with-llm-judges/llm_judges_template.py",
    "retrieval_metrics_template": "measuring-retrieval-with-hit-at-k-and-mrr/retrieval_metrics_template.py",
    "new_doc_chapter": "maintaining-docs-website/new_doc_chapter.py",
    "mlflow_tracking_template": "tracking-with-mlflow/mlflow_tracking_template.py",
    "check_native_modules": "troubleshooting-windows-env/check_native_modules.py",  # local only (git-ignored)
}
DOC = Document(id="c1", page_content="Employees get 27 vacation days.", metadata={"source": "hr/v.md", "category": "hr"})


def load(name: str):
    path = SKILLS / TEMPLATES[name]
    if not path.exists():
        pytest.skip(f"{path.name} not present (local-only skill)")
    spec = importlib.util.spec_from_file_location(f"skill_template_{name}", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses need the module to be registered
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("name", TEMPLATES)
def test_every_template_imports_and_documents_its_dependencies(name):
    module = load(name)
    assert "Dependencies:" in module.__doc__


def test_retrieval_metrics_template():
    m = load("retrieval_metrics_template")
    assert m.hit_at_k(["a", "b"], "b", 1) == 0.0 and m.reciprocal_rank(["a", "b"], "b", 5) == 0.5
    examples = [m.RetrievalExample("q1", "a", "s", "c"), m.RetrievalExample("q2", "x", "s", "c")]
    scores = m.evaluate_retriever(lambda q: ["a", "b"], examples, (1, 3))
    assert scores.as_metrics() == {"hit_at_1": 0.5, "hit_at_3": 0.5, "mrr_at_3": 0.5}
    chunks = [Document(id=str(i), page_content="x" * m.MIN_CHUNK_CHARS) for i in range(12)]
    assert m.sample_chunks(chunks, 10, seed=1) == m.sample_chunks(chunks, 10, seed=1)
    with pytest.raises(ValueError):
        m.evaluate_retriever(lambda q: [], [])
    assert m.chunks_sql('a"b').endswith('FROM "a""b" ORDER BY source, langchain_id')


def test_retrieval_metrics_template_detects_stale_labels():
    m = load("retrieval_metrics_template")
    stored = [Document(id="c1", page_content="new text")]
    fresh = m.RetrievalExample("q", "c1", "s", "c", m.content_hash("new text"))
    stale = m.RetrievalExample("q", "c1", "s", "c", m.content_hash("old text"))
    assert m.check_dataset([fresh], stored) == []
    assert m.check_dataset([stale], stored) and m.check_dataset([], stored)


def test_retrieval_metrics_template_dataset_round_trip(tmp_path):
    m = load("retrieval_metrics_template")
    examples = [m.RetrievalExample("q", "c1", "a.md", "hr")]
    m.save_dataset(tmp_path / "d.json", examples, {"seed": 1})
    assert m.load_dataset(tmp_path / "d.json") == ({"seed": 1}, examples)


def test_llm_judges_template():
    m = load("llm_judges_template")
    grades = {
        m.CorrectnessGrade: m.CorrectnessGrade(explanation="x", correct=True),
        m.RelevanceGrade: m.RelevanceGrade(explanation="x", relevant=False),
    }
    llm = MagicMock()
    llm.with_structured_output.side_effect = lambda schema, **kw: MagicMock(invoke=MagicMock(return_value=grades[schema]))
    results = m.run_all_judges(llm, "q", "a", [], "ref")  # no documents -> two judges not applicable
    assert [r.passed for r in results] == [True, False, None, None]
    assert m.pass_rates([results]) == {"correctness": (1, 1), "relevance": (0, 1),
                                       "groundedness": (0, 0), "retrieval_relevance": (0, 0)}


def test_retrieval_strategies_template():
    m = load("retrieval_strategies_template")
    store = MagicMock()
    m.metadata_filter_search(store, "q", "hr", k=3)
    store.similarity_search_with_score.assert_called_once_with("q", k=3, filter={"category": {"$eq": "hr"}})
    run = m.run_strategy("mmr", lambda: [(DOC, None)])
    assert run.score_label.startswith("no score") and run.results == [(DOC, None)]


def test_rag_chain_template():
    m = load("rag_chain_template")
    chain = m.build_rag_chain(RunnableLambda(lambda q: [DOC]), GenericFakeChatModel(messages=iter([AIMessage(content="27")])))
    assert chain.invoke("days?") == {"question": "days?", "sources": [DOC], "answer": "27"}


def test_langgraph_rag_template():
    m = load("langgraph_rag_template")
    retriever = MagicMock(invoke=MagicMock(return_value=[DOC]))
    graph = m.build_rag_graph(retriever, GenericFakeChatModel(messages=iter([AIMessage(content="27")])))
    assert m.visited_nodes(graph, {"question": "q"}) == ["retrieve_documents", "generate_answer"]


def test_agentic_rag_template_retries_then_falls_back():
    m = load("agentic_rag_template")
    llm = MagicMock()
    answers = {m.RetrievalDecision: m.RetrievalDecision(needs_retrieval=True, reason="x"),
               m.RelevanceGrade: m.RelevanceGrade(is_relevant=False)}
    llm.with_structured_output.side_effect = lambda schema: MagicMock(invoke=MagicMock(return_value=answers[schema]))
    llm.invoke.return_value = AIMessage(content="better query")
    graph = m.build_agentic_graph(MagicMock(invoke=MagicMock(return_value=[DOC])), llm)

    state = graph.invoke({"question": "dogs?", "max_retries": 1})

    assert state["retry_count"] == 1
    assert state["answer"] == m.NO_RELEVANT_DOCUMENTS_ANSWER


def test_ingestion_template(tmp_path):
    m = load("ingestion_template")
    (tmp_path / "hr").mkdir()
    (tmp_path / "hr" / "a.md").write_text("word " * 200, encoding="utf-8")
    chunks = m.split_documents(m.clean_documents(m.load_documents(tmp_path)), 300, 30)
    assert len(chunks) > 1 and chunks[0].metadata["category"] == "hr"
    assert m.make_chunk_ids(chunks) == m.make_chunk_ids(chunks)
    assert m.connection_url("u", "p@ss", "h", 5432, "db") == "postgresql+asyncpg://u:p%40ss@h:5432/db"


def test_pdf_heading_chunker():
    m = load("pdf_heading_chunker")
    lines = [m.TextLine("1 Chapter", 18.0, 1), m.TextLine("Body text " * 5, 11.0, 1),
             m.TextLine("2 Chapter", 18.0, 2), m.TextLine("More body " * 5, 11.0, 2)]
    assert m.find_heading_sizes(lines, m.find_body_size(lines)) == [18.0]
    assert [s["heading"] for s in m.split_into_sections(lines, [18.0])] == ["1 Chapter", "2 Chapter"]


def test_pgvector_docker_tools_without_docker(tmp_path):
    m = load("pgvector_docker_tools")
    assert "./postgres-data:/var/lib/postgresql/data" in m.COMPOSE_TEMPLATE
    env = tmp_path / ".env"
    env.write_text("POSTGRES_PASSWORD=secret\n", encoding="utf-8")
    assert m.read_env_value("POSTGRES_PASSWORD", env) == "secret"


def test_new_doc_chapter_scaffolds_both_languages_without_overwriting(tmp_path):
    m = load("new_doc_chapter")
    created = m.scaffold(tmp_path, 13, "caching", "Caching", "Caching")
    assert [p.relative_to(tmp_path).as_posix() for p in created] == [
        "docs/13_caching.md", "website/i18n/de/docusaurus-plugin-content-docs/current/13_caching.md"]
    assert created[1].read_text(encoding="utf-8").startswith("# 13. Caching\n\n## Was")
    assert "id: 'home.chapter.caching.title'" in m.card_snippet("caching", 13, "Caching")
    with pytest.raises(FileExistsError):
        m.scaffold(tmp_path, 13, "caching", "Caching", "Caching")


def test_mlflow_tracking_template_is_optional():
    m = load("mlflow_tracking_template")
    assert m.enable_mlflow("", "exp") is False
    with m.mlflow_run(False, run_name="x"):
        pass
    assert m.mlflow_server_available("http://127.0.0.1:9", timeout_s=0.5) is False
    assert m.mlflow_server_available("not a url", timeout_s=0.5) is False   # never raises


def test_check_native_modules(tmp_path):
    m = load("check_native_modules")
    assert m.blocked_native_modules(tmp_path) == []
    assert isinstance(m.smart_app_control_state(), str)
