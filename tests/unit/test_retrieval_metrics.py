"""Hit@k, MRR and the question generation - no database, no API calls."""

from unittest.mock import MagicMock

import pytest
from langchain_core.documents import Document

from rag_lab.evaluation.question_generation import (
    MIN_CHUNK_CHARS,
    GeneratedQuestions,
    RetrievalExample,
    build_retrieval_dataset,
    check_dataset,
    content_hash,
    load_dataset,
    sample_chunks,
    save_dataset,
)
from rag_lab.evaluation.retrieval_metrics import evaluate_retriever, gold_rank, hit_at_k, reciprocal_rank

RANKED = ["c7", "c2", "c9", "c4", "c1"]


@pytest.mark.parametrize(("k", "expected"), [(1, 0.0), (2, 1.0), (5, 1.0)])
def test_hit_at_k_checks_whether_gold_chunk_is_in_top_k(k, expected):
    assert hit_at_k(RANKED, "c2", k) == expected


@pytest.mark.parametrize(("gold", "expected"), [("c7", 1.0), ("c2", 0.5), ("c9", 1 / 3), ("c8", 0.0)])
def test_reciprocal_rank_is_one_over_position(gold, expected):
    assert reciprocal_rank(RANKED, gold, k=5) == pytest.approx(expected)


def test_reciprocal_rank_is_zero_beyond_cutoff():
    assert reciprocal_rank(RANKED, "c1", k=3) == 0.0


def test_evaluate_retriever_averages_over_questions():
    examples = [
        RetrievalExample("q1", "c7", "a.md", "hr"),   # rank 1
        RetrievalExample("q2", "c9", "a.md", "hr"),   # rank 3
        RetrievalExample("q3", "c8", "b.md", "it"),   # not found
    ]
    scores = evaluate_retriever(lambda question: RANKED, examples, k_values=(1, 3, 5))

    assert scores.hit_at == {1: pytest.approx(1 / 3), 3: pytest.approx(2 / 3), 5: pytest.approx(2 / 3)}
    assert scores.mrr == pytest.approx((1 + 1 / 3 + 0) / 3)
    assert scores.gold_ranks == [1, 3, None]


def make_chunks(count: int, length: int) -> list[Document]:
    return [Document(id=f"id{i}", page_content="x" * length, metadata={"source": f"s{i}.md", "category": "hr"})
            for i in range(count)]


def test_sample_chunks_is_reproducible_and_skips_short_chunks():
    short_chunks = [Document(id=f"short{i}", page_content="x" * (MIN_CHUNK_CHARS - 1)) for i in range(5)]
    chunks = make_chunks(20, MIN_CHUNK_CHARS) + short_chunks

    first = sample_chunks(chunks, 10, seed=42)

    assert first == sample_chunks(chunks, 10, seed=42)
    assert first != sample_chunks(chunks, 10, seed=7)
    assert len({c.id for c in first}) == 10
    assert not any(c.id.startswith("short") for c in first)


def test_sample_chunks_fails_clearly_when_too_few_chunks():
    with pytest.raises(ValueError, match="Only 3 chunks"):
        sample_chunks(make_chunks(3, MIN_CHUNK_CHARS), 10, seed=1)


def test_build_dataset_labels_each_question_with_its_source_chunk():
    llm = MagicMock()
    llm.with_structured_output.return_value.invoke.return_value = GeneratedQuestions(
        questions=["  How many days?  ", "", "Who approves?", "extra question"]
    )
    chunk = Document(id="c42", page_content="Vacation text", metadata={"source": "hr/v.md", "category": "hr"})

    examples = build_retrieval_dataset(llm, [chunk], questions_per_chunk=2)

    assert examples == [
        RetrievalExample("How many days?", "c42", "hr/v.md", "hr", content_hash("Vacation text")),
        RetrievalExample("Who approves?", "c42", "hr/v.md", "hr", content_hash("Vacation text")),
    ]
    prompt = llm.with_structured_output.return_value.invoke.call_args.args[0].to_string()
    assert "Write 2 different questions" in prompt and "Vacation text" in prompt


def test_dataset_round_trips_through_json(tmp_path):
    examples = [RetrievalExample("q", "c1", "a.md", "hr")]
    path = tmp_path / "sub" / "dataset.json"

    save_dataset(path, examples, {"seed": 42})

    assert load_dataset(path) == ({"seed": 42}, examples)


def test_evaluate_retriever_rejects_an_empty_dataset():
    with pytest.raises(ValueError, match="No examples"):
        evaluate_retriever(lambda q: RANKED, [])


@pytest.mark.parametrize(("gold", "cutoff", "expected"), [("c7", 5, 1), ("c9", 5, 3), ("c1", 3, None), ("zz", 5, None)])
def test_gold_rank_is_one_based_and_respects_cutoff(gold, cutoff, expected):
    assert gold_rank(RANKED, gold, cutoff) == expected


def test_duplicate_questions_from_different_chunks_are_kept_once():
    llm = MagicMock()
    llm.with_structured_output.return_value.invoke.return_value = GeneratedQuestions(
        questions=["How many vacation days do I get?"])
    chunks = [Document(id=f"c{i}", page_content=f"text {i}", metadata={"source": "a.md", "category": "hr"})
              for i in range(2)]

    examples = build_retrieval_dataset(llm, chunks, questions_per_chunk=1)

    assert [e.chunk_id for e in examples] == ["c0"]  # the second, identical question would be an ambiguous label


STORED = [Document(id="c1", page_content="27 vacation days"), Document(id="c2", page_content="VPN setup")]


def label(question: str, chunk_id: str, text: str) -> RetrievalExample:
    return RetrievalExample(question, chunk_id, "a.md", "hr", content_hash(text))


def test_check_dataset_accepts_current_labels():
    assert check_dataset([label("q1", "c1", "27 vacation days"), label("q2", "c2", "VPN setup")], STORED) == []


def test_check_dataset_detects_same_id_with_different_text():
    # Re-ingesting with another CHUNK_SIZE reuses ids (uuid5 of source#position) for new text.
    problems = check_dataset([label("q1", "c1", "the OLD chunk text")], STORED)
    assert len(problems) == 1 and "different text" in problems[0]


@pytest.mark.parametrize(("examples", "expected"), [
    ([], "no questions"),
    ([label("Same?", "c1", "27 vacation days"), label("same? ", "c2", "VPN setup")], "more than once"),
    ([label("q", "gone", "x")], "no longer in the database"),
    ([RetrievalExample("q", "c1", "a.md", "hr")], "no chunk hashes"),   # dataset from before hashes existed
])
def test_check_dataset_reports_problems(examples, expected):
    assert any(expected in problem for problem in check_dataset(examples, STORED))
