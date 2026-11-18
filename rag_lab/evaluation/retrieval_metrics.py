"""Retrieval metrics: did the retriever put the gold chunk near the top?

For one question the retriever returns a ranked list of chunk ids, e.g. [c7, c2, c9, ...],
and exactly one chunk (the gold chunk) is labelled relevant.

Hit@k               1 if the gold chunk is among the first k results, else 0.
                    Hit@1 = "top result is right", Hit@5 = "it is somewhere in the top 5".
Reciprocal rank     1 / position of the gold chunk: position 1 -> 1.0, 2 -> 0.5, 3 -> 0.33 ...
                    0 if it is not within the cutoff k (MRR@k).
MRR                 the mean reciprocal rank over all questions. Unlike Hit@k it rewards
                    "first" more than "third" - a single number for ranking quality.

Averaged over all questions, Hit@k is also called recall@k here (one relevant chunk per question).
No LLM is involved: the metrics are exact and free to compute.
"""

from collections.abc import Callable
from dataclasses import dataclass

from rag_lab.evaluation.question_generation import RetrievalExample

# A retriever under test: question -> ranked list of chunk ids (best first).
RankedRetriever = Callable[[str], list[str]]


def hit_at_k(ranked_ids: list[str], gold_id: str, k: int) -> float:
    return 1.0 if gold_id in ranked_ids[:k] else 0.0


def reciprocal_rank(ranked_ids: list[str], gold_id: str, k: int) -> float:
    top_k = ranked_ids[:k]
    return 1.0 / (top_k.index(gold_id) + 1) if gold_id in top_k else 0.0


@dataclass
class RetrievalScores:
    hit_at: dict[int, float]   # k -> mean Hit@k over all questions
    mrr: float                 # mean reciprocal rank at cutoff max(k_values)
    gold_ranks: list[int | None]  # per question: 1-based position of the gold chunk, None = not found

    def as_metrics(self) -> dict[str, float]:
        """MLflow metric names allow no "@": {"hit_at_1": ..., "hit_at_5": ..., "mrr_at_5": ...}."""
        metrics = {f"hit_at_{k}": value for k, value in self.hit_at.items()}
        metrics[f"mrr_at_{max(self.hit_at)}"] = self.mrr
        return metrics


def evaluate_retriever(
    retriever: RankedRetriever, examples: list[RetrievalExample], k_values: tuple[int, ...] = (1, 3, 5)
) -> RetrievalScores:
    if not examples:
        raise ValueError("No examples to evaluate - generate the retrieval dataset first.")
    cutoff = max(k_values)
    # One retrieval per question; every metric is computed from the same ranked list.
    runs = [(retriever(example.question), example.chunk_id) for example in examples]

    count = len(examples)
    return RetrievalScores(
        hit_at={k: sum(hit_at_k(ranked, gold, k) for ranked, gold in runs) / count for k in k_values},
        mrr=sum(reciprocal_rank(ranked, gold, cutoff) for ranked, gold in runs) / count,
        gold_ranks=[gold_rank(ranked, gold, cutoff) for ranked, gold in runs],
    )


def gold_rank(ranked_ids: list[str], gold_id: str, cutoff: int) -> int | None:
    """1-based position of the gold chunk within the first `cutoff` results, None if it is not there."""
    top = ranked_ids[:cutoff]
    return top.index(gold_id) + 1 if gold_id in top else None
