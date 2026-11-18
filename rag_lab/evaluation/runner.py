"""Run a RAG target over the evaluation dataset and grade every answer with all judges."""

import time
from dataclasses import dataclass

from langchain_core.language_models import BaseChatModel

from rag_lab.evaluation.dataset import EvalExample
from rag_lab.evaluation.judges import JUDGE_NAMES, JudgeResult, run_all_judges
from rag_lab.evaluation.targets import Target


@dataclass
class ExampleResult:
    question: str
    answer: str
    judgments: list[JudgeResult]
    duration_s: float  # time of the RAG target only, without judging


def evaluate_target(target: Target, examples: list[EvalExample], judge_llm: BaseChatModel) -> list[ExampleResult]:
    results = []
    for example in examples:
        start = time.perf_counter()
        output = target(example.question)
        duration_s = time.perf_counter() - start

        judgments = run_all_judges(
            judge_llm, example.question, output["answer"], output["documents"], example.reference_answer
        )
        results.append(ExampleResult(example.question, output["answer"], judgments, duration_s))
    return results


def pass_rates(results: list[ExampleResult]) -> dict[str, tuple[int, int]]:
    """Per judge: (passed, applicable). Not-applicable verdicts (None) are not counted."""
    rates = {}
    for key in JUDGE_NAMES:
        verdicts = [j.passed for r in results for j in r.judgments if j.key == key and j.passed is not None]
        rates[key] = (sum(verdicts), len(verdicts))
    return rates
