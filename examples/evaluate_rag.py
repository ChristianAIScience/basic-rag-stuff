"""Grade the three RAG implementations with LLM judges (LLM-as-a-judge).

Run:  uv run python -m examples.evaluate_rag                       (all three targets)
      uv run python -m examples.evaluate_rag agentic --limit 3     (one target, first 3 questions)
(Requires: docker compose up -d, ingestion with REAL embeddings, and a real LLM_API_KEY.)

Cost: per question and target 1 RAG run + up to 4 judge calls.
All targets x 9 questions = 27 RAG runs + up to 108 judge calls.
"""

import argparse
import time

from rag_lab.agentic_rag.graph import build_agentic_graph
from rag_lab.config import load_settings
from rag_lab.db.connection import get_engine
from rag_lab.db.vector_store import get_vector_store
from rag_lab.evaluation.dataset import EVAL_EXAMPLES
from rag_lab.evaluation.judges import JUDGE_NAMES
from rag_lab.evaluation.runner import ExampleResult, evaluate_target, pass_rates
from rag_lab.evaluation.targets import agentic_target, langchain_target, langgraph_target
from rag_lab.langchain_rag.chain import build_rag_chain
from rag_lab.langchain_rag.retriever import build_retriever
from rag_lab.langgraph_rag.graph import build_rag_graph
from rag_lab.models import get_chat_model, get_embeddings, get_judge_model

SYMBOL = {True: "pass", False: "FAIL", None: "n/a"}


def print_results(target_name: str, results: list[ExampleResult], duration_s: float) -> None:
    print(f"\n=== {target_name}  ({len(results)} questions, {duration_s:.0f} s incl. judging)")
    print(f"  {'question':<58}" + "".join(f"{name[:12]:>13}" for name in JUDGE_NAMES))
    for result in results:
        verdicts = {j.key: j.passed for j in result.judgments}
        print(f"  {result.question[:58]:<58}" + "".join(f"{SYMBOL[verdicts[name]]:>13}" for name in JUDGE_NAMES))

    rates = pass_rates(results)
    print(f"  {'pass rate':<58}" + "".join(f"{f'{p}/{n}':>13}" for p, n in (rates[name] for name in JUDGE_NAMES)))

    failed = [(r, j) for r in results for j in r.judgments if j.passed is False]
    if failed:
        print("  Failed judgments (judge reasoning):")
        for result, judgment in failed:
            print(f"   - [{judgment.key}] {result.question[:50]}")
            print(f"     answer:  {result.answer[:150]!r}")
            print(f"     reason:  {judgment.explanation[:300]}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("targets", nargs="*", choices=["langchain", "langgraph", "agentic"], default=[],
                        help="which implementations to grade (default: all)")
    parser.add_argument("--limit", type=int, default=None, help="only the first N questions")
    args = parser.parse_args()

    settings = load_settings()
    vector_store = get_vector_store(get_engine(settings), get_embeddings(settings), settings)
    retriever, llm = build_retriever(vector_store), get_chat_model(settings)
    targets = {
        "langchain": langchain_target(build_rag_chain(retriever, llm)),
        "langgraph": langgraph_target(build_rag_graph(retriever, llm)),
        "agentic": agentic_target(build_agentic_graph(retriever, llm)),
    }
    selected = args.targets or list(targets)
    examples = EVAL_EXAMPLES[: args.limit]
    judge_llm = get_judge_model(settings)
    print(f"Answer model: {settings.llm_model} | judge model: {settings.judge_model} | {len(examples)} questions")

    summary = {}
    for name in selected:
        start = time.perf_counter()
        results = evaluate_target(targets[name], examples, judge_llm)
        print_results(name, results, time.perf_counter() - start)
        summary[name] = pass_rates(results)

    print("\n=== Summary (passed / applicable)")
    print(f"  {'target':<12}" + "".join(f"{name[:19]:>20}" for name in JUDGE_NAMES))
    for name, rates in summary.items():
        print(f"  {name:<12}" + "".join(f"{f'{p}/{n}':>20}" for p, n in (rates[key] for key in JUDGE_NAMES)))
    print("\nJudges are LLMs too: check failed verdicts by reading their reasoning, and run")
    print("examples.calibrate_judges to see how often each judge agrees with a human.")


if __name__ == "__main__":
    main()
