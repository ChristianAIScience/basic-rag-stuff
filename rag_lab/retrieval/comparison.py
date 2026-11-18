"""Run a retrieval strategy, measure how long it takes, and print the results."""

import time
from collections.abc import Callable
from dataclasses import dataclass

from langchain_core.documents import Document

Results = list[tuple[Document, float | None]]


@dataclass
class StrategyRun:
    name: str
    score_label: str
    results: Results
    duration_ms: float


def run_strategy(name: str, score_label: str, search: Callable[[], Results]) -> StrategyRun:
    start = time.perf_counter()
    results = search()
    duration_ms = (time.perf_counter() - start) * 1000
    return StrategyRun(name, score_label, results, duration_ms)


def print_run(run: StrategyRun, preview_chars: int = 90) -> None:
    print(f"\n=== {run.name}  ({len(run.results)} chunks, {run.duration_ms:.0f} ms)")
    print(f"    score: {run.score_label}")
    if not run.results:
        print("    (no results)")
    for document, score in run.results:
        score_text = "   -  " if score is None else f"{score:6.3f}"
        preview = document.page_content.replace("\n", " ")[:preview_chars]
        print(f"    {score_text}  [{document.metadata.get('source')}]  {preview}...")
