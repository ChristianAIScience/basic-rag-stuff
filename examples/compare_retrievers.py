"""Run all retrieval strategies with the same question and compare the results.

Run:  uv run python -m examples.compare_retrievers
      uv run python -m examples.compare_retrievers "How many vacation days do I get?" hr
(Requires: docker compose up -d  and  uv run python -m examples.ingest_documents)
"""

import sys

from rag_lab.config import load_settings
from rag_lab.db.connection import get_engine
from rag_lab.db.vector_store import get_vector_store
from rag_lab.models import get_embeddings
from rag_lab.retrieval import hybrid, metadata_filter, mmr, score_threshold, similarity
from rag_lab.retrieval.comparison import print_run, run_strategy

# The default question contains an exact error code - a case where keyword search helps.
DEFAULT_QUESTION = "What is error TK-407?"
DEFAULT_CATEGORY = "it_support"


def main() -> None:
    question = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_QUESTION
    category = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_CATEGORY

    settings = load_settings()
    vector_store = get_vector_store(get_engine(settings), get_embeddings(settings), settings)
    print(f"Question: {question!r}")

    runs = [
        run_strategy("1. Similarity search", similarity.SCORE_LABEL,
                     lambda: similarity.similarity_search(vector_store, question)),
        run_strategy("2. MMR (lambda_mult=0.5)", mmr.SCORE_LABEL,
                     lambda: mmr.mmr_search(vector_store, question)),
        run_strategy("3. Score threshold (>= 0.4)", score_threshold.SCORE_LABEL,
                     lambda: score_threshold.score_threshold_search(vector_store, question, 0.4)),
        run_strategy(f"4. Metadata filter (category={category})", metadata_filter.SCORE_LABEL,
                     lambda: metadata_filter.metadata_filter_search(vector_store, question, category)),
        run_strategy("5. Hybrid (full-text + vector, RRF)", hybrid.SCORE_LABEL,
                     lambda: hybrid.hybrid_search(vector_store, question)),
    ]
    for run in runs:
        print_run(run)

    print("\nNote: the first run includes the embedding API call; scores of different")
    print("strategies have different meanings and must not be compared with each other.")


if __name__ == "__main__":
    main()
