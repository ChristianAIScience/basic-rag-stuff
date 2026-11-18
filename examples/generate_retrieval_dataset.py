"""Label chunks for retrieval evaluation: sample random chunks, let an LLM write questions about them.

Run:  uv run python -m examples.generate_retrieval_dataset
      uv run python -m examples.generate_retrieval_dataset --chunks 10 --questions-per-chunk 2 --seed 42
(Requires: docker compose up -d, ingested documents, and a real LLM_API_KEY.)

Writes evaluation_data/retrieval_dataset.json - review the questions there before evaluating.
Each label stores a hash of its chunk text; after re-ingesting with other chunking,
examples.evaluate_retrieval refuses the stale labels - generate the dataset again then.
"""

import argparse
from pathlib import Path

from rag_lab.config import load_settings
from rag_lab.db.chunks import fetch_all_chunks
from rag_lab.evaluation.question_generation import build_retrieval_dataset, sample_chunks, save_dataset
from rag_lab.models import get_chat_model

DEFAULT_OUTPUT = Path(__file__).resolve().parent.parent / "evaluation_data" / "retrieval_dataset.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--chunks", type=int, default=10, help="number of random chunks to label")
    parser.add_argument("--questions-per-chunk", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42, help="same seed -> same chunks")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    settings = load_settings()
    all_chunks = fetch_all_chunks(settings)
    chunks = sample_chunks(all_chunks, args.chunks, args.seed)
    print(f"Sampled {len(chunks)} of {len(all_chunks)} chunks (seed={args.seed}); generating questions ...\n")

    examples = build_retrieval_dataset(get_chat_model(settings), chunks, args.questions_per_chunk)

    for chunk in chunks:
        print(f"[{chunk.metadata['source']}] {chunk.page_content.replace(chr(10), ' ')[:90]}...")
        for example in examples:
            if example.chunk_id == chunk.id:
                print(f"   Q: {example.question}")

    info = {
        "seed": args.seed,
        "chunks": args.chunks,
        "questions_per_chunk": args.questions_per_chunk,
        "chunk_size": settings.chunk_size,
        "chunk_overlap": settings.chunk_overlap,
        "generator_model": settings.llm_model,
    }
    save_dataset(args.output, examples, info)
    print(f"\nSaved {len(examples)} labelled questions to {args.output}")


if __name__ == "__main__":
    main()
