"""Small helpers shared by the RAG chain, the graphs and the examples."""

from langchain_core.documents import Document


def format_docs(documents: list[Document]) -> str:
    """Join retrieved chunks into one context string for the prompt, tagged with their source."""
    return "\n\n".join(
        f"[Source: {document.metadata.get('source', 'unknown')}]\n{document.page_content}"
        for document in documents
    )


def print_answer(answer: str, sources: list[Document]) -> None:
    print(f"\nAnswer:\n{answer}\n")
    print(f"Sources ({len(sources)}):")
    for document in sources:
        preview = document.page_content.replace("\n", " ")[:80]
        print(f"  - [{document.metadata.get('source')}] {preview}...")
