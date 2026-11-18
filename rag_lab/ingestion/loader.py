"""Step 1 of ingestion: load .txt, .md and .pdf files into LangChain Document objects.

A Document is just text (`page_content`) plus a dict of `metadata`.
We record where each document came from, and use the folder name as its category:

    data/hr/vacation_policy.md  ->  source="hr/vacation_policy.md", category="hr"

Text files become ONE Document each. PDFs become one Document PER SECTION,
split at headings detected by font size (see pdf_loader.py).
"""

from pathlib import Path

from langchain_core.documents import Document

from rag_lab.ingestion.pdf_loader import load_pdf_sections

SUPPORTED_SUFFIXES = {".txt", ".md", ".pdf"}


def load_documents(data_dir: Path) -> list[Document]:
    documents = []
    for path in sorted(data_dir.rglob("*")):
        if path.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue
        relative_path = path.relative_to(data_dir)
        metadata = {
            "source": relative_path.as_posix(),
            "category": relative_path.parts[0] if len(relative_path.parts) > 1 else "general",
        }
        if path.suffix.lower() == ".pdf":
            documents.extend(load_pdf_sections(path, metadata))
        else:
            documents.append(Document(page_content=path.read_text(encoding="utf-8"), metadata=metadata))
    return documents
