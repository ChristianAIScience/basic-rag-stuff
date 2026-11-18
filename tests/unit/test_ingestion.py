from langchain_core.documents import Document

from rag_lab.ingestion.cleaner import clean_documents, clean_text
from rag_lab.ingestion.loader import load_documents
from rag_lab.ingestion.pipeline import make_chunk_ids
from rag_lab.ingestion.splitter import split_documents


def test_loader_reads_txt_and_md_and_sets_metadata(tmp_path):
    (tmp_path / "hr").mkdir()
    (tmp_path / "hr" / "policy.md").write_text("# Policy", encoding="utf-8")
    (tmp_path / "notes.txt").write_text("Notes", encoding="utf-8")
    (tmp_path / "image.png").write_bytes(b"\x89PNG")

    documents = load_documents(tmp_path)

    assert [d.metadata for d in documents] == [
        {"source": "hr/policy.md", "category": "hr"},
        {"source": "notes.txt", "category": "general"},
    ]


def test_clean_text_normalizes_whitespace():
    assert clean_text("  a  \r\nb\n\n\n\nc  ") == "a\nb\n\nc"


def test_clean_documents_drops_empty_documents():
    documents = [Document(page_content="   \n "), Document(page_content="text", metadata={"x": 1})]
    assert clean_documents(documents) == [Document(page_content="text", metadata={"x": 1})]


def test_splitter_respects_chunk_size_and_keeps_metadata():
    text = " ".join(f"word{i}" for i in range(200))
    chunks = split_documents([Document(page_content=text, metadata={"source": "a.txt"})], 100, 20)

    assert len(chunks) > 1
    assert all(len(chunk.page_content) <= 100 for chunk in chunks)
    assert all(chunk.metadata["source"] == "a.txt" for chunk in chunks)
    assert chunks[1].metadata["start_index"] > 0


def test_splitter_overlap_repeats_text_between_chunks():
    text = " ".join(f"word{i}" for i in range(50))
    first, second, *_ = split_documents([Document(page_content=text)], 60, 20)
    assert first.page_content.split()[-1] in second.page_content


def test_chunk_ids_are_deterministic_and_unique():
    chunks = [
        Document(page_content="a", metadata={"source": "x.md"}),
        Document(page_content="b", metadata={"source": "x.md"}),
        Document(page_content="c", metadata={"source": "y.md"}),
    ]
    ids = make_chunk_ids(chunks)
    assert ids == make_chunk_ids(chunks)
    assert len(set(ids)) == 3
