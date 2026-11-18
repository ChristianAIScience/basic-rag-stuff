"""Step 3 of ingestion: split documents into chunks.

Why split? A retriever returns whole stored units. Small, focused chunks make the
retrieved context precise and keep the prompt short.

- chunk_size:    maximum characters per chunk
- chunk_overlap: characters repeated between neighbouring chunks, so a sentence
                 cut at a boundary still appears complete in one of them

RecursiveCharacterTextSplitter first tries to split on paragraphs ("\n\n"),
then on lines, then on spaces - it keeps related text together where it can.
"""

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def split_documents(documents: list[Document], chunk_size: int, chunk_overlap: int) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True,  # stores the chunk's position in metadata["start_index"]
    )
    return splitter.split_documents(documents)
