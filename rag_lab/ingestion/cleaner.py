"""Step 2 of ingestion: light text cleaning.

Real-world text often contains trailing spaces, Windows line endings and
long runs of blank lines. They add noise to chunks and waste embedding tokens.
We keep paragraph breaks ("\n\n") because the splitter uses them as natural boundaries.
"""

import re

from langchain_core.documents import Document


def clean_text(text: str) -> str:
    text = text.replace("\r\n", "\n")
    text = re.sub(r"[ \t]+\n", "\n", text)  # trailing whitespace
    text = re.sub(r"\n{3,}", "\n\n", text)  # at most one blank line in a row
    return text.strip()


def clean_documents(documents: list[Document]) -> list[Document]:
    """Clean every document and drop the ones that end up empty."""
    cleaned = []
    for document in documents:
        text = clean_text(document.page_content)
        if text:
            cleaned.append(Document(page_content=text, metadata=document.metadata))
    return cleaned
