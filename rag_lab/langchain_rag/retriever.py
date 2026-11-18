"""The retriever: the component that turns a question into relevant chunks.

`as_retriever()` wraps the vector store in LangChain's standard Retriever
interface: `retriever.invoke("question") -> list[Document]`. Because it is a
Runnable, it can be plugged directly into a chain with the `|` operator.
"""

from langchain_core.vectorstores import VectorStore, VectorStoreRetriever


def build_retriever(vector_store: VectorStore, k: int = 4) -> VectorStoreRetriever:
    # search_type could also be "mmr" or "similarity_score_threshold" - see rag_lab/retrieval/.
    return vector_store.as_retriever(search_type="similarity", search_kwargs={"k": k})
