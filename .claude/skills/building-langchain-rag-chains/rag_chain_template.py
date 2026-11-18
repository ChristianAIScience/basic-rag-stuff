"""Basic LangChain RAG chain (LCEL) returning answer + sources - self-contained template.

Dependencies:  pip install langchain-core  (+ a chat model package, e.g. langchain-openai)

Usage:
    chain = build_rag_chain(vector_store.as_retriever(search_kwargs={"k": 4}), ChatOpenAI(model="..."))
    result = chain.invoke("How many vacation days do I get?")
    result["answer"], result["sources"]

Offline test (no API calls):
    from langchain_core.language_models import GenericFakeChatModel
    from langchain_core.messages import AIMessage
    from langchain_core.runnables import RunnableLambda
    chain = build_rag_chain(RunnableLambda(lambda q: docs), GenericFakeChatModel(messages=iter([AIMessage("27 days")])))
"""

from operator import itemgetter

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableParallel, RunnablePassthrough

# Grounding: answer ONLY from the context, admit when it is missing. Adapt the company/domain line.
RAG_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You answer questions about our internal documents.\n"
            "Use ONLY the context below. If the context does not contain the answer, "
            "say that you don't know based on the available documents.\n"
            "Keep the answer short and mention the source files you used.\n\n"
            "Context:\n{context}",
        ),
        ("human", "{question}"),
    ]
)


def format_docs(documents: list[Document]) -> str:
    """Chunks joined into one context string, each tagged with its source."""
    return "\n\n".join(
        f"[Source: {document.metadata.get('source', 'unknown')}]\n{document.page_content}" for document in documents
    )


def build_rag_chain(retriever: Runnable, llm: Runnable) -> Runnable:
    """question -> {"question", "sources", "answer"}."""
    generate_answer = (
        {
            # Two plain callables cannot be piped with |, so a lambda is used here.
            "context": lambda inputs: format_docs(inputs["sources"]),
            "question": itemgetter("question"),
        }
        | RAG_PROMPT
        | llm
        | StrOutputParser()
    )
    return RunnableParallel(sources=retriever, question=RunnablePassthrough()).assign(answer=generate_answer)
