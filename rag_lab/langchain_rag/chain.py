"""The basic RAG chain, built with LCEL (the LangChain Expression Language).

    question --> retriever --> sources ---+
       |                                  v
       +--------------------------> format prompt --> LLM --> answer

The `|` operator connects Runnables: the output of one becomes the input of the next.
The chain is a fixed pipeline: every question goes through exactly the same steps.

Input:  "How many vacation days do I get?"
Output: {"question": ..., "sources": [Document, ...], "answer": "..."}
"""

from operator import itemgetter

from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import Runnable, RunnableParallel, RunnablePassthrough

from rag_lab.langchain_rag.prompt import RAG_PROMPT
from rag_lab.utils.formatting import format_docs


def build_rag_chain(retriever: BaseRetriever, llm: BaseChatModel) -> Runnable:
    # Step 2: turn {"question", "sources"} into an answer string.
    generate_answer = (
        {
            "context": lambda inputs: format_docs(inputs["sources"]),
            "question": itemgetter("question"),
        }
        | RAG_PROMPT
        | llm
        | StrOutputParser()
    )

    # Step 1: run the retriever and keep the question; then add the answer.
    # Returning the sources (not only the answer) lets the user verify the answer.
    return RunnableParallel(
        sources=retriever,
        question=RunnablePassthrough(),
    ).assign(answer=generate_answer)
