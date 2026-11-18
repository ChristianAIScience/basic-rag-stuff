"""The RAG prompt: where retrieved context and the user's question meet.

The system message tells the model to answer ONLY from the context. This is
what makes the answer "grounded" - and lets it admit when it doesn't know.
The LangGraph examples reuse this prompt so all implementations stay comparable.
"""

from langchain_core.prompts import ChatPromptTemplate

RAG_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You answer questions about the company Lumora Systems.\n"
            "Use ONLY the context below. If the context does not contain the answer, "
            "say that you don't know based on the available documents.\n"
            "Keep the answer short and mention the source files you used.\n\n"
            "Context:\n{context}",
        ),
        ("human", "{question}"),
    ]
)
