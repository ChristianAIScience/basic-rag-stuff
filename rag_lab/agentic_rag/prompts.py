"""Prompts and structured-output schemas for the LLM-driven decisions.

`llm.with_structured_output(Schema)` makes the model return a validated Pydantic
object instead of free text - so a decision is a real `bool`, not a string to parse.
"""

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

# --- analyze_question --------------------------------------------------------


class RetrievalDecision(BaseModel):
    needs_retrieval: bool = Field(description="True if the knowledge base must be searched")
    reason: str = Field(description="One short sentence explaining the decision")


ANALYZE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You route questions for an assistant of the company Lumora Systems.\n"
            "The internal knowledge base contains HR policies, product information "
            "(Glowbeam lamps, pricing) and IT support guides.\n"
            "Decide whether answering requires searching that knowledge base.\n"
            "- Questions about Lumora, its policies, products, tools or IT: needs_retrieval = true\n"
            "- Greetings, small talk or general knowledge (math, definitions): needs_retrieval = false",
        ),
        ("human", "{question}"),
    ]
)

# --- grade_documents ---------------------------------------------------------


class RelevanceGrade(BaseModel):
    is_relevant: bool = Field(description="True if the document helps answer the question")


GRADE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You check whether a retrieved document is relevant to a user question.\n"
            "It is relevant if it contains information that helps to answer the question. "
            "It does not need to answer it completely.",
        ),
        ("human", "Question: {question}\n\nDocument:\n{document}"),
    ]
)

# --- rewrite_question --------------------------------------------------------

REWRITE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "A search in the Lumora Systems knowledge base did not return relevant documents.\n"
            "Rewrite the question into a better search query: use specific keywords, "
            "expand abbreviations, remove filler words. Return ONLY the new query.",
        ),
        ("human", "Original question: {question}\nPrevious search query: {search_query}"),
    ]
)

# --- generate_answer (no retrieval needed) -----------------------------------

DIRECT_ANSWER_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. Answer briefly."),
        ("human", "{question}"),
    ]
)

# Deterministic answer when retrieval was needed but nothing relevant was found.
NO_RELEVANT_DOCUMENTS_ANSWER = (
    "I couldn't find information about this in the Lumora Systems knowledge base, "
    "so I can't answer it reliably."
)
