"""Agentic RAG graph (analyze -> retrieve -> grade -> rewrite/answer) - self-contained template.

Dependencies:  pip install langgraph langchain-core pydantic  (+ a chat model with structured output)

Rules: the LLM judges (stored in state), plain Python routes; retry_count bounds the loop;
no relevant documents -> fixed fallback answer WITHOUT calling the LLM.

Usage:
    graph = build_agentic_graph(retriever, llm)
    state = graph.invoke({"question": "...", "max_retries": 2})
    state["answer"], state["relevant_documents"]
"""

from functools import partial
from typing import Literal, TypedDict

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

DEFAULT_MAX_RETRIES = 2
NO_RELEVANT_DOCUMENTS_ANSWER = "I couldn't find information about this in the knowledge base."


class AgenticRAGState(TypedDict, total=False):
    question: str
    max_retries: int
    needs_retrieval: bool
    search_query: str
    documents: list[Document]
    relevant_documents: list[Document]
    retry_count: int
    answer: str


class RetrievalDecision(BaseModel):
    needs_retrieval: bool = Field(description="True if the knowledge base must be searched")
    reason: str = Field(description="One short sentence explaining the decision")


class RelevanceGrade(BaseModel):
    is_relevant: bool = Field(description="True if the document helps answer the question")


ANALYZE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "Decide whether answering requires searching the internal knowledge base. "
               "Questions about the organisation, its policies, products or tools: true. "
               "Greetings or general knowledge: false."),
    ("human", "{question}"),
])
GRADE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "Is the document relevant to the question? It is relevant if it helps to answer it."),
    ("human", "Question: {question}\n\nDocument:\n{document}"),
])
REWRITE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "The search returned nothing relevant. Rewrite the question into a better search query. "
               "Return ONLY the query."),
    ("human", "Original question: {question}\nPrevious search query: {search_query}"),
])
ANSWER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "Use ONLY the context. If it does not contain the answer, say you don't know.\n\nContext:\n{context}"),
    ("human", "{question}"),
])
DIRECT_PROMPT = ChatPromptTemplate.from_messages([("system", "Answer briefly."), ("human", "{question}")])


# --- nodes ----------------------------------------------------------------------------------------

def analyze_question(state: AgenticRAGState, llm) -> dict:  # [LLM decision]
    decision = llm.with_structured_output(RetrievalDecision).invoke(ANALYZE_PROMPT.invoke({"question": state["question"]}))
    return {"needs_retrieval": decision.needs_retrieval, "search_query": state["question"],
            "retry_count": 0, "relevant_documents": []}


def retrieve_documents(state: AgenticRAGState, retriever) -> dict:  # [deterministic]
    return {"documents": retriever.invoke(state["search_query"])}


def grade_documents(state: AgenticRAGState, llm) -> dict:  # [LLM decision], graded against the ORIGINAL question
    grader = llm.with_structured_output(RelevanceGrade)
    relevant = [
        document for document in state["documents"]
        if grader.invoke(GRADE_PROMPT.invoke({"question": state["question"], "document": document.page_content})).is_relevant
    ]
    return {"relevant_documents": relevant}


def rewrite_question(state: AgenticRAGState, llm) -> dict:  # [LLM generates text]
    messages = REWRITE_PROMPT.invoke({"question": state["question"], "search_query": state["search_query"]})
    return {"search_query": llm.invoke(messages).text.strip(), "retry_count": state["retry_count"] + 1}


def generate_answer(state: AgenticRAGState, llm) -> dict:
    if not state["needs_retrieval"]:
        return {"answer": llm.invoke(DIRECT_PROMPT.invoke({"question": state["question"]})).text}
    if not state["relevant_documents"]:
        return {"answer": NO_RELEVANT_DOCUMENTS_ANSWER}  # [deterministic fallback, no LLM call]
    context = "\n\n".join(d.page_content for d in state["relevant_documents"])
    return {"answer": llm.invoke(ANSWER_PROMPT.invoke({"context": context, "question": state["question"]})).text}


# --- routing: pure Python, no LLM --------------------------------------------------------------------

def route_after_analysis(state: AgenticRAGState) -> Literal["retrieve_documents", "generate_answer"]:
    return "retrieve_documents" if state["needs_retrieval"] else "generate_answer"


def route_after_grading(state: AgenticRAGState) -> Literal["generate_answer", "rewrite_question"]:
    if state["relevant_documents"]:
        return "generate_answer"
    if state["retry_count"] >= state.get("max_retries", DEFAULT_MAX_RETRIES):
        return "generate_answer"  # give up -> fallback answer
    return "rewrite_question"


def build_agentic_graph(retriever, llm):
    graph = StateGraph(AgenticRAGState)
    graph.add_node("analyze_question", partial(analyze_question, llm=llm))
    graph.add_node("retrieve_documents", partial(retrieve_documents, retriever=retriever))
    graph.add_node("grade_documents", partial(grade_documents, llm=llm))
    graph.add_node("rewrite_question", partial(rewrite_question, llm=llm))
    graph.add_node("generate_answer", partial(generate_answer, llm=llm))
    graph.add_edge(START, "analyze_question")
    graph.add_conditional_edges("analyze_question", route_after_analysis)
    graph.add_edge("retrieve_documents", "grade_documents")
    graph.add_conditional_edges("grade_documents", route_after_grading)
    graph.add_edge("rewrite_question", "retrieve_documents")
    graph.add_edge("generate_answer", END)
    return graph.compile()
