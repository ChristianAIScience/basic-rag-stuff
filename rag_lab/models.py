"""Factories for the OpenAI models. Every part of the project gets its models from here.

- The embedding model turns text into a vector (a list of floats). Texts with similar
  meaning get vectors that point in similar directions.
- The chat model (LLM) generates answers, gradings and rewritten questions.
"""

from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from rag_lab.config import Settings


def get_embeddings(settings: Settings) -> OpenAIEmbeddings:
    return OpenAIEmbeddings(model=settings.embedding_model, api_key=settings.llm_api_key)


def get_chat_model(settings: Settings) -> ChatOpenAI:
    return ChatOpenAI(model=settings.llm_model, api_key=settings.llm_api_key)
