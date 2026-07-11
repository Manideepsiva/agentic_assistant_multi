"""Single place to construct Groq chat models."""
from functools import lru_cache

from langchain_groq import ChatGroq

from app.config import get_settings
from app.schemas import AgentPlan


@lru_cache
def get_llm() -> ChatGroq:
    """General-purpose model: text tools + final answer synthesis."""
    settings = get_settings()
    return ChatGroq(model=settings.llm_model, temperature=settings.temperature,
                    api_key=settings.groq_api_key, max_tokens=settings.llm_max_tokens)


@lru_cache
def get_planner_llm() -> ChatGroq:
    """Base model used for planning, before structured-output wrapping."""
    settings = get_settings()
    return ChatGroq(model=settings.llm_model, temperature=settings.temperature,
                    api_key=settings.groq_api_key, max_tokens=settings.planner_max_tokens)


@lru_cache
def get_planner_chain():
    """Planner as a tool-calling structured-output chain.
    """
    return get_planner_llm().with_structured_output(
        AgentPlan, method="function_calling", include_raw=True
    )


@lru_cache
def get_vision_llm() -> ChatGroq:
    """Vision-capable model for the OCR fallback (image transcription)."""
    settings = get_settings()
    return ChatGroq(model=settings.vision_model, temperature=settings.temperature,
                    api_key=settings.groq_api_key, max_tokens=settings.llm_max_tokens)
