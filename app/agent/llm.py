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

    We used to force raw JSON mode (response_format=json_object), which asks
    the model to freely write JSON text. That mode is unconstrained: if the
    model rambles even slightly (a longer "reason" field, stray commentary),
    it can run past its token ceiling before the JSON closes — and Groq's
    json_object mode is all-or-nothing, rejecting the whole response instead
    of returning what it has. Raising max_tokens only delayed that failure.

    Tool-calling structured output is different: the model fills in a
    strictly-typed function signature (AgentPlan's exact fields) and stops as
    soon as they're complete, rather than free-writing text. This is the
    reliable path Groq/Llama recommend for structured data, and it's what
    actually fixes the "max completion tokens reached" failures rather than
    just postponing them.

    include_raw=True so callers can distinguish "model refused/failed to
    produce a valid call" from a normal successful plan, without relying on
    exceptions for control flow.
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
