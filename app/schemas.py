"""Shared Pydantic schemas: agent plans, traces, extracted content, API responses."""
from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


class ExtractedItem(BaseModel):
    """One piece of content pulled out of an input (file or inline text)."""

    source: str                       # filename or "user_text"
    modality: Literal["text", "image", "pdf", "audio"]
    content: str
    method: str                       # e.g. "pdf_text", "ocr_tesseract", "whisper"
    confidence: Optional[float] = None  # OCR confidence 0-100 where available
    meta: dict[str, Any] = Field(default_factory=dict)  # pages, duration, urls...


# Every field is a plain, concrete type — no free-form dict. An earlier
# version used `args: dict[str, Any]`, which is unreliable for structured /
# tool-calling output: nothing constrains what goes in it, which both blew
# through completion-token budgets under raw JSON mode and triggered outright
# `tool_use_failed` errors under tool-calling mode. Each tool only ever needs
# a small fixed set of things from the model, so those are named explicitly.
#
# Critically, none of these fields ever carries file content itself —
# `input_source` is a closed choice ("context" or "previous_step"), and the
# executor substitutes the real extracted content or real prior tool output
# deterministically. The model can never author fabricated "content" that
# masquerades as the real thing.
#
# (Kept as a comment, not a docstring: Pydantic serializes docstrings into
# the JSON schema sent to the model on every planning call, so a long one
# here would waste tokens and could itself add confusion.)
class PlanStep(BaseModel):
    """One planned tool call."""

    tool: str
    url: Optional[str] = None                 # for fetch_youtube_transcript / fetch_url
    input_source: Literal["context", "previous_step"] = "context"
    question: Optional[str] = None            # for answer_question / compare_inputs / explain_code
    focus: Optional[str] = None               # optional emphasis for summarize
    reason: str = ""


class AgentPlan(BaseModel):
    """Structured output of the planner LLM call."""

    action: Literal["execute", "clarify"]
    clarify_question: Optional[str] = None
    steps: list[PlanStep] = Field(default_factory=list)


class TraceEvent(BaseModel):
    """One row of the visible plan / tool trace shown in the UI."""

    stage: str          # ingest | plan | tool | synthesize | error
    title: str
    detail: str = ""
    status: Literal["ok", "error", "skipped", "info"] = "ok"


class CostEstimate(BaseModel):
    input_tokens: int
    output_tokens: int
    usd: float


class ChatResponse(BaseModel):
    session_id: str
    kind: Literal["answer", "clarify", "error"]
    answer: str
    extracted: list[ExtractedItem] = Field(default_factory=list)
    trace: list[TraceEvent] = Field(default_factory=list)
    cost_estimate: Optional[CostEstimate] = None
