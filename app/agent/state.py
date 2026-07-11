"""LangGraph agent state."""
from __future__ import annotations

from typing import Optional, TypedDict

from app.schemas import AgentPlan, ExtractedItem, TraceEvent


class AgentState(TypedDict, total=False):
    # Inputs
    query: str
    history: list[dict]              # [{"role": "user"|"assistant", "content": str}]
    extracted: list[ExtractedItem]

    # Planning
    plan: Optional[AgentPlan]

    # Execution
    step_outputs: list[str]          # output of each executed plan step, in order

    # Output
    answer: str
    kind: str                        # "answer" | "clarify" | "error"
    trace: list[TraceEvent]
