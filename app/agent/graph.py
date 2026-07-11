"""LangGraph agent core.

Graph:  plan --(clarify)--> END
          \\--(execute)--> execute_tools --> synthesize --> END

Ingestion happens before the graph (in the API layer) so extracted content can
be cached per session and reused across a clarification round-trip.
"""
from __future__ import annotations

import logging

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph

from app.agent.llm import get_llm, get_planner_chain
from app.agent.prompts import PLANNER_SYSTEM, SYNTHESIZER_SYSTEM
from app.agent.state import AgentState
from app.schemas import AgentPlan, PlanStep, TraceEvent
from app.tools.registry import REGISTRY, planner_tool_manifest

logger = logging.getLogger(__name__)

MAX_STEPS = 8

# Tools whose primary job is to process real extracted content, and the
# argument name that content lands in. The model never authors this content
# itself (see PlanStep.input_source) — the executor always substitutes the
# real thing.
CONTENT_ARG_BY_TOOL = {"summarize": "text", "sentiment_analysis": "text", "explain_code": "code"}


# ---------------------------------------------------------------- helpers
def _context_block(state: AgentState, per_item_limit: int = 20000) -> str:
    parts = []
    for item in state.get("extracted", []):
        header = f"### Source: {item.source} ({item.modality}, via {item.method})"
        if item.confidence is not None:
            header += f" [OCR confidence {item.confidence}%]"
        if item.meta.get("duration_seconds"):
            header += f" [duration {item.meta['duration_seconds']}s]"
        if item.meta.get("urls"):
            header += f"\nDiscovered URLs: {', '.join(item.meta['urls'])}"
        parts.append(f"{header}\n{item.content[:per_item_limit]}")
    return "\n\n".join(parts) if parts else "(no files uploaded)"


def _history_block(state: AgentState) -> str:
    hist = state.get("history", [])
    if not hist:
        return "(none)"
    return "\n".join(f"{m['role']}: {m['content'][:2000]}" for m in hist[-6:])


def _content_for_step(step: PlanStep, step_index: int, outputs: list[str], state: AgentState) -> str:
    """The real content a text tool should operate on — never authored by the
    model, always substituted deterministically."""
    if step.input_source == "previous_step" and step_index > 0 and step_index - 1 < len(outputs):
        return outputs[step_index - 1]
    if state.get("extracted"):
        return _context_block(state)
    return state.get("query", "")


# ------------------------------------------------------------------ nodes
def plan_node(state: AgentState) -> AgentState:
    trace = state.get("trace", [])
    chain = get_planner_chain()
    user_msg = (
        f"User query: {state.get('query') or '(no text query provided)'}\n\n"
        f"Conversation history:\n{_history_block(state)}\n\n"
        f"Extracted input content:\n{_context_block(state, per_item_limit=4000)}"
    )

    def _fallback_plan(reason: str, title: str) -> AgentPlan:
        logger.warning("Planner fallback (%s): %s", title, reason)
        trace.append(TraceEvent(stage="plan", title=title, detail=reason, status="error"))
        return AgentPlan(
            action="execute",
            steps=[PlanStep(tool="answer_question", question=state.get("query", ""),
                            reason="fallback")],
        )

    try:
        result = chain.invoke(
            [
                SystemMessage(content=PLANNER_SYSTEM.format(tool_manifest=planner_tool_manifest())),
                HumanMessage(content=user_msg),
            ]
        )
    except Exception as exc:
        # Covers both transport-level failures (rate limits, timeouts) and
        # tool_use_failed / json_validate_failed API errors that the
        # structured-output call itself can still raise.
        plan = _fallback_plan(str(exc), "Planner request failed")
        plan.steps = plan.steps[:MAX_STEPS]
        return {"plan": plan, "trace": trace}

    parsed = result.get("parsed")
    if parsed is None:
        error = result.get("parsing_error")
        plan = _fallback_plan(str(error) if error else "model did not return a valid plan",
                              "Planner output was malformed")
    else:
        plan = parsed

    # HARD CAP: never clarify twice in a row. Relying on prompt wording alone
    # to make the model stop asking wasn't reliable in practice — it kept
    # re-asking even after the user explicitly said "don't ask, just answer
    # anyway." This is enforced in code instead: if the previous turn was
    # itself a clarify question, this turn is never allowed to clarify again,
    # no matter what the model decides — it must give its best-effort answer.
    force_final_answer = False
    if plan.action == "clarify" and state.get("clarification_pending"):
        logger.info("Overriding repeat clarify request; forcing best-effort answer instead.")
        trace.append(TraceEvent(
            stage="plan", title="Clarification already requested once",
            detail="Not asking again — proceeding with a best-effort answer instead.",
            status="info"))
        force_final_answer = True
        plan = AgentPlan(action="execute", steps=[
            PlanStep(
                tool="answer_question",
                question=(
                    f"The user's request was ambiguous and they were already asked to "
                    f"clarify once. Recent conversation:\n{_history_block(state)}\n\n"
                    f"They did not (or could not) provide more detail. Give the most "
                    f"helpful best-effort response you can with what's available, or "
                    f"plainly explain what specific information would be needed — do "
                    f"not ask another clarifying question."
                ),
                reason="user declined to clarify further",
            )
        ])

    plan.steps = plan.steps[:MAX_STEPS]
    if plan.action == "clarify":
        trace.append(TraceEvent(stage="plan", title="Needs clarification",
                                detail=plan.clarify_question or "", status="info"))
    else:
        detail = " -> ".join(f"{i}. {s.tool}" for i, s in enumerate(plan.steps)) or "(no steps)"
        trace.append(TraceEvent(stage="plan", title=f"Planned {len(plan.steps)} step(s)", detail=detail))
        for i, step in enumerate(plan.steps):
            trace.append(TraceEvent(stage="plan", title=f"Step {i}: {step.tool}",
                                    detail=step.reason, status="info"))
    return {"plan": plan, "trace": trace, "force_final_answer": force_final_answer}


def route_after_plan(state: AgentState) -> str:
    plan = state.get("plan")
    if plan and plan.action == "clarify":
        return "clarify"
    return "execute"


def clarify_node(state: AgentState) -> AgentState:
    question = state["plan"].clarify_question or "Could you clarify what you'd like me to do with these inputs?"
    return {"answer": question, "kind": "clarify"}


def _build_args(tool_name: str, step: PlanStep, step_index: int, outputs: list[str],
                state: AgentState) -> dict:
    """Deterministically build a tool's real arguments from the plan step's
    small, fixed fields — the model never supplies a raw args dict, and text
    content is always substituted here, never taken from model-authored text."""
    if tool_name in ("fetch_youtube_transcript", "fetch_url"):
        return {"url": step.url or ""}
    if tool_name == "summarize":
        return {"text": _content_for_step(step, step_index, outputs, state), "focus": step.focus or ""}
    if tool_name == "sentiment_analysis":
        return {"text": _content_for_step(step, step_index, outputs, state)}
    if tool_name == "explain_code":
        return {"code": _content_for_step(step, step_index, outputs, state), "question": step.question or ""}
    if tool_name == "answer_question":
        history = _history_block(state)
        context = _context_block(state)
        full_context = context if history == "(none)" else f"Conversation so far:\n{history}\n\n{context}"
        return {"question": step.question or state.get("query", ""), "context": full_context}
    if tool_name == "compare_inputs":
        return {"question": step.question or state.get("query", ""), "labeled_contents": _context_block(state)}
    return {}


def execute_node(state: AgentState) -> AgentState:
    trace = state.get("trace", [])
    outputs: list[str] = []
    state = {**state, "step_outputs": outputs}

    for i, step in enumerate(state["plan"].steps):
        tool = REGISTRY.get(step.tool)
        if tool is None:
            outputs.append(f"(unknown tool '{step.tool}')")
            trace.append(TraceEvent(stage="tool", title=f"Step {i}: unknown tool {step.tool}",
                                    status="error"))
            continue
        try:
            args = _build_args(tool.name, step, i, outputs, state)
            result = tool.fn(**args)
            outputs.append(result)
            status = "error" if result.startswith("FALLBACK") else "ok"
            trace.append(TraceEvent(stage="tool", title=f"Step {i}: {tool.name}",
                                    detail=result[:280] + ("…" if len(result) > 280 else ""),
                                    status=status))
        except Exception as exc:
            logger.exception("Tool %s failed", step.tool)
            outputs.append(f"(tool {step.tool} failed: {exc})")
            trace.append(TraceEvent(stage="tool", title=f"Step {i}: {step.tool} failed",
                                    detail=str(exc), status="error"))

    return {"step_outputs": outputs, "trace": trace}


_NO_MORE_QUESTIONS = (
    "\n\nHARD CONSTRAINT FOR THIS RESPONSE: the user already declined to provide "
    "more clarity after being asked once. Your response MUST NOT ask another "
    "question, request clarification, or end with a question mark soliciting "
    "more information from the user — not even a soft/implicit one like "
    "'let me know if...' or 'could you tell me...'. Commit to a best-effort "
    "interpretation and answer it directly, OR plainly state you don't have "
    "enough information and briefly say what would be needed — as a "
    "statement, not a question — then stop."
)


def _synthesizer_system(state: AgentState) -> str:
    if state.get("force_final_answer"):
        return SYNTHESIZER_SYSTEM + _NO_MORE_QUESTIONS
    return SYNTHESIZER_SYSTEM


def synthesize_node(state: AgentState) -> AgentState:
    trace = state.get("trace", [])
    steps_block = "\n\n".join(
        f"[Step {i}: {step.tool}] ->\n{out[:20000]}"
        for i, (step, out) in enumerate(zip(state["plan"].steps, state.get("step_outputs", [])))
    ) or "(no tool steps were executed)"

    llm = get_llm()
    answer = llm.invoke(
        [
            SystemMessage(content=_synthesizer_system(state)),
            HumanMessage(content=(
                f"User goal: {state.get('query') or '(implicit from uploads)'}\n\n"
                f"Conversation history:\n{_history_block(state)}\n\n"
                f"Extracted inputs (metadata + content):\n{_context_block(state, per_item_limit=4000)}\n\n"
                f"Tool outputs:\n{steps_block}"
            )),
        ]
    ).content.strip()

    trace.append(TraceEvent(stage="synthesize", title="Composed final answer",
                            detail=f"{len(answer)} chars"))
    return {"answer": answer, "kind": "answer", "trace": trace}


def stream_synthesize(state: AgentState):
    """Yield the final answer token-by-token (used by the SSE endpoint)."""
    steps_block = "\n\n".join(
        f"[Step {i}: {step.tool}] ->\n{out[:20000]}"
        for i, (step, out) in enumerate(zip(state["plan"].steps, state.get("step_outputs", [])))
    ) or "(no tool steps were executed)"

    llm = get_llm()
    for chunk in llm.stream(
        [
            SystemMessage(content=_synthesizer_system(state)),
            HumanMessage(content=(
                f"User goal: {state.get('query') or '(implicit from uploads)'}\n\n"
                f"Conversation history:\n{_history_block(state)}\n\n"
                f"Extracted inputs (metadata + content):\n{_context_block(state, per_item_limit=4000)}\n\n"
                f"Tool outputs:\n{steps_block}"
            )),
        ]
    ):
        if chunk.content:
            yield chunk.content


# ------------------------------------------------------------------ graph
def build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("plan", plan_node)
    graph.add_node("clarify", clarify_node)
    graph.add_node("execute", execute_node)
    graph.add_node("synthesize", synthesize_node)

    graph.set_entry_point("plan")
    graph.add_conditional_edges("plan", route_after_plan,
                                {"clarify": "clarify", "execute": "execute"})
    graph.add_edge("clarify", END)
    graph.add_edge("execute", "synthesize")
    graph.add_edge("synthesize", END)
    return graph.compile()


AGENT = build_graph()
