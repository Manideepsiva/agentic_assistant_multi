"""LLM-backed text tools. Each returns a plain string (all outputs are text-only)."""
from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage

from app.agent.llm import get_llm

# Appended to every tool's system prompt: the assignment requires the agent
# to "detect constraints (timing, format, instructions)" as part of intent
# understanding, not just pick the right tool. Constraint wording (length
# limits, format asks, tone/language, urgency) flows into `question`/`focus`
# by the planner, and this instruction makes each tool actually honor it,
# rather than only the final synthesizer trying to retrofit it afterward.
_CONSTRAINTS_NOTE = (
    "\nIf the question/focus text includes explicit constraints — a length "
    "limit, a format request (e.g. no bullets, as a table, in French), a "
    "tone/audience instruction (e.g. explain like I'm 5), or urgency wording "
    "— follow them as closely as possible without breaking the required "
    "structure above."
)


def _run(system: str, user: str) -> str:
    llm = get_llm()
    return llm.invoke([SystemMessage(content=system), HumanMessage(content=user)]).content.strip()


def summarize(text: str, focus: str = "") -> str:
    """Mandatory 3-format summary: 1-line, 3 bullets, 5-sentence."""
    focus_line = f"\nFocus the summary on: {focus}" if focus else ""
    return _run(
        "You are a precise summarizer. Always return EXACTLY this structure:\n"
        "**One-line summary:** <single sentence>\n\n"
        "**Key points:**\n- <bullet 1>\n- <bullet 2>\n- <bullet 3>\n\n"
        "**Five-sentence summary:** <exactly five sentences>" + _CONSTRAINTS_NOTE,
        f"Summarize the following content.{focus_line}\n\n---\n{text[:60000]}",
    )


def sentiment(text: str) -> str:
    """Label + confidence + one-line justification."""
    return _run(
        "You are a sentiment analyst. Return EXACTLY this structure:\n"
        "**Sentiment:** <Positive | Negative | Neutral | Mixed>\n"
        "**Confidence:** <0-100>%\n"
        "**Justification:** <one line citing specific wording or tone>" + _CONSTRAINTS_NOTE,
        f"Analyze the sentiment of:\n\n---\n{text[:30000]}",
    )


def explain_code(code: str, question: str = "") -> str:
    """Explanation + bug detection + time complexity."""
    extra = f"\nThe user also asked: {question}" if question else ""
    return _run(
        "You are a senior code reviewer. Return this structure:\n"
        "**Language:** <detected language>\n"
        "**What it does:** <clear explanation>\n"
        "**Bugs / issues:** <list any bugs, edge cases, or 'None found'>\n"
        "**Time complexity:** <Big-O with a one-line reason>" + _CONSTRAINTS_NOTE,
        f"Explain this code:{extra}\n\n```\n{code[:30000]}\n```",
    )


def answer_question(question: str, context: str = "") -> str:
    """Conversational answering, optionally grounded in extracted content."""
    ctx = f"\n\nUse this extracted content as context where relevant:\n---\n{context[:60000]}" if context else ""
    return _run(
        "You are a friendly, helpful assistant. Answer clearly and concisely in plain text. "
        "If the answer comes from provided context, ground it there and say so; "
        "quote the relevant lines when extracting specific items (e.g. action items)."
        + _CONSTRAINTS_NOTE,
        f"{question}{ctx}",
    )


def compare_inputs(question: str, labeled_contents: str) -> str:
    """Cross-input reasoning across multiple extracted sources."""
    return _run(
        "You compare and reason across multiple sources. Return:\n"
        "**Verdict:** <direct answer to the user's question>\n"
        "**Evidence per source:** <what each source says, cited by its name>\n"
        "**Analysis:** <overlaps, differences, and a reasoned conclusion>" + _CONSTRAINTS_NOTE,
        f"Question: {question}\n\nSources:\n{labeled_contents[:80000]}",
    )
