"""Tool registry.

The planner LLM picks tools by name; the executor looks them up here and
builds each tool's real arguments itself (see execute_node in graph.py) —
the model no longer authors argument dicts at all, only picks a tool and,
where relevant, a short url/question/focus string (see schemas.PlanStep).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from app.tools import text_tools, web_tools


@dataclass
class Tool:
    name: str
    description: str
    fn: Callable[..., str]


REGISTRY: dict[str, Tool] = {}


def register(tool: Tool) -> None:
    REGISTRY[tool.name] = tool


register(Tool(
    name="fetch_youtube_transcript",
    description="Fetch the transcript of a YouTube video. Use whenever a YouTube URL "
                "appears in ANY input (query, PDF text, OCR text, audio transcript) and "
                "the user's goal involves that video's content. Needs: url.",
    fn=web_tools.fetch_youtube_transcript,
))

register(Tool(
    name="fetch_url",
    description="Fetch a non-YouTube web page found in the inputs when the user's goal "
                "implies reading it. Needs: url.",
    fn=web_tools.fetch_url,
))

register(Tool(
    name="summarize",
    description="Summarize content into the mandatory 3 formats (1-line, 3 bullets, "
                "5-sentence). Set input_source='context' to summarize all extracted "
                "content, or 'previous_step' to summarize the immediately preceding "
                "step's output (e.g. a just-fetched transcript). Optionally set focus.",
    fn=text_tools.summarize,
))

register(Tool(
    name="sentiment_analysis",
    description="Sentiment label + confidence + one-line justification. Set "
                "input_source='context' or 'previous_step' as above.",
    fn=text_tools.sentiment,
))

register(Tool(
    name="explain_code",
    description="Explain code, detect bugs, and state time complexity. Use for code in "
                "any input, including OCR'd screenshots. Set input_source='context' or "
                "'previous_step'; optionally set question for a specific ask about the code.",
    fn=text_tools.explain_code,
))

register(Tool(
    name="answer_question",
    description="Answer a general or content-grounded question conversationally. Use for "
                "greetings, factual questions, and extraction-style asks like 'what are "
                "the action items?'. All extracted content is attached automatically — "
                "just set question.",
    fn=text_tools.answer_question,
))

register(Tool(
    name="compare_inputs",
    description="Cross-input reasoning: compare or combine content from MULTIPLE inputs "
                "to answer one unified question (e.g. 'do the audio and PDF discuss the "
                "same topic?'). All extracted content is attached automatically — just "
                "set question.",
    fn=text_tools.compare_inputs,
))


def planner_tool_manifest() -> str:
    """Human/LLM readable list of tools, generated from the registry."""
    return "\n".join(f"- {tool.name}: {tool.description}" for tool in REGISTRY.values())
