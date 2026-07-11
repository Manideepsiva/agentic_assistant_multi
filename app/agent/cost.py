"""

Uses the ~4 chars-per-token heuristic (good enough for a pre-execution estimate)
so we don't need a tokenizer dependency at request time.
"""
from __future__ import annotations

from app.config import get_settings
from app.schemas import AgentPlan, CostEstimate, ExtractedItem

CHARS_PER_TOKEN = 4
BASE_PROMPT_TOKENS = 900          # planner + synthesizer system prompts
OUTPUT_TOKENS_PER_STEP = 450      # typical structured tool output
SYNTH_OUTPUT_TOKENS = 500


def estimate_cost(query: str, extracted: list[ExtractedItem], plan: AgentPlan | None) -> CostEstimate:
    settings = get_settings()
    content_chars = len(query) + sum(len(item.content) for item in extracted)
    n_steps = len(plan.steps) if plan and plan.action == "execute" else 0

    # Context is re-sent to the planner, each context-consuming step, and the synthesizer.
    passes = 2 + max(n_steps, 1)
    input_tokens = BASE_PROMPT_TOKENS + passes * min(content_chars // CHARS_PER_TOKEN, 20000)
    output_tokens = SYNTH_OUTPUT_TOKENS + n_steps * OUTPUT_TOKENS_PER_STEP

    usd = (
        input_tokens / 1_000_000 * settings.price_input_per_1m
        + output_tokens / 1_000_000 * settings.price_output_per_1m
    )
    return CostEstimate(input_tokens=input_tokens, output_tokens=output_tokens, usd=round(usd, 4))
