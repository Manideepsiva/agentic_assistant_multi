
import os
from functools import lru_cache

from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv()


class Settings(BaseModel):
    groq_api_key: str = (os.getenv("GROQ_API_KEY") or "").strip()

    # Chat model: planning + text tools + synthesis.
    llm_model: str = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")
    # Vision-capable model: used only for the OCR fallback on hard images.
    vision_model: str = os.getenv("VISION_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct")
    # Audio transcription model.
    whisper_model: str = os.getenv("WHISPER_MODEL", "whisper-large-v3-turbo")

    temperature: float = float(os.getenv("LLM_TEMPERATURE", "0"))
    # Completion token caps. The planner gets more headroom than a plain
    # answer needs, because JSON mode must fit the *entire* structured plan
    # (all steps + args + reasons) within this budget or Groq raises
    # json_validate_failed / "max completion tokens reached" instead of
    # returning partial output.
    llm_max_tokens: int = int(os.getenv("LLM_MAX_TOKENS", "2048"))
    planner_max_tokens: int = int(os.getenv("PLANNER_MAX_TOKENS", "3000"))
    max_upload_mb: int = int(os.getenv("MAX_UPLOAD_MB", "25"))
    session_ttl_seconds: int = int(os.getenv("SESSION_TTL_SECONDS", "1800"))

    # Groq's free tier is $0 — the cost estimator defaults to zero.
    # Override these if you're on a paid Groq tier and want a real estimate.
    price_input_per_1m: float = float(os.getenv("PRICE_INPUT_PER_1M", "0"))
    price_output_per_1m: float = float(os.getenv("PRICE_OUTPUT_PER_1M", "0"))


@lru_cache
def get_settings() -> Settings:
    return Settings()
