"""Audio (MP3/WAV/M4A) -> text via Groq's Whisper endpoint, plus light cleanup."""
from __future__ import annotations

import io
import logging
import mimetypes

from app.config import get_settings
from app.schemas import ExtractedItem

logger = logging.getLogger(__name__)

# Groq (like OpenAI) infers the audio codec from the filename's extension on
# the multipart upload — NOT from the browser's declared MIME type. If the
# uploaded filename has a misleading or double extension (e.g. a file named
# "voice.m4a.mpeg" — the actual bytes are AAC/M4A, but the last extension
# ".mpeg" gets picked up and told to Whisper as if it were MP3/MPEG audio),
# the decoder is handed the wrong format hint and produces garbled or
# hallucinated text instead of failing loudly. We fix this by deriving the
# extension from the browser's actual MIME type whenever we recognize it,
# and only falling back to the original filename when we don't.
MIME_TO_EXT = {
    "audio/mpeg": ".mp3",
    "audio/mp3": ".mp3",
    "audio/mp4": ".m4a",
    "audio/x-m4a": ".m4a",
    "audio/m4a": ".m4a",
    "audio/wav": ".wav",
    "audio/x-wav": ".wav",
    "audio/wave": ".wav",
    "audio/ogg": ".ogg",
    "audio/flac": ".flac",
    "audio/x-flac": ".flac",
    "audio/webm": ".webm",
}
KNOWN_AUDIO_EXT = {".mp3", ".wav", ".m4a", ".ogg", ".flac", ".webm"}


def _safe_filename(original_filename: str, mime_type: str) -> str:
    mime_type = (mime_type or "").split(";")[0].strip().lower()
    ext = MIME_TO_EXT.get(mime_type)
    if ext:
        return f"upload{ext}"

    # MIME type missing or unrecognized. Only trust the filename's own
    # trailing extension if it's an unambiguous, single-purpose audio
    # extension — deliberately excluding container-y extensions like .mpeg/
    # .mp4/.mpga, since those are exactly what shows up as a misleading
    # trailing suffix on a double-extension filename (e.g. "voice.m4a.mpeg",
    # where the real codec is m4a, not whatever ".mpeg" would suggest).
    dot = original_filename.rfind(".")
    orig_ext = original_filename[dot:].lower() if dot != -1 else ""
    if orig_ext in KNOWN_AUDIO_EXT:
        return f"upload{orig_ext}"

    logger.warning(
        "Could not confidently determine audio format for '%s' (mime=%r); "
        "defaulting to .m4a. If transcription looks wrong, this file's "
        "format could not be reliably detected.", original_filename, mime_type,
    )
    return "upload.m4a"


def _cleanup(text: str) -> str:
    """Light transcript cleanup: collapse whitespace, drop filler artifacts."""
    import re

    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\b(um+|uh+|erm+)\b[,.]?\s*", "", text, flags=re.IGNORECASE)
    return text


def extract_audio(filename: str, audio_bytes: bytes, mime_type: str = "") -> ExtractedItem:
    from groq import Groq

    settings = get_settings()
    client = Groq(api_key=settings.groq_api_key)

    safe_name = _safe_filename(filename, mime_type)
    buffer = io.BytesIO(audio_bytes)
    buffer.name = safe_name  # drives the codec hint Groq/Whisper actually uses

    result = client.audio.transcriptions.create(
        model=settings.whisper_model,
        file=buffer,
        response_format="verbose_json",
    )
    duration = getattr(result, "duration", None)
    raw_text = getattr(result, "text", "") or ""

    return ExtractedItem(
        source=filename,
        modality="audio",
        content=_cleanup(raw_text),
        method="whisper",
        meta={
            "duration_seconds": round(duration, 1) if duration else None,
            "language": getattr(result, "language", None),
            "detected_format": safe_name.rsplit(".", 1)[-1],
        },
    )
