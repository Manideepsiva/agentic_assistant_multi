
from __future__ import annotations

import logging

from app.ingestion.audio_transcriber import extract_audio
from app.ingestion.image_ocr import extract_image
from app.ingestion.pdf_parser import extract_pdf
from app.schemas import ExtractedItem, TraceEvent

logger = logging.getLogger(__name__)

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp"}
PDF_EXT = {".pdf"}
AUDIO_EXT = {".mp3", ".wav", ".m4a", ".ogg", ".flac", ".webm"}


def _ext(filename: str) -> str:
    dot = filename.rfind(".")
    return filename[dot:].lower() if dot != -1 else ""


def ingest_files(
    files: list[tuple[str, bytes, str]],
) -> tuple[list[ExtractedItem], list[TraceEvent]]:
    """files: list of (filename, raw_bytes, mime_type)."""
    items: list[ExtractedItem] = []
    trace: list[TraceEvent] = []

    for filename, raw, mime in files:
        ext = _ext(filename)
        try:
            if ext in IMAGE_EXT or mime.startswith("image/"):
                item = extract_image(filename, raw, mime or "image/png")
            elif ext in PDF_EXT or mime == "application/pdf":
                item = extract_pdf(filename, raw)
            elif ext in AUDIO_EXT or mime.startswith("audio/"):
                item = extract_audio(filename, raw, mime)
            else:
                # Treat unknown types as plain text if decodable.
                item = ExtractedItem(
                    source=filename,
                    modality="text",
                    content=raw.decode("utf-8", errors="replace"),
                    method="plain_text",
                )
            items.append(item)
            detail = f"{len(item.content)} chars via {item.method}"
            if item.confidence is not None:
                detail += f", OCR confidence {item.confidence}%"
            if item.meta.get("duration_seconds"):
                detail += f", {item.meta['duration_seconds']}s audio"
            if item.meta.get("youtube_urls"):
                detail += f", found YouTube URL(s): {', '.join(item.meta['youtube_urls'])}"
            trace.append(TraceEvent(stage="ingest", title=f"Extracted {filename}", detail=detail))
        except Exception as exc:
            logger.exception("Ingestion failed for %s", filename)
            trace.append(
                TraceEvent(
                    stage="ingest",
                    title=f"Failed to extract {filename}",
                    detail=str(exc),
                    status="error",
                )
            )
    return items, trace
