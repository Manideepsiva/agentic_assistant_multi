"""Image -> text.

Primary: Tesseract OCR (gives per-word confidence scores).
Fallback / cleanup: Llama 4 Scout (Groq vision model) when Tesseract
confidence is low or empty, so screenshots of code, stylised fonts, and
photos still extract well.
"""
from __future__ import annotations

import base64
import io
import logging

from PIL import Image

from app.schemas import ExtractedItem

logger = logging.getLogger(__name__)

LOW_CONFIDENCE_THRESHOLD = 55.0


def _tesseract_ocr(image: Image.Image) -> tuple[str, float]:
    """Run tesseract, return (text, mean_word_confidence 0-100)."""
    import pytesseract

    data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
    words, confs = [], []
    for word, conf in zip(data["text"], data["conf"]):
        word = word.strip()
        if not word:
            continue
        words.append(word)
        try:
            c = float(conf)
            if c >= 0:
                confs.append(c)
        except (TypeError, ValueError):
            pass
    text = pytesseract.image_to_string(image).strip()
    mean_conf = sum(confs) / len(confs) if confs else 0.0
    return text or " ".join(words), round(mean_conf, 1)


def _vision_ocr(image_bytes: bytes, mime: str) -> str:
    """GPT-4o vision transcription fallback."""
    from langchain_core.messages import HumanMessage

    from app.agent.llm import get_vision_llm

    b64 = base64.b64encode(image_bytes).decode()
    msg = HumanMessage(
        content=[
            {
                "type": "text",
                "text": (
                    "Transcribe ALL text visible in this image exactly as written, "
                    "preserving line breaks and code indentation. "
                    "Return only the transcription, no commentary."
                ),
            },
            {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
        ]
    )
    return get_vision_llm().invoke([msg]).content.strip()


def extract_image(filename: str, image_bytes: bytes, mime: str = "image/png") -> ExtractedItem:
    text, conf, method = "", 0.0, "ocr_tesseract"
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        text, conf = _tesseract_ocr(image)
    except Exception as exc:  # tesseract missing or unreadable image
        logger.warning("Tesseract OCR failed for %s: %s", filename, exc)

    if not text or conf < LOW_CONFIDENCE_THRESHOLD:
        try:
            vision_text = _vision_ocr(image_bytes, mime)
            if len(vision_text) > len(text):
                text, method = vision_text, "ocr_gpt4o_vision"
                conf = max(conf, 90.0)  # vision transcription is high-fidelity
        except Exception as exc:
            logger.warning("Vision OCR fallback failed for %s: %s", filename, exc)

    return ExtractedItem(
        source=filename,
        modality="image",
        content=text,
        method=method,
        confidence=conf,
        meta={"bytes": len(image_bytes)},
    )
