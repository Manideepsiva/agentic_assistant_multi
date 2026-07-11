"""PDF -> text.

Primary: pypdf text layer extraction (fast, exact).
Fallback: rasterise pages with pdf2image + Tesseract OCR for scanned PDFs.
Also detects URLs (incl. YouTube links) inside the extracted text so the
planner can chain a fetch step without user prompting.
"""
from __future__ import annotations

import io
import logging
import re

from app.schemas import ExtractedItem

logger = logging.getLogger(__name__)

URL_RE = re.compile(r"https?://[^\s)>\]}\"']+")
MIN_CHARS_PER_PAGE = 20  # below this we assume the page is scanned


def _text_layer(pdf_bytes: bytes) -> tuple[list[str], int]:
    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(pdf_bytes))
    pages = [(page.extract_text() or "").strip() for page in reader.pages]
    return pages, len(reader.pages)


def _ocr_pages(pdf_bytes: bytes, page_indexes: list[int]) -> dict[int, tuple[str, float]]:
    """OCR only the pages that had no text layer."""
    import pytesseract
    from pdf2image import convert_from_bytes

    results: dict[int, tuple[str, float]] = {}
    images = convert_from_bytes(pdf_bytes, dpi=200)
    for idx in page_indexes:
        if idx >= len(images):
            continue
        img = images[idx]
        data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
        confs = [float(c) for c in data["conf"] if str(c).replace(".", "", 1).lstrip("-").isdigit() and float(c) >= 0]
        text = pytesseract.image_to_string(img).strip()
        results[idx] = (text, round(sum(confs) / len(confs), 1) if confs else 0.0)
    return results


def extract_pdf(filename: str, pdf_bytes: bytes) -> ExtractedItem:
    pages, n_pages = _text_layer(pdf_bytes)
    scanned = [i for i, p in enumerate(pages) if len(p) < MIN_CHARS_PER_PAGE]
    method = "pdf_text"
    confidence = None

    if scanned:
        try:
            ocr_results = _ocr_pages(pdf_bytes, scanned)
            confs = []
            for idx, (text, conf) in ocr_results.items():
                if text:
                    pages[idx] = text
                    confs.append(conf)
            if confs:
                method = "pdf_text+ocr" if len(scanned) < n_pages else "pdf_ocr"
                confidence = round(sum(confs) / len(confs), 1)
        except Exception as exc:
            logger.warning("PDF OCR fallback failed for %s: %s", filename, exc)

    full_text = "\n\n".join(
        f"[Page {i + 1}]\n{p}" for i, p in enumerate(pages) if p
    ).strip()
    urls = sorted(set(URL_RE.findall(full_text)))
    youtube_urls = [u for u in urls if "youtube.com" in u or "youtu.be" in u]

    return ExtractedItem(
        source=filename,
        modality="pdf",
        content=full_text,
        method=method,
        confidence=confidence,
        meta={"pages": n_pages, "urls": urls, "youtube_urls": youtube_urls},
    )
