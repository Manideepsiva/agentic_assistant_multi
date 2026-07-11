"""Web-facing tools: YouTube transcript fetching and generic URL fetching."""
from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

YT_ID_RE = re.compile(
    r"(?:youtube\.com/(?:watch\?v=|shorts/|embed/)|youtu\.be/)([A-Za-z0-9_-]{11})"
)


def extract_youtube_id(url: str) -> str | None:
    match = YT_ID_RE.search(url)
    return match.group(1) if match else None


def fetch_youtube_transcript(url: str) -> str:
    """Fetch transcript text for a YouTube URL, with a graceful fallback message."""
    video_id = extract_youtube_id(url)
    if not video_id:
        return f"FALLBACK: '{url}' does not look like a valid YouTube URL."
    try:
        from youtube_transcript_api import YouTubeTranscriptApi

        api = YouTubeTranscriptApi()
        transcript = api.fetch(video_id)
        text = " ".join(snippet.text for snippet in transcript).strip()
        if not text:
            raise ValueError("empty transcript")
        return text
    except Exception as exc:
        logger.warning("YouTube transcript fetch failed for %s: %s", url, exc)
        return (
            f"FALLBACK: A transcript could not be fetched for {url} "
            f"(reason: {type(exc).__name__}). The video may have captions disabled, "
            "be private/region-locked, or the transcript API may be blocked from this server."
        )


def fetch_url(url: str, max_chars: int = 40000) -> str:
    """Fetch a general web page and return readable text."""
    try:
        import httpx
        from bs4 import BeautifulSoup

        resp = httpx.get(url, follow_redirects=True, timeout=20,
                         headers={"User-Agent": "Mozilla/5.0 (agentic-assistant)"})
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()
        text = re.sub(r"\n{3,}", "\n\n", soup.get_text("\n")).strip()
        return text[:max_chars] or "FALLBACK: page fetched but no readable text found."
    except Exception as exc:
        logger.warning("URL fetch failed for %s: %s", url, exc)
        return f"FALLBACK: Could not fetch {url} ({type(exc).__name__}: {exc})."
