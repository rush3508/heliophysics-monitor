"""
build_corpus.py — Scrape NASA heliophysics pages and chunk into a retrieval corpus.

One-off operation. Scrapes 4–5 NASA pages, strips navigation/JS,
extracts main content, chunks into ~300-token segments with overlap,
and saves to docs/corpus/chunks.json.

Usage:
    python -m src.rag.build_corpus
"""

import json
import logging
import re
import time
from pathlib import Path

import requests
from bs4 import BeautifulSoup

from config import (
    CHUNK_OVERLAP_TOKENS,
    CHUNK_TOKEN_SIZE,
    CORPUS_URLS,
    DATA_CORPUS,
    DOCS_CORPUS,
    MAX_CHUNK_TOKENS,
    MIN_CORPUS_PAGES,
    REQUEST_TIMEOUT_SECONDS,
)

logger = logging.getLogger(__name__)

# Approximate: 1 token ≈ 4 characters for English text
CHARS_PER_TOKEN = 4


def _simple_token_count(text: str) -> int:
    """Estimate token count: ~1 token per 4 characters."""
    return max(1, len(text) // CHARS_PER_TOKEN)


def _clean_text(text: str) -> str:
    """Collapse whitespace and remove stray control characters."""
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", text)
    return text.strip()


def _extract_content(html: str, selector: str) -> str:
    """Extract main content from an HTML page."""
    soup = BeautifulSoup(html, "lxml")

    # Remove non-content elements
    for tag in soup(["nav", "footer", "header", "script", "style",
                      "noscript", "iframe", "form", "button", "input", "select"]):
        tag.decompose()

    # Try the configured selector
    content = soup.select_one(selector)
    if content:
        return _clean_text(content.get_text())

    # Fallbacks
    for sel in ("main", "article", ".content", "#content", ".main-content", "body"):
        content = soup.select_one(sel)
        if content:
            text = _clean_text(content.get_text())
            if len(text) > 200:
                return text

    body = soup.find("body")
    if body:
        return _clean_text(body.get_text())
    return ""


def _chunk_text(text: str, source_url: str, source_label: str) -> list[dict]:
    """Split text into overlapping chunks of ~CHUNK_TOKEN_SIZE tokens."""
    sentences = re.split(r"(?<=[.!?])\s+", text)
    sentences = [s.strip() for s in sentences if s.strip()]

    target_chars = CHUNK_TOKEN_SIZE * CHARS_PER_TOKEN
    overlap_chars = CHUNK_OVERLAP_TOKENS * CHARS_PER_TOKEN

    chunks = []
    current_chunk = []
    current_tokens = 0
    chunk_index = 0

    for sentence in sentences:
        sent_tokens = _simple_token_count(sentence)

        if current_tokens + sent_tokens > CHUNK_TOKEN_SIZE and current_chunk:
            chunk_text = " ".join(current_chunk)
            chunk_tokens = _simple_token_count(chunk_text)
            if chunk_tokens <= MAX_CHUNK_TOKENS:
                chunks.append({
                    "text": chunk_text,
                    "token_count": chunk_tokens,
                    "source_url": source_url,
                    "source_label": source_label,
                    "chunk_index": chunk_index,
                })
                chunk_index += 1

            # Keep overlap
            overlap_text = " ".join(current_chunk)
            if len(overlap_text) > overlap_chars:
                overlap_sents = []
                ol_len = 0
                for s in reversed(current_chunk):
                    if ol_len + len(s) > overlap_chars and overlap_sents:
                        break
                    overlap_sents.insert(0, s)
                    ol_len += len(s)
                current_chunk = overlap_sents
            else:
                current_chunk = []
            current_tokens = _simple_token_count(" ".join(current_chunk))

        current_chunk.append(sentence)
        current_tokens += sent_tokens

    # Last chunk
    if current_chunk:
        chunk_text = " ".join(current_chunk)
        chunk_tokens = _simple_token_count(chunk_text)
        if chunk_tokens <= MAX_CHUNK_TOKENS:
            chunks.append({
                "text": chunk_text,
                "token_count": chunk_tokens,
                "source_url": source_url,
                "source_label": source_label,
                "chunk_index": chunk_index,
            })

    # Filter short noise
    return [c for c in chunks if c["token_count"] >= 20]


def build_corpus() -> list[dict]:
    """Scrape all configured NASA pages, chunk them, and save."""
    DOCS_CORPUS.mkdir(parents=True, exist_ok=True)
    DATA_CORPUS.mkdir(parents=True, exist_ok=True)

    all_chunks = []
    successful_pages = 0

    for entry in CORPUS_URLS:
        url = entry["url"]
        label = entry["label"]
        selector = entry["selector"]
        optional = entry.get("optional", False)

        logger.info("Scraping: %s (%s)", label, url)

        try:
            resp = requests.get(url, timeout=REQUEST_TIMEOUT_SECONDS, headers={
                "User-Agent": "HeliophysicsMonitor/0.1 (portfolio; alex.limcl@gmail.com)",
            })
            resp.raise_for_status()
        except requests.exceptions.RequestException as e:
            if optional:
                logger.warning("  Optional page skipped: %s", e)
                continue
            else:
                logger.error("  Failed: %s — %s", label, e)
                fallback = DATA_CORPUS / f"FAILED_{label.replace(' ', '_')}.txt"
                fallback.write_text(f"Failed: {url}\n{e}\n")
                continue

        content = _extract_content(resp.text, selector)
        if not content or len(content) < 100:
            logger.warning("  Content too short (%d chars), skipping.", len(content) if content else 0)
            continue

        raw_path = DATA_CORPUS / f"{label.replace(' ', '_')}.txt"
        raw_path.write_text(content)
        logger.info("  Raw: %d chars → %s", len(content), raw_path.name)

        chunks = _chunk_text(content, url, label)
        logger.info("  Chunks: %d segments", len(chunks))
        all_chunks.extend(chunks)
        successful_pages += 1
        time.sleep(0.5)

    chunks_path = DOCS_CORPUS / "chunks.json"
    with open(chunks_path, "w") as f:
        json.dump(all_chunks, f, indent=2)
    logger.info("Saved %d chunks from %d pages → %s", len(all_chunks), successful_pages, chunks_path)

    if successful_pages < MIN_CORPUS_PAGES:
        raise RuntimeError(
            f"Only {successful_pages}/{MIN_CORPUS_PAGES} pages succeeded. Check logs."
        )

    return all_chunks


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    chunks = build_corpus()
    print(f"\nCorpus: {len(chunks)} chunks from {len(CORPUS_URLS)} sources.")
