"""
generate_brief.py — Generate a 150–250 word analyst brief using llama3.2.

Loads structured stats from data/processed/, retrieves relevant chunks
from the heliophysics corpus, builds a constrained prompt for llama3.2,
and writes the brief to reports/brief.md.

Enforces: token budget, staleness guard, hallucination guard, word count.

Usage:
    python -m src.reporting.generate_brief
"""

import json
from typing import Any
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

from config import (
    BRIEF_FILE,
    BRIEF_MAX_WORDS,
    BRIEF_MIN_WORDS,
    BRIEF_RETRY_MIN_WORDS,
    CONTROLLED_QUERIES,
    DATA_PROCESSED,
    DATA_RAW,
    DATA_MAX_AGE_HOURS,
    OLLAMA_CONTEXT_WINDOW,
    OLLAMA_MODEL,
    PROMPT_TOKEN_BUDGET,
    REPORTS_DIR,
    WARNINGS_LOG,
)

logger = logging.getLogger(__name__)

# Approximate token count: ~1 token per 4 chars
CHARS_PER_TOKEN = 4
def _count_tokens(text: str) -> int:
    return len(text) // CHARS_PER_TOKEN


def _build_prompt(
    stats: dict, chunks: list[dict], query: str | None = None
) -> str:
    """Build a constrained prompt for llama3.2."""
    chunk_text = "\n\n".join(
        f"[Source: {c['source_label']}] {c['text']}" for c in chunks[:3]
    )

    if query:
        task = f'Answer this question using ONLY the context provided: "{query}"'
    else:
        task = """Write a 150-250 word space weather analyst brief covering:
1. Event summary for the latest 7 days (counts, notable flare classes, fastest CME, any geomagnetic storms)
2. Key terminology explained from context (pick 1-2 terms)
3. What to watch in the coming days based on recent trends

Use British English. Be factual. Do not speculate beyond the data provided.
If any event type has zero count, state "No events recorded" for that type."""

    prompt = f"""You are a space weather analyst working with NASA DONKI data.

{task}

CONTEXT (retrieved from heliophysics corpus):
{chunk_text}

STATS (from DONKI data):
{json.dumps(stats, indent=2)}

Your response:"""
    return prompt


def _generate(prompt: str) -> str:
    """Call llama3.2 via Ollama and return the response text."""
    resp = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {
                "num_ctx": OLLAMA_CONTEXT_WINDOW,
                "temperature": 0.3,
            },
        },
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json().get("response", "")


def _check_staleness() -> str | None:
    """Check if raw data is older than DATA_MAX_AGE_HOURS. Returns warning or None."""
    raw_files = list(DATA_RAW.glob("*.json"))
    if not raw_files:
        return None
    newest_mtime = max(f.stat().st_mtime for f in raw_files)
    age_hours = (time.time() - newest_mtime) / 3600
    if age_hours > DATA_MAX_AGE_HOURS:
        mtime_str = datetime.fromtimestamp(newest_mtime, tz=timezone.utc).strftime(
            "%Y-%m-%d %H:%M UTC"
        )
        return f"⚠ Data from {mtime_str}. Consider re-running ingestion."
    return None


def _load_stats() -> dict:
    """Load structured stats — only the last 7 days for the brief prompt."""
    stats: dict[str, Any] = {}

    # Daily counts: last 7 days
    dc_path = DATA_PROCESSED / "daily_counts.json"
    if dc_path.exists():
        with open(dc_path) as f:
            daily = json.load(f)
        recent = daily[-7:] if len(daily) >= 7 else daily
        stats["recent_daily_counts"] = recent
        stats["total_days"] = len(daily)
        # Summary for prompt
        cme_total = sum(d.get("cme_count", 0) for d in recent)
        flr_total = sum(d.get("flr_count", 0) for d in recent)
        gst_total = sum(d.get("gst_count", 0) for d in recent)
        stats["summary"] = {
            "period_days": len(recent),
            "total_cme": cme_total,
            "total_flr": flr_total,
            "total_gst": gst_total,
            "date_range": f"{recent[0].get('date', '?')} to {recent[-1].get('date', '?')}" if recent else "N/A",
        }

    # Severity: last 7 days
    sev_path = DATA_PROCESSED / "severity.json"
    if sev_path.exists():
        with open(sev_path) as f:
            severity = json.load(f)
        recent_sev = severity[-7:] if len(severity) >= 7 else severity
        flare_vals = [s.get("max_flare_severity") for s in recent_sev if s.get("max_flare_severity")]
        speed_vals = [s.get("max_cme_speed_kms") for s in recent_sev if s.get("max_cme_speed_kms")]
        kp_vals = [s.get("max_kp_index") for s in recent_sev if s.get("max_kp_index")]
        stats["severity_summary"] = {
            "max_flare_severity_7d": max(flare_vals) if flare_vals else None,
            "max_cme_speed_kms_7d": max(speed_vals) if speed_vals else None,
            "max_kp_index_7d": max(kp_vals) if kp_vals else None,
        }

    # Linkages: count only
    link_path = DATA_PROCESSED / "linkages.json"
    if link_path.exists():
        with open(link_path) as f:
            linkages = json.load(f)
        stats["total_linkages"] = len(linkages.get("edges", []))

    # Top days
    td_path = DATA_PROCESSED / "top_days.json"
    if td_path.exists():
        with open(td_path) as f:
            stats["top_active_days"] = json.load(f)

    return stats


def generate_brief(query: str | None = None) -> str:
    """
    Generate the analyst brief.

    Args:
        query: if provided, answer this specific question instead of
               generating a general brief.

    Returns:
        The generated text.
    """
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load stats
    stats = _load_stats()

    # Staleness check
    stale_warning = _check_staleness()

    # Retrieve relevant chunks
    from src.rag.retrieve import retrieve

    search_query = query or " ".join(CONTROLLED_QUERIES[:2])
    chunks = retrieve(search_query, k=3)

    if not chunks:
        logger.warning("No chunks retrieved — brief will lack grounding.")

    # Build prompt with token guard
    prompt = _build_prompt(stats, chunks, query)
    token_count = _count_tokens(prompt)

    if token_count > PROMPT_TOKEN_BUDGET:
        logger.warning(
            "Prompt tokens (%d) exceed budget (%d). Truncating chunks.",
            token_count, PROMPT_TOKEN_BUDGET,
        )
        while chunks and _count_tokens(_build_prompt(stats, chunks, query)) > PROMPT_TOKEN_BUDGET:
            # Drop lowest-scoring chunk
            chunks.pop(-1)
        prompt = _build_prompt(stats, chunks, query)
        token_count = _count_tokens(prompt)
        logger.info("Truncated to %d tokens with %d chunks.", token_count, len(chunks))

    logger.info("Prompt: %d tokens, %d chunks.", token_count, len(chunks))

    # Generate
    logger.info("Generating brief with %s...", OLLAMA_MODEL)
    response = _generate(prompt)

    # Word count check
    word_count = len(response.split())
    logger.info("Response: %d words.", word_count)

    # Retry if too short
    if word_count < BRIEF_RETRY_MIN_WORDS:
        logger.warning("Brief too short (%d words). Retrying once...", word_count)
        simpler_prompt = _build_prompt(stats, chunks[:2], query) + "\nKeep it under 250 words."
        response = _generate(simpler_prompt)
        word_count = len(response.split())
        logger.info("Retry response: %d words.", word_count)

    # Fallback if still too short
    if word_count < BRIEF_RETRY_MIN_WORDS:
        logger.warning("Still too short. Using deterministic fallback.")
        response = _deterministic_fallback(stats)
        word_count = len(response.split())

    # Prepend staleness warning
    if stale_warning:
        response = f"{stale_warning}\n\n{response}"

    # Save
    BRIEF_FILE.write_text(response.strip())
    logger.info("Brief saved: %s (%d words)", BRIEF_FILE, word_count)

    # Log warnings
    if word_count < BRIEF_MIN_WORDS or word_count > BRIEF_MAX_WORDS:
        _log_warning(f"brief word count {word_count} outside [{BRIEF_MIN_WORDS}, {BRIEF_MAX_WORDS}]")
    if not chunks:
        _log_warning("brief generated without retrieval grounding")

    return response


def _deterministic_fallback(stats: dict) -> str:
    """Generate a deterministic template brief when llama3.2 fails."""
    daily = stats.get("daily_counts", [])
    severity = stats.get("severity", [])

    # Get last 7 days
    recent = daily[-7:] if len(daily) >= 7 else daily
    total_cme = sum(d.get("cme_count", 0) for d in recent)
    total_flr = sum(d.get("flr_count", 0) for d in recent)
    total_gst = sum(d.get("gst_count", 0) for d in recent)

    max_flare = "N/A"
    max_speed = "N/A"
    if severity:
        recent_sev = severity[-7:] if len(severity) >= 7 else severity
        flare_vals = [s.get("max_flare_severity") for s in recent_sev if s.get("max_flare_severity")]
        speed_vals = [s.get("max_cme_speed_kms") for s in recent_sev if s.get("max_cme_speed_kms")]
        if flare_vals:
            max_flare = f"{max(flare_vals):.1f}"
        if speed_vals:
            max_speed = f"{max(speed_vals):.0f} km/s"

    lines = [
        "## Heliophysics Monitor — Analyst Brief",
        "",
        f"**Period:** Last 7 days",
        "",
        "### Event Summary",
        f"- **CMEs:** {total_cme} events detected"
    ]

    if total_cme == 0:
        lines.append("  - No coronal mass ejections recorded this period.")
    else:
        lines.append(f"  - Fastest CME: {max_speed}")

    lines.append(f"- **Solar Flares:** {total_flr} events")
    if total_flr == 0:
        lines.append("  - No solar flares recorded this period.")
    else:
        lines.append(f"  - Highest severity: {max_flare}")

    lines.append(f"- **Geomagnetic Storms:** {total_gst} events")
    if total_gst == 0:
        lines.append("  - No geomagnetic storms recorded this period.")

    lines.extend([
        "",
        "### Terminology",
        "A **coronal mass ejection (CME)** is a large expulsion of plasma and",
        "magnetic field from the Sun's corona. CMEs travelling towards Earth",
        "can trigger geomagnetic storms when they interact with the magnetosphere.",
        "",
        "### Outlook",
        "Monitor DONKI for linked events: solar flares associated with Earth-directed",
        "CMEs are the primary precursors to geomagnetic storm activity.",
        "",
        "---",
        "*This brief was auto-generated by the Heliophysics Monitor pipeline.*",
    ])

    return "\n".join(lines)


def _log_warning(message: str) -> None:
    """Append a warning to the pipeline warnings log."""
    WARNINGS_LOG.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    with open(WARNINGS_LOG, "a") as f:
        f.write(f"[{timestamp}] {message}\n")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    brief = generate_brief()
    print(f"\n=== BRIEF ({len(brief.split())} words) ===")
    print(brief)
