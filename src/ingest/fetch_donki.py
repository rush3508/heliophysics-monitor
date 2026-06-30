"""
fetch_donki.py — NASA DONKI API client for space weather events.

Fetches Coronal Mass Ejections (CME), Solar Flares (FLR), and
Geomagnetic Storms (GST) from NASA's public API. Handles rate
limiting, retries, timestamp normalisation, and local JSON storage.

Usage:
    from src.ingest.fetch_donki import fetch_cme, fetch_flr, fetch_gst, ingest_all

    # Fetch individual event types
    cme_data = fetch_cme("2026-01-01", "2026-06-30")
    flr_data = fetch_flr("2026-01-01", "2026-06-30")
    gst_data = fetch_gst("2026-01-01", "2026-06-30")

    # Or fetch all three at once
    results = ingest_all("2026-01-01", "2026-06-30")
"""

import json
import logging
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests

from config import (
    DATA_RAW,
    EVENT_TYPES,
    MAX_RETRIES,
    NASA_API_BASE,
    NASA_API_KEY,
    RATE_LIMIT_BUFFER,
    REQUEST_TIMEOUT_SECONDS,
    RETRY_BACKOFF_SECONDS,
)

logger = logging.getLogger(__name__)


def _build_url(endpoint: str, start_date: str, end_date: str) -> str:
    """Construct a DONKI API URL with date range and API key."""
    return (
        f"{NASA_API_BASE}/{endpoint}"
        f"?api_key={NASA_API_KEY}"
        f"&startDate={start_date}"
        f"&endDate={end_date}"
    )


def _check_rate_limit(response: requests.Response) -> None:
    """Sleep if rate-limit remaining is at or below the safety buffer."""
    remaining = response.headers.get("X-Ratelimit-Remaining")
    if remaining is not None:
        remaining = int(remaining)
        if remaining <= RATE_LIMIT_BUFFER:
            reset_time = response.headers.get("X-Ratelimit-Reset")
            wait = 60  # default: wait 60 seconds
            if reset_time:
                try:
                    wait = max(0, int(reset_time) - int(time.time())) + 5
                except (ValueError, TypeError):
                    pass
            logger.warning(
                "Rate limit low (%d remaining). Sleeping %ds.", remaining, wait
            )
            time.sleep(wait)


def _normalise_timestamp(ts: str | None) -> str | None:
    """Convert a DONKI timestamp to UTC ISO 8601 with Z suffix."""
    if ts is None:
        return None
    # DONKI returns timestamps like "2026-06-01T04:00Z" or
    # "2026-06-01T04:00:00Z". Normalise to full ISO 8601.
    ts = ts.strip().replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(ts)
        return dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    except ValueError:
        return ts  # return as-is if parsing fails


def _normalise_event(event: dict) -> dict:
    """Normalise all datetime fields in a DONKI event to UTC ISO 8601."""
    datetime_fields = [
        "startTime",
        "endTime",
        "beginTime",
        "peakTime",
        "submissionTime",
    ]
    for field in datetime_fields:
        if field in event and event[field] is not None:
            event[field] = _normalise_timestamp(event[field])

    # Normalise nested timestamps in cmeAnalyses
    for analysis in event.get("cmeAnalyses", []) or []:
        for ts_field in ("time21_5",):
            if ts_field in analysis and analysis[ts_field] is not None:
                analysis[ts_field] = _normalise_timestamp(analysis[ts_field])

    # Normalise nested timestamps in allKpIndex
    for kp_entry in event.get("allKpIndex", []) or []:
        if "observedTime" in kp_entry and kp_entry["observedTime"] is not None:
            kp_entry["observedTime"] = _normalise_timestamp(kp_entry["observedTime"])

    return event


def _fetch_endpoint(
    endpoint: str, start_date: str, end_date: str, event_type: str
) -> list[dict]:
    """
    Fetch all events for a given DONKI endpoint within a date range.

    Handles retries, rate limiting, and response validation.
    Returns a list of normalised event dicts.
    """
    url = _build_url(endpoint, start_date, end_date)
    logger.info("Fetching %s: %s", event_type, url)

    last_error = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.get(url, timeout=REQUEST_TIMEOUT_SECONDS)
            _check_rate_limit(resp)
            resp.raise_for_status()
            data = resp.json()
            break
        except requests.exceptions.Timeout as e:
            last_error = e
            logger.warning(
                "Timeout on attempt %d/%d for %s", attempt, MAX_RETRIES, event_type
            )
        except requests.exceptions.HTTPError as e:
            last_error = e
            status = resp.status_code if "resp" in locals() else None
            logger.warning(
                "HTTP %s on attempt %d/%d for %s", status, attempt, MAX_RETRIES, event_type
            )
        except requests.exceptions.RequestException as e:
            last_error = e
            logger.warning(
                "Request error on attempt %d/%d: %s", attempt, MAX_RETRIES, e
            )

        if attempt < MAX_RETRIES:
            sleep_time = RETRY_BACKOFF_SECONDS * attempt
            logger.info("Retrying in %ds...", sleep_time)
            time.sleep(sleep_time)
    else:
        raise RuntimeError(
            f"Failed to fetch {event_type} after {MAX_RETRIES} attempts. "
            f"Last error: {last_error}"
        )

    # DONKI returns a list directly for most endpoints, or a dict with a
    # list for some. Normalise to list.
    if isinstance(data, dict):
        # Some endpoints wrap in a key matching the endpoint name
        for key in (endpoint, "data", "events"):
            if key in data and isinstance(data[key], list):
                data = data[key]
                break

    if not isinstance(data, list):
        raise ValueError(
            f"Unexpected response format for {event_type}: {type(data).__name__}"
        )

    # Normalise timestamps
    events = [_normalise_event(e) for e in data]

    logger.info("Fetched %d %s events.", len(events), event_type)
    return events


def fetch_cme(start_date: str, end_date: str) -> list[dict]:
    """Fetch Coronal Mass Ejection events."""
    return _fetch_endpoint("CME", start_date, end_date, "CME")


def fetch_flr(start_date: str, end_date: str) -> list[dict]:
    """Fetch Solar Flare events."""
    return _fetch_endpoint("FLR", start_date, end_date, "FLR")


def fetch_gst(start_date: str, end_date: str) -> list[dict]:
    """Fetch Geomagnetic Storm events."""
    return _fetch_endpoint("GST", start_date, end_date, "GST")


def ingest_all(
    start_date: str,
    end_date: str,
    save: bool = True,
    chunk_days: int = 30,
) -> dict[str, list[dict]]:
    """
    Fetch all 3 event types and optionally save to data/raw/.

    Splits the date range into chunks of chunk_days to avoid timeouts
    on large queries (e.g. 180 days of CME data can exceed NASA's
    response time limit).

    Returns a dict mapping event type keys ("CME", "FLR", "GST") to
    lists of normalised event dicts.
    """
    DATA_RAW.mkdir(parents=True, exist_ok=True)

    # Build date chunks
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")
    chunks = []
    chunk_start = start_dt
    while chunk_start < end_dt:
        chunk_end = min(chunk_start + timedelta(days=chunk_days), end_dt)
        chunks.append((
            chunk_start.strftime("%Y-%m-%d"),
            chunk_end.strftime("%Y-%m-%d"),
        ))
        chunk_start = chunk_end + timedelta(days=1)

    logger.info(
        "Ingesting %s to %s in %d chunks of ≤%d days each.",
        start_date, end_date, len(chunks), chunk_days,
    )

    results: dict[str, list[dict]] = {}
    fetchers = {"CME": fetch_cme, "FLR": fetch_flr, "GST": fetch_gst}

    for event_type, fetcher in fetchers.items():
        all_events: list[dict] = []
        seen_ids: set[str] = set()

        for i, (ch_start, ch_end) in enumerate(chunks, 1):
            logger.info(
                "[%s] Chunk %d/%d: %s → %s",
                event_type, i, len(chunks), ch_start, ch_end,
            )
            events = fetcher(ch_start, ch_end)

            # Deduplicate by event ID across chunks (boundary overlap)
            id_field = EVENT_TYPES[event_type]["id_field"]
            for event in events:
                eid = event.get(id_field)
                if eid and eid not in seen_ids:
                    seen_ids.add(eid)
                    all_events.append(event)

        results[event_type] = all_events
        logger.info(
            "%s: %d total events (deduplicated across %d chunks).",
            event_type, len(all_events), len(chunks),
        )

        if save and all_events:
            date_tag = end_date.replace("-", "")
            filename = f"{event_type}_{date_tag}.json"
            filepath = DATA_RAW / filename
            with open(filepath, "w") as f:
                json.dump(all_events, f, indent=2, default=str)
            logger.info("Saved %d %s events to %s", len(all_events), event_type, filepath)

    return results


# ── CLI entry point ────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    end = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    start = (datetime.now(timezone.utc) - timedelta(days=180)).strftime("%Y-%m-%d")

    print(f"Ingesting DONKI events from {start} to {end}...")
    results = ingest_all(start, end)

    for event_type, events in results.items():
        print(f"  {event_type}: {len(events)} events")
    print("Done.")
