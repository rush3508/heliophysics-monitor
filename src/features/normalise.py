"""
normalise.py — Normalise raw DONKI JSON into a standardised event table.

Each row is one event with:
- event_id: unique identifier (activityID, flrID, or gstID)
- event_type: "CME", "FLR", or "GST"
- event_date: UTC date (YYYY-MM-DD) for daily aggregation
- start_time, end_time: UTC ISO 8601 timestamps
- severity fields depending on event type:
  - FLR: flare_class (raw string), flare_severity (numeric)
  - CME: cme_speed_kms (float, from mostAccurate analysis)
  - GST: max_kp_index (float, from allKpIndex)
- linked_events: list of linked event IDs
- source_location: position on solar disc (if available)
- instruments: list of observing instruments

Usage:
    from src.features.normalise import normalise_events, load_and_normalise

    events_df = load_and_normalise()
"""

import json
import logging
from datetime import datetime
from pathlib import Path

import pandas as pd

from config import DATA_RAW, EVENT_TYPES, FLARE_CLASS_MULTIPLIER

logger = logging.getLogger(__name__)


def _parse_flare_severity(class_type: str | None) -> float | None:
    """
    Convert a GOES flare class (e.g. "C1.8", "M5.2", "X1.0") to a
    numeric severity score.

    Scale: C=1, M=10, X=100 base × magnitude.
    Example: "M5.2" → 10 × 5.2 = 52.0
    """
    if class_type is None or not isinstance(class_type, str) or not class_type.strip():
        return None
    class_type = class_type.strip().upper()
    for letter in ("X", "M", "C", "B", "A"):
        if class_type.startswith(letter):
            try:
                magnitude = float(class_type[1:])
            except ValueError:
                return None
            multiplier = FLARE_CLASS_MULTIPLIER.get(letter, 1)
            return multiplier * magnitude
    return None


def _extract_cme_speed(event: dict) -> float | None:
    """Extract speed (km/s) from the most accurate CME analysis."""
    analyses = event.get("cmeAnalyses")
    if not analyses:
        return None
    for analysis in analyses:
        if analysis.get("isMostAccurate"):
            speed = analysis.get("speed")
            if speed is not None:
                return float(speed)
    # Fallback: first analysis with a speed value
    for analysis in analyses:
        speed = analysis.get("speed")
        if speed is not None:
            return float(speed)
    return None


def _extract_max_kp(event: dict) -> float | None:
    """Extract the maximum Kp index from a geomagnetic storm event."""
    kp_entries = event.get("allKpIndex")
    if not kp_entries:
        return None
    kp_values = []
    for entry in kp_entries:
        kp = entry.get("kpIndex")
        if kp is not None:
            kp_values.append(float(kp))
    return max(kp_values) if kp_values else None


def _parse_timestamp(ts: str | None) -> pd.Timestamp | None:
    """Parse a UTC ISO 8601 timestamp to pandas Timestamp."""
    if ts is None:
        return None
    try:
        return pd.Timestamp(ts)
    except (ValueError, TypeError):
        return None


def normalise_events(raw_data: dict[str, list[dict]]) -> pd.DataFrame:
    """
    Normalise raw DONKI JSON into a standardised DataFrame.

    Args:
        raw_data: dict mapping event type ("CME", "FLR", "GST") to list of
                  raw event dicts from the DONKI API.

    Returns:
        DataFrame with columns:
        event_id, event_type, event_date, start_time, end_time,
        flare_class, flare_severity, cme_speed_kms, max_kp_index,
        linked_events, source_location, instruments
    """
    rows = []

    for event_type, events in raw_data.items():
        meta = EVENT_TYPES[event_type]
        id_field = meta["id_field"]

        for event in events:
            eid = event.get(id_field)

            # Determine start/end times based on event type
            if event_type == "FLR":
                start_time = event.get("beginTime")
                end_time = event.get("endTime")
            elif event_type == "CME":
                start_time = event.get("startTime")
                end_time = None  # CMEs are ongoing; no fixed end
            elif event_type == "GST":
                start_time = event.get("startTime")
                end_time = None  # storms have Kp arrays instead
            else:
                start_time = event.get("startTime")
                end_time = event.get("endTime")

            start_dt = _parse_timestamp(start_time)
            end_dt = _parse_timestamp(end_time)

            # Derive event_date (UTC date for daily aggregation)
            if start_dt is not None:
                event_date = start_dt.strftime("%Y-%m-%d")
            else:
                event_date = None

            # Severity fields
            flare_class = event.get("classType") if event_type == "FLR" else None
            flare_severity = _parse_flare_severity(flare_class)
            cme_speed = _extract_cme_speed(event) if event_type == "CME" else None
            max_kp = _extract_max_kp(event) if event_type == "GST" else None

            # Linked events
            linked = event.get("linkedEvents")
            linked_ids = []
            if linked and isinstance(linked, list):
                for link in linked:
                    if isinstance(link, dict) and "activityID" in link:
                        linked_ids.append(link["activityID"])

            # Source location (solar coordinates)
            source_location = event.get("sourceLocation")

            # Instruments
            instruments = []
            for inst in event.get("instruments", []) or []:
                if isinstance(inst, dict):
                    instruments.append(inst.get("displayName", ""))

            rows.append({
                "event_id": eid,
                "event_type": event_type,
                "event_date": event_date,
                "start_time": start_dt,
                "end_time": end_dt,
                "flare_class": flare_class,
                "flare_severity": flare_severity,
                "cme_speed_kms": cme_speed,
                "max_kp_index": max_kp,
                "linked_events": linked_ids,
                "source_location": source_location,
                "instruments": instruments,
            })

    df = pd.DataFrame(rows)

    # Sort by start time
    if "start_time" in df.columns:
        df = df.sort_values("start_time").reset_index(drop=True)

    logger.info(
        "Normalised %d events: %d CME, %d FLR, %d GST.",
        len(df),
        (df["event_type"] == "CME").sum(),
        (df["event_type"] == "FLR").sum(),
        (df["event_type"] == "GST").sum(),
    )

    return df


def load_and_normalise() -> pd.DataFrame:
    """
    Load raw JSON files from data/raw/ and normalise into a DataFrame.

    Returns the normalised events DataFrame.
    """
    raw_data: dict[str, list[dict]] = {}
    for event_type in EVENT_TYPES:
        # Find the JSON file for this event type
        pattern = f"{event_type}_*.json"
        files = sorted(DATA_RAW.glob(pattern))
        if not files:
            logger.warning("No raw data files found for %s (pattern: %s)", event_type, pattern)
            raw_data[event_type] = []
            continue
        # Use the most recent file
        filepath = files[-1]
        with open(filepath) as f:
            raw_data[event_type] = json.load(f)
        logger.info("Loaded %d %s events from %s", len(raw_data[event_type]), event_type, filepath.name)

    return normalise_events(raw_data)
