"""
aggregates.py — Compute deterministic trends from normalised DONKI events.

Produces four output files in data/processed/:
- daily_counts.json: events per day per type
- rolling_counts.json: 7-day and 30-day rolling sums per type
- severity.json: max flare class, max CME speed, max Kp index per day
- linkages.json: cross-event relationship edges (FLR→CME, CME→GST)

All computations are deterministic — no ML, no sampling, no randomness.

Usage:
    from src.features.aggregates import compute_all_aggregates
    compute_all_aggregates(events_df)
"""

import json
import logging
from pathlib import Path

import pandas as pd

from config import DATA_PROCESSED, ROLLING_WINDOWS

logger = logging.getLogger(__name__)


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def compute_daily_counts(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute daily event counts by type.

    Returns DataFrame with columns:
    date, cme_count, flr_count, gst_count, total_count
    """
    if df.empty:
        logger.warning("Empty DataFrame — returning empty daily counts.")
        return pd.DataFrame(columns=["date", "cme_count", "flr_count", "gst_count", "total_count"])

    daily = (
        df.groupby(["event_date", "event_type"])
        .size()
        .unstack(fill_value=0)
    )
    # Ensure all three types are present
    for col in ["CME", "FLR", "GST"]:
        if col not in daily.columns:
            daily[col] = 0

    daily = daily[["CME", "FLR", "GST"]]
    daily.columns = ["cme_count", "flr_count", "gst_count"]
    daily["total_count"] = daily.sum(axis=1)
    daily = daily.sort_index().reset_index()
    daily.rename(columns={"event_date": "date"}, inplace=True)

    return daily


def compute_rolling_counts(daily: pd.DataFrame) -> pd.DataFrame:
    """
    Compute rolling window sums for each event type.

    Returns DataFrame with columns:
    date, cme_7d, cme_30d, flr_7d, flr_30d, gst_7d, gst_30d
    """
    if daily.empty:
        return pd.DataFrame()

    df = daily.set_index("date").sort_index()
    result = pd.DataFrame(index=df.index)

    for event_type in ["cme", "flr", "gst"]:
        col = f"{event_type}_count"
        if col not in df.columns:
            continue
        for window in ROLLING_WINDOWS:
            result[f"{event_type}_{window}d"] = (
                df[col].rolling(window=window, min_periods=1).sum()
            )

    return result.reset_index().rename(columns={"index": "date"})


def compute_severity(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute daily maximum severity indicators.

    Returns DataFrame with columns:
    date, max_flare_severity, max_cme_speed_kms, max_kp_index
    """
    if df.empty:
        return pd.DataFrame()

    daily = df.copy()
    daily["date"] = daily["event_date"]

    # Max flare severity per day
    flares = daily[daily["event_type"] == "FLR"]
    flare_sev = (
        flares.groupby("date")["flare_severity"]
        .max()
        .reset_index()
        .rename(columns={"flare_severity": "max_flare_severity"})
    )

    # Max CME speed per day
    cmes = daily[daily["event_type"] == "CME"]
    cme_speed = (
        cmes.groupby("date")["cme_speed_kms"]
        .max()
        .reset_index()
        .rename(columns={"cme_speed_kms": "max_cme_speed_kms"})
    )

    # Max Kp index per day
    gsts = daily[daily["event_type"] == "GST"]
    kp = (
        gsts.groupby("date")["max_kp_index"]
        .max()
        .reset_index()
        .rename(columns={"max_kp_index": "max_kp_index"})
    )

    # Merge all severity indicators
    severity = pd.DataFrame({"date": daily["date"].unique()})
    severity = severity.merge(flare_sev, on="date", how="left")
    severity = severity.merge(cme_speed, on="date", how="left")
    severity = severity.merge(kp, on="date", how="left")
    severity = severity.sort_values("date").reset_index(drop=True)

    return severity


def compute_linkages(df: pd.DataFrame) -> list[dict]:
    """
    Extract cross-event relationships from linkedEvents fields.

    Returns a list of edge dicts:
    [{from: "FLR-2026-06-01T...", to: "CME-2026-06-01T...", from_type: "FLR", to_type: "CME"}, ...]
    """
    edges = []
    for _, row in df.iterrows():
        linked = row.get("linked_events")
        if not linked or not isinstance(linked, list):
            continue
        for target_id in linked:
            # Determine target type from ID prefix convention
            target_type = None
            if "-FLR-" in target_id:
                target_type = "FLR"
            elif "-CME-" in target_id:
                target_type = "CME"
            elif "-GST-" in target_id:
                target_type = "GST"
            elif "-IPS-" in target_id:
                target_type = "IPS"
            else:
                target_type = "UNKNOWN"

            edges.append({
                "from": row["event_id"],
                "to": target_id,
                "from_type": row["event_type"],
                "to_type": target_type,
            })

    logger.info("Extracted %d cross-event linkages.", len(edges))
    return edges


def compute_top_active_days(daily: pd.DataFrame, top_n: int = 5) -> list[dict]:
    """
    Return the top-N most active days by total event count.

    Returns list of {date, total_count, cme_count, flr_count, gst_count}.
    """
    if daily.empty:
        return []

    top = daily.nlargest(top_n, "total_count")
    return top.to_dict(orient="records")


def compute_all_aggregates(df: pd.DataFrame) -> dict[str, Path]:
    """
    Compute all deterministic aggregates and save to data/processed/.

    Returns a dict mapping aggregate names to output file paths.
    """
    _ensure_dir(DATA_PROCESSED)
    outputs = {}

    # 1. Daily counts
    daily = compute_daily_counts(df)
    path = DATA_PROCESSED / "daily_counts.json"
    daily.to_json(path, orient="records", date_format="iso", indent=2)
    outputs["daily_counts"] = path
    logger.info("daily_counts: %d days → %s", len(daily), path)

    # 2. Rolling counts
    rolling = compute_rolling_counts(daily)
    path = DATA_PROCESSED / "rolling_counts.json"
    rolling.to_json(path, orient="records", date_format="iso", indent=2)
    outputs["rolling_counts"] = path
    logger.info("rolling_counts: %d days → %s", len(rolling), path)

    # 3. Severity indicators
    severity = compute_severity(df)
    path = DATA_PROCESSED / "severity.json"
    severity.to_json(path, orient="records", date_format="iso", indent=2)
    outputs["severity"] = path
    logger.info("severity: %d days → %s", len(severity), path)

    # 4. Cross-event linkages
    linkages = compute_linkages(df)
    path = DATA_PROCESSED / "linkages.json"
    with open(path, "w") as f:
        json.dump({"edges": linkages}, f, indent=2)
    outputs["linkages"] = path
    logger.info("linkages: %d edges → %s", len(linkages), path)

    # 5. Top active days (for dashboard)
    top_days = compute_top_active_days(daily)
    path = DATA_PROCESSED / "top_days.json"
    with open(path, "w") as f:
        json.dump(top_days, f, indent=2)
    outputs["top_days"] = path
    logger.info("top_days: %d days → %s", len(top_days), path)

    return outputs


# ── CLI entry point ────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    from src.features.normalise import load_and_normalise

    print("Loading and normalising DONKI events...")
    events_df = load_and_normalise()
    print(f"  {len(events_df)} events loaded.")

    print("\nComputing aggregates...")
    outputs = compute_all_aggregates(events_df)
    for name, path in outputs.items():
        size = path.stat().st_size
        print(f"  {name}: {path.name} ({size:,} bytes)")
    print("Done.")
