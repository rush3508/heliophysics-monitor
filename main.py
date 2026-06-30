#!/usr/bin/env python3
"""
main.py — Heliophysics Monitor: single entry point for the full pipeline.

Usage:
    python main.py                     # full pipeline (ingest → dashboard)
    python main.py --ingest-only       # only fetch DONKI data
    python main.py --features-only     # only compute aggregates
    python main.py --build-corpus      # one-off: scrape + embed corpus
    python main.py --brief-only        # only generate analyst brief
    python main.py --dashboard-only    # only rebuild dashboard
    python main.py --backfill N        # backfill N days (default: 180)

Stages:
    1. Ingest DONKI events (180-day backfill on first run)
    2. Compute deterministic features (normalise + aggregates)
    3. Retrieve relevant chunks from the corpus
    4. Generate LLM analyst brief (llama3.2)
    5. Build static HTML dashboard (Plotly + Jinja2)
"""

import argparse
import logging
import sys
import time
from datetime import datetime, timedelta, timezone

import psutil

from config import (
    BACKFILL_DAYS,
    RAM_ABORT_THRESHOLD_GB,
)

logger = logging.getLogger("heliophysics-monitor")


def _check_ram(stage: str) -> None:
    """Abort if RAM exceeds the safety threshold."""
    mem = psutil.virtual_memory()
    used_gb = mem.used / (1024**3)
    if used_gb > RAM_ABORT_THRESHOLD_GB:
        logger.error(
            "RAM usage (%.1f GB) exceeds threshold (%.1f GB). Aborting.",
            used_gb, RAM_ABORT_THRESHOLD_GB,
        )
        sys.exit(1)
    logger.info("[RAM] %.1f GB used (threshold: %.1f GB) — %s", used_gb, RAM_ABORT_THRESHOLD_GB, stage)


def cmd_ingest(days: int) -> None:
    """Fetch DONKI events for the past N days."""
    from src.ingest.fetch_donki import ingest_all

    end = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    start = (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%d")

    print(f"\n{'='*60}")
    print(f"STAGE 1: DONKI Ingestion ({start} → {end})")
    print(f"{'='*60}")
    t0 = time.time()
    results = ingest_all(start, end)
    total = sum(len(v) for v in results.values())
    print(f"  CME: {len(results.get('CME', []))} events")
    print(f"  FLR: {len(results.get('FLR', []))} events")
    print(f"  GST: {len(results.get('GST', []))} events")
    print(f"  Total: {total} events in {time.time() - t0:.1f}s")
    _check_ram("ingest")


def cmd_features() -> None:
    """Compute deterministic aggregates."""
    from src.features.normalise import load_and_normalise
    from src.features.aggregates import compute_all_aggregates

    print(f"\n{'='*60}")
    print("STAGE 2: Feature Engineering")
    print(f"{'='*60}")
    t0 = time.time()
    df = load_and_normalise()
    print(f"  Normalised: {len(df)} events")
    outputs = compute_all_aggregates(df)
    for name, path in outputs.items():
        print(f"  {name}: {path.name} ({path.stat().st_size:,} bytes)")
    print(f"  Done in {time.time() - t0:.1f}s")
    _check_ram("features")


def cmd_corpus() -> None:
    """Build and embed the retrieval corpus (one-off)."""
    from src.rag.build_corpus import build_corpus
    from src.rag.embed import embed_corpus

    print(f"\n{'='*60}")
    print("STAGE 3a+b: Corpus Build + Embed")
    print(f"{'='*60}")
    t0 = time.time()

    print("\n-- Building corpus (scrape + chunk) --")
    chunks = build_corpus()
    print(f"  Chunks: {len(chunks)}")

    print("\n-- Embedding corpus (nomic-embed-text → ChromaDB) --")
    collection = embed_corpus()
    print(f"  Collection: {collection.name} ({collection.count()} documents)")

    print(f"  Done in {time.time() - t0:.1f}s")
    _check_ram("corpus")


def cmd_brief() -> None:
    """Generate the LLM analyst brief."""
    from src.reporting.generate_brief import generate_brief

    print(f"\n{'='*60}")
    print("STAGE 4: LLM Analyst Brief")
    print(f"{'='*60}")
    t0 = time.time()
    _check_ram("brief (pre)")
    brief = generate_brief()
    word_count = len(brief.split())
    print(f"  Words: {word_count}")
    print(f"  First 120 chars: {brief[:120]}...")
    print(f"  Done in {time.time() - t0:.1f}s")
    _check_ram("brief (post)")


def cmd_dashboard() -> None:
    """Build the static HTML dashboard."""
    from src.reporting.build_dashboard import build_dashboard

    print(f"\n{'='*60}")
    print("STAGE 5: Static Dashboard")
    print(f"{'='*60}")
    t0 = time.time()
    _check_ram("dashboard (pre)")
    path = build_dashboard()
    print(f"  Dashboard: {path} ({path.stat().st_size:,} bytes)")
    print(f"  Done in {time.time() - t0:.1f}s")
    _check_ram("dashboard (post)")


def cmd_all(days: int) -> None:
    """Run the full pipeline."""
    total_t0 = time.time()

    cmd_ingest(days)
    cmd_features()
    cmd_brief()
    cmd_dashboard()

    elapsed = time.time() - total_t0

    print(f"\n{'='*60}")
    print(f"PIPELINE COMPLETE — {elapsed:.1f}s")
    print(f"{'='*60}")

    if elapsed > 180:
        print(f"⚠ Runtime ({elapsed:.0f}s) exceeds 3-minute target. Consider optimising.")
    else:
        print(f"✓ Runtime ({elapsed:.0f}s) within 3-minute target.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Heliophysics Monitor — space weather intelligence pipeline"
    )
    parser.add_argument(
        "--ingest-only", action="store_true",
        help="Only fetch DONKI events"
    )
    parser.add_argument(
        "--features-only", action="store_true",
        help="Only compute aggregates"
    )
    parser.add_argument(
        "--build-corpus", action="store_true",
        help="One-off: scrape + embed the retrieval corpus"
    )
    parser.add_argument(
        "--brief-only", action="store_true",
        help="Only generate the analyst brief"
    )
    parser.add_argument(
        "--dashboard-only", action="store_true",
        help="Only rebuild the dashboard"
    )
    parser.add_argument(
        "--backfill", type=int, default=BACKFILL_DAYS,
        help=f"Days to backfill (default: {BACKFILL_DAYS})"
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    print("Heliophysics Monitor")
    print(f"  NASA API key: {'DEMO_KEY' if 'DEMO_KEY' in str(__import__('os').environ.get('NASA_API_KEY', '')) else 'REGISTERED'}")
    print(f"  RAM threshold: {RAM_ABORT_THRESHOLD_GB} GB")

    if args.ingest_only:
        cmd_ingest(args.backfill)
    elif args.features_only:
        cmd_features()
    elif args.build_corpus:
        cmd_corpus()
    elif args.brief_only:
        cmd_brief()
    elif args.dashboard_only:
        cmd_dashboard()
    else:
        cmd_all(args.backfill)


if __name__ == "__main__":
    main()
