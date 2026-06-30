1|#!/usr/bin/env python3
2|"""
3|main.py — Heliophysics Monitor: single entry point for the full pipeline.
4|
5|Usage:
6|    python main.py                     # full pipeline (ingest → dashboard)
7|    python main.py --ingest-only       # only fetch DONKI data
8|    python main.py --features-only     # only compute aggregates
9|    python main.py --build-corpus      # one-off: scrape + embed corpus
10|    python main.py --brief-only        # only generate analyst brief
11|    python main.py --dashboard-only    # only rebuild dashboard
12|    python main.py --backfill N        # backfill N days (default: 180)
13|
14|Stages:
15|    1. Ingest DONKI events (180-day backfill on first run)
16|    2. Compute deterministic features (normalise + aggregates)
17|    3. Retrieve relevant chunks from the corpus
18|    4. Generate LLM analyst brief (llama3.2)
19|    5. Build static HTML dashboard (Plotly + Jinja2)
20|"""
21|
22|import argparse
23|import logging
24|import sys
25|import time
26|from datetime import datetime, timedelta, timezone
27|
28|import psutil
29|
30|from config import (
31|    BACKFILL_DAYS,
32|    RAM_ABORT_THRESHOLD_GB,
33|)
34|
35|logger = logging.getLogger("heliophysics-monitor")
36|
37|
38|def _check_ram(stage: str) -> None:
39|    """Abort if RAM exceeds the safety threshold."""
40|    mem = psutil.virtual_memory()
41|    used_gb = mem.used / (1024**3)
42|    if used_gb > RAM_ABORT_THRESHOLD_GB:
43|        logger.error(
44|            "RAM usage (%.1f GB) exceeds threshold (%.1f GB). Aborting.",
45|            used_gb, RAM_ABORT_THRESHOLD_GB,
46|        )
47|        sys.exit(1)
48|    logger.info("[RAM] %.1f GB used (threshold: %.1f GB) — %s", used_gb, RAM_ABORT_THRESHOLD_GB, stage)
49|
50|
51|def cmd_ingest(days: int) -> None:
52|    """Fetch DONKI events for the past N days."""
53|    from src.ingest.fetch_donki import ingest_all
54|
55|    end = datetime.now(timezone.utc).strftime("%Y-%m-%d")
56|    start = (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%d")
57|
58|    print(f"\n{'='*60}")
59|    print(f"STAGE 1: DONKI Ingestion ({start} → {end})")
60|    print(f"{'='*60}")
61|    t0 = time.time()
62|    results = ingest_all(start, end)
63|    total = sum(len(v) for v in results.values())
64|    print(f"  CME: {len(results.get('CME', []))} events")
65|    print(f"  FLR: {len(results.get('FLR', []))} events")
66|    print(f"  GST: {len(results.get('GST', []))} events")
67|    print(f"  Total: {total} events in {time.time() - t0:.1f}s")
68|    _check_ram("ingest")
69|
70|
71|def cmd_features() -> None:
72|    """Compute deterministic aggregates."""
73|    from src.features.normalise import load_and_normalise
74|    from src.features.aggregates import compute_all_aggregates
75|
76|    print(f"\n{'='*60}")
77|    print("STAGE 2: Feature Engineering")
78|    print(f"{'='*60}")
79|    t0 = time.time()
80|    df = load_and_normalise()
81|    print(f"  Normalised: {len(df)} events")
82|    outputs = compute_all_aggregates(df)
83|    for name, path in outputs.items():
84|        print(f"  {name}: {path.name} ({path.stat().st_size:,} bytes)")
85|    print(f"  Done in {time.time() - t0:.1f}s")
86|    _check_ram("features")
87|
88|
89|def cmd_corpus() -> None:
90|    """Build and embed the retrieval corpus (one-off)."""
91|    from src.rag.build_corpus import build_corpus
92|    from src.rag.embed import embed_corpus
93|
94|    print(f"\n{'='*60}")
95|    print("STAGE 3a+b: Corpus Build + Embed")
96|    print(f"{'='*60}")
97|    t0 = time.time()
98|
99|    print("\n-- Building corpus (scrape + chunk) --")
100|    chunks = build_corpus()
101|    print(f"  Chunks: {len(chunks)}")
102|
103|    print("\n-- Embedding corpus (nomic-embed-text → ChromaDB) --")
104|    collection = embed_corpus()
105|    print(f"  Collection: {collection.name} ({collection.count()} documents)")
106|
107|    print(f"  Done in {time.time() - t0:.1f}s")
108|    _check_ram("corpus")
109|
110|
111|def cmd_brief() -> None:
112|    """Generate the LLM analyst brief."""
113|    from src.reporting.generate_brief import generate_brief
114|
115|    print(f"\n{'='*60}")
116|    print("STAGE 4: LLM Analyst Brief")
117|    print(f"{'='*60}")
118|    t0 = time.time()
119|    _check_ram("brief (pre)")
120|    brief = generate_brief()
121|    word_count = len(brief.split())
122|    print(f"  Words: {word_count}")
123|    print(f"  First 120 chars: {brief[:120]}...")
124|    print(f"  Done in {time.time() - t0:.1f}s")
125|    _check_ram("brief (post)")
126|
127|
128|def cmd_dashboard() -> None:
129|    """Build the static HTML dashboard."""
130|    from src.reporting.build_dashboard import build_dashboard
131|
132|    print(f"\n{'='*60}")
133|    print("STAGE 5: Static Dashboard")
134|    print(f"{'='*60}")
135|    t0 = time.time()
136|    _check_ram("dashboard (pre)")
137|    path = build_dashboard()
138|    print(f"  Dashboard: {path} ({path.stat().st_size:,} bytes)")
139|    print(f"  Done in {time.time() - t0:.1f}s")
140|    _check_ram("dashboard (post)")
141|
142|
143|def cmd_all(days: int) -> None:
144|    """Run the full pipeline."""
145|    total_t0 = time.time()
146|
147|    cmd_ingest(days)
148|    cmd_features()
149|    cmd_brief()
150|    cmd_dashboard()
151|
152|    elapsed = time.time() - total_t0
153|
154|    print(f"\n{'='*60}")
155|    print(f"PIPELINE COMPLETE — {elapsed:.1f}s")
156|    print(f"{'='*60}")
157|
158|    if elapsed > 180:
159|        print(f"⚠ Runtime ({elapsed:.0f}s) exceeds 3-minute target. Consider optimising.")
160|    else:
161|        print(f"✓ Runtime ({elapsed:.0f}s) within 3-minute target.")
162|
163|
164|def main() -> None:
165|    parser = argparse.ArgumentParser(
166|        description="Heliophysics Monitor — space weather intelligence pipeline"
167|    )
168|    parser.add_argument(
169|        "--ingest-only", action="store_true",
170|        help="Only fetch DONKI events"
171|    )
172|    parser.add_argument(
173|        "--features-only", action="store_true",
174|        help="Only compute aggregates"
175|    )
176|    parser.add_argument(
177|        "--build-corpus", action="store_true",
178|        help="One-off: scrape + embed the retrieval corpus"
179|    )
180|    parser.add_argument(
181|        "--brief-only", action="store_true",
182|        help="Only generate the analyst brief"
183|    )
184|    parser.add_argument(
185|        "--dashboard-only", action="store_true",
186|        help="Only rebuild the dashboard"
187|    )
188|    parser.add_argument(
189|        "--backfill", type=int, default=BACKFILL_DAYS,
190|        help=f"Days to backfill (default: {BACKFILL_DAYS})"
191|    )
192|
193|    args = parser.parse_args()
194|
195|    logging.basicConfig(
196|        level=logging.INFO,
197|        format="%(asctime)s [%(levelname)s] %(message)s",
198|    )
199|
200|    print("Heliophysics Monitor")
201|    print(f"  NASA API key: {'DEMO_KEY' if 'DEMO_KEY' in str(__import__('os').environ.get('NASA_API_KEY', '')) else 'REGISTERED'}")
202|    print(f"  RAM threshold: {RAM_ABORT_THRESHOLD_GB} GB")
203|
204|    if args.ingest_only:
205|        cmd_ingest(args.backfill)
206|    elif args.features_only:
207|        cmd_features()
208|    elif args.build_corpus:
209|        cmd_corpus()
210|    elif args.brief_only:
211|        cmd_brief()
212|    elif args.dashboard_only:
213|        cmd_dashboard()
214|    else:
215|        cmd_all(args.backfill)
216|
217|
218|if __name__ == "__main__":
219|    main()
220|