1|"""
2|config.py — Heliophysics Monitor: single source of truth.
3|
4|All paths, parameters, API settings, corpus URLs, model names,
5|chunk sizes, and seed values live here. Every module imports from
6|this file; no hardcoded values anywhere.
7|"""
8|
9|import os
10|from pathlib import Path
11|
12|# ── Paths ──────────────────────────────────────────────────────────
13|PROJECT_ROOT = Path(__file__).resolve().parent
14|DATA_RAW = PROJECT_ROOT / "data" / "raw"
15|DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
16|DATA_CORPUS = PROJECT_ROOT / "data" / "corpus"
17|DOCS_CORPUS = PROJECT_ROOT / "docs" / "corpus"
18|CHUNKS_FILE = DOCS_CORPUS / "chunks.json"
19|CHROMA_PATH = DOCS_CORPUS / "chroma"
20|DASHBOARD_DIR = PROJECT_ROOT / "dashboard"
21|DASHBOARD_DATA = DASHBOARD_DIR / "data"
22|REPORTS_DIR = PROJECT_ROOT / "reports"
23|BRIEF_FILE = REPORTS_DIR / "brief.md"
24|WARNINGS_LOG = REPORTS_DIR / "pipeline_warnings.log"
25|
26|# ── NASA DONKI API ─────────────────────────────────────────────────
27|NASA_API_BASE = "https://api.nasa.gov/DONKI"
28|NASA_API_KEY = os.environ.get("NASA_API_KEY", "DEMO_KEY")
29|
30|# Event types in scope for v1
31|EVENT_TYPES = {
32|    "CME": {"endpoint": "CME", "id_field": "activityID"},
33|    "FLR": {"endpoint": "FLR", "id_field": "flrID"},
34|    "GST": {"endpoint": "GST", "id_field": "gstID"},
35|}
36|
37|# Historical backfill window (days)
38|BACKFILL_DAYS = 180
39|
40|# Rate-limit safety margin: pause if remaining requests ≤ this
41|RATE_LIMIT_BUFFER = 2
42|
43|# Retry settings
44|MAX_RETRIES = 3
45|RETRY_BACKOFF_SECONDS = 5
46|REQUEST_TIMEOUT_SECONDS = 30
47|
48|# ── Feature Engineering ────────────────────────────────────────────
49|# Flare class severity scale: base × magnitude
50|FLARE_CLASS_MULTIPLIER = {"A": 0.1, "B": 1, "C": 1, "M": 10, "X": 100}
51|
52|# Rolling windows (days)
53|ROLLING_WINDOWS = [7, 30]
54|
55|# ── Retrieval Corpus ──────────────────────────────────────────────
56|CORPUS_URLS = [
57|    {
58|        "url": "https://ccmc.gsfc.nasa.gov/tools/DONKI/",
59|        "label": "CCMC DONKI Overview",
60|        "selector": "main",
61|    },
62|    {
63|        "url": "https://science.nasa.gov/heliophysics/",
64|        "label": "NASA Heliophysics",
65|        "selector": "article",
66|    },
67|    {
68|        "url": "https://science.nasa.gov/heliophysics/focus-areas/",
69|        "label": "NASA Heliophysics Focus Areas",
70|        "selector": "article",
71|    },
72|    {
73|        "url": "https://spdf.gsfc.nasa.gov/Glossary.html",
74|        "label": "SPDF Glossary",
75|        "selector": "body",
76|        "optional": True,
77|    },
78|    {
79|        "url": "https://science.nasa.gov/heliophysics/space-weather/",
80|        "label": "NASA Space Weather",
81|        "selector": "article",
82|        "optional": True,
83|    },
84|]
85|
86|CHUNK_TOKEN_SIZE = 120         # target tokens per chunk
87|CHUNK_OVERLAP_TOKENS = 50     # overlap between adjacent chunks
88|MAX_CHUNK_TOKENS = 500        # reject any chunk exceeding this
89|MIN_CORPUS_PAGES = 4         # abort if fewer than this many URLs succeed
90|MIN_CHUNKS = 80
91|MAX_CHUNKS = 180
92|
93|# ChromaDB
94|CHROMA_COLLECTION_NAME = "heliophysics-corpus"
95|RETRIEVAL_TOP_K = 3
96|COSINE_FLOOR = 0.25          # minimum similarity to consider a chunk relevant
97|
98|# ── Controlled Queries (SC4) ───────────────────────────────────────
99|CONTROLLED_QUERIES = [
100|    "What is a CME?",
101|    "How is a geomagnetic storm different from a solar flare?",
102|    "What does M-class or X-class flare mean?",
103|    "Why does this week's event cluster matter?",
104|]
105|
106|# ── LLM (llama3.2:3b via Ollama) ──────────────────────────────────
107|OLLAMA_MODEL = "llama3.2"
108|OLLAMA_EMBED_MODEL = "nomic-embed-text"
109|OLLAMA_CONTEXT_WINDOW = 4096
110|
111|# Token budget for llama3.2 prompt (leave headroom below context_window)
112|PROMPT_TOKEN_BUDGET = 3500
113|
114|# Brief constraints
115|BRIEF_MIN_WORDS = 150
116|BRIEF_MAX_WORDS = 250
117|BRIEF_RETRY_MIN_WORDS = 100     # retry if shorter than this on first pass
118|
119|# ── RAM Monitoring ─────────────────────────────────────────────────
120|RAM_CEILING_GB = 8.0
121|RAM_ABORT_THRESHOLD_GB = 7.5    # abort before hitting ceiling
122|
123|# ── Staleness ──────────────────────────────────────────────────────
124|DATA_MAX_AGE_HOURS = 24          # warn if raw data is older than this
125|
126|# ── Seeds ──────────────────────────────────────────────────────────
127|GLOBAL_SEED = 42
128|# (No DGP in this project — seeds are for reproducibility in any
129|#  non-deterministic operations like train/test splits if added later)
130|
131|# ── Dashboard ──────────────────────────────────────────────────────
132|DASHBOARD_PANELS = [
133|    "ingestion_timestamp",
134|    "event_counts_7d_30d",
135|    "event_timeline",
136|    "top_active_days",
137|    "terminology_card",
138|    "analyst_brief",
139|]
140|
141|# Timezone
142|LOCAL_TIMEZONE = "Asia/Kuala_Lumpur"  # UTC+8
143|