"""
config.py — Heliophysics Monitor: single source of truth.

All paths, parameters, API settings, corpus URLs, model names,
chunk sizes, and seed values live here. Every module imports from
this file; no hardcoded values anywhere.
"""

import os
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
DATA_CORPUS = PROJECT_ROOT / "data" / "corpus"
DOCS_CORPUS = PROJECT_ROOT / "docs" / "corpus"
CHUNKS_FILE = DOCS_CORPUS / "chunks.json"
CHROMA_PATH = DOCS_CORPUS / "chroma"
DASHBOARD_DIR = PROJECT_ROOT / "dashboard"
DASHBOARD_DATA = DASHBOARD_DIR / "data"
REPORTS_DIR = PROJECT_ROOT / "reports"
BRIEF_FILE = REPORTS_DIR / "brief.md"
WARNINGS_LOG = REPORTS_DIR / "pipeline_warnings.log"

# ── NASA DONKI API ─────────────────────────────────────────────────
NASA_API_BASE = "https://api.nasa.gov/DONKI"
NASA_API_KEY = os.environ.get("NASA_API_KEY", "DEMO_KEY")

# Event types in scope for v1
EVENT_TYPES = {
    "CME": {"endpoint": "CME", "id_field": "activityID"},
    "FLR": {"endpoint": "FLR", "id_field": "flrID"},
    "GST": {"endpoint": "GST", "id_field": "gstID"},
}

# Historical backfill window (days)
BACKFILL_DAYS = 180

# Rate-limit safety margin: pause if remaining requests ≤ this
RATE_LIMIT_BUFFER = 2

# Retry settings
MAX_RETRIES = 3
RETRY_BACKOFF_SECONDS = 5
REQUEST_TIMEOUT_SECONDS = 30

# ── Feature Engineering ────────────────────────────────────────────
# Flare class severity scale: base × magnitude
FLARE_CLASS_MULTIPLIER = {"A": 0.1, "B": 1, "C": 1, "M": 10, "X": 100}

# Rolling windows (days)
ROLLING_WINDOWS = [7, 30]

# ── Retrieval Corpus ──────────────────────────────────────────────
CORPUS_URLS = [
    {
        "url": "https://ccmc.gsfc.nasa.gov/tools/DONKI/",
        "label": "CCMC DONKI Overview",
        "selector": "main",
    },
    {
        "url": "https://science.nasa.gov/heliophysics/",
        "label": "NASA Heliophysics",
        "selector": "article",
    },
    {
        "url": "https://science.nasa.gov/heliophysics/focus-areas/",
        "label": "NASA Heliophysics Focus Areas",
        "selector": "article",
    },
    {
        "url": "https://spdf.gsfc.nasa.gov/Glossary.html",
        "label": "SPDF Glossary",
        "selector": "body",
        "optional": True,
    },
    {
        "url": "https://science.nasa.gov/heliophysics/space-weather/",
        "label": "NASA Space Weather",
        "selector": "article",
        "optional": True,
    },
]

CHUNK_TOKEN_SIZE = 120         # target tokens per chunk
CHUNK_OVERLAP_TOKENS = 50     # overlap between adjacent chunks
MAX_CHUNK_TOKENS = 500        # reject any chunk exceeding this
MIN_CORPUS_PAGES = 4         # abort if fewer than this many URLs succeed
MIN_CHUNKS = 80
MAX_CHUNKS = 180

# ChromaDB
CHROMA_COLLECTION_NAME = "heliophysics-corpus"
RETRIEVAL_TOP_K = 3
COSINE_FLOOR = 0.25          # minimum similarity to consider a chunk relevant

# ── Controlled Queries (SC4) ───────────────────────────────────────
CONTROLLED_QUERIES = [
    "What is a CME?",
    "How is a geomagnetic storm different from a solar flare?",
    "What does M-class or X-class flare mean?",
    "Why does this week's event cluster matter?",
]

# ── LLM (llama3.2:3b via Ollama) ──────────────────────────────────
OLLAMA_MODEL = "llama3.2"
OLLAMA_EMBED_MODEL = "nomic-embed-text"
OLLAMA_CONTEXT_WINDOW = 4096

# Token budget for llama3.2 prompt (leave headroom below context_window)
PROMPT_TOKEN_BUDGET = 3500

# Brief constraints
BRIEF_MIN_WORDS = 150
BRIEF_MAX_WORDS = 250
BRIEF_RETRY_MIN_WORDS = 100     # retry if shorter than this on first pass

# ── RAM Monitoring ─────────────────────────────────────────────────
RAM_CEILING_GB = 8.0
RAM_ABORT_THRESHOLD_GB = 7.5    # abort before hitting ceiling

# ── Staleness ──────────────────────────────────────────────────────
DATA_MAX_AGE_HOURS = 24          # warn if raw data is older than this

# ── Seeds ──────────────────────────────────────────────────────────
GLOBAL_SEED = 42
# (No DGP in this project — seeds are for reproducibility in any
#  non-deterministic operations like train/test splits if added later)

# ── Dashboard ──────────────────────────────────────────────────────
DASHBOARD_PANELS = [
    "ingestion_timestamp",
    "event_counts_7d_30d",
    "event_timeline",
    "top_active_days",
    "terminology_card",
    "analyst_brief",
]

# Timezone
LOCAL_TIMEZONE = "Asia/Kuala_Lumpur"  # UTC+8
