# CLAUDE.md — Heliophysics Monitor

> **Project bible — read this entire file before touching anything.**
> **PRD:** `v530s/Mission-Control/prd-drafts/heliophysics-monitor.md`
> **Status:** ⏳ Stage 0 in progress (scaffold)
> **Last updated:** 2026-06-30

---

## 1. What This Project Is and Why It Exists

This is a **data pipeline portfolio project** demonstrating a local-first Space Weather Intelligence
System that ingests NASA DONKI (Space Weather Database of Notifications, Knowledge, Information)
event data, computes deterministic trends, enriches analysis with a retrieval-augmented generation
(RAG) agent over heliophysics terminology, and publishes a static HTML dashboard plus a Markdown
analyst brief — all running on the V530s homelab with zero cloud costs.

**What it demonstrates:**
- API engineering against a real government data source (NASA DONKI, 3 endpoints)
- Deterministic time-series feature engineering (daily/rolling counts, severity indicators, cross-event linkages)
- RAG pipeline: web scraping → chunking → embedding (nomic-embed-text) → ChromaDB vector storage → cosine retrieval with LLM grounding
- Constrained LLM inference with phi3:mini (3.8B Q4, `context_window=4096`, serial model execution)
- Static dashboard delivery (Plotly + Jinja2, zero server dependencies)
- RAM budget discipline (< 8 GB), serial model loading, token and staleness guards
- Manual-invoke, spec-driven pipeline with 9 stages (Stages 0–7b)

**This is NOT an ML project.** There are no models, no training loops, no data-generating process
(DGP), and no stochastic target variables. Every transform is deterministic. The only non-deterministic
component is phi3:mini inference, which is constrained by strict prompt boundaries and retrieval grounding.

---

## 2. Architecture

```
╔══════════════════════════════════════════════════════════════════════════════════╗
║                            DONKI API (NASA)                                     ║
║              https://api.nasa.gov/DONKI/{CME,FLR,GST}                           ║
╚══════════════════════════════════════════════════════════════════════════════════╝
                                    │
                                    ▼
╔══════════════════════════════════════════════════════════════════════════════════╗
║                         FETCH LAYER (src/ingest/fetch_donki.py)                 ║
║  3 endpoint fetchers (CME, FLR, GST) · 180-day backfill · rate-limit handling   ║
║  Output: data/raw/{endpoint}_{date}.json                                        ║
╚══════════════════════════════════════════════════════════════════════════════════╝
                                    │
                                    ▼
╔══════════════════════════════════════════════════════════════════════════════════╗
║                      NORMALISE LAYER (src/features/normalise.py)                ║
║  Parse timestamps · Extract event types · Flare severity (C/M/X→numeric)        ║
║  CME speed (mostAccurate analysis) · Kp index for GSTs · Derive event_date      ║
╚══════════════════════════════════════════════════════════════════════════════════╝
                                    │
                                    ▼
╔══════════════════════════════════════════════════════════════════════════════════╗
║                     AGGREGATE LAYER (src/features/aggregates.py)                ║
║  Daily counts · Rolling 7d/30d · Severity indicators · Top active days          ║
║  Cross-event linkages (linkedEvents → edges)                                    ║
║  Output: data/processed/{daily_counts,rolling_counts,severity,linkages}.json     ║
╚══════════════════════════════════════════════════════════════════════════════════╝
                                    │
                    ┌───────────────┴───────────────┐
                    ▼                               ▼
╔════════════════════════════════╗   ╔═══════════════════════════════════════════╗
║  CORPUS BUILD (one-off)       ║   ║          DATA FLOW (per run)              ║
║  src/rag/build_corpus.py      ║   ║                                           ║
║  5 NASA pages scraped         ║   ║  Aggregates → Dashboard → index.html      ║
║  → chunks.json (80–180)      ║   ║  Aggregates + RAG → LLM → brief.md        ║
╚════════════════════════════════╝   ║                                           ║
              │                      ╚═══════════════════════════════════════════╝
              ▼
╔══════════════════════════════════════════════════════════════════════════════════╗
║            EMBED + RETRIEVE (src/rag/embed.py + retrieve.py)                    ║
║  nomic-embed-text via ChromaDB OllamaEmbeddingFunction                          ║
║  ChromaDB collection: heliophysics-corpus · Cosine floor 0.25                   ║
║  ── Unload nomic-embed-text (keep_alive=0) before loading phi3:mini ──          ║
╚══════════════════════════════════════════════════════════════════════════════════╝
                                    │
                                    ▼
╔══════════════════════════════════════════════════════════════════════════════════╗
║                     LLM BRIEF (src/reporting/generate_brief.py)                 ║
║  phi3:mini via Ollama · context_window=4096 · Token guard (<3500 tokens)       ║
║  Staleness guard · No-hallucination prompt · 150–250 words                      ║
║  Output: reports/brief.md                                                       ║
╚══════════════════════════════════════════════════════════════════════════════════╝
                                    │
                                    ▼
╔══════════════════════════════════════════════════════════════════════════════════╗
║                    DASHBOARD (src/reporting/build_dashboard.py)                 ║
║  6 Plotly panels · Jinja2 template · Dark NASA theme                            ║
║  Standalone static HTML · Output: dashboard/index.html                          ║
╚══════════════════════════════════════════════════════════════════════════════════╝
```

---

## 3. Project Structure

```
heliophysics-monitor/
├── README.md                    ← Setup + quick start (placeholder)
├── CLAUDE.md                    ← This file
├── config.py                    ← Single source of truth (141 lines)
├── pyproject.toml               ← Dependencies (uv-managed)
├── main.py                      ← Single entry point (Stage 6)
├── Makefile                     ← Targets: all, ingest, features, corpus, brief, dashboard, clean
├── src/
│   ├── ingest/
│   │   └── fetch_donki.py       ← DONKI API client (3 fetchers)
│   ├── features/
│   │   ├── normalise.py         ← Timestamp normalisation, event classification
│   │   └── aggregates.py        ← Daily counts, rolling stats, linkages
│   ├── rag/
│   │   ├── build_corpus.py      ← Scrape + chunk NASA pages (one-off)
│   │   ├── embed.py             ← nomic-embed-text → ChromaDB vector store
│   │   └── retrieve.py          ← Query → top-k chunks with cosine floor
│   └── reporting/
│       ├── generate_brief.py    ← phi3:mini analyst brief
│       ├── build_dashboard.py   ← Plotly → static HTML
│       └── build_report.py      ← Markdown report
├── data/
│   ├── raw/                     ← Raw DONKI JSON per endpoint per run
│   ├── processed/               ← Aggregate JSON: daily_counts, rolling_counts, severity, linkages
│   └── corpus/                  ← Scraped text files (pre-chunk)
├── docs/
│   └── corpus/                  ← Chunked JSON + ChromaDB persistence
│       └── chroma/              ← ChromaDB vector store (gitignored)
├── dashboard/
│   ├── index.html               ← Standalone static dashboard
│   └── data/                    ← Precomputed JSON for dashboard panels
├── reports/
│   ├── brief.md                 ← Latest analyst brief
│   ├── report_{date}.md         ← Timestamped reports
│   ├── latest.md                ← Symlink to latest report
│   └── pipeline_warnings.log    ← Anomalies (RAM, cosine floor, brief length)
├── notebooks/
│   ├── 00_scaffold.ipynb
│   ├── 01_ingest.ipynb
│   ├── 02_features.ipynb
│   ├── 03_rag.ipynb
│   ├── 04_brief.ipynb
│   └── 05_dashboard.ipynb
└── tests/                       ← Unit tests (future)
```

---

## 4. Key Configuration (config.py)

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `NASA_API_BASE` | `https://api.nasa.gov/DONKI` | DONKI API root — shared across all 3 endpoints |
| `NASA_API_KEY` | `DEMO_KEY` (default, from env) | Free tier; register at api.nasa.gov for 1,000 req/h if rate-limited |
| `EVENT_TYPES` | `CME`, `FLR`, `GST` | 3 event types in scope for v1; IPS, SEP, MPC, RBE, HSS out of scope |
| `BACKFILL_DAYS` | 180 | Historical window on first run; subsequent runs fetch only new events |
| `RATE_LIMIT_BUFFER` | 2 | Pause if `X-Ratelimit-Remaining` ≤ this value |
| `FLARE_CLASS_MULTIPLIER` | C=1, M=10, X=100 | Numeric severity scale from GOES X-ray class labels |
| `ROLLING_WINDOWS` | [7, 30] | Short-term and medium-term rolling sums |
| `CORPUS_URLS` | 5 URLs (4 required + 1 optional) | CCMC DONKI, NASA Heliophysics, Focus Areas, SPDF Glossary, Space Weather |
| `CHUNK_TOKEN_SIZE` | 300 | Target tokens per chunk |
| `CHUNK_OVERLAP_TOKENS` | 50 | Overlap between adjacent chunks |
| `MIN_CORPUS_PAGES` | 4 | Abort if fewer URLs succeed |
| `MIN_CHUNKS` / `MAX_CHUNKS` | 80 / 180 | Corpus size verification range |
| `CHROMA_COLLECTION_NAME` | `heliophysics-corpus` | ChromaDB collection name |
| `RETRIEVAL_TOP_K` | 3 | Chunks returned per query |
| `COSINE_FLOOR` | 0.25 | Minimum cosine similarity for relevant retrieval |
| `OLLAMA_MODEL` | `phi3:mini` | 3.8B Q4 (~2.2 GB) — LLM for brief generation |
| `OLLAMA_EMBED_MODEL` | `nomic-embed-text` | 274 MB Q4 — text embedding model |
| `OLLAMA_CONTEXT_WINDOW` | 4096 | RAM guard — prevents phi3:mini from allocating 50 GB (known SLM Deployment bug) |
| `PROMPT_TOKEN_BUDGET` | 3500 | Truncate chunks if total prompt tokens exceed this |
| `BRIEF_MIN_WORDS` / `BRIEF_MAX_WORDS` | 150 / 250 | Word count gate for analyst brief |
| `RAM_CEILING_GB` | 8.0 | Hard ceiling — pipeline aborts if exceeded |
| `RAM_ABORT_THRESHOLD_GB` | 7.5 | psutil checkpoint before each stage |
| `DATA_MAX_AGE_HOURS` | 24 | Staleness guard — warns if raw data older than this |
| `GLOBAL_SEED` | 42 | Seeds for reproducibility (only used if non-deterministic ops added later; no DGP in scope) |
| `DASHBOARD_PANELS` | 6 panels | ingestion_timestamp, event_counts_7d_30d, event_timeline, top_active_days, terminology_card, analyst_brief |
| `LOCAL_TIMEZONE` | `Asia/Kuala_Lumpur` | UTC+8 for dashboard display |

---

## 5. Key Design Decisions

1. **Data pipeline, not ML project.** No models, no training, no DGP. Every transform is deterministic. The only non-deterministic component is phi3:mini inference, which is constrained by prompt boundaries and retrieval grounding. This avoids the entire class of bugs documented in prior ML projects (feature encodes DGP, seed collision, scale_pos_weight hardcoding, etc.).

2. **Serial model execution — never load phi3:mini and nomic-embed-text simultaneously.** With RAM at ~3.2 GB idle and an 8 GB ceiling, loading both models at once would exceed the budget. The pipeline embeds the corpus with nomic-embed-text, calls `keep_alive=0` on the last embedding request to unload it, then loads phi3:mini for generation. Verified with `ollama ps` after unload.

3. **`context_window=4096` on phi3:mini (RAM guard).** Known bug from the SLM Deployment exercise: phi3:mini's default context window allocation consumed 50 GB before this fix. Explicitly setting `context_window=4096` caps RAM usage and is non-negotiable.

4. **Token guard — truncate before inference.** Before submitting the prompt to phi3:mini, count tokens in `{retrieved_chunks} + {structured_stats} + prompt_template`. If >3,500, drop lowest-cosine-similarity chunks until total <3,500. Log a warning if truncation occurred. This prevents silent context overflow.

5. **Staleness guard.** Before generating the brief, check that `data/raw/` file modification times are within the last 24 hours. If older, prepend to brief: "⚠ Data from {mtime}. Consider re-running ingestion." Never silently publish stale analysis.

6. **Cosine floor of 0.25 on retrieval.** If no chunk scores above this threshold, the answer must state "No relevant information found in the corpus" rather than confabulating. Combined with the no-hallucination prompt, this grounds the LLM in verified text.

7. **ChromaDB with OllamaEmbeddingFunction (not FAISS, not sentence-transformers).** Consistency with prior Ollama RAG Learn project. Avoids installing a separate `sentence-transformers` package. ChromaDB's built-in function integrates directly with the local nomic-embed-text model.

8. **British English throughout.** All human-facing text: code comments, docstrings, commit messages, generated prose, dashboard labels, README. Use British spellings (e.g. "normalise", "analyser", "behaviour", "acknowledgement").

9. **Static HTML dashboard (zero server dependencies).** Plotly figures are precomputed and embedded as inline JSON in a Jinja2 template. No Flask, FastAPI, or any server runtime needed. Open `dashboard/index.html` in any browser — it just works.

10. **DEMO_KEY initially; register if rate-limited.** The free NASA API tier (DEMO_KEY) allows 10 requests/hour — sufficient for the 3-endpoint pipeline. If rate-limited during development, register at api.nasa.gov for 1,000 req/h.

---

## 6. Data Schemas

### Raw DONKI JSON (from API)

#### CME (Coronal Mass Ejection)
```json
{
  "activityID": "2026-06-30T00:00:00-CME-001",
  "startTime": "2026-06-30T00:00:00Z",
  "note": "CME observed in SOHO LASCO C2/C3 imagery...",
  "cmeAnalyses": [
    {
      "isMostAccurate": true,
      "speed": 450.0,
      "type": "S",
      "latitude": 0.0,
      "longitude": 0.0,
      "halfAngle": 30.0
    }
  ],
  "linkedEvents": [
    {
      "activityID": "2026-06-29T23:59:00-FLR-001",
      "relationType": "associated_with"
    }
  ]
}
```

#### FLR (Solar Flare)
```json
{
  "flrID": "2026-06-30T00:00:00-FLR-001",
  "beginTime": "2026-06-30T00:00:00Z",
  "peakTime": "2026-06-30T00:05:00Z",
  "endTime": "2026-06-30T00:20:00Z",
  "classType": "M5.2",
  "sourceLocation": "S25E15",
  "linkedEvents": []
}
```

#### GST (Geomagnetic Storm)
```json
{
  "gstID": "2026-06-30T00:00:00-GST-001",
  "startTime": "2026-06-30T00:00:00Z",
  "allKpIndex": [
    {"observedTime": "2026-06-30T00:00:00Z", "kpIndex": 5},
    {"observedTime": "2026-06-30T03:00:00Z", "kpIndex": 6}
  ],
  "linkedEvents": [
    {
      "activityID": "2026-06-29T00:00:00-CME-001",
      "relationType": "associated_with"
    }
  ]
}
```

### Processed Aggregates

#### `daily_counts.json`
```json
{
  "dates": ["2026-06-01", "2026-06-02", ...],
  "cme_counts": [3, 1, ...],
  "flr_counts": [2, 0, ...],
  "gst_counts": [0, 1, ...]
}
```

#### `rolling_counts.json`
```json
{
  "dates": ["2026-06-01", ...],
  "cme_7d": [12, 10, ...],
  "cme_30d": [45, 44, ...],
  "flr_7d": [...],
  "flr_30d": [...],
  "gst_7d": [...],
  "gst_30d": [...]
}
```

#### `severity.json`
```json
{
  "dates": ["2026-06-01", ...],
  "max_flare_class": [5.2, null, ...],
  "max_cme_speed_kms": [450.0, 620.0, ...],
  "max_kp_index": [null, 6, ...]
}
```

#### `linkages.json`
```json
{
  "edges": [
    {"from": "FLR-2026-06-29T23:59:00-FLR-001", "to": "CME-2026-06-30T00:00:00-CME-001", "relation": "associated_with"},
    {"from": "CME-2026-06-29T00:00:00-CME-001", "to": "GST-2026-06-30T00:00:00-GST-001", "relation": "associated_with"}
  ]
}
```

---

## 7. Bugs Catalogue

### Inherited from Prior Portfolio Projects

The following bugs are documented in Allied Health Nudge, SKU Demand Forecasting, Trade Promo
Reconciliation, and RocketLab Stock Prediction. They are inherited knowledge that applies to this
project where relevant.

| # | Bug | Rule | Applied in Heliophysics? |
|---|-----|------|--------------------------|
| **1** | **Global seed collision** — `np.random.seed(42)` called globally collides with module-level seeds, causing identical random draws across components (Allied Health Bug #1, SKU Bug #2). | Always use `rng = np.random.default_rng(seed=N)` per module; never call `np.random.seed()`. | **N/A** — No DGP or random draws in this pipeline. The only source of non-determinism is phi3:mini inference, which uses Ollama's own sampling. `GLOBAL_SEED=42` defined in config.py for future use only. |
| **2** | **Feature encodes DGP directly** — Feature is a deterministic function of the same underlying variables used to generate the target label, creating circular learning (Allied Health Bug #2). | Ensure no feature is a direct algebraic combination of the same raw inputs used in label generation. | **N/A** — No DGP in this project. Every transform is deterministic feature engineering from real API data. The aggregates (daily counts, rolling sums) are pure summaries, not derived labels. |
| **3** | **Index mismatch on join/merge** — DataFrame operations produce silent NaN flags when indices don't align (Trade Promo Bug "Rules index mismatch"). | Index on key column (`claim_id` or equivalent) before merge/join operations. Always assert no NaN introduced after merge. | **⚠️ Relevant.** When joining normalised events across types (CME ↔ FLR ↔ GST), ensure indices align on `event_date`. After any merge, assert `df.isnull().sum().sum() == 0` to catch misaligned joins. |
| **4** | **String columns in numeric operations** — Categorical ID columns accidentally included in numeric operations (Trade Promo Bug "String cols in X_train"). | Filter to numeric-only columns via `select_dtypes(include='number')` before any numeric operation. | **⚠️ Relevant.** When computing rolling counts or severity aggregations, ensure only numeric columns are passed to window functions. Event IDs (`activityID`, `flrID`, `gstID`) are strings and must be excluded. |

### Potential Project-Specific Pitfalls (Preventative)

| # | Risk | Symptom | Preventative Measure |
|---|------|---------|---------------------|
| **5** | **Rate-limit exhaustion during backfill** — 3 endpoints × API calls may exceed DEMO_KEY's 10 req/h limit. | HTTP 429 responses from NASA API. | Check `X-Ratelimit-Remaining` header; if ≤2, sleep until reset. Register production API key if rate-limited during development. |
| **6** | **URL rot for corpus pages** — NASA pages restructure, breaking the scraper. | 404 or DOM changes causing empty/malformed chunks. | Log failed URLs, continue with remaining pages. Abort if fewer than 4 URLs succeed. Fallback: manually save pages and point `build_corpus.py` at local files. |
| **7** | **phi3:mini hallucinates statistics** — The model invents event counts despite no-hallucination prompt. | Brief contains numbers not present in `daily_counts.json`. | Word-count gate (150–250) + numeric verification against actual data. Retry once if <100 words; use deterministic fallback template if still short. Prompt instructs "No events recorded" for zero-count types. |
| **8** | **Zero-event GST window** — Geomagnetic storms are rare; a 180-day window may contain 0–1 GSTs. | Brief looks sparse with only CME/FLR data. | Prompt explicitly instructs phi3:mini to state "No geomagnetic storms recorded" during quiet periods. Not a bug — expected behaviour. |

---

## 8. Status Table

| Stage | Component | Status | Notes |
|-------|-----------|--------|-------|
| 0 | Scaffold & API Key | ⏳ Not started | config.py exists; CLAUDE.md being written; README.md is placeholder |
| 1 | DONKI Ingestion | ⏳ Not started | 3 fetchers (CME, FLR, GST), 180-day backfill, rate-limit handling |
| 2 | Deterministic Features | ⏳ Not started | normalise.py + aggregates.py → 4 JSON outputs |
| 3a | Corpus Build (one-off) | ⏳ Not started | 5 NASA pages, 80–180 chunks, fallback on URL failure |
| 3b | Embed & Retrieve | ⏳ Not started | ChromaDB + nomic-embed-text, cosine floor 0.25 |
| 4 | LLM Analyst Brief | ⏳ Not started | phi3:mini, token guard, staleness guard, 150–250 words |
| 5 | Static Dashboard | ⏳ Not started | 6 Plotly panels, Jinja2, dark NASA theme |
| 6 | Integration (main.py) | ⏳ Not started | Single entry point, end-to-end <180s, exit code 0 |
| 7a | GitHub Publishing | ⏳ Not started | README, gitignore, push to rush3508/heliophysics-monitor |
| 7b | Obsidian Vault | ⏳ Not started | 7 notes, YAML frontmatter, wikilinks |

---

## 9. Pipeline Stages Quick Reference

| Stage | Name | What it produces | One-line description |
|-------|------|------------------|----------------------|
| **0** | Scaffold | `config.py`, `pyproject.toml`, `CLAUDE.md`, directory tree, venv | Project skeleton, single source of truth, and dependency installation. |
| **1** | Ingestion | `data/raw/CME_*.json`, `FLR_*.json`, `GST_*.json` | Fetch 180 days of DONKI events across 3 endpoints with rate-limit handling. |
| **2** | Features | `data/processed/daily_counts.json`, `rolling_counts.json`, `severity.json`, `linkages.json` | Normalise timestamps, compute daily/rolling aggregates, extract cross-event linkages. |
| **3a** | Corpus Build | `docs/corpus/chunks.json` (80–180 chunks) | Scrape 4–5 NASA pages, strip chrome, chunk into ~300-token segments. One-off — run with `--build-corpus`. |
| **3b** | Embed & Retrieve | ChromaDB collection `heliophysics-corpus` | Embed all chunks with nomic-embed-text, persist to ChromaDB, support top-k retrieval with cosine floor. |
| **4** | LLM Brief | `reports/brief.md` (150–250 words) | Generate analyst brief via phi3:mini with token guard, staleness guard, and no-hallucination prompt. |
| **5** | Dashboard | `dashboard/index.html` (static, 6 panels) | Build Plotly figures + Jinja2 template → standalone HTML with dark NASA theme. |
| **6** | Integration | `main.py`, `Makefile` | Single entry point orchestrating stages 1–5; `python main.py` completes < 180s, exit code 0. |
| **7a** | GitHub | Public repo at `github.com/rush3508/heliophysics-monitor` | Git init, README, .gitignore, commit, push. |
| **7b** | Obsidian Vault | 7 notes at `/home/alex/obsidian_backup/heliophysics-monitor/` | Project index, architecture, ingestion, features, RAG, LLM brief, dashboard — with wikilinks. |

---

## 10. Glossary

| Term | Definition |
|------|------------|
| **CME** | Coronal Mass Ejection — a large expulsion of plasma and magnetic field from the Sun's corona into interplanetary space. |
| **FLR** | Solar Flare — a sudden, intense brightening on the Sun's surface caused by magnetic energy release. Classified by GOES X-ray flux: C (×1), M (×10), X (×100). |
| **GST** | Geomagnetic Storm — a temporary disturbance of Earth's magnetosphere caused by a solar wind shock wave, typically from a CME. Measured by Kp index (0–9). |
| **DONKI** | Space Weather Database of Notifications, Knowledge, Information — NASA's centralised space weather event database and API. |
| **Kp index** | Planetary K-index — a scale from 0–9 measuring geomagnetic storm intensity. Kp ≥5 indicates a G1 (minor) storm; Kp ≥9 indicates G5 (extreme). |
| **RAG** | Retrieval-Augmented Generation — a technique where an LLM's prompt is enriched with relevant documents retrieved from a knowledge base, grounding its responses in verified text. |
| **ChromaDB** | An open-source vector database for storing and querying text embeddings. Used here to index the heliophysics corpus for semantic retrieval. |
| **nomic-embed-text** | A 768-dimension text embedding model (274 MB, Q4) from Nomic AI, purpose-built for retrieval tasks. Runs locally via Ollama. |
| **ENLIL** | A 3D magnetohydrodynamic model that simulates solar wind and CME propagation through the heliosphere. DONKI provides ENLIL simulation results attached to CME events. |

---

## 11. Related Projects

| Project | Path | Relationship |
|---------|------|--------------|
| Allied Health Nudge | `../allied-health-nudge/` | Bugs catalogue inherited from this project's 8 documented bugs |
| SKU Demand Forecasting | `../sku-demand-forecasting/` | CLAUDE.md pattern reference; seed isolation practice |
| Trade Promo Reconciliation | `../trade-promo-reconciliation/` | Index mismatch and string-in-numeric bugs inherited |
| RocketLab Stock Prediction | `../rocketlab-stock-prediction/` | Stale checkpoint bug pattern (analogous to stale data guard) |
