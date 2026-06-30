1|# CLAUDE.md — Heliophysics Monitor
2|
3|> **Project bible — read this entire file before touching anything.**
4|> **PRD:** `v530s/Mission-Control/prd-drafts/heliophysics-monitor.md`
5|> **Status:** ⏳ Stage 0 in progress (scaffold)
6|> **Last updated:** 2026-06-30
7|
8|---
9|
10|## 1. What This Project Is and Why It Exists
11|
12|This is a **data pipeline portfolio project** demonstrating a local-first Space Weather Intelligence
13|System that ingests NASA DONKI (Space Weather Database of Notifications, Knowledge, Information)
14|event data, computes deterministic trends, enriches analysis with a retrieval-augmented generation
15|(RAG) agent over heliophysics terminology, and publishes a static HTML dashboard plus a Markdown
16|analyst brief — all running on the V530s homelab with zero cloud costs.
17|
18|**What it demonstrates:**
19|- API engineering against a real government data source (NASA DONKI, 3 endpoints)
20|- Deterministic time-series feature engineering (daily/rolling counts, severity indicators, cross-event linkages)
21|- RAG pipeline: web scraping → chunking → embedding (nomic-embed-text) → ChromaDB vector storage → cosine retrieval with LLM grounding
22|- Constrained LLM inference with llama3.2 (3B Q4_K_M, `context_window=4096`, serial model execution)
23|- Static dashboard delivery (Plotly + Jinja2, zero server dependencies)
24|- RAM budget discipline (< 8 GB), serial model loading, token and staleness guards
25|- Manual-invoke, spec-driven pipeline with 9 stages (Stages 0–7b)
26|
27|**This is NOT an ML project.** There are no models, no training loops, no data-generating process
28|(DGP), and no stochastic target variables. Every transform is deterministic. The only non-deterministic
29|component is llama3.2 inference, which is constrained by strict prompt boundaries and retrieval grounding.
30|
31|---
32|
33|## 2. Architecture
34|
35|```
36|╔══════════════════════════════════════════════════════════════════════════════════╗
37|║                            DONKI API (NASA)                                     ║
38|║              https://api.nasa.gov/DONKI/{CME,FLR,GST}                           ║
39|╚══════════════════════════════════════════════════════════════════════════════════╝
40|                                    │
41|                                    ▼
42|╔══════════════════════════════════════════════════════════════════════════════════╗
43|║                         FETCH LAYER (src/ingest/fetch_donki.py)                 ║
44|║  3 endpoint fetchers (CME, FLR, GST) · 180-day backfill · rate-limit handling   ║
45|║  Output: data/raw/{endpoint}_{date}.json                                        ║
46|╚══════════════════════════════════════════════════════════════════════════════════╝
47|                                    │
48|                                    ▼
49|╔══════════════════════════════════════════════════════════════════════════════════╗
50|║                      NORMALISE LAYER (src/features/normalise.py)                ║
51|║  Parse timestamps · Extract event types · Flare severity (C/M/X→numeric)        ║
52|║  CME speed (mostAccurate analysis) · Kp index for GSTs · Derive event_date      ║
53|╚══════════════════════════════════════════════════════════════════════════════════╝
54|                                    │
55|                                    ▼
56|╔══════════════════════════════════════════════════════════════════════════════════╗
57|║                     AGGREGATE LAYER (src/features/aggregates.py)                ║
58|║  Daily counts · Rolling 7d/30d · Severity indicators · Top active days          ║
59|║  Cross-event linkages (linkedEvents → edges)                                    ║
60|║  Output: data/processed/{daily_counts,rolling_counts,severity,linkages}.json     ║
61|╚══════════════════════════════════════════════════════════════════════════════════╝
62|                                    │
63|                    ┌───────────────┴───────────────┐
64|                    ▼                               ▼
65|╔════════════════════════════════╗   ╔═══════════════════════════════════════════╗
66|║  CORPUS BUILD (one-off)       ║   ║          DATA FLOW (per run)              ║
67|║  src/rag/build_corpus.py      ║   ║                                           ║
68|║  5 NASA pages scraped         ║   ║  Aggregates → Dashboard → index.html      ║
69|║  → chunks.json (80–180)      ║   ║  Aggregates + RAG → LLM → brief.md        ║
70|╚════════════════════════════════╝   ║                                           ║
71|              │                      ╚═══════════════════════════════════════════╝
72|              ▼
73|╔══════════════════════════════════════════════════════════════════════════════════╗
74|║            EMBED + RETRIEVE (src/rag/embed.py + retrieve.py)                    ║
75|║  nomic-embed-text via ChromaDB OllamaEmbeddingFunction                          ║
76|║  ChromaDB collection: heliophysics-corpus · Cosine floor 0.25                   ║
77|║  ── Unload nomic-embed-text (keep_alive=0) before loading llama3.2 ──          ║
78|╚══════════════════════════════════════════════════════════════════════════════════╝
79|                                    │
80|                                    ▼
81|╔══════════════════════════════════════════════════════════════════════════════════╗
82|║                     LLM BRIEF (src/reporting/generate_brief.py)                 ║
83|║  llama3.2 via Ollama · context_window=4096 · Token guard (<3500 tokens)       ║
84|║  Staleness guard · No-hallucination prompt · 150–250 words                      ║
85|║  Output: reports/brief.md                                                       ║
86|╚══════════════════════════════════════════════════════════════════════════════════╝
87|                                    │
88|                                    ▼
89|╔══════════════════════════════════════════════════════════════════════════════════╗
90|║                    DASHBOARD (src/reporting/build_dashboard.py)                 ║
91|║  6 Plotly panels · Jinja2 template · Dark NASA theme                            ║
92|║  Standalone static HTML · Output: dashboard/index.html                          ║
93|╚══════════════════════════════════════════════════════════════════════════════════╝
94|```
95|
96|---
97|
98|## 3. Project Structure
99|
100|```
101|heliophysics-monitor/
102|├── README.md                    ← Setup + quick start (placeholder)
103|├── CLAUDE.md                    ← This file
104|├── config.py                    ← Single source of truth (141 lines)
105|├── pyproject.toml               ← Dependencies (uv-managed)
106|├── main.py                      ← Single entry point (Stage 6)
107|├── Makefile                     ← Targets: all, ingest, features, corpus, brief, dashboard, clean
108|├── src/
109|│   ├── ingest/
110|│   │   └── fetch_donki.py       ← DONKI API client (3 fetchers)
111|│   ├── features/
112|│   │   ├── normalise.py         ← Timestamp normalisation, event classification
113|│   │   └── aggregates.py        ← Daily counts, rolling stats, linkages
114|│   ├── rag/
115|│   │   ├── build_corpus.py      ← Scrape + chunk NASA pages (one-off)
116|│   │   ├── embed.py             ← nomic-embed-text → ChromaDB vector store
117|│   │   └── retrieve.py          ← Query → top-k chunks with cosine floor
118|│   └── reporting/
119|│       ├── generate_brief.py    ← llama3.2 analyst brief
120|│       ├── build_dashboard.py   ← Plotly → static HTML
121|│       └── build_report.py      ← Markdown report
122|├── data/
123|│   ├── raw/                     ← Raw DONKI JSON per endpoint per run
124|│   ├── processed/               ← Aggregate JSON: daily_counts, rolling_counts, severity, linkages
125|│   └── corpus/                  ← Scraped text files (pre-chunk)
126|├── docs/
127|│   └── corpus/                  ← Chunked JSON + ChromaDB persistence
128|│       └── chroma/              ← ChromaDB vector store (gitignored)
129|├── dashboard/
130|│   ├── index.html               ← Standalone static dashboard
131|│   └── data/                    ← Precomputed JSON for dashboard panels
132|├── reports/
133|│   ├── brief.md                 ← Latest analyst brief
134|│   ├── report_{date}.md         ← Timestamped reports
135|│   ├── latest.md                ← Symlink to latest report
136|│   └── pipeline_warnings.log    ← Anomalies (RAM, cosine floor, brief length)
137|├── notebooks/
138|│   ├── 00_scaffold.ipynb
139|│   ├── 01_ingest.ipynb
140|│   ├── 02_features.ipynb
141|│   ├── 03_rag.ipynb
142|│   ├── 04_brief.ipynb
143|│   └── 05_dashboard.ipynb
144|└── tests/                       ← Unit tests (future)
145|```
146|
147|---
148|
149|## 4. Key Configuration (config.py)
150|
151|| Parameter | Value | Rationale |
152||-----------|-------|-----------|
153|| `NASA_API_BASE` | `https://api.nasa.gov/DONKI` | DONKI API root — shared across all 3 endpoints |
154|| `NASA_API_KEY` | `DEMO_KEY` (default, from env) | Free tier; register at api.nasa.gov for 1,000 req/h if rate-limited |
155|| `EVENT_TYPES` | `CME`, `FLR`, `GST` | 3 event types in scope for v1; IPS, SEP, MPC, RBE, HSS out of scope |
156|| `BACKFILL_DAYS` | 180 | Historical window on first run; subsequent runs fetch only new events |
157|| `RATE_LIMIT_BUFFER` | 2 | Pause if `X-Ratelimit-Remaining` ≤ this value |
158|| `FLARE_CLASS_MULTIPLIER` | C=1, M=10, X=100 | Numeric severity scale from GOES X-ray class labels |
159|| `ROLLING_WINDOWS` | [7, 30] | Short-term and medium-term rolling sums |
160|| `CORPUS_URLS` | 5 URLs (4 required + 1 optional) | CCMC DONKI, NASA Heliophysics, Focus Areas, SPDF Glossary, Space Weather |
161|| `CHUNK_TOKEN_SIZE` | 300 | Target tokens per chunk |
162|| `CHUNK_OVERLAP_TOKENS` | 50 | Overlap between adjacent chunks |
163|| `MIN_CORPUS_PAGES` | 4 | Abort if fewer URLs succeed |
164|| `MIN_CHUNKS` / `MAX_CHUNKS` | 80 / 180 | Corpus size verification range |
165|| `CHROMA_COLLECTION_NAME` | `heliophysics-corpus` | ChromaDB collection name |
166|| `RETRIEVAL_TOP_K` | 3 | Chunks returned per query |
167|| `COSINE_FLOOR` | 0.25 | Minimum cosine similarity for relevant retrieval |
168|| `OLLAMA_MODEL` | `llama3.2` | 3B Q4_K_M (~2.0 GB) — LLM for brief generation |
169|| `OLLAMA_EMBED_MODEL` | `nomic-embed-text` | 274 MB Q4 — text embedding model |
170|| `OLLAMA_CONTEXT_WINDOW` | 4096 | RAM guard — prevents llama3.2 from allocating excessive RAM (known SLM Deployment bug with phi3:mini) |
171|| `PROMPT_TOKEN_BUDGET` | 3500 | Truncate chunks if total prompt tokens exceed this |
172|| `BRIEF_MIN_WORDS` / `BRIEF_MAX_WORDS` | 150 / 250 | Word count gate for analyst brief |
173|| `RAM_CEILING_GB` | 8.0 | Hard ceiling — pipeline aborts if exceeded |
174|| `RAM_ABORT_THRESHOLD_GB` | 7.5 | psutil checkpoint before each stage |
175|| `DATA_MAX_AGE_HOURS` | 24 | Staleness guard — warns if raw data older than this |
176|| `GLOBAL_SEED` | 42 | Seeds for reproducibility (only used if non-deterministic ops added later; no DGP in scope) |
177|| `DASHBOARD_PANELS` | 6 panels | ingestion_timestamp, event_counts_7d_30d, event_timeline, top_active_days, terminology_card, analyst_brief |
178|| `LOCAL_TIMEZONE` | `Asia/Kuala_Lumpur` | UTC+8 for dashboard display |
179|
180|---
181|
182|## 5. Key Design Decisions
183|
184|1. **Data pipeline, not ML project.** No models, no training, no DGP. Every transform is deterministic. The only non-deterministic component is llama3.2 inference, which is constrained by prompt boundaries and retrieval grounding. This avoids the entire class of bugs documented in prior ML projects (feature encodes DGP, seed collision, scale_pos_weight hardcoding, etc.).
185|
186|2. **Serial model execution — never load llama3.2 and nomic-embed-text simultaneously.** With RAM at ~3.2 GB idle and an 8 GB ceiling, loading both models at once would exceed the budget. The pipeline embeds the corpus with nomic-embed-text, calls `keep_alive=0` on the last embedding request to unload it, then loads llama3.2 for generation. Verified with `ollama ps` after unload.
187|
188|3. **`context_window=4096` on llama3.2 (RAM guard).** Known bug from the SLM Deployment exercise: llama3.2's default context window allocation consumed 50 GB before this fix. Explicitly setting `context_window=4096` caps RAM usage and is non-negotiable.
189|
190|4. **Token guard — truncate before inference.** Before submitting the prompt to llama3.2, count tokens in `{retrieved_chunks} + {structured_stats} + prompt_template`. If >3,500, drop lowest-cosine-similarity chunks until total <3,500. Log a warning if truncation occurred. This prevents silent context overflow.
191|
192|5. **Staleness guard.** Before generating the brief, check that `data/raw/` file modification times are within the last 24 hours. If older, prepend to brief: "⚠ Data from {mtime}. Consider re-running ingestion." Never silently publish stale analysis.
193|
194|6. **Cosine floor of 0.25 on retrieval.** If no chunk scores above this threshold, the answer must state "No relevant information found in the corpus" rather than confabulating. Combined with the no-hallucination prompt, this grounds the LLM in verified text.
195|
196|7. **ChromaDB with OllamaEmbeddingFunction (not FAISS, not sentence-transformers).** Consistency with prior Ollama RAG Learn project. Avoids installing a separate `sentence-transformers` package. ChromaDB's built-in function integrates directly with the local nomic-embed-text model.
197|
198|8. **British English throughout.** All human-facing text: code comments, docstrings, commit messages, generated prose, dashboard labels, README. Use British spellings (e.g. "normalise", "analyser", "behaviour", "acknowledgement").
199|
200|9. **Static HTML dashboard (zero server dependencies).** Plotly figures are precomputed and embedded as inline JSON in a Jinja2 template. No Flask, FastAPI, or any server runtime needed. Open `dashboard/index.html` in any browser — it just works.
201|
202|10. **DEMO_KEY initially; register if rate-limited.** The free NASA API tier (DEMO_KEY) allows 10 requests/hour — sufficient for the 3-endpoint pipeline. If rate-limited during development, register at api.nasa.gov for 1,000 req/h.
203|
204|---
205|
206|## 6. Data Schemas
207|
208|### Raw DONKI JSON (from API)
209|
210|#### CME (Coronal Mass Ejection)
211|```json
212|{
213|  "activityID": "2026-06-30T00:00:00-CME-001",
214|  "startTime": "2026-06-30T00:00:00Z",
215|  "note": "CME observed in SOHO LASCO C2/C3 imagery...",
216|  "cmeAnalyses": [
217|    {
218|      "isMostAccurate": true,
219|      "speed": 450.0,
220|      "type": "S",
221|      "latitude": 0.0,
222|      "longitude": 0.0,
223|      "halfAngle": 30.0
224|    }
225|  ],
226|  "linkedEvents": [
227|    {
228|      "activityID": "2026-06-29T23:59:00-FLR-001",
229|      "relationType": "associated_with"
230|    }
231|  ]
232|}
233|```
234|
235|#### FLR (Solar Flare)
236|```json
237|{
238|  "flrID": "2026-06-30T00:00:00-FLR-001",
239|  "beginTime": "2026-06-30T00:00:00Z",
240|  "peakTime": "2026-06-30T00:05:00Z",
241|  "endTime": "2026-06-30T00:20:00Z",
242|  "classType": "M5.2",
243|  "sourceLocation": "S25E15",
244|  "linkedEvents": []
245|}
246|```
247|
248|#### GST (Geomagnetic Storm)
249|```json
250|{
251|  "gstID": "2026-06-30T00:00:00-GST-001",
252|  "startTime": "2026-06-30T00:00:00Z",
253|  "allKpIndex": [
254|    {"observedTime": "2026-06-30T00:00:00Z", "kpIndex": 5},
255|    {"observedTime": "2026-06-30T03:00:00Z", "kpIndex": 6}
256|  ],
257|  "linkedEvents": [
258|    {
259|      "activityID": "2026-06-29T00:00:00-CME-001",
260|      "relationType": "associated_with"
261|    }
262|  ]
263|}
264|```
265|
266|### Processed Aggregates
267|
268|#### `daily_counts.json`
269|```json
270|{
271|  "dates": ["2026-06-01", "2026-06-02", ...],
272|  "cme_counts": [3, 1, ...],
273|  "flr_counts": [2, 0, ...],
274|  "gst_counts": [0, 1, ...]
275|}
276|```
277|
278|#### `rolling_counts.json`
279|```json
280|{
281|  "dates": ["2026-06-01", ...],
282|  "cme_7d": [12, 10, ...],
283|  "cme_30d": [45, 44, ...],
284|  "flr_7d": [...],
285|  "flr_30d": [...],
286|  "gst_7d": [...],
287|  "gst_30d": [...]
288|}
289|```
290|
291|#### `severity.json`
292|```json
293|{
294|  "dates": ["2026-06-01", ...],
295|  "max_flare_class": [5.2, null, ...],
296|  "max_cme_speed_kms": [450.0, 620.0, ...],
297|  "max_kp_index": [null, 6, ...]
298|}
299|```
300|
301|#### `linkages.json`
302|```json
303|{
304|  "edges": [
305|    {"from": "FLR-2026-06-29T23:59:00-FLR-001", "to": "CME-2026-06-30T00:00:00-CME-001", "relation": "associated_with"},
306|    {"from": "CME-2026-06-29T00:00:00-CME-001", "to": "GST-2026-06-30T00:00:00-GST-001", "relation": "associated_with"}
307|  ]
308|}
309|```
310|
311|---
312|
313|## 7. Bugs Catalogue
314|
315|### Inherited from Prior Portfolio Projects
316|
317|The following bugs are documented in Allied Health Nudge, SKU Demand Forecasting, Trade Promo
318|Reconciliation, and RocketLab Stock Prediction. They are inherited knowledge that applies to this
319|project where relevant.
320|
321|| # | Bug | Rule | Applied in Heliophysics? |
322||---|-----|------|--------------------------|
323|| **1** | **Global seed collision** — `np.random.seed(42)` called globally collides with module-level seeds, causing identical random draws across components (Allied Health Bug #1, SKU Bug #2). | Always use `rng = np.random.default_rng(seed=N)` per module; never call `np.random.seed()`. | **N/A** — No DGP or random draws in this pipeline. The only source of non-determinism is llama3.2 inference, which uses Ollama's own sampling. `GLOBAL_SEED=42` defined in config.py for future use only. |
324|| **2** | **Feature encodes DGP directly** — Feature is a deterministic function of the same underlying variables used to generate the target label, creating circular learning (Allied Health Bug #2). | Ensure no feature is a direct algebraic combination of the same raw inputs used in label generation. | **N/A** — No DGP in this project. Every transform is deterministic feature engineering from real API data. The aggregates (daily counts, rolling sums) are pure summaries, not derived labels. |
325|| **3** | **Index mismatch on join/merge** — DataFrame operations produce silent NaN flags when indices don't align (Trade Promo Bug "Rules index mismatch"). | Index on key column (`claim_id` or equivalent) before merge/join operations. Always assert no NaN introduced after merge. | **⚠️ Relevant.** When joining normalised events across types (CME ↔ FLR ↔ GST), ensure indices align on `event_date`. After any merge, assert `df.isnull().sum().sum() == 0` to catch misaligned joins. |
326|| **4** | **String columns in numeric operations** — Categorical ID columns accidentally included in numeric operations (Trade Promo Bug "String cols in X_train"). | Filter to numeric-only columns via `select_dtypes(include='number')` before any numeric operation. | **⚠️ Relevant.** When computing rolling counts or severity aggregations, ensure only numeric columns are passed to window functions. Event IDs (`activityID`, `flrID`, `gstID`) are strings and must be excluded. |
327|
328|### Potential Project-Specific Pitfalls (Preventative)
329|
330|| # | Risk | Symptom | Preventative Measure |
331||---|------|---------|---------------------|
332|| **5** | **Rate-limit exhaustion during backfill** — 3 endpoints × API calls may exceed DEMO_KEY's 10 req/h limit. | HTTP 429 responses from NASA API. | Check `X-Ratelimit-Remaining` header; if ≤2, sleep until reset. Register production API key if rate-limited during development. |
333|| **6** | **URL rot for corpus pages** — NASA pages restructure, breaking the scraper. | 404 or DOM changes causing empty/malformed chunks. | Log failed URLs, continue with remaining pages. Abort if fewer than 4 URLs succeed. Fallback: manually save pages and point `build_corpus.py` at local files. |
334|| **7** | **llama3.2 hallucinates statistics** — The model invents event counts despite no-hallucination prompt. | Brief contains numbers not present in `daily_counts.json`. | Word-count gate (150–250) + numeric verification against actual data. Retry once if <100 words; use deterministic fallback template if still short. Prompt instructs "No events recorded" for zero-count types. |
335|| **8** | **Zero-event GST window** — Geomagnetic storms are rare; a 180-day window may contain 0–1 GSTs. | Brief looks sparse with only CME/FLR data. | Prompt explicitly instructs llama3.2 to state "No geomagnetic storms recorded" during quiet periods. Not a bug — expected behaviour. |
336|
337|---
338|
339|## 8. Status Table
340|
341|| Stage | Component | Status | Notes |
342||-------|-----------|--------|-------|
343|| 0 | Scaffold & API Key | ⏳ Not started | config.py exists; CLAUDE.md being written; README.md is placeholder |
344|| 1 | DONKI Ingestion | ⏳ Not started | 3 fetchers (CME, FLR, GST), 180-day backfill, rate-limit handling |
345|| 2 | Deterministic Features | ⏳ Not started | normalise.py + aggregates.py → 4 JSON outputs |
346|| 3a | Corpus Build (one-off) | ⏳ Not started | 5 NASA pages, 80–180 chunks, fallback on URL failure |
347|| 3b | Embed & Retrieve | ⏳ Not started | ChromaDB + nomic-embed-text, cosine floor 0.25 |
348|| 4 | LLM Analyst Brief | ⏳ Not started | llama3.2, token guard, staleness guard, 150–250 words |
349|| 5 | Static Dashboard | ⏳ Not started | 6 Plotly panels, Jinja2, dark NASA theme |
350|| 6 | Integration (main.py) | ⏳ Not started | Single entry point, end-to-end <180s, exit code 0 |
351|| 7a | GitHub Publishing | ⏳ Not started | README, gitignore, push to rush3508/heliophysics-monitor |
352|| 7b | Obsidian Vault | ⏳ Not started | 7 notes, YAML frontmatter, wikilinks |
353|
354|---
355|
356|## 9. Pipeline Stages Quick Reference
357|
358|| Stage | Name | What it produces | One-line description |
359||-------|------|------------------|----------------------|
360|| **0** | Scaffold | `config.py`, `pyproject.toml`, `CLAUDE.md`, directory tree, venv | Project skeleton, single source of truth, and dependency installation. |
361|| **1** | Ingestion | `data/raw/CME_*.json`, `FLR_*.json`, `GST_*.json` | Fetch 180 days of DONKI events across 3 endpoints with rate-limit handling. |
362|| **2** | Features | `data/processed/daily_counts.json`, `rolling_counts.json`, `severity.json`, `linkages.json` | Normalise timestamps, compute daily/rolling aggregates, extract cross-event linkages. |
363|| **3a** | Corpus Build | `docs/corpus/chunks.json` (80–180 chunks) | Scrape 4–5 NASA pages, strip chrome, chunk into ~300-token segments. One-off — run with `--build-corpus`. |
364|| **3b** | Embed & Retrieve | ChromaDB collection `heliophysics-corpus` | Embed all chunks with nomic-embed-text, persist to ChromaDB, support top-k retrieval with cosine floor. |
365|| **4** | LLM Brief | `reports/brief.md` (150–250 words) | Generate analyst brief via llama3.2 with token guard, staleness guard, and no-hallucination prompt. |
366|| **5** | Dashboard | `dashboard/index.html` (static, 6 panels) | Build Plotly figures + Jinja2 template → standalone HTML with dark NASA theme. |
367|| **6** | Integration | `main.py`, `Makefile` | Single entry point orchestrating stages 1–5; `python main.py` completes < 180s, exit code 0. |
368|| **7a** | GitHub | Public repo at `github.com/rush3508/heliophysics-monitor` | Git init, README, .gitignore, commit, push. |
369|| **7b** | Obsidian Vault | 7 notes at `/home/alex/obsidian_backup/heliophysics-monitor/` | Project index, architecture, ingestion, features, RAG, LLM brief, dashboard — with wikilinks. |
370|
371|---
372|
373|## 10. Glossary
374|
375|| Term | Definition |
376||------|------------|
377|| **CME** | Coronal Mass Ejection — a large expulsion of plasma and magnetic field from the Sun's corona into interplanetary space. |
378|| **FLR** | Solar Flare — a sudden, intense brightening on the Sun's surface caused by magnetic energy release. Classified by GOES X-ray flux: C (×1), M (×10), X (×100). |
379|| **GST** | Geomagnetic Storm — a temporary disturbance of Earth's magnetosphere caused by a solar wind shock wave, typically from a CME. Measured by Kp index (0–9). |
380|| **DONKI** | Space Weather Database of Notifications, Knowledge, Information — NASA's centralised space weather event database and API. |
381|| **Kp index** | Planetary K-index — a scale from 0–9 measuring geomagnetic storm intensity. Kp ≥5 indicates a G1 (minor) storm; Kp ≥9 indicates G5 (extreme). |
382|| **RAG** | Retrieval-Augmented Generation — a technique where an LLM's prompt is enriched with relevant documents retrieved from a knowledge base, grounding its responses in verified text. |
383|| **ChromaDB** | An open-source vector database for storing and querying text embeddings. Used here to index the heliophysics corpus for semantic retrieval. |
384|| **nomic-embed-text** | A 768-dimension text embedding model (274 MB, Q4) from Nomic AI, purpose-built for retrieval tasks. Runs locally via Ollama. |
385|| **ENLIL** | A 3D magnetohydrodynamic model that simulates solar wind and CME propagation through the heliosphere. DONKI provides ENLIL simulation results attached to CME events. |
386|
387|---
388|
389|## 11. Related Projects
390|
391|| Project | Path | Relationship |
392||---------|------|--------------|
393|| Allied Health Nudge | `../allied-health-nudge/` | Bugs catalogue inherited from this project's 8 documented bugs |
394|| SKU Demand Forecasting | `../sku-demand-forecasting/` | CLAUDE.md pattern reference; seed isolation practice |
395|| Trade Promo Reconciliation | `../trade-promo-reconciliation/` | Index mismatch and string-in-numeric bugs inherited |
396|| RocketLab Stock Prediction | `../rocketlab-stock-prediction/` | Stale checkpoint bug pattern (analogous to stale data guard) |
397|