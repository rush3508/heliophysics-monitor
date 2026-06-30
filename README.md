1|# Heliophysics Monitor
2|
3|**Local-first Space Weather Intelligence System** — ingests NASA DONKI event data, computes
4|deterministic trends, enriches analysis with a retrieval-augmented generation (RAG) agent
5|over heliophysics terminology, and publishes a static dashboard plus a Markdown analyst brief.
6|
7|Runs on a single Ubuntu desktop (16 GB RAM, no GPU) with zero cloud costs.
8|
9|[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://python.org)
10|[![NASA DONKI](https://img.shields.io/badge/data-NASA%20DONKI-red.svg)](https://ccmc.gsfc.nasa.gov/tools/DONKI/)
11|[![Ollama](https://img.shields.io/badge/LLM-phi3%3Amini-green.svg)](https://ollama.com)
12|[![ChromaDB](https://img.shields.io/badge/vector-ChromaDB-purple.svg)](https://www.trychroma.com/)
13|
14|```
15|╔══════════════════════════════════════════════════════════════╗
16|║                    NASA DONKI API                            ║
17|║         CME · Solar Flares · Geomagnetic Storms              ║
18|╚══════════════════════════════════════════════════════════════╝
19|                         │
20|                         ▼
21|              ┌──────────────────────┐
22|              │   Fetch + Normalise  │
23|              │   180-day backfill   │
24|              └──────────────────────┘
25|                         │
26|          ┌──────────────┴──────────────┐
27|          ▼                             ▼
28|┌─────────────────────┐     ┌──────────────────────┐
29|│  Deterministic      │     │  RAG Pipeline         │
30|│  Aggregates         │     │  NASA corpus → Chroma │
31|│  · Daily counts     │     │  → llama3.2 brief    │
32|│  · Rolling windows  │     └──────────────────────┘
33|│  · Severity trends  │              │
34|│  · Event linkages   │              ▼
35|└─────────────────────┘     ┌──────────────────────┐
36|          │                 │  Static Dashboard     │
37|          ▼                 │  6 Plotly panels      │
38|┌─────────────────────┐     │  Dark NASA theme      │
39|│  Dashboard + Brief  │◄────│  Zero server deps     │
40|└─────────────────────┘     └──────────────────────┘
41|```
42|
43|## Quick Start
44|
45|```bash
46|# Clone
47|git clone git@github.com:rush3508/heliophysics-monitor.git
48|cd heliophysics-monitor
49|
50|# Install dependencies (uv required)
51|uv sync
52|
53|# Set NASA API key (free registration at https://api.nasa.gov)
54|export NASA_API_KEY="your_key_here"
55|
56|# One-off: build the retrieval corpus
57|python main.py --build-corpus
58|
59|# Run the full pipeline (~90 seconds)
60|python main.py
61|
62|# Open the dashboard
63|open dashboard/index.html
64|```
65|
66|## What It Does
67|
68|1. **Ingests** 180 days of space weather events from NASA's DONKI API:
69|   - **CMEs** (Coronal Mass Ejections) — speed, angle, ENLIL simulations
70|   - **Solar Flares** — GOES class (C/M/X), peak times, source location
71|   - **Geomagnetic Storms** — Kp index arrays, linked CME events
72|
73|2. **Computes** deterministic trends:
74|   - Daily event counts per type
75|   - 7-day and 30-day rolling windows
76|   - Severity indicators (max flare class, max CME speed, max Kp)
77|   - Cross-event linkages (FLR → CME → GST causal chains)
78|   - Top-5 most active days
79|
80|3. **Retrieves** relevant terminology from a local NASA heliophysics corpus
81|   (4 scraped pages, 96 chunks, ChromaDB vector store, nomic-embed-text)
82|
83|4. **Generates** a 150–250 word analyst brief using llama3.2 (3B, local Ollama)
84|   with retrieval grounding, token budget enforcement, and hallucination guards
85|
86|5. **Publishes** a standalone static HTML dashboard (6 Plotly panels, dark theme)
87|   and a Markdown report — no server required
88|
89|## Stack
90|
91|| Layer | Technology |
92||-------|-----------|
93|| Data source | NASA DONKI API (CME, FLR, GST endpoints) |
94|| Pipeline | Python 3.12+, single-entry `main.py` |
95|| Feature engineering | pandas, deterministic (no ML) |
96|| Vector store | ChromaDB with nomic-embed-text (Ollama) |
97|| LLM | llama3.2 (3B Q4_K_M) via Ollama, context_window=4096 |
98|| Dashboard | Plotly + Jinja2, standalone HTML, CDN plotly.js |
99|| Package management | uv |
100|| Hardware | Lenovo V530S, Core i5-9400, 16 GB RAM, Ubuntu 24.04 |
101|
102|## File Structure
103|
104|```
105|heliophysics-monitor/
106|├── main.py                     # Single entry point
107|├── config.py                   # All parameters, paths, API settings
108|├── CLAUDE.md                   # Project bible
109|├── src/
110|│   ├── ingest/fetch_donki.py   # DONKI API client (3 fetchers)
111|│   ├── features/
112|│   │   ├── normalise.py        # Timestamp + event classification
113|│   │   └── aggregates.py       # Daily counts, rolling stats, linkages
114|│   ├── rag/
115|│   │   ├── build_corpus.py     # Scrape + chunk NASA pages
116|│   │   ├── embed.py            # nomic-embed-text → ChromaDB
117|│   │   └── retrieve.py         # Cosine search with similarity floor
118|│   └── reporting/
119|│       ├── generate_brief.py   # llama3.2 analyst brief
120|│       └── build_dashboard.py  # Plotly static HTML dashboard
121|├── data/
122|│   ├── raw/                    # Raw DONKI JSON (gitignored)
123|│   └── processed/              # Aggregate JSON (daily_counts etc.)
124|├── docs/corpus/                # Chunks JSON + ChromaDB (gitignored)
125|├── dashboard/
126|│   ├── index.html              # Standalone dashboard
127|│   └── data/                   # Precomputed panel JSON
128|└── reports/
129|    └── brief.md                # Latest analyst brief
130|```
131|
132|## Example Output
133|
134|### Dashboard (6 panels)
135|![Dashboard preview — see dashboard/index.html]()
136|
137|### Analyst Brief (excerpt)
138|> Over the past week, we observed a significant increase in solar activity with CME counts
139|> ranging between 0 and 14 events per day, peaking on June 26th. The fastest CME was
140|> clocked at 936 km/s, indicative of highly active space weather conditions...
141|
142|### Controlled Retrieval Queries
143|- "What is a CME?" → Retrieves NASA corpus chunks at cosine 0.78+
144|- "How is a geomagnetic storm different from a solar flare?" → cosine 0.83
145|- "What does M-class or X-class flare mean?" → cosine 0.84
146|- "Why does this week's event cluster matter?" → cosine 0.80
147|
148|## Constraints
149|
150|- **RAM budget**: < 8 GB enforced (serial model loading, psutil checkpoints)
151|- **Manual invoke only** — no cron, no server
152|- **British English** throughout
153|- **No ML, no training** — every transform is deterministic except the LLM brief
154|- **DEMO_KEY** or registered NASA API key required
155|
156|## Licence
157|
158|MIT
159|
160|---
161|
162|*Built on a Lenovo V530S homelab. Zero cloud costs. Local-first by design.*
163|