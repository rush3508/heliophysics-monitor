# Heliophysics Monitor

**Local-first Space Weather Intelligence System** — ingests NASA DONKI event data, computes
deterministic trends, enriches analysis with a retrieval-augmented generation (RAG) agent
over heliophysics terminology, and publishes a static dashboard plus a Markdown analyst brief.

Runs on a single Ubuntu desktop (16 GB RAM, no GPU) with zero cloud costs.

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://python.org)
[![NASA DONKI](https://img.shields.io/badge/data-NASA%20DONKI-red.svg)](https://ccmc.gsfc.nasa.gov/tools/DONKI/)
[![Ollama](https://img.shields.io/badge/LLM-phi3%3Amini-green.svg)](https://ollama.com)
[![ChromaDB](https://img.shields.io/badge/vector-ChromaDB-purple.svg)](https://www.trychroma.com/)

```
╔══════════════════════════════════════════════════════════════╗
║                    NASA DONKI API                            ║
║         CME · Solar Flares · Geomagnetic Storms              ║
╚══════════════════════════════════════════════════════════════╝
                         │
                         ▼
              ┌──────────────────────┐
              │   Fetch + Normalise  │
              │   180-day backfill   │
              └──────────────────────┘
                         │
          ┌──────────────┴──────────────┐
          ▼                             ▼
┌─────────────────────┐     ┌──────────────────────┐
│  Deterministic      │     │  RAG Pipeline         │
│  Aggregates         │     │  NASA corpus → Chroma │
│  · Daily counts     │     │  → phi3:mini brief    │
│  · Rolling windows  │     └──────────────────────┘
│  · Severity trends  │              │
│  · Event linkages   │              ▼
└─────────────────────┘     ┌──────────────────────┐
          │                 │  Static Dashboard     │
          ▼                 │  6 Plotly panels      │
┌─────────────────────┐     │  Dark NASA theme      │
│  Dashboard + Brief  │◄────│  Zero server deps     │
└─────────────────────┘     └──────────────────────┘
```

## Quick Start

```bash
# Clone
git clone git@github.com:rush3508/heliophysics-monitor.git
cd heliophysics-monitor

# Install dependencies (uv required)
uv sync

# Set NASA API key (free registration at https://api.nasa.gov)
export NASA_API_KEY="your_key_here"

# One-off: build the retrieval corpus
python main.py --build-corpus

# Run the full pipeline (~90 seconds)
python main.py

# Open the dashboard
open dashboard/index.html
```

## What It Does

1. **Ingests** 180 days of space weather events from NASA's DONKI API:
   - **CMEs** (Coronal Mass Ejections) — speed, angle, ENLIL simulations
   - **Solar Flares** — GOES class (C/M/X), peak times, source location
   - **Geomagnetic Storms** — Kp index arrays, linked CME events

2. **Computes** deterministic trends:
   - Daily event counts per type
   - 7-day and 30-day rolling windows
   - Severity indicators (max flare class, max CME speed, max Kp)
   - Cross-event linkages (FLR → CME → GST causal chains)
   - Top-5 most active days

3. **Retrieves** relevant terminology from a local NASA heliophysics corpus
   (4 scraped pages, 96 chunks, ChromaDB vector store, nomic-embed-text)

4. **Generates** a 150–250 word analyst brief using phi3:mini (3.8B, local Ollama)
   with retrieval grounding, token budget enforcement, and hallucination guards

5. **Publishes** a standalone static HTML dashboard (6 Plotly panels, dark theme)
   and a Markdown report — no server required

## Stack

| Layer | Technology |
|-------|-----------|
| Data source | NASA DONKI API (CME, FLR, GST endpoints) |
| Pipeline | Python 3.12+, single-entry `main.py` |
| Feature engineering | pandas, deterministic (no ML) |
| Vector store | ChromaDB with nomic-embed-text (Ollama) |
| LLM | phi3:mini (3.8B Q4) via Ollama, context_window=4096 |
| Dashboard | Plotly + Jinja2, standalone HTML, CDN plotly.js |
| Package management | uv |
| Hardware | Lenovo V530S, Core i5-9400, 16 GB RAM, Ubuntu 24.04 |

## File Structure

```
heliophysics-monitor/
├── main.py                     # Single entry point
├── config.py                   # All parameters, paths, API settings
├── CLAUDE.md                   # Project bible
├── src/
│   ├── ingest/fetch_donki.py   # DONKI API client (3 fetchers)
│   ├── features/
│   │   ├── normalise.py        # Timestamp + event classification
│   │   └── aggregates.py       # Daily counts, rolling stats, linkages
│   ├── rag/
│   │   ├── build_corpus.py     # Scrape + chunk NASA pages
│   │   ├── embed.py            # nomic-embed-text → ChromaDB
│   │   └── retrieve.py         # Cosine search with similarity floor
│   └── reporting/
│       ├── generate_brief.py   # phi3:mini analyst brief
│       └── build_dashboard.py  # Plotly static HTML dashboard
├── data/
│   ├── raw/                    # Raw DONKI JSON (gitignored)
│   └── processed/              # Aggregate JSON (daily_counts etc.)
├── docs/corpus/                # Chunks JSON + ChromaDB (gitignored)
├── dashboard/
│   ├── index.html              # Standalone dashboard
│   └── data/                   # Precomputed panel JSON
└── reports/
    └── brief.md                # Latest analyst brief
```

## Example Output

### Dashboard (6 panels)
![Dashboard preview — see dashboard/index.html]()

### Analyst Brief (excerpt)
> Over the past week, we observed a significant increase in solar activity with CME counts
> ranging between 0 and 14 events per day, peaking on June 26th. The fastest CME was
> clocked at 936 km/s, indicative of highly active space weather conditions...

### Controlled Retrieval Queries
- "What is a CME?" → Retrieves NASA corpus chunks at cosine 0.78+
- "How is a geomagnetic storm different from a solar flare?" → cosine 0.83
- "What does M-class or X-class flare mean?" → cosine 0.84
- "Why does this week's event cluster matter?" → cosine 0.80

## Constraints

- **RAM budget**: < 8 GB enforced (serial model loading, psutil checkpoints)
- **Manual invoke only** — no cron, no server
- **British English** throughout
- **No ML, no training** — every transform is deterministic except the LLM brief
- **DEMO_KEY** or registered NASA API key required

## Licence

MIT

---

*Built on a Lenovo V530S homelab. Zero cloud costs. Local-first by design.*
