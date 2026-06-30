"""
build_dashboard.py — Generate a standalone static HTML dashboard with 6 Plotly panels.

Uses a dark NASA-themed aesthetic, Jinja2 template, and precomputed JSON data
for offline reference. All 6 panels defined in config.DASHBOARD_PANELS are
rendered into a single responsive grid.

Workflow:
    1.  Load processed data (daily_counts, top_days, severity, linkages, brief)
    2.  Build each Plotly figure / text card
    3.  Render via inline Jinja2 template → dashboard/index.html
    4.  Save precomputed panel data → dashboard/data/*.json

Run:
    python -m src.reporting.build_dashboard
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone, timedelta
from pathlib import Path

import plotly.graph_objects as go
from jinja2 import Template

from src.reporting.solar_disc import build_solar_disc_figure

from config import (
    DASHBOARD_DIR,
    DASHBOARD_DATA,
    DASHBOARD_PANELS,
    DATA_RAW,
    DATA_PROCESSED,
    BRIEF_FILE,
    LOCAL_TIMEZONE,
)
from src.rag.retrieve import retrieve

logger = logging.getLogger(__name__)

# ── NASA Dark Theme Palette ─────────────────────────────────────────
NASA_BG = "#0a0a1a"
NASA_CARD_BG = "#111128"
NASA_BORDER = "#2a2a4a"
NASA_TEXT = "#e0e0e0"
NASA_TEXT_MUTED = "#8888aa"
NASA_ACCENT = "#4fc3f7"

EVENT_COLOURS = {
    "CME": "#ff6b35",
    "FLR": "#ffd700",
    "GST": "#00d4ff",
}
EVENT_TYPES = ["CME", "FLR", "GST"]

MYT_OFFSET = timedelta(hours=8)

# ── Helpers ─────────────────────────────────────────────────────────


def _load_json(path: Path) -> list | dict | None:
    """Load a JSON file, returning None if it doesn't exist or is invalid."""
    try:
        if path.exists() and path.stat().st_size > 0:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Could not load %s: %s", path, exc)
    return None


def _save_json(data, filename: str) -> None:
    """Save precomputed panel data to dashboard/data/."""
    DASHBOARD_DATA.mkdir(parents=True, exist_ok=True)
    path = DASHBOARD_DATA / filename
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)
    logger.info("Saved precomputed data → %s", path)


def _apply_dark_theme(fig: go.Figure) -> go.Figure:
    """Apply the NASA dark theme to a Plotly figure."""
    fig.update_layout(
        paper_bgcolor=NASA_BG,
        plot_bgcolor=NASA_CARD_BG,
        font=dict(color=NASA_TEXT, family="system-ui, -apple-system, sans-serif"),
        margin=dict(l=40, r=20, t=10, b=40),
        legend=dict(
            font=dict(color=NASA_TEXT),
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
        xaxis=dict(
            gridcolor="#1a1a3a",
            title=dict(font=dict(color=NASA_TEXT_MUTED)),
            tickfont=dict(color=NASA_TEXT_MUTED),
            zeroline=False,
        ),
        yaxis=dict(
            gridcolor="#1a1a3a",
            title=dict(font=dict(color=NASA_TEXT_MUTED)),
            tickfont=dict(color=NASA_TEXT_MUTED),
            zeroline=False,
        ),
        hovermode="x unified",
    )
    return fig


# ── Panel Builders ──────────────────────────────────────────────────


def build_ingestion_timestamp() -> dict:
    """Panel (a): text card showing last data fetch time (UTC + MYT).

    Derives the timestamp from the newest file mtime in DATA_RAW.
    Returns a dict with ``utc`` and ``myt`` formatted strings.
    """
    result = {"utc": "No data", "myt": "No data", "source_file": ""}

    if not DATA_RAW.is_dir():
        return result

    raw_files = sorted(DATA_RAW.glob("*"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not raw_files:
        return result

    newest = raw_files[0]
    mtime_utc = datetime.fromtimestamp(newest.stat().st_mtime, tz=timezone.utc)
    mtime_myt = mtime_utc + MYT_OFFSET  # timedelta shift

    result["utc"] = mtime_utc.strftime("%Y-%m-%d %H:%M UTC")
    result["myt"] = mtime_myt.strftime("%Y-%m-%d %H:%M MYT")
    result["source_file"] = newest.name

    _save_json(result, "ingestion_timestamp.json")
    return result


def build_event_counts_7d_30d(daily_counts: list[dict] | None) -> go.Figure | None:
    """Panel (b): grouped bar chart — 7d / 30d event counts by type.

    Computes rolling sums for the last 7 and 30 calendar days from the
    most recent date in the dataset.
    """
    if not daily_counts:
        return None

    # Sort by date, find latest
    sorted_counts = sorted(daily_counts, key=lambda d: d.get("date", ""))
    latest_date = datetime.strptime(sorted_counts[-1]["date"], "%Y-%m-%d").date()

    def _window_counts(days: int) -> dict[str, int]:
        cutoff = latest_date - timedelta(days=days - 1)
        counts = {"CME": 0, "FLR": 0, "GST": 0}
        for entry in sorted_counts:
            entry_date = datetime.strptime(entry["date"], "%Y-%m-%d").date()
            if entry_date >= cutoff:
                counts["CME"] += entry.get("cme_count", 0)
                counts["FLR"] += entry.get("flr_count", 0)
                counts["GST"] += entry.get("gst_count", 0)
        return counts

    counts_7d = _window_counts(7)
    counts_30d = _window_counts(30)

    # Save precomputed data
    _save_json({"7d": counts_7d, "30d": counts_30d}, "event_counts_7d_30d.json")

    fig = go.Figure()
    for event_type in EVENT_TYPES:
        fig.add_trace(go.Bar(
            name=event_type,
            x=["Last 7 Days", "Last 30 Days"],
            y=[counts_7d[event_type], counts_30d[event_type]],
            marker_color=EVENT_COLOURS[event_type],
            hovertemplate="%{y} events<extra>%{x}</extra>",
        ))

    fig.update_layout(
        barmode="group",
        barnorm="",
        yaxis=dict(title=dict(text="Event Count")),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
    )
    fig = _apply_dark_theme(fig)
    return fig


def build_event_timeline(daily_counts: list[dict] | None) -> go.Figure | None:
    """Panel (c): scatter plot — daily event counts over time, coloured by type.

    Three separate traces (lines) for CME, FLR, GST.
    """
    if not daily_counts:
        return None

    dates = [d["date"] for d in daily_counts]
    cme_vals = [d.get("cme_count", 0) for d in daily_counts]
    flr_vals = [d.get("flr_count", 0) for d in daily_counts]
    gst_vals = [d.get("gst_count", 0) for d in daily_counts]

    # Save precomputed data
    timeline_data = [
        {"date": d, "cme": c, "flr": f, "gst": g}
        for d, c, f, g in zip(dates, cme_vals, flr_vals, gst_vals)
    ]
    _save_json(timeline_data, "event_timeline.json")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dates, y=cme_vals,
        mode="lines",
        name="CME",
        line=dict(color=EVENT_COLOURS["CME"], width=2),
        hovertemplate="%{y} CME<extra>%{x}</extra>",
    ))
    fig.add_trace(go.Scatter(
        x=dates, y=flr_vals,
        mode="lines",
        name="FLR",
        line=dict(color=EVENT_COLOURS["FLR"], width=2),
        hovertemplate="%{y} FLR<extra>%{x}</extra>",
    ))
    fig.add_trace(go.Scatter(
        x=dates, y=gst_vals,
        mode="lines",
        name="GST",
        line=dict(color=EVENT_COLOURS["GST"], width=2),
        hovertemplate="%{y} GST<extra>%{x}</extra>",
    ))

    fig.update_layout(
        yaxis=dict(title=dict(text="Daily Event Count")),
        xaxis=dict(title=dict(text="Date")),
    )
    fig = _apply_dark_theme(fig)
    return fig


def build_top_active_days(top_days: list[dict] | None) -> go.Figure | None:
    """Panel (d): horizontal bar chart — top 5 days by total event count."""
    if not top_days:
        return None

    # Sort descending so the highest value is at the top of the horizontal bar
    sorted_days = sorted(top_days, key=lambda d: d.get("total_count", 0), reverse=True)

    # Save precomputed data
    _save_json(sorted_days, "top_active_days.json")

    # Format dates for display: "Mon 01 Feb" style
    from datetime import datetime as dt
    def _fmt_date(d):
        try:
            parsed = dt.strptime(d, "%Y-%m-%d")
            return parsed.strftime("%a %d %b")  # e.g. "Sun 01 Feb"
        except (ValueError, TypeError):
            return d

    dates = [_fmt_date(d["date"]) for d in sorted_days]
    totals = [d.get("total_count", 0) for d in sorted_days]
    cme = [d.get("cme_count", 0) for d in sorted_days]
    flr = [d.get("flr_count", 0) for d in sorted_days]
    gst = [d.get("gst_count", 0) for d in sorted_days]

    # Stacked horizontal bar to show composition
    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=dates, x=cme,
        name="CME",
        orientation="h",
        marker_color=EVENT_COLOURS["CME"],
        hovertemplate="CME: %{x}<extra>%{y}</extra>",
    ))
    fig.add_trace(go.Bar(
        y=dates, x=flr,
        name="FLR",
        orientation="h",
        marker_color=EVENT_COLOURS["FLR"],
        hovertemplate="FLR: %{x}<extra>%{y}</extra>",
    ))
    fig.add_trace(go.Bar(
        y=dates, x=gst,
        name="GST",
        orientation="h",
        marker_color=EVENT_COLOURS["GST"],
        hovertemplate="GST: %{x}<extra>%{y}</extra>",
    ))

    fig.update_layout(
        barmode="stack",
        xaxis=dict(title=dict(text="Total Event Count")),
        yaxis=dict(
            title=dict(text=""),
            autorange="reversed",  # highest at top
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
        hovermode="y unified",
        margin=dict(l=120, r=20, t=10, b=40),
        height=300,
    )
    fig = _apply_dark_theme(fig)
    # Override hovermode back — _apply_dark_theme sets "x unified" which is wrong for horizontal
    fig.update_layout(hovermode="y unified")
    return fig


def build_terminology_card() -> dict:
    """Panel (e): 'Term of the Day' — RAG-retrieved explanation.

    Queries the corpus for a broad heliophysics terminology question and
    returns the top chunk with its source label.
    """
    fallback = {
        "text": "No retrieval results available. The corpus may not have been built yet.",
        "source_label": "",
        "cosine_score": 0.0,
    }

    try:
        results = retrieve(
            "What is a coronal mass ejection CME solar flare geomagnetic storm",
            k=1,
        )
    except Exception as exc:
        logger.warning("Retrieval failed for terminology card: %s", exc)
        _save_json(fallback, "terminology_card.json")
        return fallback

    if not results:
        _save_json(fallback, "terminology_card.json")
        return fallback

    top = results[0]
    card = {
        "text": top.get("text", fallback["text"]),
        "source_label": top.get("source_label", ""),
        "cosine_score": top.get("cosine_score", 0.0),
    }
    _save_json(card, "terminology_card.json")
    return card


def build_analyst_brief() -> dict:
    """Panel (f): LLM analyst brief — render the contents of brief.md."""
    fallback = {
        "text": "No analyst brief available. Run the generate_brief pipeline stage first.",
        "word_count": 0,
    }

    if not BRIEF_FILE.exists():
        _save_json(fallback, "analyst_brief.json")
        return fallback

    try:
        text = BRIEF_FILE.read_text(encoding="utf-8").strip()
    except OSError as exc:
        logger.warning("Could not read brief file %s: %s", BRIEF_FILE, exc)
        _save_json(fallback, "analyst_brief.json")
        return fallback

    if not text:
        _save_json(fallback, "analyst_brief.json")
        return fallback

    brief = {
        "text": text,
        "word_count": len(text.split()),
    }
    _save_json(brief, "analyst_brief.json")
    return brief


# ── Jinja2 Template (inline) ────────────────────────────────────────

DASHBOARD_TEMPLATE_STR = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Heliophysics Monitor — Space Weather Dashboard</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    background: #0a0a1a;
    color: #e0e0e0;
    font-family: system-ui, -apple-system, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
    padding: 1.5rem;
    min-height: 100vh;
  }
  .dashboard-header {
    text-align: center;
    margin-bottom: 2rem;
    padding-bottom: 1rem;
    border-bottom: 1px solid #2a2a4a;
  }
  .dashboard-header h1 {
    font-size: 1.8rem;
    font-weight: 300;
    letter-spacing: 0.05em;
    color: #4fc3f7;
  }
  .dashboard-header p {
    color: #8888aa;
    font-size: 0.85rem;
    margin-top: 0.3rem;
  }

  .dashboard-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 1.25rem;
    max-width: 1600px;
    margin: 0 auto;
  }

  .panel {
    background: #111128;
    border: 1px solid #2a2a4a;
    border-radius: 8px;
    padding: 1rem;
    overflow: hidden;
  }
  .panel h2 {
    font-size: 0.95rem;
    font-weight: 600;
    color: #4fc3f7;
    margin-bottom: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.04em;
  }
  .panel-full { grid-column: 1 / -1; }
  .panel-wide { grid-column: span 2; }

  /* Text card panels */
  .text-card { line-height: 1.6; }
  .text-card .timestamp-row {
    display: flex; justify-content: space-between; align-items: center;
    padding: 0.6rem 0; border-bottom: 1px solid #1a1a3a;
  }
  .text-card .timestamp-row:last-child { border-bottom: none; }
  .text-card .label { color: #8888aa; font-size: 0.85rem; }
  .text-card .value  { color: #e0e0e0; font-family: 'SF Mono', 'Cascadia Code', 'Consolas', monospace; font-size: 0.9rem; }

  .brief-text {
    white-space: pre-wrap;
    word-wrap: break-word;
    overflow-wrap: break-word;
    font-size: 0.85rem;
    line-height: 1.65;
    color: #d0d0e0;
    max-height: 480px;
    overflow-y: auto;
  }

  .terminology-source {
    margin-top: 0.6rem;
    font-size: 0.75rem;
    color: #8888aa;
    font-style: italic;
  }
  .terminology-score {
    margin-top: 0.3rem;
    font-size: 0.75rem;
    color: #5555aa;
  }
  .no-data-msg {
    color: #5555aa;
    text-align: center;
    padding: 2rem 1rem;
    font-style: italic;
  }

  @media (max-width: 1024px) {
    .dashboard-grid { grid-template-columns: repeat(2, 1fr); }
  }
  @media (max-width: 640px) {
    .dashboard-grid { grid-template-columns: 1fr; }
    body { padding: 0.75rem; }
  }
</style>
</head>
<body>
<div class="dashboard-header">
  <h1>&#9733; Heliophysics Monitor</h1>
  <p>Space Weather Dashboard &mdash; NASA DONKI &middot; {{ generation_time }}</p>
</div>

<div class="dashboard-grid">
{% for panel in panels %}
  <div class="panel {{ panel.css_class }}">
    <h2>{{ panel.title }}</h2>
    {% if panel.type == "plotly" and panel.figure_html %}
      {{ panel.figure_html }}
    {% elif panel.type == "text" %}
      <div class="text-card">
      {% if panel.key == "ingestion" %}
        <div class="timestamp-row"><span class="label">UTC</span><span class="value">{{ panel.data.utc }}</span></div>
        <div class="timestamp-row"><span class="label">Malaysia Time (UTC+8)</span><span class="value">{{ panel.data.myt }}</span></div>
        <div class="timestamp-row"><span class="label">Source File</span><span class="value">{{ panel.data.source_file }}</span></div>
      {% elif panel.key == "terminology" %}
        {% if panel.data.text %}
          <p>{{ panel.data.text }}</p>
          {% if panel.data.source_label %}
            <div class="terminology-source">Source: {{ panel.data.source_label }}</div>
          {% endif %}
          <div class="terminology-score">Cosine similarity: {{ panel.data.cosine_score }}</div>
        {% else %}
          <p class="no-data-msg">No data.</p>
        {% endif %}
      {% elif panel.key == "brief" %}
        {% if panel.data.text %}
          <div class="brief-text">{{ panel.data.text }}</div>
          <div style="margin-top:0.5rem;font-size:0.75rem;color:#5555aa;">{{ panel.data.word_count }} words</div>
        {% else %}
          <p class="no-data-msg">No data.</p>
        {% endif %}
      {% endif %}
      </div>
    {% else %}
      <p class="no-data-msg">No data available for this panel.</p>
    {% endif %}
  </div>
{% endfor %}
</div>

<div style="text-align:center;margin-top:2rem;padding-top:1rem;border-top:1px solid #2a2a4a;font-size:0.75rem;color:#5555aa;">
  Heliophysics Monitor &mdash; Generated {{ generation_time_utc }}
</div>
</body>
</html>"""


# ── Dashboard Assembler ──────────────────────────────────────────────


def build_dashboard() -> Path:
    """Main entry point — load data, build panels, render HTML, save files."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    logger.info("Building Heliophysics Monitor dashboard…")

    # ── Ensure output directories exist ────────────────────────────
    DASHBOARD_DIR.mkdir(parents=True, exist_ok=True)
    DASHBOARD_DATA.mkdir(parents=True, exist_ok=True)

    # ── Load processed data ────────────────────────────────────────
    daily_counts = _load_json(DATA_PROCESSED / "daily_counts.json")
    top_days = _load_json(DATA_PROCESSED / "top_days.json")

    if daily_counts is not None and not isinstance(daily_counts, list):
        logger.warning("daily_counts.json is not a list; treating as empty.")
        daily_counts = None
    if top_days is not None and not isinstance(top_days, list):
        logger.warning("top_days.json is not a list; treating as empty.")
        top_days = None

    # ── Build panels ───────────────────────────────────────────────

    now_utc = datetime.now(timezone.utc)

    panels = []

    # (a) Ingestion timestamp — text card
    ingest = build_ingestion_timestamp()
    panels.append({
        "css_class": "",
        "title": "Last Data Ingestion",
        "type": "text",
        "key": "ingestion",
        "data": ingest,
        "figure_html": "",
    })

    # (b) 7d/30d event counts — grouped bar chart
    fig_b = build_event_counts_7d_30d(daily_counts)
    panels.append({
        "css_class": "",
        "title": "Event Counts — 7 & 30 Day Windows",
        "type": "plotly",
        "key": "counts",
        "data": {},
        "figure_html": fig_b.to_html(full_html=False, include_plotlyjs=False) if fig_b else "",
    })

    # (c) Event timeline — scatter / line chart
    fig_c = build_event_timeline(daily_counts)
    panels.append({
        "css_class": "",
        "title": "Event Timeline",
        "type": "plotly",
        "key": "timeline",
        "data": {},
        "figure_html": fig_c.to_html(full_html=False, include_plotlyjs=False) if fig_c else "",
    })

    # (d) Top active days — horizontal bar chart
    fig_d = build_top_active_days(top_days)
    panels.append({
        "css_class": "",
        "title": "Top 5 Most Active Days",
        "type": "plotly",
        "key": "top_days",
        "data": {},
        "figure_html": fig_d.to_html(full_html=False, include_plotlyjs=False) if fig_d else "",
    })

    # (e) Solar disc — CME origins on Stonyhurst projection (last 30 days)
    fig_e = build_solar_disc_figure(days=30)
    panels.append({
        "css_class": "panel-full",
        "title": "CME Origins — Solar Disc (last 30 days)",
        "type": "plotly",
        "key": "solar_disc",
        "data": {},
        "figure_html": fig_e.to_html(full_html=False, include_plotlyjs=False) if fig_e else "<p>No CME origin data available.</p>",
    })

    # (f) Analyst brief — rendered brief.md
    brief = build_analyst_brief()
    panels.append({
        "css_class": "",
        "title": "LLM Analyst Brief",
        "type": "text",
        "key": "brief",
        "data": brief,
        "figure_html": "",
    })

    # ── Render via Jinja2 ──────────────────────────────────────────
    template = Template(DASHBOARD_TEMPLATE_STR)
    html = template.render(
        panels=panels,
        generation_time=now_utc.strftime("%Y-%m-%d %H:%M UTC"),
        generation_time_utc=now_utc.strftime("%Y-%m-%d %H:%M UTC"),
    )

    index_path = DASHBOARD_DIR / "index.html"
    with open(index_path, "w", encoding="utf-8") as f:
        f.write(html)

    logger.info("Dashboard written to %s", index_path)
    print(f"Dashboard written to {index_path}")
    return index_path


if __name__ == "__main__":
    build_dashboard()
