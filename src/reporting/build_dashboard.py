"""
build_dashboard.py — Generate a standalone static HTML dashboard with 6 D3.js panels.

Uses a dark NASA-themed aesthetic, Jinja2 template, and precomputed JSON data
for offline reference. All visualisations are rendered via D3.js v7 (no Plotly).

Panels:
    (a) Ingestion Timestamp — text card
    (b) Event Counts — 7 & 30 Day Windows — D3 grouped bar chart
    (c) Event Timeline — D3 line chart
    (d) Top 5 Most Active Days — D3 horizontal stacked bar chart
    (e) CME Origins — Solar Disc — D3 SVG custom visualisation
    (f) LLM Analyst Brief — text card

Workflow:
    1.  Load processed data (daily_counts, top_days, severity, linkages, brief)
    2.  Build each panel's JSON payload
    3.  Render via inline Jinja2 template → dashboard/index.html
    4.  Save precomputed panel data → dashboard/data/*.json

Run:
    python -m src.reporting.build_dashboard
"""

from __future__ import annotations

import json
import logging
import math
from datetime import datetime, timezone, timedelta
from pathlib import Path

from jinja2 import Template

from config import (
    DASHBOARD_DIR,
    DASHBOARD_DATA,
    DATA_RAW,
    DATA_PROCESSED,
    BRIEF_FILE,
)
logger = logging.getLogger(__name__)

# ── Colour Palette (consistent across all panels) ──────────────────
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


def _parse_stonyhurst(loc: str) -> tuple[float, float] | None:
    """
    Parse a Stonyhurst coordinate string like "N25E35" -> (lat, lon).
    Returns (latitude, longitude) in degrees, or None if unparseable.
    """
    if not loc or not isinstance(loc, str) or not loc.strip():
        return None
    loc = loc.strip().upper()
    try:
        if loc[0] not in "NS":
            return None
        lat_sign = 1 if loc[0] == "N" else -1
        ew_idx = None
        for i, ch in enumerate(loc[1:], 1):
            if ch in "EW":
                ew_idx = i
                break
        if ew_idx is None:
            return None
        lat_val = float(loc[1:ew_idx])
        lon_sign = 1 if loc[ew_idx] == "E" else -1
        lon_val = float(loc[ew_idx + 1:])
        return (lat_sign * lat_val, lon_sign * lon_val)
    except (ValueError, IndexError):
        return None


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
    mtime_myt = mtime_utc + MYT_OFFSET

    result["utc"] = mtime_utc.strftime("%Y-%m-%d %H:%M UTC")
    result["myt"] = mtime_myt.strftime("%Y-%m-%d %H:%M MYT")
    result["source_file"] = newest.name

    _save_json(result, "ingestion_timestamp.json")
    return result


def build_event_counts_7d_30d(daily_counts: list[dict] | None) -> dict:
    """Panel (b): prepare grouped bar chart data — 7d / 30d event counts by type.

    Returns a dict with JSON-safe data for D3.
    """
    payload = {"groups": ["Last 7 Days", "Last 30 Days"], "types": EVENT_TYPES, "values": {}}

    if not daily_counts:
        for t in EVENT_TYPES:
            payload["values"]["7d_" + t] = 0
            payload["values"]["30d_" + t] = 0
        _save_json(payload, "event_counts_7d_30d.json")
        return payload

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

    for t in EVENT_TYPES:
        payload["values"]["7d_" + t] = counts_7d[t]
        payload["values"]["30d_" + t] = counts_30d[t]

    _save_json(payload, "event_counts_7d_30d.json")
    return payload


def build_event_timeline(daily_counts: list[dict] | None) -> dict:
    """Panel (c): prepare timeline data — daily event counts over time.

    Returns a dict with dates and per-type counts for D3 line chart.
    """
    if not daily_counts:
        payload = {"dates": [], "cme": [], "flr": [], "gst": []}
        _save_json(payload, "event_timeline.json")
        return payload

    dates = [d["date"] for d in daily_counts]
    cme_vals = [d.get("cme_count", 0) for d in daily_counts]
    flr_vals = [d.get("flr_count", 0) for d in daily_counts]
    gst_vals = [d.get("gst_count", 0) for d in daily_counts]

    payload = {
        "dates": dates,
        "cme": cme_vals,
        "flr": flr_vals,
        "gst": gst_vals,
    }
    _save_json(payload, "event_timeline.json")
    return payload


def build_top_active_days(top_days: list[dict] | None) -> dict:
    """Panel (d): prepare top 5 days by total event count for horizontal stacked bar."""
    if not top_days:
        payload = {"days": []}
        _save_json(payload, "top_active_days.json")
        return payload

    sorted_days = sorted(top_days, key=lambda d: d.get("total_count", 0), reverse=True)

    def _fmt_date(d):
        try:
            parsed = datetime.strptime(d, "%Y-%m-%d")
            return parsed.strftime("%a %d %b")
        except (ValueError, TypeError):
            return d

    days = []
    for d in sorted_days:
        days.append({
            "date": d["date"],
            "label": _fmt_date(d["date"]),
            "total": d.get("total_count", 0),
            "cme": d.get("cme_count", 0),
            "flr": d.get("flr_count", 0),
            "gst": d.get("gst_count", 0),
        })

    payload = {"days": days}
    _save_json(payload, "top_active_days.json")
    return payload


def build_terminology_card() -> dict:
    """Panel (e, replaced terminology card with solar disc): keep for data, but not displayed.

    Terminology is still computed and saved to JSON, but the dashboard now
    uses the solar disc as panel (e). Terminology data is available if needed.
    """
    fallback = {
        "text": "No retrieval results available. The corpus may not have been built yet.",
        "source_label": "",
        "cosine_score": 0.0,
    }

    try:
        from src.rag.retrieve import retrieve
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


def build_solar_disc_data(days: int = 30) -> dict:
    """Panel (e): prepare CME solar disc data for D3 rendering.

    Loads the latest CME JSON from DATA_RAW, filters to recent window,
    parses Stonyhurst coordinates, and returns a dict with all fields
    needed for the D3 visualisation.
    """
    payload = {"points": [], "now_iso": "", "days": days, "has_data": False}

    cme_files = sorted(DATA_RAW.glob("CME_*.json"))
    if not cme_files:
        logger.warning("No CME data files found for solar disc.")
        _save_json(payload, "solar_disc_data.json")
        return payload

    with open(cme_files[-1]) as f:
        cmes = json.load(f)

    now = datetime.now(timezone.utc)
    payload["now_iso"] = now.strftime("%Y-%m-%dT%H:%M:%SZ")
    cutoff_ts = now.timestamp() - days * 86400

    points = []
    for cme in cmes:
        try:
            start = datetime.fromisoformat(cme["startTime"].replace("Z", "+00:00"))
            if start.timestamp() < cutoff_ts:
                continue
        except (KeyError, ValueError):
            continue

        lat = None
        lon = None
        speed = None
        cme_type = None
        half_angle = None
        loc_str = cme.get("sourceLocation", "")

        parsed = _parse_stonyhurst(loc_str)
        if parsed:
            lat, lon = parsed

        if lat is None or lon is None:
            for analysis in cme.get("cmeAnalyses", []) or []:
                if analysis.get("latitude") is not None and analysis.get("longitude") is not None:
                    lat = analysis["latitude"]
                    lon = analysis["longitude"]
                    break

        if lat is None or lon is None:
            continue

        for analysis in cme.get("cmeAnalyses", []) or []:
            if analysis.get("isMostAccurate") and analysis.get("speed"):
                speed = analysis["speed"]
                break
        if speed is None:
            for analysis in cme.get("cmeAnalyses", []) or []:
                if analysis.get("speed"):
                    speed = analysis["speed"]
                    break

        for analysis in cme.get("cmeAnalyses", []) or []:
            if analysis.get("type"):
                cme_type = analysis["type"]
                half_angle = analysis.get("halfAngle")
                break

        days_ago = (now - start).total_seconds() / 86400
        radial_dist = math.sqrt(lat**2 + lon**2)
        visible = radial_dist <= 90.0

        points.append({
            "activityID": cme.get("activityID", "")[-40:],
            "lat": lat,
            "lon": lon,
            "speed": speed or 0,
            "type": cme_type or "",
            "halfAngle": half_angle,
            "locStr": loc_str or f"lat={lat:.0f} lon={lon:.0f}",
            "daysAgo": round(days_ago, 1),
            "visible": visible,
            "startTime": cme.get("startTime", ""),
        })

    payload["points"] = points
    payload["has_data"] = len(points) > 0

    _save_json(payload, "solar_disc_data.json")
    logger.info("Solar disc: %d CME points prepared.", len(points))
    return payload


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
<script src="https://d3js.org/d3.v7.min.js"></script>
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

  .no-data-msg {
    color: #5555aa;
    text-align: center;
    padding: 2rem 1rem;
    font-style: italic;
  }

  /* Chart containers */
  .chart-container {
    width: 100%;
    position: relative;
  }
  .chart-container svg {
    display: block;
    margin: 0 auto;
  }

  /* Tooltip */
  .d3-tooltip {
    position: absolute;
    padding: 8px 12px;
    background: #1a1a2e;
    color: #e0e0e0;
    border: 1px solid #555;
    border-radius: 6px;
    font-size: 0.8rem;
    pointer-events: none;
    opacity: 0;
    transition: opacity 0.15s;
    z-index: 100;
    max-width: 260px;
    line-height: 1.5;
    box-shadow: 0 4px 12px rgba(0,0,0,0.5);
  }
  .d3-tooltip.visible { opacity: 1; }
  .d3-tooltip strong { color: #4fc3f7; }

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

  <!-- (a) Ingestion Timestamp -->
  <div class="panel">
    <h2>Last Data Ingestion</h2>
    <div class="text-card">
      <div class="timestamp-row"><span class="label">UTC</span><span class="value">{{ ingest.utc }}</span></div>
      <div class="timestamp-row"><span class="label">Malaysia Time (UTC+8)</span><span class="value">{{ ingest.myt }}</span></div>
      <div class="timestamp-row"><span class="label">Source File</span><span class="value">{{ ingest.source_file }}</span></div>
    </div>
  </div>

  <!-- (b) Event Counts — 7 & 30 Day Windows -->
  <div class="panel">
    <h2>Event Counts — 7 &amp; 30 Day Windows</h2>
    <div class="chart-container" id="chart-counts"></div>
  </div>

  <!-- (c) Event Timeline -->
  <div class="panel panel-wide">
    <h2>Event Timeline</h2>
    <div class="chart-container" id="chart-timeline"></div>
  </div>

  <!-- (d) Top 5 Most Active Days -->
  <div class="panel">
    <h2>Top 5 Most Active Days</h2>
    <div class="chart-container" id="chart-topdays"></div>
  </div>

  <!-- (e) CME Origins — Solar Disc (full width) -->
  <div class="panel panel-full">
    <h2>CME Origins — Solar Disc (last 30 days)</h2>
    <div class="chart-container" id="chart-solardisc"></div>
  </div>

  <!-- (f) LLM Analyst Brief -->
  <div class="panel">
    <h2>LLM Analyst Brief</h2>
    <div class="text-card">
      {% if brief.text %}
        <div class="brief-text">{{ brief.text }}</div>
        <div style="margin-top:0.5rem;font-size:0.75rem;color:#5555aa;">{{ brief.word_count }} words</div>
      {% else %}
        <p class="no-data-msg">No analyst brief available.</p>
      {% endif %}
    </div>
  </div>

</div>

<div style="text-align:center;margin-top:2rem;padding-top:1rem;border-top:1px solid #2a2a4a;font-size:0.75rem;color:#5555aa;">
  Heliophysics Monitor &mdash; Generated {{ generation_time_utc }}
</div>

<!-- Embedded JSON data -->
<script>
var COUNTS_DATA = {{ counts_json|safe }};
var TIMELINE_DATA = {{ timeline_json|safe }};
var TOPDAYS_DATA = {{ topdays_json|safe }};
var SOLARDISC_DATA = {{ solardisc_json|safe }};
</script>

<!-- D3.js Visualisation Code -->
<script>
(function() {
// ── Colour palette ─────────────────────────────────────────────────
var C = {
  bg: "#0a0a1a",
  cardBg: "#111128",
  border: "#2a2a4a",
  text: "#e0e0e0",
  muted: "#8888aa",
  accent: "#4fc3f7",
  cme: "#ff6b35",
  flr: "#ffd700",
  gst: "#00d4ff",
};

// ── Tooltip ─────────────────────────────────────────────────────────
var tooltip = d3.select("body").append("div")
    .attr("class", "d3-tooltip");

function showTooltip(html, ev) {
  tooltip.html(html).classed("visible", true);
  var rect = tooltip.node().getBoundingClientRect();
  var x = ev.pageX + 14;
  var y = ev.pageY - 14;
  // Keep within viewport
  if (x + rect.width > window.innerWidth - 10) x = ev.pageX - rect.width - 14;
  if (y < 10) y = ev.pageY + 14;
  tooltip.style("left", x + "px").style("top", y + "px");
}
function hideTooltip() {
  tooltip.classed("visible", false);
}

// ── Responsive sizing helper ───────────────────────────────────────
function getContainerSize(sel) {
  var node = document.querySelector(sel);
  if (!node) return {w: 400, h: 300};
  var w = node.clientWidth;
  if (w < 100) w = 400;
  return {w: w, h: Math.max(200, Math.round(w * 0.55))};
}

// ══════════════════════════════════════════════════════════════════
// PANEL (b): Event Counts — Grouped Bar Chart
// ══════════════════════════════════════════════════════════════════
function renderCounts() {
  var sel = "#chart-counts";
  var size = getContainerSize(sel);
  var margin = {top: 20, right: 20, bottom: 40, left: 50};
  var w = size.w - margin.left - margin.right;
  var h = size.h - margin.top - margin.bottom;
  if (w < 100) return;

  var svg = d3.select(sel).selectAll("svg").data([null]);
  svg = svg.enter().append("svg").merge(svg);
  svg.attr("width", size.w).attr("height", size.h);

  var g = svg.selectAll(".chart-g").data([null]);
  g = g.enter().append("g").attr("class", "chart-g").merge(g);
  g.attr("transform", "translate(" + margin.left + "," + margin.top + ")");

  var data = COUNTS_DATA;
  var groups = data.groups;
  var types = data.types;
  var values = data.values;

  var x0 = d3.scaleBand().domain(groups).range([0, w]).padding(0.2);
  var x1 = d3.scaleBand().domain(types).range([0, x0.bandwidth()]).padding(0.1);
  var yMax = d3.max(types, function(t) { return Math.max(values["7d_" + t] || 0, values["30d_" + t] || 0); });
  var yMaxR = Math.ceil((yMax || 10) * 1.15);
  var y = d3.scaleLinear().domain([0, yMaxR]).range([h, 0]).nice();

  var colourMap = {CME: C.cme, FLR: C.flr, GST: C.gst};

  // Bars
  var barGroups = g.selectAll(".bar-group").data(groups);
  barGroups.exit().remove();
  var barGroupsEnter = barGroups.enter().append("g").attr("class", "bar-group");
  barGroups = barGroupsEnter.merge(barGroups);
  barGroups.attr("transform", function(d) { return "translate(" + x0(d) + ",0)"; });

  types.forEach(function(type) {
    var bars = barGroups.selectAll(".bar-" + type).data(function(group) {
      return [{group: group, type: type, value: values[(group === "Last 7 Days" ? "7d" : "30d") + "_" + type]}];
    });
    bars.exit().remove();
    var barsEnter = bars.enter().append("rect").attr("class", "bar-" + type);
    bars = barsEnter.merge(bars);
    bars.attr("x", function(d) { return x1(d.type); })
        .attr("width", x1.bandwidth())
        .attr("y", function(d) { return y(d.value || 0); })
        .attr("height", function(d) { return h - y(d.value || 0); })
        .attr("fill", function(d) { return colourMap[d.type]; })
        .attr("rx", 2)
        .attr("ry", 2)
        .style("opacity", 0.85)
        .on("mouseover", function(ev, d) {
          showTooltip("<strong>" + d.type + "</strong><br>" + d.group + ": " + (d.value || 0) + " events", ev);
        })
        .on("mouseout", hideTooltip);
  });

  // Axes
  var xAxis = d3.axisBottom(x0).tickSize(0);
  g.selectAll(".x-axis").remove();
  g.append("g").attr("class", "x-axis").attr("transform", "translate(0," + h + ")")
    .call(xAxis)
    .style("color", C.muted);

  var yAxis = d3.axisLeft(y).ticks(5);
  g.selectAll(".y-axis").remove();
  g.append("g").attr("class", "y-axis")
    .call(yAxis)
    .style("color", C.muted);

  // Y-axis label
  g.selectAll(".y-label").remove();
  g.append("text").attr("class", "y-label")
    .attr("x", -40).attr("y", 10).attr("transform", "rotate(-90)")
    .style("fill", C.muted).style("font-size", "0.75rem")
    .style("text-anchor", "end")
    .text("Event Count");

  // Legend
  g.selectAll(".legend").remove();
  var legend = g.append("g").attr("class", "legend").attr("transform", "translate(0," + (h + 25) + ")");
  types.forEach(function(type, i) {
    var lg = legend.append("g").attr("transform", "translate(" + (i * 80) + ",0)");
    lg.append("rect").attr("x", 0).attr("y", 0).attr("width", 12).attr("height", 12)
      .attr("fill", colourMap[type]).attr("rx", 2);
    lg.append("text").attr("x", 16).attr("y", 10)
      .style("fill", C.muted).style("font-size", "0.7rem")
      .text(type);
  });
}

// ══════════════════════════════════════════════════════════════════
// PANEL (c): Event Timeline — Line Chart
// ══════════════════════════════════════════════════════════════════
function renderTimeline() {
  var sel = "#chart-timeline";
  var size = getContainerSize(sel);
  var margin = {top: 20, right: 20, bottom: 45, left: 50};
  var w = size.w - margin.left - margin.right;
  var h = size.h - margin.top - margin.bottom;
  if (w < 100) return;

  var svg = d3.select(sel).selectAll("svg").data([null]);
  svg = svg.enter().append("svg").merge(svg);
  svg.attr("width", size.w).attr("height", size.h);

  var g = svg.selectAll(".chart-g").data([null]);
  g = g.enter().append("g").attr("class", "chart-g").merge(g);
  g.attr("transform", "translate(" + margin.left + "," + margin.top + ")");

  var data = TIMELINE_DATA;
  if (!data.dates || data.dates.length === 0) {
    g.append("text").attr("x", w/2).attr("y", h/2)
      .style("fill", C.muted).style("text-anchor", "middle")
      .text("No timeline data");
    return;
  }

  var parseDate = d3.timeParse("%Y-%m-%d");
  var dates = data.dates.map(parseDate);

  var x = d3.scaleTime().domain(d3.extent(dates)).range([0, w]);
  var yMax = d3.max(data.dates, function(_, i) {
    return Math.max(data.cme[i] || 0, data.flr[i] || 0, data.gst[i] || 0);
  });
  var yMaxR = Math.ceil((yMax || 10) * 1.15);
  var y = d3.scaleLinear().domain([0, yMaxR]).range([h, 0]).nice();

  var series = [
    {key: "CME", values: data.cme, color: C.cme},
    {key: "FLR", values: data.flr, color: C.flr},
    {key: "GST", values: data.gst, color: C.gst},
  ];

  var line = d3.line()
    .x(function(_, i) { return x(dates[i]); })
    .y(function(v) { return y(v); })
    .curve(d3.curveMonotoneX);

  series.forEach(function(s) {
    var path = g.selectAll(".line-" + s.key).data([s]);
    path.exit().remove();
    var pathEnter = path.enter().append("path").attr("class", "line-" + s.key);
    path = pathEnter.merge(path);
    path.attr("d", function() { return line(s.values); })
        .style("fill", "none")
        .style("stroke", s.color)
        .style("stroke-width", 2)
        .style("opacity", 0.85);

    // Dot overlay for tooltip
    var dots = g.selectAll(".dots-" + s.key).data(dates);
    dots.exit().remove();
    var dotsEnter = dots.enter().append("circle").attr("class", "dots-" + s.key);
    dots = dotsEnter.merge(dots);
    dots.attr("r", 3)
        .attr("cx", function(_, i) { return x(dates[i]); })
        .attr("cy", function(_, i) { return y(s.values[i] || 0); })
        .style("fill", s.color)
        .style("opacity", 0)
        .style("stroke", "none")
        .on("mouseover", function(ev, d) {
          var i = dates.indexOf(d);
          if (i < 0) return;
          var html = "<strong>" + data.dates[i] + "</strong><br>";
          html += "CME: " + data.cme[i] + "<br>";
          html += "FLR: " + data.flr[i] + "<br>";
          html += "GST: " + data.gst[i];
          showTooltip(html, ev);
        })
        .on("mouseout", hideTooltip);
  });

  // Make dots visible on hover over the whole chart area
  var overlay = g.selectAll(".overlay").data([null]);
  overlay = overlay.enter().append("rect").attr("class", "overlay").merge(overlay);
  overlay.attr("width", w).attr("height", h)
    .style("fill", "none")
    .style("pointer-events", "all")
    .on("mouseover", function() {
      g.selectAll("circle").style("opacity", 0.8);
    })
    .on("mouseout", function() {
      g.selectAll("circle").style("opacity", 0);
      hideTooltip();
    });

  // Axes
  g.selectAll(".x-axis").remove();
  g.append("g").attr("class", "x-axis").attr("transform", "translate(0," + h + ")")
    .call(d3.axisBottom(x).ticks(8).tickFormat(d3.timeFormat("%b %d")))
    .style("color", C.muted);

  g.selectAll(".y-axis").remove();
  g.append("g").attr("class", "y-axis")
    .call(d3.axisLeft(y).ticks(5))
    .style("color", C.muted);

  g.selectAll(".y-label").remove();
  g.append("text").attr("class", "y-label")
    .attr("x", -40).attr("y", 10).attr("transform", "rotate(-90)")
    .style("fill", C.muted).style("font-size", "0.75rem")
    .style("text-anchor", "end")
    .text("Daily Event Count");

  // Legend
  g.selectAll(".legend").remove();
  var legend = g.append("g").attr("class", "legend").attr("transform", "translate(" + (w - 180) + "," + (-5) + ")");
  series.forEach(function(s, i) {
    var lg = legend.append("g").attr("transform", "translate(" + (i * 65) + ",0)");
    lg.append("line").attr("x1", 0).attr("y1", 6).attr("x2", 16).attr("y2", 6)
      .style("stroke", s.color).style("stroke-width", 2);
    lg.append("text").attr("x", 20).attr("y", 10)
      .style("fill", C.muted).style("font-size", "0.7rem")
      .text(s.key);
  });
}

// ══════════════════════════════════════════════════════════════════
// PANEL (d): Top 5 Most Active Days — Horizontal Stacked Bar
// ══════════════════════════════════════════════════════════════════
function renderTopDays() {
  var sel = "#chart-topdays";
  var size = getContainerSize(sel);
  // For horizontal bars, make it taller
  size.h = Math.max(size.h, 220);
  var margin = {top: 10, right: 30, bottom: 35, left: 95};
  var w = size.w - margin.left - margin.right;
  var h = size.h - margin.top - margin.bottom;
  if (w < 100) return;

  var svg = d3.select(sel).selectAll("svg").data([null]);
  svg = svg.enter().append("svg").merge(svg);
  svg.attr("width", size.w).attr("height", size.h);

  var g = svg.selectAll(".chart-g").data([null]);
  g = g.enter().append("g").attr("class", "chart-g").merge(g);
  g.attr("transform", "translate(" + margin.left + "," + margin.top + ")");

  var data = TOPDAYS_DATA;
  if (!data.days || data.days.length === 0) {
    g.append("text").attr("x", w/2).attr("y", h/2)
      .style("fill", C.muted).style("text-anchor", "middle")
      .text("No data");
    return;
  }

  var days = data.days;
  var types = ["cme", "flr", "gst"];
  var typeLabels = {cme: "CME", flr: "FLR", gst: "GST"};
  var typeColours = {cme: C.cme, flr: C.flr, gst: C.gst};

  // Stack the data
  var stacked = days.map(function(d) {
    var sum = 0;
    var items = types.map(function(t) {
      var val = d[t] || 0;
      var entry = {key: t, val: val, x0: sum, x1: sum + val, label: typeLabels[t], color: typeColours[t]};
      sum += val;
      return entry;
    });
    return {label: d.label, total: d.total, items: items, date: d.date};
  });

  var y = d3.scaleBand().domain(stacked.map(function(d) { return d.label; })).range([0, h]).padding(0.25);
  var xMax = d3.max(stacked, function(d) { return d.total; });
  var x = d3.scaleLinear().domain([0, Math.ceil(xMax * 1.1)]).range([0, w]);

  // Bars for each type segment
  types.forEach(function(type) {
    var bars = g.selectAll(".bar-" + type).data(stacked);
    bars.exit().remove();
    var barsEnter = bars.enter().append("rect").attr("class", "bar-" + type);
    bars = barsEnter.merge(bars);
    bars.attr("y", function(d) { return y(d.label); })
        .attr("height", y.bandwidth())
        .attr("x", function(d) {
          var item = d.items.find(function(i) { return i.key === type; });
          return x(item ? item.x0 : 0);
        })
        .attr("width", function(d) {
          var item = d.items.find(function(i) { return i.key === type; });
          return item ? x(item.x1) - x(item.x0) : 0;
        })
        .attr("fill", typeColours[type])
        .style("opacity", 0.85)
        .on("mouseover", function(ev, d) {
          var html = "<strong>" + d.date + "</strong><br>";
          d.items.forEach(function(i) {
            html += i.label + ": " + i.val + "<br>";
          });
          html += "<em>Total: " + d.total + "</em>";
          showTooltip(html, ev);
        })
        .on("mouseout", hideTooltip);
  });

  // Axes
  g.selectAll(".x-axis").remove();
  g.append("g").attr("class", "x-axis").attr("transform", "translate(0," + h + ")")
    .call(d3.axisBottom(x).ticks(4))
    .style("color", C.muted);

  g.selectAll(".y-axis").remove();
  g.append("g").attr("class", "y-axis")
    .call(d3.axisLeft(y).tickSize(0))
    .style("color", C.text)
    .style("font-size", "0.75rem");

  g.selectAll(".x-label").remove();
  g.append("text").attr("class", "x-label")
    .attr("x", w/2).attr("y", h + 25)
    .style("fill", C.muted).style("font-size", "0.75rem")
    .style("text-anchor", "middle")
    .text("Total Event Count");
}

// ══════════════════════════════════════════════════════════════════
// PANEL (e): CME Origins — Solar Disc (SVG visualisation)
// ══════════════════════════════════════════════════════════════════
function renderSolarDisc() {
  var sel = "#chart-solardisc";
  var size = getContainerSize(sel);
  // Force 1:1 aspect ratio (square)
  var dim = Math.min(size.w, 560);
  var margin = 30;
  var radius = (dim - margin * 2) / 2;
  var cx = dim / 2;
  var cy = dim / 2;
  // Maximum Stonyhurst degrees visible = 90
  var scale = radius / 90;  // px per degree

  var svg = d3.select(sel).selectAll("svg").data([null]);
  svg = svg.enter().append("svg").merge(svg);
  svg.attr("width", dim).attr("height", dim)
    .attr("viewBox", "0 0 " + dim + " " + dim)
    .style("background", "#0a0a1a")
    .style("border-radius", "8px");

  var data = SOLARDISC_DATA;
  if (!data.has_data || !data.points || data.points.length === 0) {
    svg.selectAll("*").remove();
    svg.append("text").attr("x", dim/2).attr("y", dim/2)
      .style("fill", C.muted).style("text-anchor", "middle")
      .style("font-size", "0.85rem")
      .text("No CME origin data available");
    return;
  }

  // Clear previous content but keep the svg
  svg.selectAll("defs, g").remove();

  // ── Defs: gradients ────────────────────────────────────────────
  var defs = svg.append("defs");

  // Solar disc radial gradient
  var solarGrad = defs.append("radialGradient").attr("id", "solarGrad")
    .attr("cx", "50%").attr("cy", "50%").attr("r", "50%");
  solarGrad.append("stop").attr("offset", "0%")
    .attr("stop-color", "#ffdd44").attr("stop-opacity", 0.4);
  solarGrad.append("stop").attr("offset", "40%")
    .attr("stop-color", "#ff8800").attr("stop-opacity", 0.15);
  solarGrad.append("stop").attr("offset", "100%")
    .attr("stop-color", "#ff4400").attr("stop-opacity", 0.02);

  // Corona glow (outer ring gradient)
  var coronaGrad = defs.append("radialGradient").attr("id", "coronaGrad")
    .attr("cx", "50%").attr("cy", "50%").attr("r", "50%");
  coronaGrad.append("stop").attr("offset", "85%")
    .attr("stop-color", "#ffaa44").attr("stop-opacity", 0.12);
  coronaGrad.append("stop").attr("offset", "92%")
    .attr("stop-color", "#ff6600").attr("stop-opacity", 0.06);
  coronaGrad.append("stop").attr("offset", "100%")
    .attr("stop-color", "#ff3300").attr("stop-opacity", 0);

  // ── Outer corona glow ──────────────────────────────────────────
  // Multiple concentric circles for glow effect
  for (var i = 5; i > 0; i--) {
    var glowR = radius + i * 6;
    var glowOpacity = 0.04 - i * 0.007;
    svg.append("circle")
      .attr("cx", cx).attr("cy", cy)
      .attr("r", glowR)
      .style("fill", "none")
      .style("stroke", "#ffaa44")
      .style("stroke-width", 2)
      .style("opacity", Math.max(0, glowOpacity));
  }

  // Corona fill
  svg.append("circle")
    .attr("cx", cx).attr("cy", cy)
    .attr("r", radius + 12)
    .style("fill", "url(#coronaGrad)");

  // ── Solar disc ─────────────────────────────────────────────────
  svg.append("circle")
    .attr("cx", cx).attr("cy", cy)
    .attr("r", radius)
    .style("fill", "url(#solarGrad)")
    .style("stroke", "#ffd700")
    .style("stroke-width", 2)
    .style("opacity", 0.9);

  // ── Grid: concentric rings at 30° and 60° ──────────────────────
  var ringRadii = [
    {r: scale * 30, label: "30°"},
    {r: scale * 60, label: "60°"},
  ];
  ringRadii.forEach(function(rr) {
    svg.append("circle")
      .attr("cx", cx).attr("cy", cy)
      .attr("r", rr.r)
      .style("fill", "none")
      .style("stroke", "rgba(255,255,255,0.12)")
      .style("stroke-width", 0.5)
      .style("stroke-dasharray", "3,3");
  });

  // ── Grid: radial lines every 30° ────────────────────────────────
  for (var a = 0; a < 360; a += 30) {
    var rad = a * Math.PI / 180;
    var x2 = cx + radius * Math.cos(rad);
    var y2 = cy + radius * Math.sin(rad);
    svg.append("line")
      .attr("x1", cx).attr("y1", cy)
      .attr("x2", x2).attr("y2", y2)
      .style("stroke", "rgba(255,255,255,0.08)")
      .style("stroke-width", 0.5);
  }

  // ── Earth-directed zone (central ~30°) ───────────────────────────
  svg.append("circle")
    .attr("cx", cx).attr("cy", cy)
    .attr("r", scale * 30)
    .style("fill", "rgba(0,180,255,0.08)")
    .style("stroke", "rgba(0,212,255,0.2)")
    .style("stroke-width", 1)
    .style("stroke-dasharray", "4,4");

  // ── Compass labels ──────────────────────────────────────────────
  var compass = [
    {label: "N", x: cx, y: cy - radius - 16},
    {label: "S", x: cx, y: cy + radius + 18},
    {label: "E", x: cx + radius + 16, y: cy + 4},
    {label: "W", x: cx - radius - 16, y: cy + 4},
  ];
  compass.forEach(function(cp) {
    svg.append("text")
      .attr("x", cp.x).attr("y", cp.y)
      .style("fill", C.muted)
      .style("font-size", "0.75rem")
      .style("font-weight", "bold")
      .style("text-anchor", "middle")
      .style("dominant-baseline", "central")
      .text(cp.label);
  });

  // ── "EARTH" label ───────────────────────────────────────────────
  svg.append("text")
    .attr("x", cx).attr("y", cy + 3)
    .style("fill", C.gst)
    .style("font-size", "0.7rem")
    .style("font-weight", "bold")
    .style("text-anchor", "middle")
    .style("dominant-baseline", "central")
    .style("opacity", 0.7)
    .text("EARTH");

  // ── CME markers ─────────────────────────────────────────────────
  var points = data.points;
  var now = new Date();
  var maxDays = 30;

  // Speed range for marker sizing
  var speeds = points.map(function(p) { return p.speed || 0; });
  var speedMin = d3.min(speeds) || 100;
  var speedMax = d3.max(speeds) || 3000;
  var speedRange = speedMax - speedMin || 1;

  // Colour scale by days ago
  function colourByDays(daysAgo) {
    var frac = Math.min(1, Math.max(0, daysAgo / maxDays));
    // Gradient: #ff4500 -> #ff8c00 -> #ffd700 -> #87ceeb -> #4169e1
    var stops = [
      {t: 0.0, c: d3.rgb("#ff4500")},
      {t: 0.25, c: d3.rgb("#ff8c00")},
      {t: 0.5, c: d3.rgb("#ffd700")},
      {t: 0.75, c: d3.rgb("#87ceeb")},
      {t: 1.0, c: d3.rgb("#4169e1")},
    ];
    for (var i = 0; i < stops.length - 1; i++) {
      if (frac >= stops[i].t && frac <= stops[i+1].t) {
        var local = (frac - stops[i].t) / (stops[i+1].t - stops[i].t);
        return d3.interpolateRgb(stops[i].c, stops[i+1].c)(local);
      }
    }
    return stops[stops.length-1].c;
  }

  function markerSize(speed) {
    if (!speed || speed <= 0) return 4;
    // Log scale: log10(100)->4px, log10(3000)->20px
    var logS = Math.log10(Math.max(speed, 10));
    var frac = (logS - 2.0) / (3.48 - 2.0);
    frac = Math.max(0, Math.min(1, frac));
    return 4 + frac * 16;
  }

  var markerGroup = svg.append("g").attr("class", "cme-markers");

  points.forEach(function(p) {
    var px = cx + p.lon * scale;
    var py = cy - p.lat * scale;  // SVG y-axis inverted
    var r = markerSize(p.speed);
    var col = colourByDays(p.daysAgo);
    var isVisible = p.visible;

    var circle = markerGroup.append("circle")
      .attr("cx", px).attr("cy", py)
      .attr("r", isVisible ? r : Math.max(3, r * 0.5))
      .style("fill", isVisible ? col : "rgba(120,120,120,0.3)")
      .style("stroke", isVisible ? "rgba(255,255,255,0.5)" : "rgba(150,150,150,0.2)")
      .style("stroke-width", isVisible ? 1.5 : 0.5)
      .style("opacity", isVisible ? 0.85 : 0.3)
      .style("cursor", "pointer");

    // Glow effect for visible points
    if (isVisible) {
      markerGroup.append("circle")
        .attr("cx", px).attr("cy", py)
        .attr("r", r * 2)
        .style("fill", "none")
        .style("stroke", col)
        .style("stroke-width", 1)
        .style("opacity", 0.15);
    }

    // Tooltip
    circle.on("mouseover", function(ev) {
      var html = "<strong>" + p.locStr + "</strong><br>";
      html += "Date: " + p.startTime.slice(0, 10) + "<br>";
      html += "Speed: " + p.speed + " km/s<br>";
      html += "Type: " + (p.type || "N/A") + "<br>";
      html += "Days ago: " + p.daysAgo + "<br>";
      if (!p.visible) html += "<em>Far side</em>";
      showTooltip(html, ev);
      d3.select(this).style("stroke", "#fff").style("stroke-width", 2.5);
    })
    .on("mouseout", function() {
      hideTooltip();
      d3.select(this).style("stroke", isVisible ? "rgba(255,255,255,0.5)" : "rgba(150,150,150,0.2)")
        .style("stroke-width", isVisible ? 1.5 : 0.5);
    });
  });

  // ── Colour legend bar (days-ago gradient) ──────────────────────
  var legendY = dim - 28;
  var legendW = 180;
  var legendH = 10;
  var legendX = (dim - legendW) / 2;

  // Draw gradient legend
  var legendDefs = defs.append("linearGradient").attr("id", "legendGrad")
    .attr("x1", "0%").attr("y1", "0%").attr("x2", "100%").attr("y2", "0%");
  var gradStops = [
    {t: "0%", c: "#ff4500"},
    {t: "25%", c: "#ff8c00"},
    {t: "50%", c: "#ffd700"},
    {t: "75%", c: "#87ceeb"},
    {t: "100%", c: "#4169e1"},
  ];
  gradStops.forEach(function(s) {
    legendDefs.append("stop").attr("offset", s.t).attr("stop-color", s.c);
  });

  svg.append("rect")
    .attr("x", legendX).attr("y", legendY)
    .attr("width", legendW).attr("height", legendH)
    .style("fill", "url(#legendGrad)")
    .style("rx", 3);

  svg.append("text")
    .attr("x", legendX - 4).attr("y", legendY + legendH / 2 + 1)
    .style("fill", C.muted).style("font-size", "0.6rem")
    .style("text-anchor", "end").style("dominant-baseline", "central")
    .text("Today");

  svg.append("text")
    .attr("x", legendX + legendW + 4).attr("y", legendY + legendH / 2 + 1)
    .style("fill", C.muted).style("font-size", "0.6rem")
    .style("text-anchor", "start").style("dominant-baseline", "central")
    .text("30 days ago");
}

// ══════════════════════════════════════════════════════════════════
// Initialise all D3 panels
// ══════════════════════════════════════════════════════════════════
function init() {
  renderCounts();
  renderTimeline();
  renderTopDays();
  renderSolarDisc();
}

// Debounced resize handler
var resizeTimer;
window.addEventListener("resize", function() {
  clearTimeout(resizeTimer);
  resizeTimer = setTimeout(init, 200);
});

// Run on load
if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", init);
} else {
  init();
}

})();
</script>

</body>
</html>"""


# ── Dashboard Assembler ──────────────────────────────────────────────


def build_dashboard() -> Path:
    """Main entry point — load data, build panels, render HTML, save files."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    logger.info("Building Heliophysics Monitor dashboard (D3.js)…")

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

    # ── Build panel data ───────────────────────────────────────────
    now_utc = datetime.now(timezone.utc)

    # (a) Ingestion timestamp — text card
    ingest = build_ingestion_timestamp()

    # (b) Event counts — JSON payload for D3
    counts_data = build_event_counts_7d_30d(daily_counts)

    # (c) Event timeline — JSON payload for D3
    timeline_data = build_event_timeline(daily_counts)

    # (d) Top active days — JSON payload for D3
    topdays_data = build_top_active_days(top_days)

    # (e) Solar disc — CME origins JSON payload for D3
    solardisc_data = build_solar_disc_data(days=30)

    # Terminology card (computed but not displayed in dashboard; saved for reference)
    build_terminology_card()

    # (f) Analyst brief — text card
    brief = build_analyst_brief()

    # ── Render via Jinja2 ──────────────────────────────────────────
    template = Template(DASHBOARD_TEMPLATE_STR)
    html = template.render(
        ingest=ingest,
        brief=brief,
        counts_json=json.dumps(counts_data),
        timeline_json=json.dumps(timeline_data),
        topdays_json=json.dumps(topdays_data),
        solardisc_json=json.dumps(solardisc_data),
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
