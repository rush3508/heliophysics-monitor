"""
solar_disc.py — Plot CME origin points on a solar disc using Stonyhurst coordinates.

Stonyhurst heliographic coordinates: latitude (N/S, ±90°) and longitude (E/W, ±90°).
Longitude 0° = central meridian (Earth-facing). The visible solar disc is the circle
where sqrt(lat² + lon²) ≤ 90°. Points outside this are on the far side.

Each CME is plotted as a circle marker:
- Position: (longitude, latitude) on the Stonyhurst projection
- Size: proportional to speed (km/s)
- Colour: gradient by recency (hot = recent, cool = older)
- Tooltip: date, speed, sourceLocation, type, halfAngle

Usage:
    from src.reporting.solar_disc import build_solar_disc_figure
    fig = build_solar_disc_figure()
"""

import json
import logging
from datetime import datetime, timezone

import numpy as np
import plotly.graph_objects as go

from config import DATA_RAW

logger = logging.getLogger(__name__)

# Solar disc styling
DISC_BG = "#0a0a1a"
DISC_RADIUS = 90  # degrees (the solar limb)
GRID_COLOUR = "rgba(255,255,255,0.08)"
LIMB_COLOUR = "rgba(255,200,100,0.4)"
EARTH_DIRECTED_COLOUR = "rgba(0,212,255,0.15)"

# Colour scale: recency (days ago → colour)
COLOUR_SCALE = [
    [0.0, "#ff4500"],   # today = red-orange
    [0.25, "#ff8c00"],  # recent = orange
    [0.5, "#ffd700"],   # moderate = gold
    [0.75, "#87ceeb"],  # older = light blue
    [1.0, "#4169e1"],   # oldest = royal blue
]


def _parse_stonyhurst(loc: str) -> tuple[float, float] | None:
    """
    Parse a Stonyhurst coordinate string like "N25E35" → (lat, lon).

    Returns (latitude, longitude) in degrees, or None if unparseable.
    """
    if not loc or not isinstance(loc, str) or not loc.strip():
        return None
    loc = loc.strip().upper()
    # Expected pattern: [NS]dd[EW]dd, e.g. N25E35, S18W59
    try:
        ns_idx = 0 if loc[0] in "NS" else None
        if ns_idx is None:
            return None
        lat_sign = 1 if loc[0] == "N" else -1
        # Find the E/W separator
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


def _is_visible(lat: float, lon: float) -> bool:
    """Check if a heliographic point is on the visible solar disc."""
    return np.sqrt(lat**2 + lon**2) <= DISC_RADIUS


def _build_disc_background() -> go.Figure:
    """Create a Plotly figure with solar disc grid and limb outline."""
    fig = go.Figure()

    # Solar disc outline
    theta = np.linspace(0, 2 * np.pi, 200)
    disc_x = DISC_RADIUS * np.cos(theta)
    disc_y = DISC_RADIUS * np.sin(theta)

    fig.add_trace(go.Scatter(
        x=disc_x, y=disc_y,
        mode="lines",
        line=dict(color=LIMB_COLOUR, width=2, dash="solid"),
        fill="toself",
        fillcolor="rgba(255,200,100,0.03)",
        name="Solar limb",
        hoverinfo="skip",
        showlegend=False,
    ))

    # Earth-direction zone (central 30°)
    inner_theta = np.linspace(0, 2 * np.pi, 100)
    inner_r = 30
    inner_x = inner_r * np.cos(inner_theta)
    inner_y = inner_r * np.sin(inner_theta)
    fig.add_trace(go.Scatter(
        x=inner_x, y=inner_y,
        mode="lines",
        line=dict(color="rgba(0,212,255,0.2)", width=1, dash="dot"),
        fill="toself",
        fillcolor=EARTH_DIRECTED_COLOUR,
        name="Earth-directed zone",
        hoverinfo="skip",
        showlegend=False,
    ))

    # Radial grid lines
    for angle in np.linspace(0, 2 * np.pi, 13)[:-1]:
        fig.add_trace(go.Scatter(
            x=[0, DISC_RADIUS * np.cos(angle)],
            y=[0, DISC_RADIUS * np.sin(angle)],
            mode="lines",
            line=dict(color=GRID_COLOUR, width=0.5),
            hoverinfo="skip",
            showlegend=False,
        ))

    # Concentric rings
    for r in [30, 60]:
        ring_theta = np.linspace(0, 2 * np.pi, 100)
        fig.add_trace(go.Scatter(
            x=r * np.cos(ring_theta),
            y=r * np.sin(ring_theta),
            mode="lines",
            line=dict(color=GRID_COLOUR, width=0.5),
            hoverinfo="skip",
            showlegend=False,
        ))

    return fig


def _compute_marker_size(speed: float | None, min_size: int = 6, max_size: int = 28) -> int:
    """Map CME speed (km/s) to marker size. Faster = bigger."""
    if speed is None or speed <= 0:
        return min_size
    # Log-ish scale: typical CME speeds range 100–3000 km/s
    log_speed = np.log10(max(speed, 10))
    # Map log10(10)=1 → min_size, log10(3000)≈3.48 → max_size
    frac = (log_speed - 1.0) / (3.48 - 1.0)
    frac = max(0.0, min(1.0, frac))
    return int(min_size + frac * (max_size - min_size))


def build_solar_disc_figure(days: int = 30) -> go.Figure | None:
    """
    Build a Plotly figure showing CME origins on a solar disc.

    Args:
        days: number of recent days to include (default: 30)

    Returns:
        Plotly Figure or None if no CME data is available.
    """
    # Load CME data
    cme_files = sorted(DATA_RAW.glob("CME_*.json"))
    if not cme_files:
        logger.warning("No CME data files found.")
        return None

    with open(cme_files[-1]) as f:
        cmes = json.load(f)

    # Filter to recent window
    now = datetime.now(timezone.utc)
    cutoff = now.timestamp() - days * 86400
    recent_cmes = []
    for cme in cmes:
        try:
            start = datetime.fromisoformat(cme["startTime"].replace("Z", "+00:00"))
            if start.timestamp() >= cutoff:
                recent_cmes.append(cme)
        except (KeyError, ValueError):
            continue

    logger.info("Solar disc: %d CMEs in last %d days.", len(recent_cmes), days)

    if not recent_cmes:
        return None

    # Extract visible CME origin points
    points = []
    for cme in recent_cmes:
        # Try sourceLocation first (Stonyhurst), fall back to cmeAnalyses lat/lon
        lat = None
        lon = None
        speed = None
        loc_str = cme.get("sourceLocation", "")

        # Parse sourceLocation
        parsed = _parse_stonyhurst(loc_str)
        if parsed:
            lat, lon = parsed

        # If no sourceLocation, use mostAccurate cmeAnalyses
        if lat is None or lon is None:
            for analysis in cme.get("cmeAnalyses", []) or []:
                if analysis.get("latitude") is not None and analysis.get("longitude") is not None:
                    lat = analysis["latitude"]
                    lon = analysis["longitude"]
                    break

        if lat is None or lon is None:
            continue

        # Get speed from mostAccurate analysis
        for analysis in cme.get("cmeAnalyses", []) or []:
            if analysis.get("isMostAccurate") and analysis.get("speed"):
                speed = analysis["speed"]
                break
        if speed is None:
            for analysis in cme.get("cmeAnalyses", []) or []:
                if analysis.get("speed"):
                    speed = analysis["speed"]
                    break

        # Determine type (SCORE scale: S, C, etc.)
        cme_type = None
        for analysis in cme.get("cmeAnalyses", []) or []:
            if analysis.get("type"):
                cme_type = analysis["type"]
                break

        # Days ago for colour mapping
        try:
            start = datetime.fromisoformat(cme["startTime"].replace("Z", "+00:00"))
            days_ago = (now - start).total_seconds() / 86400
        except (KeyError, ValueError):
            days_ago = days / 2

        visible = _is_visible(lat, lon)
        points.append({
            "activityID": cme.get("activityID", "")[-30:],
            "lat": lat,
            "lon": lon,
            "speed": speed,
            "type": cme_type,
            "loc_str": loc_str or f"lat={lat:.0f} lon={lon:.0f}",
            "days_ago": days_ago,
            "visible": visible,
            "start_time": cme.get("startTime", ""),
        })

    logger.info("Solar disc: %d CMEs with coordinates.", len(points))

    if not points:
        return None

    # Separate visible and far-side
    visible_pts = [p for p in points if p["visible"]]
    hidden_pts = [p for p in points if not p["visible"]]

    # Build figure
    fig = _build_disc_background()

    # Far-side points (dimmed)
    if hidden_pts:
        fig.add_trace(go.Scatter(
            x=[p["lon"] for p in hidden_pts],
            y=[p["lat"] for p in hidden_pts],
            mode="markers",
            marker=dict(
                size=[max(4, _compute_marker_size(p["speed"]) // 2) for p in hidden_pts],
                color="rgba(100,100,100,0.3)",
                line=dict(width=0.5, color="rgba(150,150,150,0.2)"),
            ),
            text=[f"<b>{p['loc_str']}</b><br>Speed: {p['speed']:.0f} km/s<br>Type: {p['type']}<br>Days ago: {p['days_ago']:.1f}<br><i>Far side</i>" for p in hidden_pts],
            hoverinfo="text",
            name="Far side",
            showlegend=True,
        ))

    # Visible disc points (coloured by recency)
    if visible_pts:
        colours = []
        for p in visible_pts:
            frac = min(1.0, p["days_ago"] / days)
            # Interpolate colour scale
            for i in range(len(COLOUR_SCALE) - 1):
                if COLOUR_SCALE[i][0] <= frac <= COLOUR_SCALE[i + 1][0]:
                    colours.append(COLOUR_SCALE[i][1])
                    break
            else:
                colours.append(COLOUR_SCALE[-1][1])

        fig.add_trace(go.Scatter(
            x=[p["lon"] for p in visible_pts],
            y=[p["lat"] for p in visible_pts],
            mode="markers",
            marker=dict(
                size=[_compute_marker_size(p["speed"]) for p in visible_pts],
                color=colours,
                line=dict(width=0.8, color="rgba(255,255,255,0.4)"),
                opacity=0.85,
            ),
            text=[f"<b>{p['loc_str']}</b><br>Speed: {p['speed']:.0f} km/s<br>Type: {p['type']}<br>{p['days_ago']:.1f} days ago<br><i>{p['start_time'][:16]}</i>" for p in visible_pts],
            hoverinfo="text",
            name="CME origins",
            showlegend=True,
        ))

    # Layout
    fig.update_layout(
        title=dict(
            text=f"CME Origins — Solar Disc (last {days} days)",
            font=dict(size=14, color="#e0e0e0"),
        ),
        xaxis=dict(
            range=[-100, 100],
            scaleanchor="y",
            scaleratio=1,
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            title=None,
        ),
        yaxis=dict(
            range=[-100, 100],
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            title=None,
        ),
        plot_bgcolor=DISC_BG,
        paper_bgcolor=DISC_BG,
        font=dict(color="#e0e0e0"),
        margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(
            x=0.01, y=0.99,
            font=dict(size=10, color="#888"),
            bgcolor="rgba(0,0,0,0.3)",
        ),
        hoverlabel=dict(
            bgcolor="#1a1a2e",
            font=dict(size=11, color="#e0e0e0"),
            bordercolor="#444",
        ),
    )

    # Annotations: compass points
    annotations = [
        dict(x=0, y=98, text="<b>N</b>", showarrow=False, font=dict(color="#888", size=11)),
        dict(x=0, y=-102, text="<b>S</b>", showarrow=False, font=dict(color="#888", size=11)),
        dict(x=98, y=0, text="<b>E</b>", showarrow=False, font=dict(color="#888", size=11)),
        dict(x=-102, y=0, text="<b>W</b>", showarrow=False, font=dict(color="#888", size=11)),
        dict(x=0, y=12, text="<b>EARTH</b>", showarrow=False, font=dict(color="#00d4ff", size=8)),
    ]
    fig.update_layout(annotations=annotations)

    return fig
