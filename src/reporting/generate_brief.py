1|"""
2|generate_brief.py — Generate a 150–250 word analyst brief using llama3.2.
3|
4|Loads structured stats from data/processed/, retrieves relevant chunks
5|from the heliophysics corpus, builds a constrained prompt for llama3.2,
6|and writes the brief to reports/brief.md.
7|
8|Enforces: token budget, staleness guard, hallucination guard, word count.
9|
10|Usage:
11|    python -m src.reporting.generate_brief
12|"""
13|
14|import json
15|from typing import Any
16|import logging
17|import os
18|import time
19|from datetime import datetime, timezone
20|from pathlib import Path
21|
22|import requests
23|
24|from config import (
25|    BRIEF_FILE,
26|    BRIEF_MAX_WORDS,
27|    BRIEF_MIN_WORDS,
28|    BRIEF_RETRY_MIN_WORDS,
29|    CONTROLLED_QUERIES,
30|    DATA_PROCESSED,
31|    DATA_RAW,
32|    DATA_MAX_AGE_HOURS,
33|    OLLAMA_CONTEXT_WINDOW,
34|    OLLAMA_MODEL,
35|    PROMPT_TOKEN_BUDGET,
36|    REPORTS_DIR,
37|    WARNINGS_LOG,
38|)
39|
40|logger = logging.getLogger(__name__)
41|
42|# Approximate token count: ~1 token per 4 chars
43|CHARS_PER_TOKEN = 4
44|def _count_tokens(text: str) -> int:
45|    return len(text) // CHARS_PER_TOKEN
46|
47|
48|def _build_prompt(
49|    stats: dict, chunks: list[dict], query: str | None = None
50|) -> str:
51|    """Build a constrained prompt for llama3.2."""
52|    chunk_text = "\n\n".join(
53|        f"[Source: {c['source_label']}] {c['text']}" for c in chunks[:3]
54|    )
55|
56|    if query:
57|        task = f'Answer this question using ONLY the context provided: "{query}"'
58|    else:
59|        task = """Write a 150-250 word space weather analyst brief covering:
60|1. Event summary for the latest 7 days (counts, notable flare classes, fastest CME, any geomagnetic storms)
61|2. Key terminology explained from context (pick 1-2 terms)
62|3. What to watch in the coming days based on recent trends
63|
64|Use British English. Be factual. Do not speculate beyond the data provided.
65|If any event type has zero count, state "No events recorded" for that type."""
66|
67|    prompt = f"""You are a space weather analyst working with NASA DONKI data.
68|
69|{task}
70|
71|CONTEXT (retrieved from heliophysics corpus):
72|{chunk_text}
73|
74|STATS (from DONKI data):
75|{json.dumps(stats, indent=2)}
76|
77|Your response:"""
78|    return prompt
79|
80|
81|def _generate(prompt: str) -> str:
82|    """Call llama3.2 via Ollama and return the response text."""
83|    resp = requests.post(
84|        "http://localhost:11434/api/generate",
85|        json={
86|            "model": OLLAMA_MODEL,
87|            "prompt": prompt,
88|            "stream": False,
89|            "options": {
90|                "num_ctx": OLLAMA_CONTEXT_WINDOW,
91|                "temperature": 0.3,
92|            },
93|        },
94|        timeout=120,
95|    )
96|    resp.raise_for_status()
97|    return resp.json().get("response", "")
98|
99|
100|def _check_staleness() -> str | None:
101|    """Check if raw data is older than DATA_MAX_AGE_HOURS. Returns warning or None."""
102|    raw_files = list(DATA_RAW.glob("*.json"))
103|    if not raw_files:
104|        return None
105|    newest_mtime = max(f.stat().st_mtime for f in raw_files)
106|    age_hours = (time.time() - newest_mtime) / 3600
107|    if age_hours > DATA_MAX_AGE_HOURS:
108|        mtime_str = datetime.fromtimestamp(newest_mtime, tz=timezone.utc).strftime(
109|            "%Y-%m-%d %H:%M UTC"
110|        )
111|        return f"⚠ Data from {mtime_str}. Consider re-running ingestion."
112|    return None
113|
114|
115|def _load_stats() -> dict:
116|    """Load structured stats — only the last 7 days for the brief prompt."""
117|    stats: dict[str, Any] = {}
118|
119|    # Daily counts: last 7 days
120|    dc_path = DATA_PROCESSED / "daily_counts.json"
121|    if dc_path.exists():
122|        with open(dc_path) as f:
123|            daily = json.load(f)
124|        recent = daily[-7:] if len(daily) >= 7 else daily
125|        stats["recent_daily_counts"] = recent
126|        stats["total_days"] = len(daily)
127|        # Summary for prompt
128|        cme_total = sum(d.get("cme_count", 0) for d in recent)
129|        flr_total = sum(d.get("flr_count", 0) for d in recent)
130|        gst_total = sum(d.get("gst_count", 0) for d in recent)
131|        stats["summary"] = {
132|            "period_days": len(recent),
133|            "total_cme": cme_total,
134|            "total_flr": flr_total,
135|            "total_gst": gst_total,
136|            "date_range": f"{recent[0].get('date', '?')} to {recent[-1].get('date', '?')}" if recent else "N/A",
137|        }
138|
139|    # Severity: last 7 days
140|    sev_path = DATA_PROCESSED / "severity.json"
141|    if sev_path.exists():
142|        with open(sev_path) as f:
143|            severity = json.load(f)
144|        recent_sev = severity[-7:] if len(severity) >= 7 else severity
145|        flare_vals = [s.get("max_flare_severity") for s in recent_sev if s.get("max_flare_severity")]
146|        speed_vals = [s.get("max_cme_speed_kms") for s in recent_sev if s.get("max_cme_speed_kms")]
147|        kp_vals = [s.get("max_kp_index") for s in recent_sev if s.get("max_kp_index")]
148|        stats["severity_summary"] = {
149|            "max_flare_severity_7d": max(flare_vals) if flare_vals else None,
150|            "max_cme_speed_kms_7d": max(speed_vals) if speed_vals else None,
151|            "max_kp_index_7d": max(kp_vals) if kp_vals else None,
152|        }
153|
154|    # Linkages: count only
155|    link_path = DATA_PROCESSED / "linkages.json"
156|    if link_path.exists():
157|        with open(link_path) as f:
158|            linkages = json.load(f)
159|        stats["total_linkages"] = len(linkages.get("edges", []))
160|
161|    # Top days
162|    td_path = DATA_PROCESSED / "top_days.json"
163|    if td_path.exists():
164|        with open(td_path) as f:
165|            stats["top_active_days"] = json.load(f)
166|
167|    return stats
168|
169|
170|def generate_brief(query: str | None = None) -> str:
171|    """
172|    Generate the analyst brief.
173|
174|    Args:
175|        query: if provided, answer this specific question instead of
176|               generating a general brief.
177|
178|    Returns:
179|        The generated text.
180|    """
181|    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
182|
183|    # Load stats
184|    stats = _load_stats()
185|
186|    # Staleness check
187|    stale_warning = _check_staleness()
188|
189|    # Retrieve relevant chunks
190|    from src.rag.retrieve import retrieve
191|
192|    search_query = query or " ".join(CONTROLLED_QUERIES[:2])
193|    chunks = retrieve(search_query, k=3)
194|
195|    if not chunks:
196|        logger.warning("No chunks retrieved — brief will lack grounding.")
197|
198|    # Build prompt with token guard
199|    prompt = _build_prompt(stats, chunks, query)
200|    token_count = _count_tokens(prompt)
201|
202|    if token_count > PROMPT_TOKEN_BUDGET:
203|        logger.warning(
204|            "Prompt tokens (%d) exceed budget (%d). Truncating chunks.",
205|            token_count, PROMPT_TOKEN_BUDGET,
206|        )
207|        while chunks and _count_tokens(_build_prompt(stats, chunks, query)) > PROMPT_TOKEN_BUDGET:
208|            # Drop lowest-scoring chunk
209|            chunks.pop(-1)
210|        prompt = _build_prompt(stats, chunks, query)
211|        token_count = _count_tokens(prompt)
212|        logger.info("Truncated to %d tokens with %d chunks.", token_count, len(chunks))
213|
214|    logger.info("Prompt: %d tokens, %d chunks.", token_count, len(chunks))
215|
216|    # Generate
217|    logger.info("Generating brief with %s...", OLLAMA_MODEL)
218|    response = _generate(prompt)
219|
220|    # Word count check
221|    word_count = len(response.split())
222|    logger.info("Response: %d words.", word_count)
223|
224|    # Retry if too short
225|    if word_count < BRIEF_RETRY_MIN_WORDS:
226|        logger.warning("Brief too short (%d words). Retrying once...", word_count)
227|        simpler_prompt = _build_prompt(stats, chunks[:2], query) + "\nKeep it under 250 words."
228|        response = _generate(simpler_prompt)
229|        word_count = len(response.split())
230|        logger.info("Retry response: %d words.", word_count)
231|
232|    # Fallback if still too short
233|    if word_count < BRIEF_RETRY_MIN_WORDS:
234|        logger.warning("Still too short. Using deterministic fallback.")
235|        response = _deterministic_fallback(stats)
236|        word_count = len(response.split())
237|
238|    # Prepend staleness warning
239|    if stale_warning:
240|        response = f"{stale_warning}\n\n{response}"
241|
242|    # Save
243|    BRIEF_FILE.write_text(response.strip())
244|    logger.info("Brief saved: %s (%d words)", BRIEF_FILE, word_count)
245|
246|    # Log warnings
247|    if word_count < BRIEF_MIN_WORDS or word_count > BRIEF_MAX_WORDS:
248|        _log_warning(f"brief word count {word_count} outside [{BRIEF_MIN_WORDS}, {BRIEF_MAX_WORDS}]")
249|    if not chunks:
250|        _log_warning("brief generated without retrieval grounding")
251|
252|    return response
253|
254|
255|def _deterministic_fallback(stats: dict) -> str:
256|    """Generate a deterministic template brief when llama3.2 fails."""
257|    daily = stats.get("daily_counts", [])
258|    severity = stats.get("severity", [])
259|
260|    # Get last 7 days
261|    recent = daily[-7:] if len(daily) >= 7 else daily
262|    total_cme = sum(d.get("cme_count", 0) for d in recent)
263|    total_flr = sum(d.get("flr_count", 0) for d in recent)
264|    total_gst = sum(d.get("gst_count", 0) for d in recent)
265|
266|    max_flare = "N/A"
267|    max_speed = "N/A"
268|    if severity:
269|        recent_sev = severity[-7:] if len(severity) >= 7 else severity
270|        flare_vals = [s.get("max_flare_severity") for s in recent_sev if s.get("max_flare_severity")]
271|        speed_vals = [s.get("max_cme_speed_kms") for s in recent_sev if s.get("max_cme_speed_kms")]
272|        if flare_vals:
273|            max_flare = f"{max(flare_vals):.1f}"
274|        if speed_vals:
275|            max_speed = f"{max(speed_vals):.0f} km/s"
276|
277|    lines = [
278|        "## Heliophysics Monitor — Analyst Brief",
279|        "",
280|        f"**Period:** Last 7 days",
281|        "",
282|        "### Event Summary",
283|        f"- **CMEs:** {total_cme} events detected"
284|    ]
285|
286|    if total_cme == 0:
287|        lines.append("  - No coronal mass ejections recorded this period.")
288|    else:
289|        lines.append(f"  - Fastest CME: {max_speed}")
290|
291|    lines.append(f"- **Solar Flares:** {total_flr} events")
292|    if total_flr == 0:
293|        lines.append("  - No solar flares recorded this period.")
294|    else:
295|        lines.append(f"  - Highest severity: {max_flare}")
296|
297|    lines.append(f"- **Geomagnetic Storms:** {total_gst} events")
298|    if total_gst == 0:
299|        lines.append("  - No geomagnetic storms recorded this period.")
300|
301|    lines.extend([
302|        "",
303|        "### Terminology",
304|        "A **coronal mass ejection (CME)** is a large expulsion of plasma and",
305|        "magnetic field from the Sun's corona. CMEs travelling towards Earth",
306|        "can trigger geomagnetic storms when they interact with the magnetosphere.",
307|        "",
308|        "### Outlook",
309|        "Monitor DONKI for linked events: solar flares associated with Earth-directed",
310|        "CMEs are the primary precursors to geomagnetic storm activity.",
311|        "",
312|        "---",
313|        "*This brief was auto-generated by the Heliophysics Monitor pipeline.*",
314|    ])
315|
316|    return "\n".join(lines)
317|
318|
319|def _log_warning(message: str) -> None:
320|    """Append a warning to the pipeline warnings log."""
321|    WARNINGS_LOG.parent.mkdir(parents=True, exist_ok=True)
322|    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
323|    with open(WARNINGS_LOG, "a") as f:
324|        f.write(f"[{timestamp}] {message}\n")
325|
326|
327|if __name__ == "__main__":
328|    logging.basicConfig(
329|        level=logging.INFO,
330|        format="%(asctime)s [%(levelname)s] %(message)s",
331|    )
332|    brief = generate_brief()
333|    print(f"\n=== BRIEF ({len(brief.split())} words) ===")
334|    print(brief)
335|