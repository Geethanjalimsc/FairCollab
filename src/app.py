"""Streamlit UI for FairCollab: run assessments, plus a Live Data sidebar section that fetches
GitHub/Form data into robotics_project.json and rebuilds the RAG index."""

import streamlit as st
import json
import os
import re
import sys
import html
import time
import pandas as pd
import plotly.graph_objects as go

# Puts project root on sys.path so `from src...` imports work under `streamlit run`.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agent import run_assessment_streaming, MAX_REVISIONS
from src.rag_pipeline import STUDENTS_PATH, RESEARCH_STUDENTS_PATH, ROBOTICS_PATH, build_and_save_index
from src.connectors.github_connector import fetch_and_map_commits, detect_contributors, GitHubConnectorError
from src.connectors.forms_connector import (
    fetch_task_assignments,
    fetch_peer_reviews,
    cross_reference_with_commits,
    FormsConnectorError,
)

st.set_page_config(page_title="FairCollab", page_icon=":material/balance:", layout="centered")

# SINGLE source of truth for rating-word coloring, reused by every place a rating (High/Medium/
# Low) renders: the top overall-rating badge, each factor card's at-a-glance header, and (via
# RATING_BADGE_CLASSES below) the CSS classes those two use. Values are the pre-existing neon
# colors the overall-rating badge already used (confirmed by reading the CSS before this change,
# not re-guessed) -- kept as-is rather than swapped for a different green/amber/rose so the top
# badge doesn't visibly change color out from under existing users.
RATING_COLORS = {"High": "#39FF14", "Medium": "#FFEA00", "Low": "#FF1053", "Unknown": "#94A3B8"}


def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    """Convert '#RRGGBB' to 'rgba(r, g, b, a)', so the CSS glow effect below derives from
    RATING_COLORS's hex values instead of a second, independently-hardcoded rgba() triplet."""
    hex_color = hex_color.lstrip("#")
    r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
    return f"rgba({r}, {g}, {b}, {alpha})"


def _material_icon(name: str, size_px: int = 18, color: str = "currentColor") -> str:
    """Build one Material Symbols Outlined icon span (see the stylesheet import + base
    .material-symbols-outlined rules in _CUSTOM_CSS). `name` is the icon's ligature name (e.g.
    "description", "forum", "warning" -- see Google's Material Symbols directory), not literal
    text. Replaces every emoji glyph used throughout the dashboard for a consistent,
    theme-matched icon system. color defaults to "currentColor" so each icon inherits whatever
    color its surrounding element already establishes (a severity-tinted flag row, the trust
    badge's green/amber, ...) instead of flattening those existing color systems to one fixed
    hue -- callers only pass an explicit color where none of that context exists (e.g. the
    evidence-card type icon, styled to match the adjacent E# tag's own color)."""
    return f'<span class="material-symbols-outlined" style="font-size:{size_px}px; color:{color};">{name}</span>'


# Loads Inter/JetBrains Mono and styles the custom badge/card/section-header
# components that native Streamlit theming (.streamlit/config.toml) can't reach.
_CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');
@import url('https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined:opsz,wght,FILL,GRAD@20..48,100..700,0..1,-50..200&display=block');

html, body, [class*="css"] {
    font-family: 'Space Grotesk', sans-serif;
}
code, pre, [data-testid="stCodeBlock"] {
    font-family: 'JetBrains Mono', monospace;
}

/* Base rules Google's own Material Symbols docs recommend -- the icon glyph is a font
   ligature keyed by the icon's name (e.g. "description", "forum"), not raw text, so these
   properties (feature-settings in particular) are required for the name to render as a glyph
   instead of literal letters. Used by _material_icon(), which replaces every emoji glyph
   elsewhere in this file with one of these spans. */
.material-symbols-outlined {
    font-family: 'Material Symbols Outlined';
    font-weight: normal;
    font-style: normal;
    display: inline-block;
    line-height: 1;
    white-space: nowrap;
    word-wrap: normal;
    direction: ltr;
    vertical-align: middle;
    -webkit-font-feature-settings: 'liga';
    font-feature-settings: 'liga';
}

#MainMenu, footer {
    visibility: hidden;
}

.fc-brand {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    margin-bottom: 0.15rem;
}
.fc-brand-word {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.35rem;
    font-weight: 700;
    letter-spacing: 0.03em;
    color: #E2E8F0;
}

/* Whole-app framing: a bordered panel around the main content area, and a
   strengthened separator on the sidebar's right edge. */
[data-testid="stMain"] {
    border: 1px solid #334155 !important;
    border-radius: 16px !important;
    padding: 1rem 1.25rem !important;
}
[data-testid="stSidebar"] {
    border-right: 1px solid #334155 !important;
}

[data-testid="stExpander"] {
    border: 1px solid #334155 !important;
    border-radius: 0.75rem !important;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.25);
    background: #1E293B;
    margin-bottom: 0.75rem;
}
[data-testid="stExpander"] summary {
    text-transform: uppercase;
    letter-spacing: 0.07em;
    font-weight: 600;
    font-size: 0.8rem;
    color: #818CF8;
}

.stButton > button {
    transition: transform 0.15s ease, box-shadow 0.15s ease;
}
.stButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 4px 10px rgba(129, 140, 246, 0.28);
}

.fc-rating-badge {
    display: flex;
    width: 100%;
    align-items: baseline;
    flex-wrap: wrap;
    gap: 0.7rem;
    margin: 0.5rem 0 1rem 0;
}
.fc-rating-label {
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.09em;
    text-transform: uppercase;
    color: #94A3B8;
}
.fc-rating-value {
    font-size: 2rem;
    font-weight: 800;
    letter-spacing: -0.01em;
}
.fc-rating-high { color: __RATING_HIGH__; text-shadow: 0 0 8px __RATING_HIGH_SHADOW__; }
.fc-rating-medium { color: __RATING_MEDIUM__; text-shadow: 0 0 8px __RATING_MEDIUM_SHADOW__; }
.fc-rating-low { color: __RATING_LOW__; text-shadow: 0 0 8px __RATING_LOW_SHADOW__; }
.fc-rating-unknown { color: __RATING_UNKNOWN__; }

.fc-evidence-card {
    background: #1E293B;
    border: 1px solid #334155;
    border-radius: 0.6rem;
    padding: 0.7rem 0.9rem;
    margin-bottom: 0.55rem;
}
.fc-evidence-tag {
    font-family: 'JetBrains Mono', monospace;
    font-weight: 700;
    color: #22D3EE;
}
.fc-evidence-meta {
    margin: 0 0 0.3rem 0;
    font-size: 0.78rem;
}
.fc-evidence-text {
    margin: 0;
    font-size: 0.92rem;
    line-height: 1.55;
    color: #E2E8F0;
}

/* Custom loading indicator: repurposes st.spinner()'s icon slot (a stable
   data-testid, unlike its auto-generated class names) into an indeterminate
   bicycle progress bar, applying to every st.spinner() call automatically. */
[data-testid="stSpinner"] > div {
    display: flex !important;
    flex-direction: column !important;
    align-items: center !important;
    gap: 0.6rem !important;
    padding: 0.6rem 0 !important;
}
[data-testid="stSpinnerIcon"] {
    width: 220px !important;
    height: 8px !important;
    border: none !important;
    border-radius: 999px !important;
    background: #334155 !important;
    position: relative !important;
    overflow: visible !important;
    animation: none !important;
}
[data-testid="stSpinnerIcon"]::before {
    content: "";
    position: absolute;
    top: 0;
    left: 0;
    height: 100%;
    width: 35%;
    border-radius: 999px;
    background: #818CF8;
    animation: fc-bar-fill-slide 1.6s ease-in-out infinite alternate;
}
[data-testid="stSpinnerIcon"]::after {
    content: "🚲";
    position: absolute;
    top: -11px;
    left: 35%;
    font-size: 15px;
    line-height: 1;
    animation: fc-bar-bike-slide 1.6s ease-in-out infinite alternate;
}
@keyframes fc-bar-fill-slide {
    0%   { left: 0%; }
    100% { left: 65%; }
}
@keyframes fc-bar-bike-slide {
    0%   { left: 35%; }
    100% { left: 100%; }
}

/* Determinate progress bar for Generate Assessment: fill width and bike position
   are set from real pipeline state (run_assessment_streaming), not animated. Only
   the wheels spin continuously -- realistic for a bike that's actually moving. */
.fc-progress-wrap {
    max-width: 360px;
    margin: 0.75rem auto 1.1rem auto;
}
.fc-progress-track {
    position: relative;
    width: 100%;
    height: 8px;
    border-radius: 999px;
    background: #334155;
    overflow: visible;
}
.fc-progress-fill {
    position: absolute;
    top: 0;
    left: 0;
    height: 100%;
    border-radius: 999px;
    background: #818CF8;
    transition: width 0.5s ease;
}
.fc-progress-bike {
    position: absolute;
    top: -19px;
    transform: translateX(-50%);
    color: #818CF8;
    transition: left 0.5s ease;
}
.fc-bike-svg {
    width: 34px;
    height: 21px;
    display: block;
}
.fc-bike-wheel {
    /* view-box (not fill-box) so the pixel transform-origin below lines up with
       each wheel's own cx/cy in SVG user-space, instead of its own bounding box. */
    transform-box: view-box;
    animation: fc-wheel-spin 0.7s linear infinite;
}
@keyframes fc-wheel-spin {
    from { transform: rotate(0deg); }
    to   { transform: rotate(360deg); }
}
.fc-progress-meta {
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    margin-top: 0.5rem;
    font-size: 0.85rem;
}
.fc-progress-pct {
    font-family: 'JetBrains Mono', monospace;
    font-weight: 700;
    color: #818CF8;
}
.fc-progress-label {
    color: #94A3B8;
}

/* App-boot overlay: covers the gap between Streamlit's shell connecting and this
   script's first run finishing (theme/CSS applied, sidebar built). Fixed + high
   z-index so it sits above the sidebar and everything else until cleared. */
.fc-boot-overlay {
    position: fixed;
    inset: 0;
    z-index: 999999;
    display: flex;
    align-items: center;
    justify-content: center;
    background: #0F172A;
}
.fc-boot-bike-svg {
    width: 104px;
    height: 65px;
    display: block;
    color: #818CF8;
}

.fc-trust-badge {
    font-size: 0.82rem;
    font-weight: 600;
    padding: 0.15rem 0.6rem;
    border-radius: 999px;
    display: inline-flex;
    align-items: center;
    gap: 0.3rem;
    white-space: nowrap;
}
.fc-trust-clean {
    color: #34D399;
    background: rgba(52, 211, 153, 0.12);
}
.fc-trust-flagged {
    color: #FBBF24;
    background: rgba(251, 191, 36, 0.12);
}

.fc-radar-caption {
    font-size: 0.82rem;
    color: #94A3B8;
    margin: -0.4rem 0 0.75rem 0;
}

.fc-headline {
    font-size: 1.05rem;
    font-weight: 500;
    color: #E2E8F0;
    line-height: 1.5;
    margin: 0.25rem 0 1.1rem 0;
}

/* Factor card top row (inside its st.container(border=True)): name on the left, colored rating
   word (+ confidence note) right-aligned on the same row -- st.expander's own label can only
   carry plain Markdown (bold/italic/links/images, no color), so the colored rating word has to
   live here instead of inside the expander's label text, with the expander placed directly
   below it inside the same bordered container so the whole thing reads as one card. */
.fc-factor-row {
    display: flex;
    align-items: baseline;
    justify-content: space-between;
    flex-wrap: wrap;
    gap: 0.5rem;
    margin: 0 0 0.6rem 0;
}
/* Reads as a clear section header now that the reasoning box sits directly below it, always
   visible -- bigger/bolder than body text, still Space Grotesk (no new font). */
.fc-factor-name {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.1rem;
    font-weight: 700;
    color: #E2E8F0;
}
.fc-factor-rating-group {
    display: flex;
    align-items: baseline;
    flex-wrap: wrap;
    gap: 0.4rem;
    font-size: 0.95rem;
    font-weight: 600;
}
.fc-factor-rating-word {
    font-weight: 700;
}
.fc-factor-confidence-note {
    font-size: 0.82rem;
    font-weight: 400;
    color: #94A3B8;
}

/* At-a-glance summary row: all six factors as small pills, shown before the detailed cards so
   the full picture is visible with zero clicks. */
.fc-glance-row {
    display: flex;
    flex-wrap: wrap;
    gap: 0.5rem;
    margin: 0.5rem 0 1.25rem 0;
}
.fc-glance-pill {
    background: #1E293B;
    border: 1px solid #334155;
    border-radius: 999px;
    padding: 0.3rem 0.75rem;
    font-size: 0.85rem;
    font-weight: 500;
    color: #94A3B8;
    white-space: nowrap;
}
.fc-glance-pill-rating {
    font-weight: 700;
}

/* Header: full-width left-aligned title that wraps naturally, rating badge left-aligned on its
   own row below it (not beside it) -- both share the same left edge. */
.fc-header-title {
    text-align: left;
    font-size: 2rem;
    font-weight: 700;
    line-height: 1.3;
    color: #E2E8F0;
    margin: 0.5rem 0 0.25rem 0;
    word-wrap: break-word;
}

.fc-insight-box {
    background: rgba(129, 140, 246, 0.10);
    border-left: 3px solid #818CF8;
    border-radius: 0.4rem;
    padding: 0.6rem 0.85rem;
    margin: 0.2rem 0 0.75rem 0;
    font-size: 0.92rem;
    line-height: 1.55;
    color: #E2E8F0;
}

/* Integrity flag rows -- severity-differentiated (LOW = neutral/grey, MEDIUM = amber). */
.fc-flag-row {
    display: flex;
    align-items: flex-start;
    gap: 0.6rem;
    padding: 0.6rem 0.8rem;
    border-radius: 0.5rem;
    margin-bottom: 0.5rem;
    border: 1px solid #334155;
}
.fc-flag-low {
    background: #1E293B;
    color: #94A3B8;
}
.fc-flag-medium {
    background: rgba(251, 191, 36, 0.08);
    border-color: rgba(251, 191, 36, 0.35);
    color: #E2E8F0;
}
.fc-flag-icon {
    line-height: 1.4;
    flex-shrink: 0;
}
.fc-flag-body {
    flex: 1;
    min-width: 0;
}
.fc-flag-checking {
    font-size: 0.72rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    color: #64748B;
    margin-bottom: 0.25rem;
}
.fc-flag-text {
    font-size: 0.9rem;
    line-height: 1.55;
}
.fc-flag-cross-tag {
    display: inline-flex;
    align-items: center;
    gap: 0.25rem;
    font-size: 0.76rem;
    font-weight: 600;
    color: #22D3EE;
    background: rgba(34, 211, 238, 0.12);
    padding: 0.08rem 0.5rem;
    border-radius: 999px;
    margin-left: 0.5rem;
    white-space: nowrap;
}
.fc-integrity-caveat {
    font-size: 0.8rem;
    color: #94A3B8;
    margin-top: 0.4rem;
}
</style>
"""
# Substituted (not an f-string) so the hundreds of literal CSS braces above don't all need
# escaping -- RATING_COLORS is still the only place these hex values are written down.
_CUSTOM_CSS = (
    _CUSTOM_CSS.replace("__RATING_HIGH__", RATING_COLORS["High"])
    .replace("__RATING_HIGH_SHADOW__", _hex_to_rgba(RATING_COLORS["High"], 0.5))
    .replace("__RATING_MEDIUM__", RATING_COLORS["Medium"])
    .replace("__RATING_MEDIUM_SHADOW__", _hex_to_rgba(RATING_COLORS["Medium"], 0.5))
    .replace("__RATING_LOW__", RATING_COLORS["Low"])
    .replace("__RATING_LOW_SHADOW__", _hex_to_rgba(RATING_COLORS["Low"], 0.5))
    .replace("__RATING_UNKNOWN__", RATING_COLORS["Unknown"])
)
st.markdown(_CUSTOM_CSS, unsafe_allow_html=True)

# CSS class per rating value, for the pill badge rendered by _render_rating_badge().
RATING_BADGE_CLASSES = {"High": "fc-rating-high", "Medium": "fc-rating-medium", "Low": "fc-rating-low", "Unknown": "fc-rating-unknown"}

# Numeric mapping for the radar chart's axes (Low=1, Medium=2, High=3 -- a full hexagon means
# High on every factor). Unrecognized ratings map to 0 so a parsing miss shows as an empty
# vertex rather than crashing.
FACTOR_RATING_VALUES = {"Low": 1, "Medium": 2, "High": 3}

# Icon per evidence record_type (from Document.metadata), distinguishing peer testimony from
# logged work at a glance inside factor cards. Values are Material Symbols Outlined ligature
# names (see _material_icon()), not emoji.
EVIDENCE_TYPE_ICONS = {"peer_feedback": "forum", "contribution": "description"}

# Human-readable label and target fill percentage per graph node, keyed to run_assessment_streaming()'s
# yielded "node" values. "done" is the generator's synthetic final sentinel, not a real graph node.
NODE_LABELS = {
    "retrieve_evidence": "Retrieving evidence",
    "integrity_analysis": "Checking integrity",
    "factor_analysis": "Analysing factors",
    "overall_assessment": "Synthesising assessment",
    "validate": "Validating claims",
}
NODE_PROGRESS = {
    "retrieve_evidence": 20,
    "integrity_analysis": 40,
    "factor_analysis": 60,
    "overall_assessment": 80,
    "validate": 95,
    "done": 100,
}

# A real inline SVG (not the CSS spinner's emoji glyph) so the two wheels are separate elements
# that .fc-bike-wheel's CSS animation can spin independently of the marker's position. Shared by
# the progress bar and the boot overlay below -- css_class controls size only, geometry is fixed.
def _bike_svg(css_class: str) -> str:
    """Build the inline bicycle SVG at a given CSS size class."""
    return (
        f'<svg viewBox="0 0 48 30" class="{css_class}" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">'
        '<path d="M9,23 L22,9 L31,9 L39,23 M22,9 L16,23 M25,9 L29,4 L34,4" fill="none" '
        'stroke="currentColor" stroke-width="2.3" stroke-linecap="round" stroke-linejoin="round"/>'
        '<g class="fc-bike-wheel" style="transform-origin:9px 23px;">'
        '<circle cx="9" cy="23" r="7" fill="none" stroke="currentColor" stroke-width="2"/>'
        '<line x1="9" y1="17" x2="9" y2="29" stroke="currentColor" stroke-width="1.3"/>'
        '<line x1="3" y1="23" x2="15" y2="23" stroke="currentColor" stroke-width="1.3"/>'
        "</g>"
        '<g class="fc-bike-wheel" style="transform-origin:39px 23px;">'
        '<circle cx="39" cy="23" r="7" fill="none" stroke="currentColor" stroke-width="2"/>'
        '<line x1="39" y1="17" x2="39" y2="29" stroke="currentColor" stroke-width="1.3"/>'
        '<line x1="33" y1="23" x2="45" y2="23" stroke="currentColor" stroke-width="1.3"/>'
        "</g>"
        "</svg>"
    )


_BIKE_SVG = _bike_svg("fc-bike-svg")

# Streamlit has no supported hook to inject HTML into index.html before its own JS mounts:
# `streamlit config show` has no such key, and index.html itself lives inside the installed
# package under venv/, not this project's source -- hand-editing it wouldn't survive a fresh
# `pip install` or reach anyone else who runs this app. So this covers the next best gap: the
# moment after Streamlit's shell has connected but before this script's first run has finished
# painting the real page (theme CSS, sidebar, data). st.session_state marks it done so later
# reruns (button clicks, widget changes) never show it again.
_BOOT_SPINNER_HTML = f'<div class="fc-boot-overlay">{_bike_svg("fc-boot-bike-svg")}</div>'
_boot_placeholder = st.empty()
if "_app_booted" not in st.session_state:
    _boot_placeholder.markdown(_BOOT_SPINNER_HTML, unsafe_allow_html=True)


def _render_progress_html(pct: int, label: str) -> str:
    """Build the determinate progress bar's HTML: a real inline width plus a bike marker at the fill edge."""
    pct = max(0, min(100, pct))
    return (
        f'<div class="fc-progress-wrap">'
        f'<div class="fc-progress-track">'
        f'<div class="fc-progress-fill" style="width:{pct}%;"></div>'
        f'<span class="fc-progress-bike" style="left:{pct}%;">{_BIKE_SVG}</span>'
        f"</div>"
        f'<div class="fc-progress-meta">'
        f'<span class="fc-progress-pct">{pct}%</span>'
        f'<span class="fc-progress-label">{html.escape(label)}</span>'
        f"</div>"
        f"</div>"
    )


def _render_rating_badge(rating: str, trust_html: str = "", justify: str = "flex-start") -> None:
    """Render the overall rating as plain neon-glow text (no pill background), optionally with
    a small integrity trust signal (trust_html, pre-built markup) directly next to it. justify
    is a CSS justify-content value ("flex-start"/"center"/"flex-end") -- .fc-rating-badge is a
    full-width flex row (not inline-flex), so this actually positions the badge within the page
    rather than just arranging its own children within a content-sized box."""
    badge_class = RATING_BADGE_CLASSES.get(rating, "fc-rating-unknown")
    st.markdown(
        f'<div class="fc-rating-badge" style="justify-content:{justify};">'
        f'<span class="fc-rating-label">Overall Rating</span>'
        f'<span class="fc-rating-value {badge_class}">{html.escape(rating)}</span>'
        f"{trust_html}"
        f"</div>",
        unsafe_allow_html=True,
    )


_CONTRIBUTION_CONTENT_RE = re.compile(
    r"^Student:\s*.+?\s*\|\s*Project:\s*.+?\s*\|\s*Date:\s*(?P<date>.+?)\s*\|\s*Type:\s*"
    r"(?P<type>.+?)\s*\|\s*Description:\s*(?P<description>.+)$"
)
_PEER_FEEDBACK_CONTENT_RE = re.compile(
    r"^Student:\s*.+?\s*\|\s*Project:\s*.+?\s*\|\s*Peer feedback:\s*(?P<feedback>.+)$"
)

# Matches the trailing "(complexity: X, status: Y, ...)" metadata parenthetical
# _contribution_to_document() (src/rag_pipeline.py) appends to Description -- only these 4
# known keys, in this fixed order, only the ones actually present on the record. Anchored at
# the end of the string and restricted to these exact keys (not "any trailing parenthetical")
# so a genuine descriptive parenthetical a student wrote themselves (e.g. "fixed a bug (see
# issue #42)") is never mistaken for metadata and stripped out.
_KNOWN_EXTRA_KEYS = ("complexity", "status", "role", "recipient")
_TRAILING_EXTRAS_RE = re.compile(
    r"^(?P<description>.*?)\s*\("
    r"(?P<extras>(?:(?:complexity|status|role|recipient):\s*[^,()]+)"
    r"(?:,\s*(?:complexity|status|role|recipient):\s*[^,()]+)*)"
    r"\)\s*$"
)


def _split_description_extras(description: str) -> tuple:
    """Split a contribution's Description text into the clean sentence and its trailing
    metadata parenthetical (complexity/status/role/recipient -- see _TRAILING_EXTRAS_RE).
    Returns (clean_description, {key: value, ...}) with only the keys actually present."""
    match = _TRAILING_EXTRAS_RE.match(description)
    if not match:
        return description.strip(), {}
    clean = match.group("description").strip()
    extras = {}
    for pair in match.group("extras").split(","):
        key, _, value = pair.partition(":")
        extras[key.strip()] = value.strip()
    return clean, extras


def _parse_evidence_content(page_content: str) -> dict:
    """Split a Document's page_content (see _contribution_to_document()/_feedback_to_document()
    in src/rag_pipeline.py) into the redundant "Student: X | Project: Y |" prefix -- dropped,
    since the whole page is already scoped to one student -- and the actually informative part.
    For contributions, complexity/status/role/recipient are further split out of the
    description into their own fields (see _split_description_extras()) rather than left as a
    trailing parenthetical buried in prose. Falls back to {"kind": "raw", "text": page_content}
    on any shape mismatch (a display-layer parse of a plain string, not a structured field), so
    an unexpected page_content shape still renders in full rather than silently dropping
    content."""
    contribution_match = _CONTRIBUTION_CONTENT_RE.match(page_content)
    if contribution_match:
        clean_description, extras = _split_description_extras(
            contribution_match.group("description").strip()
        )
        return {
            "kind": "contribution",
            "date": contribution_match.group("date").strip(),
            "type": contribution_match.group("type").strip(),
            "description": clean_description,
            "complexity": extras.get("complexity"),
            "status": extras.get("status"),
            "role": extras.get("role"),
            "recipient": extras.get("recipient"),
        }
    feedback_match = _PEER_FEEDBACK_CONTENT_RE.match(page_content)
    if feedback_match:
        return {"kind": "peer_feedback", "feedback": feedback_match.group("feedback").strip()}
    return {"kind": "raw", "text": page_content}


def _render_evidence_card(index: int, document) -> None:
    """Render one evidence citation as icon + a metadata line of small distinct tags ("E4 ·
    Documentation · Medium complexity · Completed · 2026-06-27", date always last) + the
    cleaned description/quote as the main text below -- not the raw page_content string, which
    repeats "Student: X | Project: Y |" on every card and buries complexity/status/role as a
    trailing parenthetical inside the sentence. Only tags for fields actually present on this
    record are rendered -- no empty/placeholder tags for missing fields."""
    icon_name = EVIDENCE_TYPE_ICONS.get(document.metadata.get("record_type"), "description")
    icon_html = _material_icon(icon_name, size_px=17, color="#22D3EE")
    parsed = _parse_evidence_content(document.page_content)

    if parsed["kind"] == "contribution":
        tags = [f"E{index}", parsed["type"].replace("_", " ").title()]
        if parsed["complexity"]:
            tags.append(f"{parsed['complexity'].title()} complexity")
        if parsed["status"]:
            tags.append(parsed["status"].title())
        if parsed["role"]:
            tags.append(parsed["role"].title())
        if parsed["recipient"]:
            tags.append(f"Re: {parsed['recipient']}")
        tags.append(parsed["date"])  # date always last
        meta_line = " · ".join(tags)
        body = html.escape(parsed["description"])
    elif parsed["kind"] == "peer_feedback":
        meta_line = f"E{index} · Peer feedback"
        # Quotation marks since this is someone's direct words, not a logged work record.
        body = f'&ldquo;{html.escape(parsed["feedback"])}&rdquo;'
    else:
        meta_line = f"E{index}"
        body = html.escape(parsed["text"])

    st.markdown(
        f'<div class="fc-evidence-card">'
        f'<p class="fc-evidence-meta">{icon_html} <span class="fc-evidence-tag">{html.escape(meta_line)}</span></p>'
        f'<p class="fc-evidence-text">{body}</p>'
        f"</div>",
        unsafe_allow_html=True,
    )

def _parse_factor_blocks(factor_analysis_text: str) -> list:
    """Parse the six "### Factor: ..." blocks out of node_factor_analysis()'s raw LLM output
    (see FACTOR_PROMPT in src/prompts.py for the exact structure each block follows: Rating/
    Confidence/Reasoning/Evidence lines). Returns a list of dicts in the order the LLM produced
    them (which follows the prompt's fixed six-factor order):
    {"name": str, "rating": "High"|"Medium"|"Low"|"Unknown", "confidence": "High"|"Medium"|"Low"|"Unknown",
     "reasoning": str, "evidence_indices": [int, ...]}

    Confidence's regex accepts "Medium" too even though FACTOR_PROMPT only allows High/Low --
    confirmed against real output (Student A / Live Data assessment) that the LLM sometimes
    writes "Confidence: Medium" despite the prompt's constraint. Silently degrading that to
    "Unknown" would have been actively wrong here, not just imprecise: it also broke
    _render_factor_card()'s auto-expand rule (which checks confidence == "Low"), so a
    genuinely equivocal factor would render collapsed by default -- the opposite of this
    section's purpose. auto-expand below therefore treats anything short of "High" as
    expand-worthy, not just a literal "Low".

    A block that doesn't match the expected shape degrades to "Unknown" fields rather than
    raising, since this is a display-layer parse of LLM text, not a strict schema.
    """
    factors = []
    blocks = re.split(r"(?=^### Factor:)", factor_analysis_text, flags=re.MULTILINE)
    for block in blocks:
        block = block.strip()
        if not block.startswith("### Factor:"):
            continue
        name_match = re.search(r"^### Factor:\s*(.+)$", block, re.MULTILINE)
        rating_match = re.search(r"^Rating:\s*(High|Medium|Low)", block, re.MULTILINE | re.IGNORECASE)
        confidence_match = re.search(r"^Confidence:\s*(High|Medium|Low)", block, re.MULTILINE | re.IGNORECASE)
        reasoning_match = re.search(r"^Reasoning:\s*(.+?)(?=^Evidence:|\Z)", block, re.MULTILINE | re.DOTALL)
        evidence_match = re.search(r"^Evidence:\s*(.+)", block, re.MULTILINE | re.DOTALL)
        evidence_str = evidence_match.group(1).strip() if evidence_match else ""
        factors.append(
            {
                "name": name_match.group(1).strip() if name_match else "Unknown Factor",
                "rating": rating_match.group(1).title() if rating_match else "Unknown",
                "confidence": confidence_match.group(1).title() if confidence_match else "Unknown",
                "reasoning": reasoning_match.group(1).strip() if reasoning_match else "",
                "evidence_indices": [int(n) for n in re.findall(r"\[E(\d+)\]", evidence_str)],
            }
        )
    return factors


def _extract_headline(overall_assessment_text: str) -> str:
    """Pull the first sentence of OVERALL_PROMPT's "Reasoning:" line -- no extra LLM call,
    text extraction only, per the explicit instruction not to spend another API call on this."""
    match = re.search(r"^Reasoning:\s*(.+)", overall_assessment_text, re.MULTILINE | re.DOTALL)
    if not match:
        return ""
    reasoning = match.group(1).strip()
    sentence_match = re.search(r"(.+?[.!?])(\s|$)", reasoning, re.DOTALL)
    return (sentence_match.group(1).strip() if sentence_match else reasoning)


# --- Integrity flag severity classification ------------------------------------------------
# NOTE: these patterns are matched against the EXACT flag wording node_integrity_analysis()
# produces in src/agent.py. If that wording ever changes, the patterns below must be updated
# to match -- this is a display-layer parse of a plain string, not a structured field, so there
# is no compiler/type-checker to catch drift here.

# Checking-purpose label per flag type -- stated once here (not duplicated across branches),
# since the weak and "none" corroboration tiers share the exact same underlying check and
# therefore the exact same "Checking:" text; only their finding text differs by confidence.
_CHECKING_CORROBORATION = "whether teammates independently confirm this claimed contribution"
_CHECKING_CROSS_STUDENT_DUPLICATE = (
    "whether this contribution looks copied from or coordinated with another student's work"
)
_CHECKING_MUTUAL_ISOLATION = (
    "whether these two students only vouch for each other with no outside confirmation"
)
_CHECKING_ZERO_CORROBORATORS = (
    "whether any teammate has independently mentioned or confirmed this student's contributions"
)
_CHECKING_COUNT_INVARIANT = "whether logged completion counts stay within what was actually assigned"
_CHECKING_WITHIN_STUDENT_DUPLICATE = (
    "whether this student's own contribution log contains duplicate entries"
)
_CHECKING_FALLBACK = "an automated pre-check"


def _classify_one_flag(text: str, student_name: str) -> dict:
    """Classify a single integrity flag line into a severity tier + icon + "Checking:" label +
    (if applicable) the other student it involves. Returns {"text": str, "severity":
    "low"|"medium", "icon": str, "checking": str, "cross_student": str or None}. "icon" is a
    Material Symbols Outlined ligature name (see _material_icon()), not an emoji glyph -- built
    into an actual icon span only at render time in _render_integrity_section(), so this
    function stays plain data."""
    if "has only weak, unverified support" in text:
        # Stage 1c "weak" tier: possible but unverified -- the only LOW-severity flag type.
        return {
            "text": text, "severity": "low", "icon": "help",
            "checking": _CHECKING_CORROBORATION, "cross_student": None,
        }

    if "closely resembles one logged by" in text:
        # Stage 2 cross-student duplicate-contribution flag.
        match = re.search(r"logged by ([^:]+):", text)
        other = match.group(1).strip() if match else None
        return {
            "text": text, "severity": "medium", "icon": "content_copy",
            "checking": _CHECKING_CROSS_STUDENT_DUPLICATE, "cross_student": other,
        }

    if "exclusively corroborate each other" in text:
        # Stage 4 mutual-isolation flag: agent.py always writes "{this student} and {other
        # student} exclusively corroborate...", with this student (the one being viewed) first.
        match = re.search(
            rf"^{re.escape(student_name)} and (.+?) exclusively corroborate each other", text
        )
        other = match.group(1).strip() if match else None
        return {
            "text": text, "severity": "medium", "icon": "sync_alt",
            "checking": _CHECKING_MUTUAL_ISOLATION, "cross_student": other,
        }

    if "no corroborating mentions from any other group member" in text:
        # Stage 4b zero-corroborators flag: about the viewed student's total isolation, not a
        # specific named pair -- no second student to extract despite being a coordination signal.
        return {
            "text": text, "severity": "medium", "icon": "person_off",
            "checking": _CHECKING_ZERO_CORROBORATORS, "cross_student": None,
        }

    if "exceeds" in text:
        # Count-invariant violation (tasks_completed > tasks_assigned, meetings_attended > total).
        return {
            "text": text, "severity": "medium", "icon": "rule",
            "checking": _CHECKING_COUNT_INVARIANT, "cross_student": None,
        }

    if "duplicate contribution description" in text:
        return {
            "text": text, "severity": "medium", "icon": "file_copy",
            "checking": _CHECKING_WITHIN_STUDENT_DUPLICATE, "cross_student": None,
        }

    if "no meaningful corroboration" in text:
        # Stage 1c "none" tier -- same underlying check as the "weak" tier above, hence the
        # same _CHECKING_CORROBORATION label; only the finding text differs by confidence.
        return {
            "text": text, "severity": "medium", "icon": "flag",
            "checking": _CHECKING_CORROBORATION, "cross_student": None,
        }

    # Fallback for any flag wording not matched above (e.g. peer_support pointing at a
    # nonexistent recipient) -- still surfaced, just without a specific icon/checking-purpose
    # signal beyond the generic one.
    return {
        "text": text, "severity": "medium", "icon": "warning",
        "checking": _CHECKING_FALLBACK, "cross_student": None,
    }


def _classify_integrity_flags(integrity_notes: str, student_name: str) -> list:
    """Split integrity_notes into its individual "- " bulleted flag lines and classify each.
    Returns [] for the clean state (integrity_notes has no bullet lines at all)."""
    flags = []
    for line in integrity_notes.split("\n"):
        line = line.strip()
        if not line.startswith("- "):
            continue
        flags.append(_classify_one_flag(line[2:], student_name))
    return flags


# Known technical -> plain-language substring rewrites, applied to a flag's text at RENDER time
# only -- node_integrity_analysis() in src/agent.py, and the integrity_notes string it produces,
# are never touched. Matched against the same exact flag wording _classify_one_flag() depends on
# (see that function's note about wording drift), so if agent.py's phrasing ever changes, these
# need updating too.
#
# Full-sentence grammar audit (this pass) against every real flag-producing scenario found the
# following, beyond the one previously-known zero-corroborators issue:
#
# - "peer_support claim of helping X" (both the weak- and "none"-tier corroboration flags) leaks
#   the raw snake_case field name "peer_support" into otherwise plain-language text -- not
#   something a person would write. New entry below rewrites this prefix on both flags.
# - The "none"-tier corroboration flag ("has no meaningful corroboration") had NO rewrite rule
#   at all -- it rendered fully raw/technical. New entry added, phrased as a "has X" -> "is Y"
#   swap (same convention as the weak-tier entry) so the transition from the "peer_support
#   claim..." prefix rewrite above reads as one clean sentence, not two mismatched clauses.
# - Both corroboration tiers can render "1 days away" (agent.py's days_desc hardcodes the
#   plural "days" regardless of count) -- a genuine number-agreement error a person would
#   never write. Can't fix at the source (agent.py) here, so _plain_language_flag() applies a
#   second, regex-based pass (_SINGULAR_DAY_RE) after the substring rewrites below, specifically
#   for the N=1 case (0 and 2+ are already correct plural "days").
# - Cross-student-duplicate and mutual-isolation flags were re-checked in full and read cleanly
#   as rewritten in the prior pass -- no further changes needed there.
# - The truncated quoted excerpts inside the corroboration and cross-student-duplicate flags
#   (agent.py hard-slices descriptions to a fixed character count, e.g. "...after gettin") are a
#   real readability rough edge but NOT fixable here without either touching agent.py (out of
#   scope) or fragile heuristic re-parsing of an already-truncated string with no reliable way
#   to tell "genuinely ends here" apart from "cut mid-word" -- left as-is, noted rather than
#   silently ignored.
#
# The 4th entry's matched phrase is deliberately longer than just "exclusively corroborate each
# other": agent.py's actual sentence continues "...each other with no independent mentions from
# any other group member", and replacing only the first few words would have left that trailing
# clause dangling after the plain-language replacement, duplicating what it already says.
_PLAIN_LANGUAGE_REWRITES = [
    ("peer_support claim of helping", "The claim of having helped"),
    (
        "no corroborating mentions from any other group member in peer_support claims or peer_feedback",
        "no teammate who has independently mentioned or confirmed their contributions",
    ),
    ("has only weak, unverified support", "is only loosely supported by the available evidence"),
    ("has no meaningful corroboration", "is not supported by any teammate's logged activity"),
    ("closely resembles one logged by", "looks very similar to one submitted by"),
    (
        "exclusively corroborate each other with no independent mentions from any other group member",
        "only vouch for each other, with no one else confirming either",
    ),
]

# Fixes "1 days away" -> "1 day away" (agent.py's days_desc hardcodes plural "days" regardless
# of count; 0 and 2+ are already grammatically correct plural, only N=1 needs singularizing).
_SINGULAR_DAY_RE = re.compile(r"\b1 days\b")


def _plain_language_flag(text: str) -> str:
    """Rewrite known technical phrasing in one flag's text to plain language. A flag whose
    wording doesn't match any known pattern passes through completely unchanged, rather than
    guessing at a rewrite."""
    for technical, plain in _PLAIN_LANGUAGE_REWRITES:
        text = text.replace(technical, plain)
    text = _SINGULAR_DAY_RE.sub("1 day", text)
    return text


# Matches a trailing " -- Group Project <year>" or " -- Group Research Report <year>" suffix
# (em-dash, then one of these two known literal labels, then a 4-digit year) on a project_name
# value -- e.g. data/students.json's "AI Web Application -- Group Project 2026" -> "AI Web
# Application", and data/research_students.json's "AI in Healthcare -- Group Research Report
# 2026" -> "AI in Healthcare". Deliberately narrow: only these two exact known suffixes, not a
# broad "any trailing label + year" catch-all -- robotics_project.json's "Intelligent Robotics
# Project" (no trailing year suffix at all) still doesn't match and is left completely
# unchanged, rather than guessing at a different truncation for a differently-worded suffix.
_PROJECT_NAME_SUFFIX_RE = re.compile(r"\s*—\s*Group (?:Project|Research Report)\s+\d{4}\s*$")


def _display_project_name(project_name: str) -> str:
    """Strip a trailing generic " -- Group Project <year>" / " -- Group Research Report <year>"
    suffix from project_name for display only -- never touches the stored project_name (data
    files, agent.py, result dict). A project_name that doesn't match either pattern is returned
    completely unchanged."""
    return _PROJECT_NAME_SUFFIX_RE.sub("", project_name)


def _render_radar_chart(factors: list) -> None:
    """Render the six-factor overview as a Plotly radar/spider chart (go.Scatterpolar), styled
    to match the app's existing dark theme/accent color rather than Streamlit's chart theme."""
    names = [f["name"] for f in factors]
    values = [FACTOR_RATING_VALUES.get(f["rating"], 0) for f in factors]
    ratings = [f["rating"] for f in factors]
    # Close the polygon by repeating the first vertex.
    theta = names + names[:1]
    r = values + values[:1]
    customdata = ratings + ratings[:1]

    fig = go.Figure()
    fig.add_trace(
        go.Scatterpolar(
            r=r,
            theta=theta,
            fill="toself",
            fillcolor="rgba(129, 140, 246, 0.35)",
            line=dict(color="#818CF8", width=2),
            marker=dict(color="#818CF8", size=6),
            customdata=customdata,
            hovertemplate="%{theta}: %{customdata}<extra></extra>",
        )
    )
    fig.update_layout(
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(
                visible=True,
                range=[0, 3],
                tickvals=[1, 2, 3],
                ticktext=["Low", "Medium", "High"],
                tickfont=dict(color="#94A3B8", size=10),
                gridcolor="#334155",
                linecolor="#334155",
            ),
            angularaxis=dict(
                tickfont=dict(color="#E2E8F0", size=11),
                gridcolor="#334155",
                linecolor="#334155",
            ),
        ),
        showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=50, r=50, t=30, b=30),
        height=380,
        font=dict(family="Space Grotesk, sans-serif"),
    )
    st.plotly_chart(fig, width="stretch", theme=None, config={"displayModeBar": False})
    st.markdown(
        f'<div class="fc-radar-caption">{_material_icon("info", size_px=14)} Each point is '
        "one of six contribution factors; a fuller shape means stronger, more even "
        "performance.</div>",
        unsafe_allow_html=True,
    )


def _render_at_a_glance_row(factors: list) -> None:
    """Compact horizontal row of small pills, one per factor (name + colored rating word, same
    RATING_COLORS mapping as everywhere else), shown above the six detailed cards so the full
    six-factor picture is visible with zero clicks. Wraps to multiple lines on narrow screens
    via flex-wrap (see .fc-glance-row), not a fixed-width layout."""
    pills = []
    for factor in factors:
        color = RATING_COLORS.get(factor["rating"], RATING_COLORS["Unknown"])
        pills.append(
            f'<span class="fc-glance-pill">{html.escape(factor["name"])}: '
            f'<span class="fc-glance-pill-rating" style="color:{color};">'
            f'{html.escape(factor["rating"])}</span></span>'
        )
    st.markdown(f'<div class="fc-glance-row">{"".join(pills)}</div>', unsafe_allow_html=True)


def _render_factor_card(factor: dict, evidence_docs: list, student_key: str) -> None:
    """Render one factor as a single bordered strip: a top row (name left, colored rating word
    right-aligned), the reasoning/insight box directly below it -- ALWAYS visible, not gated
    behind a click -- and, below that inside the SAME st.container(border=True), an expander
    holding only the evidence citations, labeled generically ("Evidence (N)") rather than
    repeating the factor name. One cohesive card, not two floating elements. Expanded-state
    defaults are computed once (the first time this factor is seen for this student_key) and
    then live entirely in st.session_state, so "Expand all"/"Collapse all" and the user's own
    clicks both just work across reruns without being overwritten by the auto-expand logic on
    every rerun -- it now governs whether the Evidence sub-section starts open, since the
    reasoning above it is unconditionally visible either way."""
    exp_key = f"fc_factor_exp::{student_key}::{factor['name']}"
    if exp_key not in st.session_state:
        # "Confidence is Low" per spec; also expands on "Medium" and "Unknown" (a parse miss)
        # since both mean the reviewer can't be sure this factor was rated with full
        # confidence -- collapsing those by default would hide exactly the content this
        # section exists to surface.
        st.session_state[exp_key] = factor["rating"] in ("Medium", "Low") or factor["confidence"] != "High"

    with st.container(border=True):
        # The rating word carries color directly (via RATING_COLORS -- the same mapping the top
        # overall-rating badge's CSS classes derive from), not a colored dot next to it:
        # st.expander labels can only hold plain Markdown (bold/italic/links/images -- no
        # color), so this row is rendered as real HTML rather than as the expander's own label
        # text. It stays visible whether the expander below it is open or closed, which is what
        # makes it an "at-a-glance" row in the first place.
        color = RATING_COLORS.get(factor["rating"], RATING_COLORS["Unknown"])
        confidence_note = (
            f'<span class="fc-factor-confidence-note">· {html.escape(factor["confidence"])} confidence</span>'
            if factor["confidence"] != "High"
            else ""
        )
        st.markdown(
            f'<div class="fc-factor-row">'
            f'<span class="fc-factor-name">{html.escape(factor["name"])}</span>'
            f'<span class="fc-factor-rating-group">'
            f'<span class="fc-factor-rating-word" style="color:{color};">{html.escape(factor["rating"])}</span>'
            f"{confidence_note}"
            f"</span>"
            f"</div>",
            unsafe_allow_html=True,
        )

        st.markdown(
            f'<div class="fc-insight-box">{html.escape(factor["reasoning"]) or "No reasoning provided."}</div>',
            unsafe_allow_html=True,
        )

        # on_change="rerun" is required here -- this Streamlit version's st.expander only
        # tracks expanded/collapsed state via st.session_state[key] when state-tracking is
        # turned on; with the default on_change="ignore" the key is inert and "Expand
        # all"/"Collapse all" (which write directly to st.session_state[exp_key]) would have no
        # visible effect.
        evidence_count = len(factor["evidence_indices"])
        with st.expander(
            f"Evidence ({evidence_count})", expanded=st.session_state[exp_key], key=exp_key, on_change="rerun"
        ):
            if not factor["evidence_indices"]:
                st.caption("No evidence cited for this factor.")
            for idx in factor["evidence_indices"]:
                if 1 <= idx <= len(evidence_docs):
                    _render_evidence_card(idx, evidence_docs[idx - 1])


def _render_integrity_section(integrity_notes: str, student_name: str, student_key: str) -> list:
    """Render the severity-aware integrity section as its own expander. Returns the parsed
    flags list so the caller can also drive the header's trust-signal count from it (same
    parse, not a second independent count)."""
    flags = _classify_integrity_flags(integrity_notes, student_name)
    exp_key = f"fc_integrity_exp::{student_key}"
    if exp_key not in st.session_state:
        st.session_state[exp_key] = any(flag["severity"] == "medium" for flag in flags)

    if flags:
        label = f"Integrity Check — {len(flags)} concern{'s' if len(flags) != 1 else ''} flagged"
        expander_icon = ":material/warning:"
    else:
        label = "Integrity Check — No concerns detected"
        expander_icon = ":material/check_circle:"

    # icon= is st.expander's own native mechanism for this (see the note above _render_factor_card
    # on why a colored/custom icon can't be embedded in the label text itself). See the matching
    # on_change="rerun" note in _render_factor_card() for why that's required here too.
    with st.expander(
        label, expanded=st.session_state[exp_key], key=exp_key, on_change="rerun", icon=expander_icon
    ):
        if not flags:
            st.markdown("No integrity concerns detected during automated pre-check.")
            st.markdown(
                '<div class="fc-integrity-caveat">Automated checks cover corroboration, '
                "duplication, and coordination patterns; not all manipulation types are "
                "detected.</div>",
                unsafe_allow_html=True,
            )
        else:
            for flag in flags:
                severity_class = "fc-flag-medium" if flag["severity"] == "medium" else "fc-flag-low"
                cross_tag = ""
                if flag["cross_student"]:
                    cross_tag = (
                        f'<span class="fc-flag-cross-tag">{_material_icon("group", size_px=13)} '
                        f'Also involves {html.escape(flag["cross_student"])}</span>'
                    )
                plain_text = _plain_language_flag(flag["text"])
                icon_html = _material_icon(flag["icon"], size_px=17)
                st.markdown(
                    f'<div class="fc-flag-row {severity_class}">'
                    f'<span class="fc-flag-icon">{icon_html}</span>'
                    f'<div class="fc-flag-body">'
                    f'<div class="fc-flag-checking">Checking: {html.escape(flag["checking"])}</div>'
                    f'<span class="fc-flag-text">{html.escape(plain_text)}{cross_tag}</span>'
                    f"</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
    return flags


LIVE_CONFIG_PATH = os.path.join(os.path.dirname(STUDENTS_PATH), "live_config.json")

# Deliberately excludes the GitHub PAT -- never write a credential to disk.
DEFAULT_LIVE_CONFIG = {
    "github_url": "",
    "github_email_mapping": [],
    "task_spreadsheet_id": "",
    "peer_spreadsheet_id": "",
    "service_account_path": "google_service_account.json",
    "project_name": "",
}


def _load_project(path: str) -> dict:
    """Read one contribution-log JSON file and return it as a dict."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def load_project_options() -> dict:
    """Load all three project files once and cache them across reruns."""
    return {
        "Software Project": _load_project(STUDENTS_PATH),
        "Research Project": _load_project(RESEARCH_STUDENTS_PATH),
        "Live Data": _load_project(ROBOTICS_PATH),
    }


def _load_live_config() -> dict:
    """Load data/live_config.json, falling back to defaults if it's missing or unreadable."""
    if not os.path.exists(LIVE_CONFIG_PATH):
        return dict(DEFAULT_LIVE_CONFIG)
    try:
        with open(LIVE_CONFIG_PATH, "r", encoding="utf-8") as f:
            loaded = json.load(f)
    except (OSError, json.JSONDecodeError):
        return dict(DEFAULT_LIVE_CONFIG)
    merged = dict(DEFAULT_LIVE_CONFIG)
    merged.update(loaded)
    return merged


def _save_live_config(config: dict) -> None:
    """Write the given config dict to data/live_config.json."""
    try:
        with open(LIVE_CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
    except OSError:
        pass  # best-effort autosave; ignore transient disk errors


def _persist_live_config(mapping_df: pd.DataFrame) -> None:
    """Save the Live Data section's current field values, matching DEFAULT_LIVE_CONFIG's schema."""
    config = {
        "github_url": st.session_state.get("github_repo_url", ""),
        "github_email_mapping": mapping_df.fillna("").to_dict(orient="records"),
        "task_spreadsheet_id": st.session_state.get("live_task_spreadsheet_id", ""),
        "peer_spreadsheet_id": st.session_state.get("live_peer_spreadsheet_id", ""),
        # Tabs 2 and 3 share one config slot; most recently touched wins.
        "service_account_path": st.session_state.get(
            "live_peer_service_account_path",
            st.session_state.get("live_service_account_path", DEFAULT_LIVE_CONFIG["service_account_path"]),
        ),
        "project_name": st.session_state.get("live_project_name", ""),
    }
    _save_live_config(config)


def _load_existing_contributions() -> dict:
    """Read robotics_project.json fresh and return {student: [contribution record, ...]}."""
    project = _load_project(ROBOTICS_PATH)
    return {student["student_name"]: student.get("contributions", []) for student in project["students"]}


def _replace_fields_in_robotics(updates_by_field: dict, project_name: str = None) -> set:
    """Overwrite (not merge) given fields on matching students in robotics_project.json; returns unmatched names."""
    all_names = set()
    for per_student in updates_by_field.values():
        all_names.update(per_student.keys())
    unmatched_names = set(all_names)
    project = _load_project(ROBOTICS_PATH)
    changed = False
    for student in project["students"]:
        name = student["student_name"]
        if name in all_names:
            unmatched_names.discard(name)
            for field_name, per_student in updates_by_field.items():
                if name in per_student:
                    student[field_name] = per_student[name]
            changed = True
    if project_name and project_name.strip() and project.get("project_name") != project_name.strip():
        project["project_name"] = project_name.strip()
        changed = True
    if changed:
        with open(ROBOTICS_PATH, "w", encoding="utf-8") as f:
            json.dump(project, f, indent=2, ensure_ascii=False)
        load_project_options.clear()
    return unmatched_names


def _is_quota_error(error: Exception) -> bool:
    """Best-effort check for whether an exception is a Gemini quota/rate-limit error."""
    message = str(error).upper()
    return "RESOURCE_EXHAUSTED" in message or "429" in message or "QUOTA" in message


def _mapping_df_to_dict(mapping_df: pd.DataFrame) -> dict:
    """Build the {email: student name} dict from the editable mapping table."""
    return {
        str(row["Email"]).strip(): str(row["Student Name"]).strip()
        for _, row in mapping_df.iterrows()
        if str(row["Email"]).strip() and str(row["Student Name"]).strip()
    }


# Streamlit persists session_state across reruns; seed defaults once per session.
for _key, _default in {
    "result": None,
    "error": None,
    "github_result": None,
    "github_unmatched": set(),
    "github_error": None,
    "task_result": None,
    "task_completed": None,
    "task_unmatched": set(),
    "task_error": None,
    "peer_result": None,
    "peer_unmatched": set(),
    "peer_error": None,
    "fetch_all_summary": None,
    "detected_contributors": None,
    "detect_error": None,
}.items():
    if _key not in st.session_state:
        st.session_state[_key] = _default

# Loaded once per rerun; seeds each Live Data widget's first-ever value.
_live_config = _load_live_config()

with st.sidebar:
    st.markdown(
        f'<div class="fc-brand">{_material_icon("balance", size_px=22, color="#818CF8")}'
        f'<span class="fc-brand-word">FairCollab</span></div>',
        unsafe_allow_html=True,
    )
    st.caption("AI-powered group contribution assessment")
    st.divider()

    project_options = load_project_options()
    project_type_label = st.selectbox("Project Type", options=list(project_options.keys()))
    selected_project = project_options[project_type_label]
    student_names = [student["student_name"] for student in selected_project["students"]]
    selected_student = st.selectbox("Student", options=student_names)

    st.write("")
    generate_clicked = st.button(
        "Generate Assessment", type="primary", width="stretch"
    )

    if project_type_label == "Live Data":
        st.divider()
        st.subheader("Live Data Sources")
        st.caption("Connect live sources for the robotics project below.")

        if "live_project_name" not in st.session_state:
            st.session_state["live_project_name"] = _live_config["project_name"]
        st.text_input("Project Name", key="live_project_name", placeholder="e.g. Autonomous Robotics Team")

        tab_github, tab_tasks, tab_peer = st.tabs(["GitHub", "Task Assignment", "Peer Review"])

        # --- Tab 1: GitHub -- writes to robotics_project.json, replacing contributions ---
        with tab_github:
            st.caption("Pulls commits from a GitHub repository via the GitHub REST API.")
            if "github_repo_url" not in st.session_state:
                st.session_state["github_repo_url"] = _live_config["github_url"]
            repo_url = st.text_input(
                "GitHub Repository URL",
                placeholder="https://github.com/owner/repo",
                key="github_repo_url",
            )
            pat = st.text_input(
                "Personal Access Token",
                type="password",  # never persisted to live_config.json either
                key="github_pat",
            )

            detect_clicked = st.button(
                "Detect Contributors", icon=":material/search:", width="stretch", key="detect_contributors_button"
            )
            if detect_clicked:
                st.session_state.detected_contributors = None
                st.session_state.detect_error = None
                if not repo_url.strip() or not pat.strip():
                    st.session_state.detect_error = "Please provide both a repository URL and a Personal Access Token."
                else:
                    with st.spinner("Detecting contributors..."):
                        try:
                            contributors = detect_contributors(repo_url.strip(), pat.strip())
                            st.session_state.detected_contributors = contributors
                            # Drop the editor's remembered state so it re-seeds from the fresh rows below.
                            st.session_state.pop("github_mapping_editor", None)
                        except GitHubConnectorError as exc:
                            st.session_state.detect_error = str(exc)
                        except Exception as exc:
                            st.session_state.detect_error = f"Unexpected error while detecting contributors: {exc}"

            if st.session_state.detect_error:
                st.error(st.session_state.detect_error)
            if st.session_state.detected_contributors:
                st.info(
                    f"Found {len(st.session_state.detected_contributors)} contributors. "
                    f"Assign each to a student name below."
                )
                # SelectboxColumn hides values outside its options -- this table shows the real names.
                st.dataframe(
                    pd.DataFrame(st.session_state.detected_contributors).rename(
                        columns={"email": "Email", "name": "GitHub Name"}
                    ),
                    width="stretch",
                    hide_index=True,
                )

            st.caption("Map each commit author's email to a student name:")
            student_name_options = ["Student A", "Student B", "Student C", "Student D"]
            detected = st.session_state.detected_contributors
            if detected:
                # Cycling A/B/C/D suggestion: always a valid dropdown value, never a repeat.
                default_mapping = pd.DataFrame(
                    {
                        "Email": [c["email"] for c in detected],
                        "Student Name": [
                            student_name_options[i % len(student_name_options)] for i in range(len(detected))
                        ],
                    }
                )
            else:
                saved_mapping = _live_config.get("github_email_mapping") or []
                if saved_mapping:
                    default_mapping = pd.DataFrame(saved_mapping)
                else:
                    default_mapping = pd.DataFrame({"Email": ["", "", "", ""], "Student Name": ["", "", "", ""]})
            mapping_df = st.data_editor(
                default_mapping,
                num_rows="dynamic",
                width="stretch",
                key="github_mapping_editor",
                column_config={
                    "Student Name": st.column_config.SelectboxColumn(
                        "Student Name",
                        options=student_name_options,
                    )
                },
            )
            fetch_clicked = st.button("Fetch Commits", width="stretch", key="fetch_commits_button")

            if fetch_clicked:
                st.session_state.github_result = None
                st.session_state.github_unmatched = set()
                st.session_state.github_error = None
                student_mapping = _mapping_df_to_dict(mapping_df)
                if not repo_url.strip() or not pat.strip():
                    st.session_state.github_error = "Please provide both a repository URL and a Personal Access Token."
                elif not student_mapping:
                    st.session_state.github_error = "Please fill in at least one email -> student name mapping row."
                else:
                    with st.spinner("Fetching commits from GitHub..."):
                        try:
                            fetched = fetch_and_map_commits(repo_url.strip(), pat.strip(), student_mapping)
                            unmatched = _replace_fields_in_robotics(
                                {"contributions": fetched},
                                project_name=st.session_state.get("live_project_name"),
                            )
                            st.session_state.github_result = fetched
                            st.session_state.github_unmatched = unmatched
                        except GitHubConnectorError as exc:
                            st.session_state.github_error = str(exc)
                        except Exception as exc:
                            st.session_state.github_error = f"Unexpected error while fetching commits: {exc}"

            if st.session_state.github_error:
                st.error(st.session_state.github_error)
            if st.session_state.github_result:
                total_commits = sum(len(records) for records in st.session_state.github_result.values())
                st.success(f"Fetched and replaced {total_commits} commit(s) for {len(st.session_state.github_result)} student(s).")
                st.dataframe(
                    pd.DataFrame(
                        [{"Student": name, "Commits": len(records)} for name, records in st.session_state.github_result.items()]
                    ),
                    width="stretch",
                    hide_index=True,
                )
                if st.session_state.github_unmatched:
                    st.warning(
                        "These mapped names don't match any student in the Live Data roster: "
                        + ", ".join(sorted(st.session_state.github_unmatched))
                    )

        with tab_tasks:
            st.caption("Reads task-assignment responses from a Google Form's linked spreadsheet.")
            if "live_task_spreadsheet_id" not in st.session_state:
                st.session_state["live_task_spreadsheet_id"] = _live_config["task_spreadsheet_id"]
            task_spreadsheet_id = st.text_input(
                "Task Assignment Spreadsheet ID",
                placeholder="paste spreadsheet ID here",
                key="live_task_spreadsheet_id",
            )
            if "live_service_account_path" not in st.session_state:
                st.session_state["live_service_account_path"] = _live_config["service_account_path"]
            task_service_account_path = st.text_input(
                "Service Account JSON path",
                key="live_service_account_path",
                help="JSON file in your project folder",
            )
            fetch_tasks_clicked = st.button("Fetch Task Assignments", width="stretch", key="fetch_tasks_button")

            if fetch_tasks_clicked:
                st.session_state.task_result = None
                st.session_state.task_completed = None
                st.session_state.task_unmatched = set()
                st.session_state.task_error = None
                if not task_spreadsheet_id.strip():
                    st.session_state.task_error = "Please provide the Task Assignment spreadsheet ID."
                elif not task_service_account_path.strip():
                    st.session_state.task_error = "Please provide the service account JSON path."
                else:
                    with st.spinner("Fetching task assignments..."):
                        try:
                            task_assignments, submitted_students = fetch_task_assignments(
                                task_spreadsheet_id.strip(), task_service_account_path.strip()
                            )
                            # Cross-reference against whatever GitHub data is already stored.
                            existing_contributions = _load_existing_contributions()
                            tasks_completed = cross_reference_with_commits(task_assignments, existing_contributions)
                            unmatched = _replace_fields_in_robotics(
                                {
                                    "tasks_assigned": {
                                        name: data["tasks_assigned"] for name, data in task_assignments.items()
                                    },
                                    "task_descriptions": {
                                        name: data["task_descriptions"] for name, data in task_assignments.items()
                                    },
                                    "tasks_completed": tasks_completed,
                                    "task_form_submitted": {name: True for name in submitted_students},
                                },
                                project_name=st.session_state.get("live_project_name"),
                            )
                            st.session_state.task_result = task_assignments
                            st.session_state.task_completed = tasks_completed
                            st.session_state.task_unmatched = unmatched
                        except FormsConnectorError as exc:
                            st.session_state.task_error = str(exc)
                        except Exception as exc:
                            st.session_state.task_error = f"Unexpected error while fetching task assignments: {exc}"

            if st.session_state.task_error:
                st.error(st.session_state.task_error)
            if st.session_state.task_result:
                total_tasks = sum(data["tasks_assigned"] for data in st.session_state.task_result.values())
                st.success(f"Found {total_tasks} tasks for {len(st.session_state.task_result)} students")
                st.dataframe(
                    pd.DataFrame(
                        [
                            {
                                "Student": name,
                                "Assigned": data["tasks_assigned"],
                                "Completed": st.session_state.task_completed.get(name, 0),
                            }
                            for name, data in st.session_state.task_result.items()
                        ]
                    ),
                    width="stretch",
                    hide_index=True,
                )
                if st.session_state.task_unmatched:
                    st.warning(
                        "These names don't match any student in the Live Data roster: "
                        + ", ".join(sorted(st.session_state.task_unmatched))
                    )

        with tab_peer:
            st.caption("Reads peer-review responses from a second Google Form's linked spreadsheet.")
            if "live_peer_spreadsheet_id" not in st.session_state:
                st.session_state["live_peer_spreadsheet_id"] = _live_config["peer_spreadsheet_id"]
            peer_spreadsheet_id = st.text_input(
                "Peer Review Spreadsheet ID",
                placeholder="paste spreadsheet ID here",
                key="live_peer_spreadsheet_id",
            )
            if "live_peer_service_account_path" not in st.session_state:
                st.session_state["live_peer_service_account_path"] = _live_config["service_account_path"]
            peer_service_account_path = st.text_input(
                "Service Account JSON path",
                key="live_peer_service_account_path",
                help="JSON file in your project folder",
            )
            fetch_peer_clicked = st.button("Fetch Peer Reviews", width="stretch", key="fetch_peer_button")

            if fetch_peer_clicked:
                st.session_state.peer_result = None
                st.session_state.peer_unmatched = set()
                st.session_state.peer_error = None
                if not peer_spreadsheet_id.strip():
                    st.session_state.peer_error = "Please provide the Peer Review spreadsheet ID."
                elif not peer_service_account_path.strip():
                    st.session_state.peer_error = "Please provide the service account JSON path."
                else:
                    with st.spinner("Fetching peer reviews..."):
                        try:
                            peer_reviews = fetch_peer_reviews(
                                peer_spreadsheet_id.strip(), peer_service_account_path.strip()
                            )
                            unmatched = _replace_fields_in_robotics(
                                {"peer_feedback": peer_reviews},
                                project_name=st.session_state.get("live_project_name"),
                            )
                            st.session_state.peer_result = peer_reviews
                            st.session_state.peer_unmatched = unmatched
                        except FormsConnectorError as exc:
                            st.session_state.peer_error = str(exc)
                        except Exception as exc:
                            st.session_state.peer_error = f"Unexpected error while fetching peer reviews: {exc}"

            if st.session_state.peer_error:
                st.error(st.session_state.peer_error)
            if st.session_state.peer_result:
                total_reviews = sum(len(reviews) for reviews in st.session_state.peer_result.values())
                st.success(f"Found {total_reviews} reviews for {len(st.session_state.peer_result)} students")
                st.dataframe(
                    pd.DataFrame(
                        [{"Student": name, "Reviews": len(reviews)} for name, reviews in st.session_state.peer_result.items()]
                    ),
                    width="stretch",
                    hide_index=True,
                )
                if st.session_state.peer_unmatched:
                    st.warning(
                        "These names don't match any student in the Live Data roster: "
                        + ", ".join(sorted(st.session_state.peer_unmatched))
                    )

        # Autosave now that every field above holds this rerun's live value.
        _persist_live_config(mapping_df)

        st.divider()
        fetch_all_clicked = st.button(
            "Fetch All Data Sources",
            icon=":material/sync:",
            type="primary",
            width="stretch",
            key="fetch_all_button",
        )

        if fetch_all_clicked:
            summary_lines = []
            any_success = False
            with st.spinner("Fetching all data sources... this can take a while."):
                if repo_url.strip():
                    try:
                        student_mapping = _mapping_df_to_dict(mapping_df)
                        fetched = fetch_and_map_commits(repo_url.strip(), pat.strip(), student_mapping)
                        _replace_fields_in_robotics({"contributions": fetched})
                        total_commits = sum(len(records) for records in fetched.values())
                        summary_lines.append(f"GitHub: {total_commits} commit(s) for {len(fetched)} student(s).")
                        any_success = True
                    except GitHubConnectorError as exc:
                        summary_lines.append(f"GitHub: failed -- {exc}")
                    except Exception as exc:
                        summary_lines.append(f"GitHub: unexpected error -- {exc}")
                else:
                    summary_lines.append("GitHub: skipped (no repository URL provided).")

                if task_spreadsheet_id.strip():
                    try:
                        task_assignments, submitted_students = fetch_task_assignments(
                            task_spreadsheet_id.strip(), task_service_account_path.strip()
                        )
                        # Reads fresh from disk, so this picks up step 1's commits if it ran.
                        existing_contributions = _load_existing_contributions()
                        tasks_completed = cross_reference_with_commits(task_assignments, existing_contributions)
                        _replace_fields_in_robotics(
                            {
                                "tasks_assigned": {
                                    name: data["tasks_assigned"] for name, data in task_assignments.items()
                                },
                                "task_descriptions": {
                                    name: data["task_descriptions"] for name, data in task_assignments.items()
                                },
                                "tasks_completed": tasks_completed,
                                "task_form_submitted": {name: True for name in submitted_students},
                            }
                        )
                        total_tasks = sum(data["tasks_assigned"] for data in task_assignments.values())
                        summary_lines.append(f"Tasks: {total_tasks} task(s) for {len(task_assignments)} student(s).")
                        any_success = True
                    except FormsConnectorError as exc:
                        summary_lines.append(f"Tasks: failed -- {exc}")
                    except Exception as exc:
                        summary_lines.append(f"Tasks: unexpected error -- {exc}")
                else:
                    summary_lines.append("Tasks: skipped (no spreadsheet ID provided).")

                if peer_spreadsheet_id.strip():
                    try:
                        peer_reviews = fetch_peer_reviews(
                            peer_spreadsheet_id.strip(), peer_service_account_path.strip()
                        )
                        _replace_fields_in_robotics({"peer_feedback": peer_reviews})
                        total_reviews = sum(len(reviews) for reviews in peer_reviews.values())
                        summary_lines.append(f"Peer reviews: {total_reviews} review(s) for {len(peer_reviews)} student(s).")
                        any_success = True
                    except FormsConnectorError as exc:
                        summary_lines.append(f"Peer reviews: failed -- {exc}")
                    except Exception as exc:
                        summary_lines.append(f"Peer reviews: unexpected error -- {exc}")
                else:
                    summary_lines.append("Peer reviews: skipped (no spreadsheet ID provided).")

                if any_success:
                    try:
                        build_and_save_index()
                        summary_lines.append("RAG index: rebuilt successfully.")
                    except Exception as exc:
                        summary_lines.append(f"RAG index: rebuild failed -- {exc}")
                else:
                    summary_lines.append("RAG index: skipped (nothing new was fetched).")

            st.session_state.fetch_all_summary = summary_lines
            _replace_fields_in_robotics({}, project_name=st.session_state.get("live_project_name"))
            _persist_live_config(mapping_df)

        if st.session_state.fetch_all_summary:
            st.success("Fetch All complete:\n" + "\n".join(f"- {line}" for line in st.session_state.fetch_all_summary))

# Sidebar (and everything above) has now rendered -- drop the boot overlay and never show it again.
_boot_placeholder.empty()
st.session_state["_app_booted"] = True

if generate_clicked:
    st.session_state.result = None
    st.session_state.error = None
    progress_slot = st.empty()
    displayed_pct = 0
    try:
        # project_id scopes retrieval, since every project reuses "Student A"/"B"/"C"/"D".
        for update in run_assessment_streaming(selected_student, project_id=selected_project["project_id"]):
            node = update["node"]
            if node == "done":
                progress_slot.markdown(_render_progress_html(100, "Complete"), unsafe_allow_html=True)
                time.sleep(0.4)  # briefly show 100% before the result replaces it
                st.session_state.result = update["final_state"]
                break
            revision_count = update["chunk"].get("revision_count", 0)
            # Revisions loop back to factor_analysis, whose target% is lower than validate's --
            # never let the displayed number regress, only the label communicates the retry.
            displayed_pct = max(displayed_pct, NODE_PROGRESS.get(node, displayed_pct))
            if node == "factor_analysis" and revision_count > 0:
                label = f"Revising analysis (attempt {revision_count + 1} of {MAX_REVISIONS + 1})"
            else:
                label = NODE_LABELS.get(node, node)
            progress_slot.markdown(_render_progress_html(displayed_pct, label), unsafe_allow_html=True)
    except Exception as exc:
        if _is_quota_error(exc):
            st.session_state.error = (
                "Gemini API quota exceeded. Please wait a minute and try again, "
                "or check your plan/billing at "
                "https://ai.google.dev/gemini-api/docs/rate-limits."
            )
        else:
            st.session_state.error = f"Assessment failed: {exc}"
    finally:
        progress_slot.empty()

if st.session_state.error:
    st.error(st.session_state.error)

if st.session_state.result:
    result = st.session_state.result
    student_name = result["student_name"]
    # Scopes session_state keys to this specific (project, student) pair, not just student_name --
    # several demo projects in this codebase deliberately reuse "Student A"/"B"/"C"/"D" (see
    # _find_student()'s docstring in src/agent.py), so student_name alone would let two different
    # students' expand/collapse state leak into each other across projects.
    student_key = f"{result.get('project_name', '')}::{student_name}"
    rating = result.get("overall_rating", "Unknown")

    # Parsed once up front (before any widget is created) so the "Expand all"/"Collapse all"
    # buttons below can set every relevant session_state key before its widget is instantiated.
    # Reused verbatim by the integrity section itself, so the header's "N concerns" count and
    # the section's own flag list never disagree from being parsed two different ways.
    factors = _parse_factor_blocks(result["factor_analysis"])
    flags_preview = _classify_integrity_flags(result["integrity_notes"], student_name)

    # 1. HEADER -- student/project title full-width and left-aligned (wraps naturally across
    # lines), rating + integrity trust signal left-aligned on their OWN row below it, not
    # beside it -- both rows share the same left edge.
    display_project_name = _display_project_name(result["project_name"])
    st.markdown(
        f'<div class="fc-header-title">{html.escape(student_name)} — {html.escape(display_project_name)}</div>',
        unsafe_allow_html=True,
    )
    if not flags_preview:
        trust_html = (
            f'<span class="fc-trust-badge fc-trust-clean">{_material_icon("check_circle", size_px=15)} '
            f"No integrity concerns</span>"
        )
    else:
        trust_html = (
            f'<span class="fc-trust-badge fc-trust-flagged">{_material_icon("warning", size_px=15)} '
            f'{len(flags_preview)} concern{"s" if len(flags_preview) != 1 else ""} flagged</span>'
        )
    _render_rating_badge(rating, trust_html=trust_html, justify="flex-start")

    if result.get("max_revisions_reached"):
        st.warning(
            f"Maximum of {MAX_REVISIONS} revisions reached; this assessment is "
            f"returned best-effort with some validation concerns still unresolved."
        )

    # 2. RADAR CHART -- six-factor overview.
    if factors:
        _render_radar_chart(factors)

    # 3. ONE-LINE HEADLINE SUMMARY -- extracted from OVERALL_PROMPT's Reasoning line, no extra LLM call.
    headline = _extract_headline(result["overall_assessment"])
    if headline:
        st.markdown(f'<div class="fc-headline">{html.escape(headline)}</div>', unsafe_allow_html=True)

    # AT-A-GLANCE ROW -- all six factors' names + colored ratings, zero clicks required, shown
    # before the six detailed (collapsible) cards below.
    if factors:
        _render_at_a_glance_row(factors)

    # Collect every expander key up front so "Expand all"/"Collapse all" can drive both the
    # factor cards AND the integrity section below, regardless of render order.
    all_exp_keys = [f"fc_factor_exp::{student_key}::{f['name']}" for f in factors]
    all_exp_keys.append(f"fc_integrity_exp::{student_key}")

    # 4. SIX FACTOR CARDS -- with manual expand-all/collapse-all override.
    toggle_col1, toggle_col2, _spacer = st.columns([1, 1, 3])
    with toggle_col1:
        if st.button("Expand all", width="stretch", key="expand_all_button"):
            for key in all_exp_keys:
                st.session_state[key] = True
    with toggle_col2:
        if st.button("Collapse all", width="stretch", key="collapse_all_button"):
            for key in all_exp_keys:
                st.session_state[key] = False

    for factor in factors:
        _render_factor_card(factor, result["evidence_docs"], student_key)

    st.write("")

    # 5. INTEGRITY CHECK -- severity-aware, not a flat bulleted list.
    _render_integrity_section(result["integrity_notes"], student_name, student_key)
