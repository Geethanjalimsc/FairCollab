"""Streamlit UI for FairCollab: run assessments, plus a Live Data sidebar section that fetches
GitHub/Form data into robotics_project.json and rebuilds the RAG index."""

import streamlit as st
import json
import os
import sys
import html
import time
import pandas as pd

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

# Loads Inter/JetBrains Mono and styles the custom badge/card/section-header
# components that native Streamlit theming (.streamlit/config.toml) can't reach.
_CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Space Grotesk', sans-serif;
}
code, pre, [data-testid="stCodeBlock"] {
    font-family: 'JetBrains Mono', monospace;
}

#MainMenu, footer {
    visibility: hidden;
}

.fc-brand {
    display: flex;
    align-items: center;
    gap: 0.55rem;
    font-size: 1.35rem;
    font-weight: 700;
    letter-spacing: -0.01em;
    color: #E2E8F0;
    margin-bottom: 0.15rem;
}
.fc-brand-mark {
    width: 0.65rem;
    height: 0.65rem;
    border-radius: 0.2rem;
    background: #818CF8;
    flex-shrink: 0;
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
    display: inline-flex;
    align-items: baseline;
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
.fc-rating-high { color: #39FF14; text-shadow: 0 0 8px rgba(57, 255, 20, 0.5); }
.fc-rating-medium { color: #FFEA00; text-shadow: 0 0 8px rgba(255, 234, 0, 0.5); }
.fc-rating-low { color: #FF1053; text-shadow: 0 0 8px rgba(255, 16, 83, 0.5); }
.fc-rating-unknown { color: #94A3B8; }

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
</style>
"""
st.markdown(_CUSTOM_CSS, unsafe_allow_html=True)

# CSS class per rating value, for the pill badge rendered by _render_rating_badge().
RATING_BADGE_CLASSES = {"High": "fc-rating-high", "Medium": "fc-rating-medium", "Low": "fc-rating-low", "Unknown": "fc-rating-unknown"}

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


def _render_rating_badge(rating: str) -> None:
    """Render the overall rating as plain neon-glow text (no pill background)."""
    badge_class = RATING_BADGE_CLASSES.get(rating, "fc-rating-unknown")
    st.markdown(
        f'<div class="fc-rating-badge">'
        f'<span class="fc-rating-label">Overall Rating</span>'
        f'<span class="fc-rating-value {badge_class}">{html.escape(rating)}</span>'
        f"</div>",
        unsafe_allow_html=True,
    )


def _render_evidence_card(index: int, page_content: str) -> None:
    """Render one evidence citation as a card with an [E#] tag inline before the text, no tag background."""
    st.markdown(
        f'<div class="fc-evidence-card">'
        f'<p class="fc-evidence-text"><span class="fc-evidence-tag">E{index}</span> {html.escape(page_content)}</p>'
        f"</div>",
        unsafe_allow_html=True,
    )

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
                    student[field_name] = per_student[name]  # replace, not append
            changed = True
    if project_name and project_name.strip() and project.get("project_name") != project_name.strip():
        project["project_name"] = project_name.strip()
        changed = True
    if changed:
        with open(ROBOTICS_PATH, "w", encoding="utf-8") as f:
            json.dump(project, f, indent=2, ensure_ascii=False)
        load_project_options.clear()
    return unmatched_names


def _with_hard_line_breaks(text: str) -> str:
    """Turn every single newline into a Markdown hard line break."""
    # Markdown collapses a lone "\n" into a space; two trailing spaces force a real break.
    return text.replace("\n", "  \n")


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
    st.markdown('<div class="fc-brand"><span class="fc-brand-mark"></span>FairCollab</div>', unsafe_allow_html=True)
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

        # --- Tab 2: Task Assignment Form ---
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

        # --- Tab 3: Peer Review Form ---
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
    st.header(f"{result['student_name']} — {result['project_name']}")

    rating = result.get("overall_rating", "Unknown")
    _render_rating_badge(rating)

    if result.get("max_revisions_reached"):
        st.warning(
            f"Maximum of {MAX_REVISIONS} revisions reached; this assessment is "
            f"returned best-effort with some validation concerns still unresolved."
        )

    st.write("")

    with st.expander("Integrity Check"):
        st.write(result["integrity_notes"])

    with st.expander("Evidence Retrieved"):
        for index, document in enumerate(result["evidence_docs"], start=1):
            _render_evidence_card(index, document.page_content)

    with st.expander("Factor Analysis"):
        st.markdown(_with_hard_line_breaks(result["factor_analysis"]))

    with st.expander("Overall Assessment"):
        st.markdown(_with_hard_line_breaks(result["overall_assessment"]))
