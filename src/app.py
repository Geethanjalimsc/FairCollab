"""
Streamlit web interface for FairCollab.

Lets a user pick a project type and a student from the sidebar, click
"Generate Assessment", and see the five-node LangGraph agent's result
(src/agent.py's run_assessment()) laid out in the main area: an overall
rating badge plus four expandable sections for integrity, evidence, factor
analysis, and the overall reasoning.

Also hosts the "Live Data" project type: a sidebar section with three tabs
(GitHub, Task Assignment Form, Peer Review Form) that fetch real data via
src/connectors and write it straight into data/robotics_project.json, plus
a "Fetch All" button that runs every source and rebuilds the RAG index.
Every field in that section is auto-saved to data/live_config.json so the
values survive an app restart.
"""

import streamlit as st
import json
import os
import sys
import pandas as pd

# `streamlit run src/app.py` puts this file's own directory on sys.path
# instead of the project root, so "from src..." imports below would fail
# without this.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agent import run_assessment, MAX_REVISIONS
from src.rag_pipeline import STUDENTS_PATH, RESEARCH_STUDENTS_PATH, ROBOTICS_PATH, build_and_save_index
from src.connectors.github_connector import fetch_and_map_commits, detect_contributors, GitHubConnectorError
from src.connectors.forms_connector import (
    fetch_task_assignments,
    fetch_peer_reviews,
    cross_reference_with_commits,
    FormsConnectorError,
)

st.set_page_config(page_title="FairCollab", page_icon="⚖️", layout="centered")

RATING_COLORS = {"High": "green", "Medium": "orange", "Low": "red", "Unknown": "gray"}

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
        # Tabs 2 and 3 each have their own field but share one config slot;
        # whichever was touched most recently wins.
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
    """Overwrite (not merge) one or more fields on matching students in robotics_project.json.

    updates_by_field: {field_name: {student_name: new_value}}.
    Returns the set of student names that didn't match any student in
    robotics_project.json, so the caller can warn about them.
    """
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
    st.title("⚖️ FairCollab")
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

        tab_github, tab_tasks, tab_peer = st.tabs(["🐙 GitHub", "📋 Task Assignment Form", "📝 Peer Review Form"])

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

            detect_clicked = st.button("🔍 Detect Contributors", width="stretch", key="detect_contributors_button")
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
                # Read-only reference table: a SelectboxColumn can't display a
                # raw name outside its options, so this is how the user sees who's who.
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
                            task_assignments = fetch_task_assignments(
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
            "🔄 Fetch All Data Sources", type="primary", width="stretch", key="fetch_all_button"
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
                        task_assignments = fetch_task_assignments(
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

if generate_clicked:
    st.session_state.result = None
    st.session_state.error = None
    with st.spinner(f"Assessing {selected_student}... this runs multiple AI calls and may take a minute."):
        try:
            # project_id scopes retrieval, since every project reuses "Student A"/"B"/"C"/"D".
            st.session_state.result = run_assessment(selected_student, project_id=selected_project["project_id"])
        except Exception as exc:
            if _is_quota_error(exc):
                st.session_state.error = (
                    "Gemini API quota exceeded. Please wait a minute and try again, "
                    "or check your plan/billing at "
                    "https://ai.google.dev/gemini-api/docs/rate-limits."
                )
            else:
                st.session_state.error = f"Assessment failed: {exc}"

if st.session_state.error:
    st.error(st.session_state.error)

if st.session_state.result:
    result = st.session_state.result
    st.header(f"{result['student_name']} — {result['project_name']}")

    rating = result.get("overall_rating", "Unknown")
    color = RATING_COLORS.get(rating, "gray")
    st.markdown(f"## Overall Rating: :{color}[{rating}]")

    if result.get("max_revisions_reached"):
        st.warning(
            f"Maximum of {MAX_REVISIONS} revisions reached; this assessment is "
            f"returned best-effort with some validation concerns still unresolved."
        )

    st.write("")

    with st.expander("🔍 Integrity Check"):
        st.write(result["integrity_notes"])

    with st.expander("📄 Evidence Retrieved"):
        for index, document in enumerate(result["evidence_docs"], start=1):
            st.markdown(f"**[E{index}]** {document.page_content}")

    with st.expander("📊 Factor Analysis"):
        st.markdown(_with_hard_line_breaks(result["factor_analysis"]))

    with st.expander("✅ Overall Assessment"):
        st.markdown(_with_hard_line_breaks(result["overall_assessment"]))
