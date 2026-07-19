"""
Streamlit web interface for FairCollab.

Lets a user pick a project type and a student from the sidebar, click
"Generate Assessment", and see the five-node LangGraph agent's result
(src/agent.py's run_assessment()) laid out in the main area: an overall
rating badge plus four expandable sections for integrity, evidence, factor
analysis, and the overall reasoning.
"""

import streamlit as st

import json

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agent import run_assessment, MAX_REVISIONS

from src.rag_pipeline import STUDENTS_PATH, RESEARCH_STUDENTS_PATH

st.set_page_config(page_title="FairCollab", page_icon="⚖️", layout="centered")

RATING_COLORS = {"High": "green", "Medium": "orange", "Low": "red", "Unknown": "gray"}


def _load_project(path: str) -> dict:
    """Read one contribution-log JSON file and return it as a dict."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def load_project_options() -> dict:
    """Load both project files once and cache them across reruns."""
    return {
        "Software Project": _load_project(STUDENTS_PATH),
        "Research Project": _load_project(RESEARCH_STUDENTS_PATH),
    }


def _with_hard_line_breaks(text: str) -> str:
    """Turn every single newline into a Markdown hard line break."""
    return text.replace("\n", "  \n")


def _is_quota_error(error: Exception) -> bool:
    """Best-effort check for whether an exception is a Gemini quota/rate-limit error."""
    message = str(error).upper()
    return "RESOURCE_EXHAUSTED" in message or "429" in message or "QUOTA" in message


if "result" not in st.session_state:
    st.session_state.result = None
if "error" not in st.session_state:
    st.session_state.error = None

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
        "Generate Assessment", type="primary", use_container_width=True
    )

if generate_clicked:
    st.session_state.result = None
    st.session_state.error = None
    with st.spinner(f"Assessing {selected_student}... this runs multiple AI calls and may take a minute."):
        try:
            st.session_state.result = run_assessment(selected_student)
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
