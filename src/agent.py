"""LangGraph assessment agent for FairCollab -- five nodes: evidence, integrity, factor analysis, overall rating, validation."""

import os

import json

import re

from typing import TypedDict

from dotenv import load_dotenv

from langgraph.graph import StateGraph, END

from langchain_google_genai import ChatGoogleGenerativeAI

from src.rag_pipeline import retrieve_evidence, STUDENTS_PATH, RESEARCH_STUDENTS_PATH, ROBOTICS_PATH

from src.prompts import FACTOR_PROMPT, OVERALL_PROMPT, VALIDATION_PROMPT

load_dotenv()

LLM_MODEL = "gemini-2.5-flash"

MAX_REVISIONS = 2


class AgentState(TypedDict, total=False):
    """Shape of the state dict threaded through every node in the graph."""

    student_name: str

    # Optional: scopes lookup/retrieval when a student_name exists in multiple projects.
    project_id: str

    project_data: dict

    student_record: dict

    project_name: str
    project_type: str
    duration_weeks: int

    stats: str

    evidence_docs: list

    evidence_text: str

    integrity_notes: str

    factor_analysis: str

    overall_assessment: str

    overall_rating: str

    validation_result: str

    revision_count: int

    max_revisions_reached: bool

    final_output: str


def _get_llm() -> ChatGoogleGenerativeAI:
    """Construct the Gemini chat client, failing fast if the API key is missing."""
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GOOGLE_API_KEY is not set. Add it to your .env file before running this script."
        )
    return ChatGoogleGenerativeAI(model=LLM_MODEL, google_api_key=api_key, temperature=0.2)


def _load_project(path: str) -> dict:
    """Read one contribution-log JSON file and return it as a dict."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _find_student(student_name: str, project_id: str = None) -> tuple:
    """Search project files for a student, optionally scoped to one project_id."""
    for path in (STUDENTS_PATH, RESEARCH_STUDENTS_PATH, ROBOTICS_PATH):
        project = _load_project(path)
        if project_id is not None and project.get("project_id") != project_id:
            continue
        for student in project["students"]:
            if student["student_name"] == student_name:
                return project, student
    raise ValueError(f"No student named '{student_name}' found in the expected project data file(s).")


def _format_stats(student_record: dict) -> str:
    """Turn a student's raw counters into the preformatted STUDENT STATS block."""
    assigned = student_record.get("tasks_assigned", 0)
    completed = student_record.get("tasks_completed", 0)
    attended = student_record.get("meetings_attended", 0)
    total_meetings = student_record.get("meetings_total", 0)
    task_submitted = student_record.get("task_form_submitted", False)
    meeting_submitted = student_record.get("meeting_data_submitted", False)

    # Also checks == 0 so legacy records with real counts aren't misreported as missing.
    if not task_submitted and assigned == 0:
        task_line = "Task completion: no data available (task assignment form not yet submitted)"
    else:
        completion_pct = round(100 * completed / assigned) if assigned else 0
        task_line = f"Tasks completed: {completed}/{assigned} ({completion_pct}%)"

    if not meeting_submitted and total_meetings == 0:
        meeting_line = "Meeting attendance: no data available (no meeting data submitted)"
    else:
        attendance_pct = round(100 * attended / total_meetings) if total_meetings else 0
        meeting_line = f"Meetings attended: {attended}/{total_meetings} ({attendance_pct}%)"

    return f"{task_line}\n{meeting_line}"


def _format_evidence(documents: list) -> str:
    """Turn retrieve_evidence()'s Document list into the [E#]-tagged block prompts expect."""
    if not documents:
        return "No evidence retrieved for this student."
    lines = [f"[E{i + 1}] {doc.page_content}" for i, doc in enumerate(documents)]
    return "\n".join(lines)


def node_retrieve_evidence(state: AgentState) -> AgentState:
    """Node 1: gather everything downstream nodes need -- evidence, stats, project context."""
    student_name = state["student_name"]
    project_id = state.get("project_id")
    project, student_record = _find_student(student_name, project_id=project_id)
    documents = retrieve_evidence(student_name, project_id=project_id)
    state["project_data"] = project
    state["student_record"] = student_record
    state["evidence_docs"] = documents
    state["project_name"] = project["project_name"]
    state["project_type"] = project["project_type"]
    state["duration_weeks"] = project["duration_weeks"]
    state["stats"] = _format_stats(student_record)
    state["evidence_text"] = _format_evidence(documents)
    state["revision_count"] = 0
    return state


def node_integrity_analysis(state: AgentState) -> AgentState:
    """Node 2: run cheap, deterministic fraud-signal heuristics over the raw data."""
    student_record = state["student_record"]
    project = state["project_data"]
    flags = []

    if student_record.get("tasks_completed", 0) > student_record.get("tasks_assigned", 0):
        flags.append(
            f"tasks_completed ({student_record.get('tasks_completed')}) exceeds "
            f"tasks_assigned ({student_record.get('tasks_assigned')})"
        )

    if student_record.get("meetings_attended", 0) > student_record.get("meetings_total", 0):
        flags.append(
            f"meetings_attended ({student_record.get('meetings_attended')}) exceeds "
            f"meetings_total ({student_record.get('meetings_total')})"
        )

    descriptions = [c.get("description", "") for c in student_record.get("contributions", [])]
    seen_descriptions = set()
    duplicate_descriptions = set()
    for description in descriptions:
        if description in seen_descriptions:
            duplicate_descriptions.add(description)
        seen_descriptions.add(description)
    if duplicate_descriptions:
        example = next(iter(duplicate_descriptions))[:60]
        flags.append(
            f"{len(duplicate_descriptions)} duplicate contribution description(s) found, "
            f"e.g. \"{example}...\""
        )

    for contribution in student_record.get("contributions", []):
        if contribution.get("type") == "peer_support" and contribution.get("recipient"):
            recipient_name = contribution["recipient"]
            recipient_record = next(
                (s for s in project["students"] if s["student_name"] == recipient_name), None
            )
            if recipient_record is None:
                flags.append(
                    f"peer_support claims to have helped '{recipient_name}', who is not "
                    f"a student on this project"
                )
            else:
                helper_name = student_record["student_name"]
                mentioned_in_feedback = any(
                    helper_name in quote for quote in recipient_record.get("peer_feedback", [])
                )
                mentioned_in_contributions = any(
                    helper_name in c.get("description", "")
                    for c in recipient_record.get("contributions", [])
                )
                if not (mentioned_in_feedback or mentioned_in_contributions):
                    flags.append(
                        f"peer_support claim of helping {recipient_name} has no corroborating "
                        f"mention in {recipient_name}'s own feedback or contributions"
                    )

    if flags:
        integrity_notes = "Potential integrity concerns detected during automated pre-check:\n" + "\n".join(
            f"- {flag}" for flag in flags
        )
    else:
        integrity_notes = "No integrity concerns detected during automated pre-check."

    state["integrity_notes"] = integrity_notes
    state["evidence_text"] = f"[INTEGRITY CHECK]\n{integrity_notes}\n\n{state['evidence_text']}"
    return state


def node_factor_analysis(state: AgentState) -> AgentState:
    """Node 3: run FACTOR_PROMPT to score the student on all six fairness-weighted factors."""
    llm = _get_llm()
    evidence_for_prompt = state["evidence_text"]
    prior_validation = state.get("validation_result")
    if prior_validation and prior_validation.strip().upper().startswith("REVISION NEEDED"):
        evidence_for_prompt += (
            f"\n\n[VALIDATION FEEDBACK FROM PRIOR ATTEMPT]\n{prior_validation}\n"
            f"Address this specific concern in your revised analysis.\n"
        )
    prompt_text = FACTOR_PROMPT.format(
        student_name=state["student_name"],
        project_name=state["project_name"],
        project_type=state["project_type"],
        duration_weeks=state["duration_weeks"],
        stats=state["stats"],
        evidence=evidence_for_prompt,
    )
    response = llm.invoke(prompt_text)
    state["factor_analysis"] = response.content
    return state


def node_overall_assessment(state: AgentState) -> AgentState:
    """Node 4: run OVERALL_PROMPT to roll the factor analysis up into one rating."""
    llm = _get_llm()
    prompt_text = OVERALL_PROMPT.format(
        student_name=state["student_name"],
        factor_analysis=state["factor_analysis"],
    )
    response = llm.invoke(prompt_text)
    state["overall_assessment"] = response.content
    rating_match = re.search(r"Overall Rating:\s*(High|Medium|Low)", response.content, re.IGNORECASE)
    state["overall_rating"] = rating_match.group(1).title() if rating_match else "Unknown"
    return state


def _format_final_output(state: AgentState) -> str:
    """Assemble the human-readable report once Node 5 decides to stop looping."""
    lines = [
        f"FairCollab Contribution Assessment — {state['student_name']}",
        "",
        "Integrity Check:",
        state["integrity_notes"],
        "",
        "Factor Analysis:",
        state["factor_analysis"],
        "",
        "Overall Assessment:",
        state["overall_assessment"],
        "",
        "Validation:",
        state["validation_result"],
    ]
    if state["max_revisions_reached"]:
        lines.append("")
        lines.append(
            f"Note: maximum of {MAX_REVISIONS} revisions reached; returning this "
            f"best-effort assessment despite unresolved validation concerns."
        )
    return "\n".join(lines)


def node_validate(state: AgentState) -> AgentState:
    """Node 5: run VALIDATION_PROMPT to fact-check the assessment, looping back if needed."""
    llm = _get_llm()
    prompt_text = VALIDATION_PROMPT.format(
        student_name=state["student_name"],
        stats=state["stats"],
        evidence=state["evidence_text"],
        factor_analysis=state["factor_analysis"],
        overall_assessment=state["overall_assessment"],
    )
    response = llm.invoke(prompt_text)
    validation_result = response.content.strip()
    state["validation_result"] = validation_result
    needs_revision = (
        validation_result.upper().startswith("REVISION NEEDED")
        and state["revision_count"] < MAX_REVISIONS
    )
    if needs_revision:
        state["revision_count"] += 1
    else:
        state["max_revisions_reached"] = validation_result.upper().startswith("REVISION NEEDED")
        state["final_output"] = _format_final_output(state)
    return state


def _route_after_validation(state: AgentState) -> str:
    """Conditional-edge router: decide whether Node 5 wants to loop or finish."""
    if state.get("final_output"):
        return "end"
    return "retry"


def build_graph():
    """Wire the five nodes and the validation loop into a compiled LangGraph app."""
    graph = StateGraph(AgentState)
    graph.add_node("retrieve_evidence", node_retrieve_evidence)
    graph.add_node("integrity_analysis", node_integrity_analysis)
    graph.add_node("factor_analysis", node_factor_analysis)
    graph.add_node("overall_assessment", node_overall_assessment)
    graph.add_node("validate", node_validate)
    graph.set_entry_point("retrieve_evidence")
    graph.add_edge("retrieve_evidence", "integrity_analysis")
    graph.add_edge("integrity_analysis", "factor_analysis")
    graph.add_edge("factor_analysis", "overall_assessment")
    graph.add_edge("overall_assessment", "validate")
    graph.add_conditional_edges(
        "validate",
        _route_after_validation,
        {"retry": "factor_analysis", "end": END},
    )
    return graph.compile()


def run_assessment(student_name: str, project_id: str = None) -> AgentState:
    """Run the full assessment for one student; project_id disambiguates same-named students across projects."""
    app = build_graph()
    return app.invoke({"student_name": student_name, "project_id": project_id})


def run_assessment_streaming(student_name: str, project_id: str = None):
    """Same as run_assessment(), but yields a progress update after each node completes.

    Yields {"node": <node_name>, "chunk": <cumulative state so far>} as each node
    finishes, then a final {"node": "done", "final_state": <the complete AgentState>}.
    Node names can repeat (factor_analysis/overall_assessment/validate re-run on
    each revision loop), so callers should key off "node" plus the chunk's
    revision_count rather than assuming each name appears once.
    """
    app = build_graph()
    final_state = None
    for chunk in app.stream({"student_name": student_name, "project_id": project_id}):
        node_name = list(chunk.keys())[0]
        node_state = chunk[node_name]
        final_state = node_state
        yield {"node": node_name, "chunk": node_state}
    yield {"node": "done", "final_state": final_state}


if __name__ == "__main__":
    import sys

    name = sys.argv[1] if len(sys.argv) > 1 else "Student A"
    print(run_assessment(name)["final_output"])
