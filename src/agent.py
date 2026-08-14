"""LangGraph assessment agent for FairCollab -- five nodes: evidence, integrity, factor analysis, overall rating, validation."""

import os

import re

from datetime import date

from typing import TypedDict

from dotenv import load_dotenv

from langgraph.graph import StateGraph, END

from langchain_google_genai import ChatGoogleGenerativeAI

from src.rag_pipeline import (
    retrieve_evidence,
    check_semantic_corroboration,
    find_cross_student_similar_contributions,
    load_all_projects,
)

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


def _find_student(student_name: str, project_id: str = None) -> tuple:
    """Search project files for a student, optionally scoped to one project_id."""
    for project in load_all_projects():
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


def check_temporal_plausibility(claim_date: str, recipient_record: dict, window_days: int = 3) -> dict:
    """Check whether recipient_record shows ANY contribution activity within window_days of
    claim_date -- an independent, non-semantic signal that something was happening around
    that time. This does NOT confirm the specific claimed event happened, only that the
    recipient wasn't completely silent near that date; total silence near the claimed date
    is itself informative, regardless of what text similarity says.

    peer_feedback entries have no date field at all in this schema (confirmed against the
    real project JSON: each entry is a plain string, not a dated record), so only
    recipient_record's "contributions" list can be checked here -- peer_feedback is
    excluded because there is nothing to compare it against, not as an oversight.

    Dates are "YYYY-MM-DD" strings throughout the codebase (confirmed against real JSON),
    parsed with date.fromisoformat().

    Also returns the nearest contribution's own description/complexity (not just that it
    exists), so callers like assess_corroboration_confidence() can judge whether that nearby
    activity is meaningful competing evidence or just incidental noise.

    Returns: {
        "plausible": bool,
        "nearest_contribution_date": str or None,
        "nearest_contribution_description": str or None,
        "nearest_contribution_complexity": str or None,
        "days_difference": int or None,
    }
    """
    empty_result = {
        "plausible": False,
        "nearest_contribution_date": None,
        "nearest_contribution_description": None,
        "nearest_contribution_complexity": None,
        "days_difference": None,
    }
    try:
        claim_dt = date.fromisoformat(claim_date)
    except (TypeError, ValueError):
        return dict(empty_result)

    nearest_date = None
    nearest_description = None
    nearest_complexity = None
    nearest_diff = None
    for contribution in recipient_record.get("contributions", []):
        raw_date = contribution.get("date")
        if not raw_date:
            continue
        try:
            contribution_dt = date.fromisoformat(raw_date)
        except (TypeError, ValueError):
            continue
        diff = abs((contribution_dt - claim_dt).days)
        if nearest_diff is None or diff < nearest_diff:
            nearest_diff = diff
            nearest_date = raw_date
            nearest_description = contribution.get("description")
            nearest_complexity = contribution.get("complexity")

    if nearest_diff is None:
        return dict(empty_result)

    return {
        "plausible": nearest_diff <= window_days,
        "nearest_contribution_date": nearest_date,
        "nearest_contribution_description": nearest_description,
        "nearest_contribution_complexity": nearest_complexity,
        "days_difference": nearest_diff,
    }


# Both bounds are empirically calibrated from Stage 1's real-data testing (see
# check_semantic_corroboration()'s docstring for the raw numbers), not arbitrary guesses:
# - STRONG_SEMANTIC_THRESHOLD (0.75): a genuinely uncorroborated real claim topped out at
#   ~0.606 relevance score, while a real corroborating paraphrase -- once actually indexed --
#   scored 0.806. 0.75 sits in the gap Stage 1 found clean separation at, above which a match
#   can be trusted on semantics alone.
# - WEAK_SEMANTIC_MIN (0.60): scores below this are at or under the known-negative case's own
#   ceiling (~0.606) -- no more convincing than a case already confirmed uncorroborated, so
#   temporal activity shouldn't be able to rescue a score this low into "weak".
STRONG_SEMANTIC_THRESHOLD = 0.75
WEAK_SEMANTIC_MIN = 0.60


def assess_corroboration_confidence(
    claim_text: str,
    claim_date: str,
    recipient_name: str,
    recipient_record: dict,
    project_id: str = None,
) -> dict:
    """Combine semantic and temporal signals into a THREE-TIER confidence assessment, not a
    binary verdict -- Stage 1b showed that OR-combining the two signals into one boolean lets
    a coincidental, topically-unrelated nearby activity fully clear a claim that the
    recipient's own peer feedback directly contradicts. Reuses check_semantic_corroboration()
    and check_temporal_plausibility() as-is; this function only combines their outputs.

    Tiers:
    - "strong": semantic_score >= STRONG_SEMANTIC_THRESHOLD -- confident corroboration on
      semantics alone, temporal signal not needed.
    - "weak": semantic_score is in the ambiguous band [WEAK_SEMANTIC_MIN, STRONG_SEMANTIC_THRESHOLD)
      AND the recipient has non-trivial activity within the temporal window -- reported as
      possible but explicitly unverified, never treated as corroborated.
    - "none": semantic_score < WEAK_SEMANTIC_MIN (regardless of temporal activity -- a low
      topical match isn't rescued by nearby-but-unrelated activity), OR the score is in the
      ambiguous band but the recipient has no nearby activity, or only trivial/irrelevant
      nearby activity.

    "Non-trivial" means the nearest nearby contribution has a real, non-empty description AND
    isn't itself marked complexity="trivial" -- reusing the complexity scale already
    established by connectors/github_connector.py's _infer_complexity() and used throughout
    the project data, rather than inventing a new heuristic. A non-empty-but-trivial entry
    (e.g. a real "Add files via upload" placeholder commit) does not count as competing
    evidence for a peer_support claim.

    Returns: {
        "tier": "strong" | "weak" | "none",
        "semantic_score": float,
        "nearest_activity_date": str or None,
        "nearest_activity_description": str or None,
        "days_difference": int or None,
    }
    """
    semantic = check_semantic_corroboration(claim_text, recipient_name, project_id=project_id)
    temporal = check_temporal_plausibility(claim_date, recipient_record)

    # No documents at all for this recipient (best_match_score is None) is treated as the
    # lowest possible score, not a crash or a free pass.
    semantic_score = semantic["best_match_score"] if semantic["best_match_score"] is not None else 0.0

    if semantic_score >= STRONG_SEMANTIC_THRESHOLD:
        tier = "strong"
    else:
        description = (temporal.get("nearest_contribution_description") or "").strip()
        is_non_trivial = bool(description) and temporal.get("nearest_contribution_complexity") != "trivial"
        if semantic_score >= WEAK_SEMANTIC_MIN and temporal["plausible"] and is_non_trivial:
            tier = "weak"
        else:
            tier = "none"

    return {
        "tier": tier,
        "semantic_score": semantic_score,
        "nearest_activity_date": temporal["nearest_contribution_date"],
        "nearest_activity_description": temporal["nearest_contribution_description"],
        "days_difference": temporal["days_difference"],
    }


def build_mention_graph(project: dict) -> dict:
    """Build a directed mention graph across all students in a project: an edge A -> B means
    A's own record contains something that references/credits B. Pure graph logic over the
    existing JSON structure, no embeddings -- deliberately kept out of rag_pipeline.py.

    Two source types:
    1. peer_support contributions with an explicit "recipient" field -- already structured.
       The contribution lives under the helper's own record and names the recipient, so
       helper -> recipient (helper's own record references the recipient).
    2. peer_feedback free-text entries -- NOT structured with a recipient field, and their
       true author is anonymous/untracked in this schema (a feedback quote is filed under the
       SUBJECT it praises, not under whoever wrote it). A plain substring check for another
       student's name is used here rather than semantic matching: this is graph construction
       from already-authored, already-attributed-to-a-record text (we just need "does this
       string literally name student Y", not a judgment call about corroboration strength),
       so the precision concerns that motivated semantic matching in Stage 1 don't apply the
       same way here. Checked in practice (see Stage 4 verification notes): no false-positive
       substring collisions occurred against the real project rosters.

    EDGE DIRECTION for peer_feedback (this is the part that's easy to get backwards): a
    feedback string is filed under student X's own record. If it names student Y, that is
    X's own record referencing Y -- the SAME "owner's record references someone" pattern as
    peer_support above, applied consistently. So the edge is X -> Y, not Y -> X. This matters
    because Y did not write the entry and cannot be credited as its source; X's record is
    what structurally contains the reference to Y. (Getting this backwards would attribute
    mentions to the wrong student and silently invert who looks "corroborated" by whom.)

    Returns: {student_name: set of student_names who mention/vouch for them} -- i.e. an
    IN-EDGES map (who corroborates this student), since what matters downstream is "who has
    corroborated me", not "who have I corroborated".
    """
    student_names = [s["student_name"] for s in project["students"]]
    in_edges = {name: set() for name in student_names}

    for student in project["students"]:
        source_name = student["student_name"]

        for contribution in student.get("contributions", []):
            if contribution.get("type") == "peer_support" and contribution.get("recipient"):
                recipient_name = contribution["recipient"]
                if recipient_name in in_edges and recipient_name != source_name:
                    in_edges[recipient_name].add(source_name)

        for feedback_text in student.get("peer_feedback", []):
            for other_name in student_names:
                if other_name != source_name and other_name in feedback_text:
                    in_edges[other_name].add(source_name)

    return in_edges


def find_mutual_isolation_pairs(project: dict, in_edges: dict = None) -> list:
    """Using build_mention_graph(), find pairs (A, B) that mutually mention/vouch for each
    other AND have zero corroboration from anyone else in the project -- the actual suspicious
    pattern, not just "any pair who mention each other" (which would flag ordinary close
    collaborators, who are supposed to mention each other often).

    in_edges: pass a graph already built by build_mention_graph(project) to avoid rebuilding
    it (node_integrity_analysis() reuses the same graph for its zero-corroborators check);
    left as None for standalone callers, who get one built automatically.

    Returns list of dicts: {"student_a": str, "student_b": str}
    """
    if in_edges is None:
        in_edges = build_mention_graph(project)
    student_names = list(in_edges.keys())
    pairs = []
    for i, a in enumerate(student_names):
        for b in student_names[i + 1 :]:
            mutual = a in in_edges.get(b, set()) and b in in_edges.get(a, set())
            if not mutual:
                continue
            a_others = in_edges[a] - {b}
            b_others = in_edges[b] - {a}
            if not a_others and not b_others:
                pairs.append({"student_a": a, "student_b": b})
    return pairs


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

    project_id = state.get("project_id")
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
                claim_text = contribution.get("description", "")
                claim_date = contribution.get("date")
                assessment = assess_corroboration_confidence(
                    claim_text, claim_date, recipient_name, recipient_record, project_id=project_id
                )
                score = round(assessment["semantic_score"], 3)
                days_diff = assessment["days_difference"]
                # "strong" suppresses the flag entirely (confident corroboration). "weak" and
                # "none" both flag, but with deliberately different wording -- Stage 1b's bug was
                # treating ambiguous-but-unverified cases as fully cleared with no flag at all, so
                # "weak" must read as genuinely uncertain (not corroborated, not disproven) while
                # "none" reads as the stronger concern, so the downstream LLM and a human reading
                # integrity_notes can tell the two apart instead of seeing a flat bullet list.
                if assessment["tier"] == "weak":
                    nearest_desc = (assessment["nearest_activity_description"] or "unknown activity")[:60]
                    flags.append(
                        f"peer_support claim of helping {recipient_name} has only weak, unverified "
                        f"support: topical similarity is moderate ({score}) and {recipient_name} logged "
                        f"activity {days_diff} days away (\"{nearest_desc}\"), which may or may not "
                        f"relate to this specific claim"
                    )
                elif assessment["tier"] == "none":
                    days_desc = f"{days_diff} days away" if days_diff is not None else "no dated activity at all"
                    flags.append(
                        f"peer_support claim of helping {recipient_name} has no meaningful corroboration: "
                        f"topical similarity is low ({score}) and {recipient_name}'s nearest logged "
                        f"activity is {days_desc}"
                    )

    # KNOWN INEFFICIENCY, not addressed here: find_cross_student_similar_contributions() scans
    # every contribution in the whole project, not just this student's, then filters down below.
    # run_assessment() assesses one student per call, so assessing every student in a project
    # (e.g. a teacher clicking "Generate Assessment" for all 4) redundantly reruns this same
    # whole-project scan once per student -- measured at ~8s / ~30 FAISS queries for the 4-student
    # software project and ~7s / ~26 queries for the 8-student research project (see Stage 2
    # calibration notes). Not fixed now because at this dataset size the redundant cost is a few
    # seconds per assessment, not worth the added complexity of a cache that would need explicit
    # invalidation whenever the index is rebuilt (Live Data fetches, etc.). If this grows into a
    # real cost -- bigger classes, more contributions per student -- the natural fix is caching
    # this scan per project_id (e.g. via st.cache_data in app.py, invalidated the same way
    # load_project_options() already is) rather than changing anything in this function.
    if project_id is not None:
        cross_student_pairs = find_cross_student_similar_contributions(project_id)
        student_name = student_record["student_name"]
        for pair in cross_student_pairs:
            if student_name == pair["student_a"]:
                this_text, other_student, other_text = pair["text_a"], pair["student_b"], pair["text_b"]
            elif student_name == pair["student_b"]:
                this_text, other_student, other_text = pair["text_b"], pair["student_a"], pair["text_a"]
            else:
                continue
            this_desc = this_text.rsplit(" | ", 1)[-1][:80]
            other_desc = other_text.rsplit(" | ", 1)[-1][:80]
            flags.append(
                f"Contribution closely resembles one logged by {other_student}: "
                f"\"{this_desc}\" vs \"{other_desc}\" (similarity {round(pair['score'], 3)}, dated "
                f"around {pair['date_a']} and {pair['date_b']}) -- flagged for review, not "
                f"confirmed as duplicated or copied work"
            )

    # Built once and reused for both mention-graph checks below (Stage 4's mutual-isolation
    # pairs and Stage 4b's zero-corroborators flag), rather than letting
    # find_mutual_isolation_pairs() rebuild it a second time internally.
    mention_graph = build_mention_graph(project)

    isolation_pairs = find_mutual_isolation_pairs(project, in_edges=mention_graph)
    student_name = student_record["student_name"]
    for pair in isolation_pairs:
        if student_name == pair["student_a"]:
            other_student = pair["student_b"]
        elif student_name == pair["student_b"]:
            other_student = pair["student_a"]
        else:
            continue
        flags.append(
            f"{student_name} and {other_student} exclusively corroborate each other with no "
            f"independent mentions from any other group member -- flagged for review as a "
            f"potential coordination pattern, not confirmed"
        )

    # Project-wide sparse-data guard: skip the zero-corroborators check entirely if the WHOLE
    # project has essentially no raw peer-interaction data recorded at all (e.g. peer review
    # forms not yet submitted by anyone), rather than flagging every single student as having
    # "no corroborators" when the real cause is missing data collection, not missing evidence.
    #
    # This counts RAW peer_feedback entries + peer_support-with-recipient contributions across
    # every student -- NOT the resulting graph edge count. Those are different things here: in
    # every real project tested so far, peer_feedback text is written in natural language
    # ("She helped me debug my code...") that essentially never contains a formal "Student X"
    # identifier, so build_mention_graph()'s substring matching produces few or zero edges from
    # it even when substantial real peer_feedback data exists (e.g. the research project has 12
    # real peer_feedback entries across its 4 students but 0 graph edges). Using edge count as
    # the sparsity signal would misclassify "peer review was done but doesn't name-match" as "no
    # peer review happened" and wrongly suppress the check project-wide. Raw entry count doesn't
    # have that problem: it reflects whether peer-interaction data was actually collected.
    #
    # Cutoff: fewer than 2 total raw entries across the whole project. Every real project in
    # this codebase has 12+ (3 peer_feedback entries per student, times 4-8 students), so this
    # threshold sits far below any real usage while still catching the genuine "nothing
    # submitted yet" case the exclusion exists for.
    total_peer_entries = sum(
        len(s.get("peer_feedback", []))
        + sum(1 for c in s.get("contributions", []) if c.get("type") == "peer_support" and c.get("recipient"))
        for s in project["students"]
    )
    if total_peer_entries >= 2 and not mention_graph.get(student_name):
        flags.append(
            f"{student_name} has no corroborating mentions from any other group member in "
            f"peer_support claims or peer_feedback -- flagged for review, not confirmed as an "
            f"integrity concern (this may simply reflect limited peer interaction data rather "
            f"than any wrongdoing)"
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
