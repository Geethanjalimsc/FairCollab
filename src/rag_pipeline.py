"""RAG pipeline for FairCollab: builds a searchable evidence index from student contribution logs."""

import os

import json

from dotenv import load_dotenv

from langchain_core.documents import Document

from langchain_google_genai import GoogleGenerativeAIEmbeddings

from langchain_community.vectorstores import FAISS

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

STUDENTS_PATH = os.path.join(BASE_DIR, "data", "students.json")

RESEARCH_STUDENTS_PATH = os.path.join(BASE_DIR, "data", "research_students.json")

ROBOTICS_PATH = os.path.join(BASE_DIR, "data", "robotics_project.json")

# Experiment 4 (fraud detection evaluation): a separate, permanent dataset containing four
# independent scenarios with known ground-truth fraud patterns, kept apart from the core
# students.json/research_students.json (which test fairness/rating spread, not fraud
# detection) the same way robotics_project.json is kept apart as the live-data case.
FRAUD_SCENARIOS_PATH = os.path.join(BASE_DIR, "data", "fraud_scenarios.json")

INDEX_DIR = os.path.join(BASE_DIR, "data", "faircollab_index")

EMBEDDING_MODEL = "models/gemini-embedding-001"

load_dotenv()


def _load_json(path: str) -> dict:
    """Read one contribution-log JSON file and return it as a dict."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_all_projects() -> list:
    """Load every known project file and flatten into one list of project dicts.

    Most project files (students.json, research_students.json, robotics_project.json) hold
    ONE project object at the top level. fraud_scenarios.json is the one exception: it holds a
    JSON list of FOUR independent project objects (one per Experiment 4 scenario), since that
    experiment specifically needs several small, separately-scoped projects in one file rather
    than one file per scenario. This normalizes both shapes into a single flat list so callers
    (build_documents(), find_cross_student_similar_contributions(), agent.py's _find_student())
    don't need their own special-casing for "this path might contain several projects."
    """
    projects = []
    for path in (STUDENTS_PATH, RESEARCH_STUDENTS_PATH, ROBOTICS_PATH, FRAUD_SCENARIOS_PATH):
        loaded = _load_json(path)
        if isinstance(loaded, list):
            projects.extend(loaded)
        else:
            projects.append(loaded)
    return projects


def _contribution_to_document(project: dict, student: dict, contribution: dict) -> Document:
    """Turn a single contribution entry into one embeddable Document."""
    date = contribution.get("date", "unknown date")
    ctype = contribution.get("type", "unknown type")
    description = contribution.get("description", "")

    extra_bits = []
    for key in ("complexity", "status", "role", "recipient"):
        if key in contribution:
            extra_bits.append(f"{key}: {contribution[key]}")
    extra_text = f" ({', '.join(extra_bits)})" if extra_bits else ""

    page_content = (
        f"Student: {student['student_name']} | Project: {project['project_name']} | "
        f"Date: {date} | Type: {ctype} | Description: {description}{extra_text}"
    )

    metadata = {
        "student_name": student["student_name"],
        "student_id": student.get("student_id"),
        "project_id": project.get("project_id"),
        "project_name": project.get("project_name"),
        "record_type": "contribution",
        "contribution_type": ctype,
        "date": date,
    }

    return Document(page_content=page_content, metadata=metadata)


def _feedback_to_document(project: dict, student: dict, feedback_text: str, index: int) -> Document:
    """Turn a single peer-feedback quote into one embeddable Document."""
    page_content = (
        f"Student: {student['student_name']} | Project: {project['project_name']} | "
        f"Peer feedback: {feedback_text}"
    )
    metadata = {
        "student_name": student["student_name"],
        "student_id": student.get("student_id"),
        "project_id": project.get("project_id"),
        "project_name": project.get("project_name"),
        "record_type": "peer_feedback",
        "feedback_index": index,
    }
    return Document(page_content=page_content, metadata=metadata)


def build_documents() -> list:
    """Load all project data files and flatten every record into a list of Documents."""
    projects = load_all_projects()

    documents = []
    for project in projects:
        for student in project["students"]:
            for contribution in student.get("contributions", []):
                documents.append(_contribution_to_document(project, student, contribution))

            for i, feedback_text in enumerate(student.get("peer_feedback", [])):
                documents.append(_feedback_to_document(project, student, feedback_text, i))

    return documents


def _get_embeddings() -> GoogleGenerativeAIEmbeddings:
    """Construct the Gemini embeddings client, failing fast if the key is missing."""
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GOOGLE_API_KEY is not set. Add it to your .env file before running this script."
        )
    return GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL, google_api_key=api_key)


def build_and_save_index() -> FAISS:
    """Embed every contribution/feedback record and persist the FAISS index to disk."""
    documents = build_documents()
    embeddings = _get_embeddings()

    vector_store = FAISS.from_documents(documents, embeddings)

    vector_store.save_local(INDEX_DIR)

    return vector_store


def _load_index() -> FAISS:
    """Load a previously saved FAISS index from disk, embeddings client included."""
    embeddings = _get_embeddings()
    return FAISS.load_local(INDEX_DIR, embeddings, allow_dangerous_deserialization=True)


def retrieve_evidence(student_name: str, k: int = 8, project_id: str = None) -> list:
    """Return top-k contribution records for one student plus a smaller top slice of peer feedback."""
    vector_store = _load_index()

    # Student names repeat across projects, so scope by project_id too when given.
    base_filter = {"student_name": student_name}
    if project_id is not None:
        base_filter["project_id"] = project_id

    # Searched separately so contribution volume can't crowd out peer feedback.
    contribution_results = vector_store.similarity_search(
        student_name, k=k, filter={**base_filter, "record_type": "contribution"}, fetch_k=200
    )

    # Peer feedback is supporting evidence, so its cap is lower than k.
    peer_feedback_k = max(1, round(k * 0.4))
    peer_feedback_results = vector_store.similarity_search(
        student_name, k=peer_feedback_k, filter={**base_filter, "record_type": "peer_feedback"}, fetch_k=200
    )

    results = contribution_results + peer_feedback_results

    if not results:
        results = vector_store.similarity_search(student_name, k=k)

    return results


def check_semantic_corroboration(
    claim_text: str,
    recipient_name: str,
    project_id: str = None,
    similarity_threshold: float = 0.65,
) -> dict:
    """Check whether claim_text (e.g. a peer_support claim of helping recipient_name) is
    semantically corroborated by anything in recipient_name's own documents (their
    peer_feedback or contributions) -- not just a literal mention of the helper's name.

    Score direction (verified empirically, not assumed): langchain_community's FAISS
    wrapper defaults to Euclidean distance, so its raw similarity_search_with_score()
    returns a DISTANCE where LOWER means more similar -- the opposite of typical cosine-
    similarity intuition, and easy to get backwards. similarity_search_with_relevance_scores()
    is used here instead: it applies FAISS's own relevance conversion (1 - distance/sqrt(2))
    to give a [0, 1] score where HIGHER means more similar, which is what similarity_threshold
    compares against below.

    similarity_threshold's default (0.65) is itself empirically calibrated against this
    project's real data, not a guess: a genuinely uncorroborated claim (Student A's
    peer_support claim about Student B, who has no matching record) tops out at ~0.606
    relevance score across all of Student B's own documents, while realistic paraphrases
    of a real corroborating event that drop the helper's name but keep the topic (e.g.
    "a groupmate helped resolve a merge conflict I was stuck on...") score ~0.65-0.67.
    Known limitation: paraphrases sharing almost no vocabulary with the claim at all (full
    synonym replacement) scored *lower* in testing than genuinely unrelated text -- this
    embedding model leans on some lexical/topical overlap as signal at this text length, so
    this check reliably catches name-free paraphrases that still share a topic anchor, but
    not fully alienated rewordings. Worth revisiting in a later stage.

    Returns: {
        "corroborated": bool,
        "best_match_score": float or None,
        "best_match_text": str or None,  # the recipient's document that came closest
    }
    """
    vector_store = _load_index()

    search_filter = {"student_name": recipient_name}
    if project_id is not None:
        search_filter["project_id"] = project_id

    # fetch_k=200 (matching retrieve_evidence()'s convention) so the filter is applied
    # over the whole index, not just FAISS's small default candidate pool -- without it,
    # a narrow filter can silently miss most of the recipient's own documents.
    results = vector_store.similarity_search_with_relevance_scores(
        claim_text, k=20, filter=search_filter, fetch_k=200
    )

    if not results:
        return {"corroborated": False, "best_match_score": None, "best_match_text": None}

    best_doc, best_score = max(results, key=lambda pair: pair[1])
    best_score = float(best_score)  # FAISS returns numpy scalars; keep the return dict plain Python
    return {
        "corroborated": best_score >= similarity_threshold,
        "best_match_score": best_score,
        "best_match_text": best_doc.page_content,
    }


def find_cross_student_similar_contributions(
    project_id: str,
    non_meeting_threshold: float = 0.85,
    meeting_threshold: float = None,
) -> list:
    """Scan all 'contribution' record_type documents within one project and find pairs
    belonging to DIFFERENT students whose descriptions are near-duplicates -- a signal for
    possible coordinated or copied work, distinct from the within-student duplicate check
    already in node_integrity_analysis(). Scope: "contribution" documents only.

    peer_feedback was deliberately tried and rejected as a scope extension, not just left
    unexplored: real-data testing (fraud_scenarios.json's Scenario 2 vs the three genuine
    clean projects) found complete score overlap between innocent and coordinated cases --
    the single highest genuine cross-student peer_feedback pair scored 0.8140, HIGHER than
    the lowest pair in the deliberately-constructed suspicious set (0.8107). No threshold
    separates them, at 0.65, 0.85, or anywhere else. Root cause: contribution text describes
    specific, itemizable technical work (a large, specific vocabulary space where coincidental
    overlap is rare), while peer_feedback text characterizes a *performance level* from a
    small, common vocabulary ("reliable", "unresponsive", "took initiative") -- so two
    genuinely different students who happen to both be strong (or both weak) contributors
    naturally produce similar-sounding feedback with zero coordination involved. Don't
    re-attempt this as a simple filter-scope extension without a fundamentally different
    mechanism (e.g. shared-rare-vocabulary matching showed more promise in early testing but
    was never calibrated to a shippable standard).

    For each contribution, this queries the FAISS index with that contribution's own
    page_content, filtered to {"project_id": project_id, "record_type": "contribution"} with
    fetch_k=200 (matching the fetch_k convention used elsewhere in this file). The dict filter
    here only supports equality (plus this LangChain version's $eq/$neq/$in/$and/$or/$not
    operators -- verified by reading FAISS._create_filter_func's source), but this function
    still excludes same-student matches in Python after retrieval rather than via a $neq
    filter clause, to stay consistent with the plain-equality-filter style used by every other
    filter in this file (retrieve_evidence, check_semantic_corroboration).

    TYPE-AWARE THRESHOLD, empirically calibrated (not guessed) against the real software and
    research projects' contribution data:
    - Innocent NON-meeting pairs (different students, genuinely different work, same project
      vocabulary) scored up to 0.8244 in testing -- e.g. one student writing API documentation
      and another creating the doc template for the same feature naturally score high on
      topic alone, with no copying involved.
    - Innocent MEETING pairs scored even higher: up to 0.8315 -- HIGHER than the non-meeting
      ceiling in the same dataset. Students independently describing attendance at the same
      real meeting ("Attended sprint planning meeting" / "Organised sprint planning meeting")
      routinely land in the 0.75-0.83 band. There is no innocent-vs-suspicious separation to
      threshold against for meeting-type pairs -- the innocent ceiling for meetings sits at or
      above the innocent ceiling for everything else, so no single bar can safely admit
      genuine meeting overlap while excluding a fabricated meeting-type duplicate, because a
      fabricated one wouldn't even need to be that close to still look "normal." Per this
      finding, meeting-type pairs are excluded from flagging entirely by default
      (meeting_threshold=None) rather than assigned a "much higher" numeric bar that testing
      does not support. meeting_threshold is kept as a parameter (not hardcoded away) so a
      future stage can revisit this with a different method; passing an explicit value here
      re-enables meeting-pair flagging at that bar.
    - A deliberately near-duplicate contribution (copied description, reworded by one word)
      scored 0.8730 against its source; an exact-text duplicate's self-match scored 0.8817 --
      this dataset/embedding model's practical ceiling is well under 1.0 even for identical
      text. non_meeting_threshold=0.85 sits in the real gap this produced: above every
      observed innocent non-meeting score (max 0.8244), below both observed true-duplicate
      scores (0.8730, 0.8817). The margin is real but narrow (~0.02-0.03 either side), so this
      should be read as "flagged for review," never as confirmed copying.

    Deduplicates symmetric pairs (A-vs-B and B-vs-A are the same finding, reported once).

    Returns a list of dicts:
    {
        "student_a": str, "student_b": str,
        "text_a": str, "text_b": str,
        "score": float,
        "contribution_type": str,
        "date_a": str, "date_b": str,
    }
    """
    vector_store = _load_index()

    project = next((p for p in load_all_projects() if p.get("project_id") == project_id), None)
    if project is None:
        return []

    seen_pairs = set()
    findings = []

    for student in project["students"]:
        for contribution in student.get("contributions", []):
            query_doc = _contribution_to_document(project, student, contribution)
            this_type = query_doc.metadata.get("contribution_type")

            results = vector_store.similarity_search_with_relevance_scores(
                query_doc.page_content,
                k=10,
                filter={"project_id": project_id, "record_type": "contribution"},
                fetch_k=200,
            )
            for other_doc, score in results:
                other_student = other_doc.metadata.get("student_name")
                if other_student == student["student_name"]:
                    continue

                other_type = other_doc.metadata.get("contribution_type")
                is_meeting_pair = this_type == "meeting" and other_type == "meeting"
                threshold = meeting_threshold if is_meeting_pair else non_meeting_threshold
                if threshold is None or score < threshold:
                    continue

                pair_key = tuple(
                    sorted(
                        [
                            (student["student_name"], query_doc.page_content),
                            (other_student, other_doc.page_content),
                        ]
                    )
                )
                if pair_key in seen_pairs:
                    continue
                seen_pairs.add(pair_key)

                findings.append(
                    {
                        "student_a": student["student_name"],
                        "student_b": other_student,
                        "text_a": query_doc.page_content,
                        "text_b": other_doc.page_content,
                        "score": float(score),
                        "contribution_type": this_type if this_type == other_type else f"{this_type}/{other_type}",
                        "date_a": contribution.get("date"),
                        "date_b": other_doc.metadata.get("date"),
                    }
                )

    return findings


def find_best_matching_contribution(task_description: str, student_name: str, project_id: str = None) -> dict:
    """Find the single best-matching contribution (record_type "contribution" only, not
    peer_feedback) belonging to student_name for a claimed task_description -- reuses
    check_semantic_corroboration()'s exact scoring approach and confirmed score direction
    (similarity_search_with_relevance_scores(): [0, 1], higher means more similar), not
    re-derived here.

    A task may legitimately match several real commits (e.g. "Implement SLAM" spread across
    5 commits); this only needs the SINGLE best match, since the question is "is there any
    real evidence at all for this claim", not full credit allocation across matches.

    Returns: {
        "best_score": float,
        "best_match_text": str or None,
    }
    """
    vector_store = _load_index()

    search_filter = {"student_name": student_name, "record_type": "contribution"}
    if project_id is not None:
        search_filter["project_id"] = project_id

    # fetch_k=200, same convention as check_semantic_corroboration() and retrieve_evidence().
    results = vector_store.similarity_search_with_relevance_scores(
        task_description, k=20, filter=search_filter, fetch_k=200
    )

    if not results:
        return {"best_score": 0.0, "best_match_text": None}

    best_doc, best_score = max(results, key=lambda pair: pair[1])
    return {"best_score": float(best_score), "best_match_text": best_doc.page_content}


if __name__ == "__main__":
    print(f"Loading data from {STUDENTS_PATH} and {RESEARCH_STUDENTS_PATH}...")
    docs = build_documents()
    print(f"Built {len(docs)} documents. Embedding and building FAISS index...")
    build_and_save_index()
    print(f"Index saved to {INDEX_DIR}")
