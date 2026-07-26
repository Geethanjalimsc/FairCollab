"""
Google Forms data connector for FairCollab.

Reads two Google Sheets that Google Forms writes responses into -- a task
assignment form and a peer review form -- via the Google Sheets API using a
service account, and converts each into FairCollab's data shapes. Also
cross-references assigned tasks against GitHub commit messages (simple
keyword overlap, not NLP) to estimate how many assigned tasks were completed.

The fully combined per-student shape this module documents (and that
build_student_summary() produces) is:

    {
        "Student A": {
            "tasks_assigned": 3,
            "tasks_completed": 2,
            "peer_feedback": ["Rating 4/5: ...", "Rating 5/5: ..."],
            "task_descriptions": ["Implement SLAM algorithm", "Write Docker setup"],
        },
        ...
    }
"""

import os
import re
import gspread

from gspread.exceptions import SpreadsheetNotFound, APIError
from google.oauth2.service_account import Credentials

SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets.readonly",
    "https://www.googleapis.com/auth/drive.readonly",
]

# Must match the form's column headers verbatim -- gspread keys rows by header text.
TASK_FORM_COL_STUDENT_NAME = "Your Name"
TASK_FORM_COL_TASKS = "List all tasks you are responsible for (one per line)"
PEER_FORM_COL_REVIEWER = "Your Name"

PEER_FORM_STUDENT_LETTERS = ["A", "B", "C", "D"]

# Filler words dropped before keyword-matching a task against a commit message.
_STOPWORDS = {
    "the", "and", "for", "with", "that", "from", "this", "have", "will",
    "been", "into", "your", "you", "are", "was", "were", "all", "any",
    "our", "not", "but", "can", "get", "got", "out", "use", "used",
}


class FormsConnectorError(Exception):
    """Base class for every error this connector can raise."""


class ServiceAccountFileNotFoundError(FormsConnectorError):
    """Raised when the given service account JSON file doesn't exist or isn't valid."""


class SpreadsheetNotFoundError(FormsConnectorError):
    """Raised when Google reports no such spreadsheet (or no access to it)."""


class InvalidSpreadsheetIDError(FormsConnectorError):
    """Raised when the given spreadsheet ID is blank or malformed."""


def _get_gspread_client(service_account_path: str) -> gspread.Client:
    """Build an authorized gspread client from a service account JSON file."""
    if not service_account_path or not os.path.exists(service_account_path):
        raise ServiceAccountFileNotFoundError(
            f"Service account file not found at '{service_account_path}'. "
            f"Download it from the Google Cloud console and place it there."
        )
    try:
        credentials = Credentials.from_service_account_file(service_account_path, scopes=SCOPES)
    except (ValueError, KeyError) as exc:
        raise ServiceAccountFileNotFoundError(
            f"'{service_account_path}' doesn't look like a valid service account JSON file: {exc}"
        ) from exc
    return gspread.authorize(credentials)


def _get_all_records(client: gspread.Client, spreadsheet_id: str, context: str) -> list:
    """Open a spreadsheet by ID and return every response row as a list of dicts."""
    try:
        spreadsheet = client.open_by_key(spreadsheet_id)
    except SpreadsheetNotFound as exc:
        raise SpreadsheetNotFoundError(
            f"No spreadsheet found while {context} (ID: {spreadsheet_id}). Check the ID is "
            f"correct and that the spreadsheet is shared with the service account's email."
        ) from exc
    except APIError as exc:
        status_code = getattr(getattr(exc, "response", None), "status_code", None)
        if status_code == 404:
            raise SpreadsheetNotFoundError(
                f"No spreadsheet found while {context} (ID: {spreadsheet_id})."
            ) from exc
        if status_code == 400:
            raise InvalidSpreadsheetIDError(
                f"'{spreadsheet_id}' is not a valid spreadsheet ID."
            ) from exc
        raise FormsConnectorError(f"Google Sheets API error while {context}: {exc}") from exc
    # sheet1 is the tab Google Forms writes responses into.
    worksheet = spreadsheet.sheet1
    # Strip header whitespace -- Google Forms sometimes pads question text
    # with spaces, which would otherwise make dict lookups silently miss.
    records = worksheet.get_all_records()
    return [{k.strip(): v for k, v in row.items()} for row in records]


def _split_tasks(raw_tasks_cell: str) -> list:
    """Split one task-list form answer into individual, trimmed task strings."""
    lines = str(raw_tasks_cell).split("\n")
    return [line.strip() for line in lines if line.strip()]


def _parse_task_assignment_records(records: list) -> dict:
    """Turn task-assignment response rows into {student: {tasks_assigned, task_descriptions}}."""
    result = {}
    for row in records:
        student_name = str(row.get(TASK_FORM_COL_STUDENT_NAME, "")).strip()
        if not student_name:
            continue
        tasks = _split_tasks(row.get(TASK_FORM_COL_TASKS, ""))
        # setdefault so multiple submissions from the same student accumulate.
        entry = result.setdefault(student_name, {"tasks_assigned": 0, "task_descriptions": []})
        entry["tasks_assigned"] += len(tasks)
        entry["task_descriptions"].extend(tasks)
    return result


def fetch_task_assignments(spreadsheet_id: str, service_account_path: str) -> dict:
    """Fetch every task-assignment response and return {student: {tasks_assigned, task_descriptions}}."""
    if not spreadsheet_id or not spreadsheet_id.strip():
        raise InvalidSpreadsheetIDError("No spreadsheet ID was provided.")
    client = _get_gspread_client(service_account_path)
    records = _get_all_records(client, spreadsheet_id.strip(), context="reading task assignments")
    return _parse_task_assignment_records(records)


def _rating_column(letter: str) -> str:
    return f"Overall contribution rating (1-5) for Student {letter}"

def _comment_column(letter: str) -> str:
    return f"Written comment for Student {letter}"


def _parse_rating(raw_rating) -> int:
    """Parse a rating cell (int, float, or numeric string) into an int, or None."""
    if raw_rating in ("", None):
        return None
    try:
        # float() first handles Sheets returning "4.0" for an integer rating.
        return int(float(raw_rating))
    except (TypeError, ValueError):
        return None


def _parse_peer_review_records(records: list) -> dict:
    """Turn peer-review response rows into {reviewed student: [feedback string, ...]}."""
    result = {}
    for row in records:
        reviewer = str(row.get(PEER_FORM_COL_REVIEWER, "")).strip()
        for letter in PEER_FORM_STUDENT_LETTERS:
            reviewed_student = f"Student {letter}"
            if reviewer == reviewed_student:
                continue  # skip self-review
            rating = _parse_rating(row.get(_rating_column(letter)))
            comment = str(row.get(_comment_column(letter), "")).strip()
            if rating is None and not comment:
                continue  # this reviewer didn't review this student
            if rating is None:
                feedback_line = comment
            elif comment:
                feedback_line = f"Rating {rating}/5: {comment}"
            else:
                feedback_line = f"Rating {rating}/5"
            result.setdefault(reviewed_student, []).append(feedback_line)
    return result


def fetch_peer_reviews(spreadsheet_id: str, service_account_path: str) -> dict:
    """Fetch every peer-review response and return {reviewed student: [feedback string, ...]}."""
    if not spreadsheet_id or not spreadsheet_id.strip():
        raise InvalidSpreadsheetIDError("No spreadsheet ID was provided.")
    client = _get_gspread_client(service_account_path)
    records = _get_all_records(client, spreadsheet_id.strip(), context="reading peer reviews")
    return _parse_peer_review_records(records)


def _keywords(text: str) -> set:
    """Reduce free text to a lowercase set of "meaningful" word tokens for simple matching."""
    words = re.findall(r"[a-z0-9]+", str(text).lower())
    return {word for word in words if len(word) > 3 and word not in _STOPWORDS}


def _keywords_overlap(task_keywords: set, commit_keywords: set) -> bool:
    """Simple keyword matching (not NLP): any shared meaningful word counts as a match."""
    if not task_keywords or not commit_keywords:
        return False
    return bool(task_keywords & commit_keywords)


def cross_reference_with_commits(task_assignments: dict, contributions_by_student: dict) -> dict:
    """Estimate tasks_completed per student by keyword-matching tasks against commit messages.

    task_assignments: fetch_task_assignments()'s output.
    contributions_by_student: {student: [contribution record, ...]}, each with a "description".
    Returns {student: tasks_completed_count}.
    """
    tasks_completed = {}
    for student, assignment in task_assignments.items():
        commit_keyword_sets = [
            _keywords(record.get("description", ""))
            for record in contributions_by_student.get(student, [])
        ]
        completed_count = 0
        for task_description in assignment.get("task_descriptions", []):
            task_keywords = _keywords(task_description)
            if any(_keywords_overlap(task_keywords, ck) for ck in commit_keyword_sets):
                completed_count += 1
        tasks_completed[student] = completed_count
    return tasks_completed


def build_student_summary(task_assignments: dict, peer_reviews: dict, tasks_completed: dict) -> dict:
    """Combine the three fetch results into the module's documented per-student shape."""
    all_students = set(task_assignments) | set(peer_reviews) | set(tasks_completed)
    summary = {}
    for student in sorted(all_students):
        assignment = task_assignments.get(student, {})
        summary[student] = {
            "tasks_assigned": assignment.get("tasks_assigned", 0),
            "tasks_completed": tasks_completed.get(student, 0),
            "peer_feedback": peer_reviews.get(student, []),
            "task_descriptions": assignment.get("task_descriptions", []),
        }
    return summary
