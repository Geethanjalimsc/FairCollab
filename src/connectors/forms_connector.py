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

# os is used to check whether the service account file actually exists
# before attempting to use it, so that failure mode gets a clear message.
import os

# re is used for the simple lowercase-word tokenizer the keyword-matching
# cross-reference step relies on.
import re

# gspread is the Google Sheets client this module reads every response
# through, once authorized with the google-auth credentials built below.
import gspread

# The gspread-specific exceptions raised when a spreadsheet can't be opened,
# so they can be translated into this module's own exception types.
from gspread.exceptions import SpreadsheetNotFound, APIError

# Credentials builds an OAuth2 credential object straight from a downloaded
# service account JSON key file -- this is the "google-auth" half of the
# "use google-auth and gspread" requirement (gspread itself only handles the
# Sheets API calls once it's been handed valid credentials).
from google.oauth2.service_account import Credentials

# The OAuth scopes this service account needs: read-only Sheets access to
# read responses, plus read-only Drive access, since gspread's open_by_key()
# looks the spreadsheet up via the Drive API under the hood.
SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets.readonly",
    "https://www.googleapis.com/auth/drive.readonly",
]

# The exact column header Google Forms writes for the task-assignment form's
# name question -- must match verbatim, since gspread keys rows by header text.
TASK_FORM_COL_STUDENT_NAME = "Your Name"

# The exact column header for the task-assignment form's task-list question.
TASK_FORM_COL_TASKS = "List all tasks you are responsible for (one per line)"

# The exact column header for the peer-review form's reviewer-name question.
PEER_FORM_COL_REVIEWER = "Your Name"

# The four students the peer review form has one rating/comment pair for.
PEER_FORM_STUDENT_LETTERS = ["A", "B", "C", "D"]

# A small set of common English filler words to drop before keyword-matching
# a task description against a commit message, so matches aren't driven by
# words that carry no real information about what the work actually was.
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
    # Check existence explicitly first, so a missing file gets our own clear
    # error instead of a lower-level "file not found" from inside google-auth.
    if not service_account_path or not os.path.exists(service_account_path):
        raise ServiceAccountFileNotFoundError(
            f"Service account file not found at '{service_account_path}'. "
            f"Download it from the Google Cloud console and place it there."
        )
    try:
        # Loads and parses the JSON key file into an OAuth2 credential object
        # scoped to exactly the two read-only permissions this module needs.
        credentials = Credentials.from_service_account_file(service_account_path, scopes=SCOPES)
    except (ValueError, KeyError) as exc:
        # from_service_account_file raises these when the file exists but
        # isn't a well-formed service account key (bad JSON, missing fields).
        raise ServiceAccountFileNotFoundError(
            f"'{service_account_path}' doesn't look like a valid service account JSON file: {exc}"
        ) from exc
    # gspread.authorize() wraps the credentials in a ready-to-use Sheets client.
    return gspread.authorize(credentials)


def _get_all_records(client: gspread.Client, spreadsheet_id: str, context: str) -> list:
    """Open a spreadsheet by ID and return every response row as a list of dicts."""
    try:
        # open_by_key() looks the spreadsheet up by its ID (the long string
        # in the spreadsheet's URL between /d/ and /edit).
        spreadsheet = client.open_by_key(spreadsheet_id)
    except SpreadsheetNotFound as exc:
        # gspread raises this specific exception when the ID doesn't resolve
        # to any spreadsheet the service account can see.
        raise SpreadsheetNotFoundError(
            f"No spreadsheet found while {context} (ID: {spreadsheet_id}). Check the ID is "
            f"correct and that the spreadsheet is shared with the service account's email."
        ) from exc
    except APIError as exc:
        # Other Sheets API failures (bad ID format, no permission, etc.) come
        # back as a generic APIError with an HTTP status code attached.
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
    # sheet1 is the first (leftmost) worksheet tab -- where Google Forms
    # writes every response by default.
    worksheet = spreadsheet.sheet1
    # get_all_records() reads the header row once and returns every
    # following row as {header: cell value} -- an empty list means the form
    # simply has no responses yet, which every caller treats as a normal,
    # non-error outcome rather than something to crash on.
    # Strip leading/trailing whitespace from all column header keys
    # because Google Forms sometimes adds spaces to question text
    # which causes dictionary lookups to silently return nothing.
    records = worksheet.get_all_records()
    return [{k.strip(): v for k, v in row.items()} for row in records]


def _split_tasks(raw_tasks_cell: str) -> list:
    """Split one task-list form answer into individual, trimmed task strings."""
    # Google Sheets stores a Google Forms "long answer" with embedded line
    # breaks as literal "\n" characters within the cell.
    lines = str(raw_tasks_cell).split("\n")
    # Strip whitespace off each line and drop any that are blank (e.g. a
    # trailing empty line from the respondent pressing Enter one extra time).
    return [line.strip() for line in lines if line.strip()]


def _parse_task_assignment_records(records: list) -> dict:
    """Turn task-assignment response rows into {student: {tasks_assigned, task_descriptions}}."""
    result = {}
    for row in records:
        # .strip() guards against a name entered with stray leading/trailing spaces.
        student_name = str(row.get(TASK_FORM_COL_STUDENT_NAME, "")).strip()
        if not student_name:
            # A row with no name can't be attributed to anyone -- skip it
            # rather than crashing the whole fetch over one malformed row.
            continue
        tasks = _split_tasks(row.get(TASK_FORM_COL_TASKS, ""))
        # setdefault so a student who submitted the form more than once has
        # their tasks accumulated across every response, not just the last one.
        entry = result.setdefault(student_name, {"tasks_assigned": 0, "task_descriptions": []})
        entry["tasks_assigned"] += len(tasks)
        entry["task_descriptions"].extend(tasks)
    return result


def fetch_task_assignments(spreadsheet_id: str, service_account_path: str) -> dict:
    """Fetch every task-assignment response and return {student: {tasks_assigned, task_descriptions}}."""
    if not spreadsheet_id or not spreadsheet_id.strip():
        # Validate before touching the network, since a blank ID can only ever fail.
        raise InvalidSpreadsheetIDError("No spreadsheet ID was provided.")
    client = _get_gspread_client(service_account_path)
    records = _get_all_records(client, spreadsheet_id.strip(), context="reading task assignments")
    # If records is [] (no responses yet), this simply returns {} -- a valid,
    # non-error result every caller can treat the same as "nothing fetched".
    return _parse_task_assignment_records(records)


def _rating_column(letter: str) -> str:
    return f"Overall contribution rating (1-5) for Student {letter}"

def _comment_column(letter: str) -> str:
    return f"Written comment for Student {letter}"


def _parse_rating(raw_rating) -> int:
    """Parse a rating cell (which may arrive as int, float, or numeric string) into an int, or None."""
    if raw_rating in ("", None):
        # An entirely blank rating cell means this reviewer left that
        # student's block empty -- not a parse failure, just "no rating given".
        return None
    try:
        # float() first handles Sheets sometimes returning "4.0"-style values
        # for what was really an integer 1-5 rating in the form.
        return int(float(raw_rating))
    except (TypeError, ValueError):
        # Anything that isn't numeric at all (e.g. stray text) is treated as
        # unusable rather than raising and aborting the whole fetch.
        return None


def _parse_peer_review_records(records: list) -> dict:
    """Turn peer-review response rows into {reviewed student: [feedback string, ...]}."""
    result = {}
    for row in records:
        reviewer = str(row.get(PEER_FORM_COL_REVIEWER, "")).strip()
        for letter in PEER_FORM_STUDENT_LETTERS:
            reviewed_student = f"Student {letter}"
            if reviewer == reviewed_student:
                # Skip self-reviews -- a student's own opinion of their own
                # contribution isn't peer evidence.
                continue
            rating = _parse_rating(row.get(_rating_column(letter)))
            comment = str(row.get(_comment_column(letter), "")).strip()
            if rating is None and not comment:
                # Both the rating and comment for this student are blank,
                # meaning this reviewer simply didn't review them in this
                # response -- not every reviewer rates every student.
                continue
            if rating is None:
                # There's a comment but no parseable rating -- still worth
                # keeping as qualitative evidence, just without a "N/5" prefix.
                feedback_line = comment
            elif comment:
                feedback_line = f"Rating {rating}/5: {comment}"
            else:
                # A rating with no comment -- still valid, standalone evidence.
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
    # Pull out alphanumeric runs only, lowercased, so punctuation/case never
    # affects whether two pieces of text are considered to "share a word".
    words = re.findall(r"[a-z0-9]+", str(text).lower())
    # Very short words (1-3 letters) carry little meaning on their own and
    # would cause spurious matches (e.g. "the", "for", "add"), so both a
    # minimum length and the explicit stopword list filter them out.
    return {word for word in words if len(word) > 3 and word not in _STOPWORDS}


def _keywords_overlap(task_keywords: set, commit_keywords: set) -> bool:
    """Decide whether a task and a commit are "about the same thing", by simple keyword overlap."""
    if not task_keywords or not commit_keywords:
        # If either side has no meaningful keywords at all (e.g. a task
        # description that was just "misc"), there's nothing to match on.
        return False
    # This is deliberately simple keyword matching, not semantic/NLP
    # matching: any single shared meaningful word counts as a match. It will
    # produce some false positives/negatives, but that's the explicitly
    # requested trade-off in exchange for being cheap and easy to reason about.
    return bool(task_keywords & commit_keywords)


def cross_reference_with_commits(task_assignments: dict, contributions_by_student: dict) -> dict:
    """Estimate tasks_completed per student by keyword-matching tasks against commit messages.

    task_assignments: fetch_task_assignments()'s output, {student: {tasks_assigned, task_descriptions}}.
    contributions_by_student: {student: [contribution record, ...]} -- each
        record is expected to have a "description" field (as produced by
        src.connectors.github_connector's contribution records).

    Returns {student: tasks_completed_count}.
    """
    tasks_completed = {}
    for student, assignment in task_assignments.items():
        # Pre-compute this student's commit descriptions' keyword sets once,
        # rather than re-tokenizing the same commit messages per task.
        commit_keyword_sets = [
            _keywords(record.get("description", ""))
            for record in contributions_by_student.get(student, [])
        ]
        completed_count = 0
        for task_description in assignment.get("task_descriptions", []):
            task_keywords = _keywords(task_description)
            # A task counts as completed if it matches at least one commit;
            # once one match is found there's no need to check the rest.
            if any(_keywords_overlap(task_keywords, ck) for ck in commit_keyword_sets):
                completed_count += 1
        tasks_completed[student] = completed_count
    return tasks_completed


def build_student_summary(task_assignments: dict, peer_reviews: dict, tasks_completed: dict) -> dict:
    """Combine the three fetch results into the module's documented per-student shape."""
    # A student might appear in only one or two of the three inputs (e.g. a
    # student with peer reviews but who hasn't submitted the task form yet),
    # so the summary covers the union of every student name seen anywhere.
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
