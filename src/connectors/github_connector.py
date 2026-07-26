"""
GitHub data connector for FairCollab.

Fetches every commit from a GitHub repository via the REST API v3, maps each
commit to a student using an email -> student-name dictionary, and converts
the matched commits into FairCollab's existing contribution-record schema
(the same shape used by data/students.json and data/research_students.json)
so they can be merged into the rest of the assessment pipeline.
"""

import re
import logging
import requests

logger = logging.getLogger(__name__)

GITHUB_API_BASE = "https://api.github.com"

# Pinned so responses don't silently change shape if GitHub bumps the default.
GITHUB_API_VERSION = "2022-11-28"

REQUEST_TIMEOUT_SECONDS = 30

# 100 is GitHub's max per page.
COMMITS_PER_PAGE = 100


class GitHubConnectorError(Exception):
    """Base class for every error this connector can raise."""


class InvalidRepositoryURLError(GitHubConnectorError):
    """Raised when the given repository URL isn't a recognizable GitHub URL."""


class AuthenticationError(GitHubConnectorError):
    """Raised when the PAT is missing, wrong, expired, or lacks access."""


class RepositoryNotFoundError(GitHubConnectorError):
    """Raised when GitHub reports no such repository (or no access to it)."""


class RateLimitExceededError(GitHubConnectorError):
    """Raised when GitHub's rate limit (5000 requests/hour with a PAT) is hit."""


def _parse_repo_url(repo_url: str) -> tuple:
    """Extract (owner, repo) from a GitHub URL, raising if it isn't one."""
    cleaned_url = repo_url.strip()
    # HTTPS form, with/without www./trailing slash/.git; SSH form as a fallback.
    https_pattern = r"^(?:https?://)?(?:www\.)?github\.com/([A-Za-z0-9_.-]+)/([A-Za-z0-9_.-]+?)(?:\.git)?/?$"
    ssh_pattern = r"^git@github\.com:([A-Za-z0-9_.-]+)/([A-Za-z0-9_.-]+?)(?:\.git)?$"
    match = re.match(https_pattern, cleaned_url) or re.match(ssh_pattern, cleaned_url)
    if not match:
        raise InvalidRepositoryURLError(
            f"'{repo_url}' doesn't look like a GitHub repository URL. "
            f"Expected something like https://github.com/owner/repo."
        )
    return match.group(1), match.group(2)


def _build_headers(pat: str) -> dict:
    """Build the headers every GitHub API request needs."""
    return {
        "Authorization": f"Bearer {pat}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": GITHUB_API_VERSION,
    }


def _raise_for_response_errors(response: requests.Response, context: str) -> None:
    """Translate a non-OK GitHub API response into a specific connector error."""
    if response.status_code == 401:
        raise AuthenticationError(
            f"GitHub rejected the Personal Access Token while {context}. "
            f"Check that it's correct and hasn't expired."
        )
    if response.status_code == 403:
        # GitHub overloads 403 for both "no permission" and "rate limited";
        # X-RateLimit-Remaining: 0 is how you tell them apart.
        remaining = response.headers.get("X-RateLimit-Remaining")
        if remaining == "0":
            reset_timestamp = response.headers.get("X-RateLimit-Reset", "unknown")
            raise RateLimitExceededError(
                f"GitHub API rate limit exceeded while {context}. "
                f"The limit resets at Unix timestamp {reset_timestamp} "
                f"(GitHub allows 5000 requests/hour with a PAT)."
            )
        raise AuthenticationError(
            f"GitHub denied access while {context} (HTTP 403). The PAT may be "
            f"missing the required scope, or the repository may be private."
        )
    if response.status_code == 404:
        # GitHub returns 404 (not 403) for a private repo the token can't see,
        # to avoid leaking that the repo exists at all.
        raise RepositoryNotFoundError(
            f"Repository not found while {context}. Check the URL is correct, "
            f"and that the PAT has access if the repository is private."
        )
    if not response.ok:
        raise GitHubConnectorError(
            f"GitHub API request failed while {context}: "
            f"HTTP {response.status_code} -- {response.text[:200]}"
        )


def _get_json(url: str, headers: dict, params: dict, context: str):
    """Perform one GET request and return its parsed JSON, with unified error handling."""
    try:
        response = requests.get(url, headers=headers, params=params, timeout=REQUEST_TIMEOUT_SECONDS)
    except requests.exceptions.RequestException as exc:
        raise GitHubConnectorError(f"Network error while {context}: {exc}") from exc
    _raise_for_response_errors(response, context)
    return response.json()


def _fetch_commit_list(owner: str, repo: str, pat: str) -> list:
    """Fetch every commit's summary (SHA, author, date, message) via pagination."""
    headers = _build_headers(pat)
    all_commits = []
    page = 1
    while True:
        url = f"{GITHUB_API_BASE}/repos/{owner}/{repo}/commits"
        params = {"per_page": COMMITS_PER_PAGE, "page": page}
        page_commits = _get_json(url, headers, params, context=f"listing commits (page {page})")
        if not page_commits:
            break
        all_commits.extend(page_commits)
        page += 1
    return all_commits


def _fetch_commit_detail(owner: str, repo: str, sha: str, pat: str) -> dict:
    """Fetch one commit's full detail (stats + changed files) by SHA."""
    headers = _build_headers(pat)
    url = f"{GITHUB_API_BASE}/repos/{owner}/{repo}/commits/{sha}"
    # Only the single-commit endpoint includes "stats"/"files"; the list endpoint omits both.
    return _get_json(url, headers, params=None, context=f"fetching details for commit {sha[:7]}")


def _infer_complexity(total_lines_changed: int) -> str:
    """Map a total lines-changed count onto FairCollab's complexity scale."""
    if total_lines_changed < 10:
        return "trivial"
    if total_lines_changed < 50:
        return "low"
    if total_lines_changed < 200:
        return "medium"
    return "high"


def _commit_to_record(commit_summary: dict, commit_detail: dict) -> dict:
    """Convert one commit's list-entry + detail into a contribution record."""
    git_commit = commit_summary.get("commit", {})
    # Use just the summary line -- commit bodies can be long/multi-line.
    full_message = git_commit.get("message", "")
    description = full_message.split("\n", 1)[0].strip()
    # ISO 8601 timestamp -- first 10 chars are always "YYYY-MM-DD".
    author_date = git_commit.get("author", {}).get("date", "")
    date_only = author_date[:10]
    stats = commit_detail.get("stats", {})
    additions = stats.get("additions", 0)
    deletions = stats.get("deletions", 0)
    total_lines_changed = additions + deletions
    # Keep filenames only, not the diff/patch text.
    files_changed = [f.get("filename") for f in commit_detail.get("files", [])]
    return {
        "date": date_only,
        "type": "code",
        "description": description,
        "complexity": _infer_complexity(total_lines_changed),
        "status": "completed",
        "metadata": {
            "sha": commit_summary.get("sha"),
            "additions": additions,
            "deletions": deletions,
            "files_changed": files_changed,
            "platform": "github",
        },
    }


def fetch_and_map_commits(repo_url: str, pat: str, student_mapping: dict) -> dict:
    """Fetch every commit from a GitHub repo and group it into per-student contribution records.

    student_mapping: {commit author email: student name}.
    Returns {student name: [contribution record, ...]}. Commits whose author
    email isn't in student_mapping are skipped (and logged), not raised as errors.
    """
    if not pat:
        raise AuthenticationError("No Personal Access Token was provided.")
    owner, repo = _parse_repo_url(repo_url)
    commit_summaries = _fetch_commit_list(owner, repo, pat)
    student_records = {}
    for commit_summary in commit_summaries:
        author_email = commit_summary.get("commit", {}).get("author", {}).get("email")
        if not author_email or author_email not in student_mapping:
            logger.warning(
                "Skipping commit %s: author email %r is not in the student mapping.",
                commit_summary.get("sha", "?")[:7],
                author_email,
            )
            continue
        student_name = student_mapping[author_email]
        # Only fetch detail (extra API call) once we know the commit is actually used.
        commit_detail = _fetch_commit_detail(owner, repo, commit_summary["sha"], pat)
        record = _commit_to_record(commit_summary, commit_detail)
        student_records.setdefault(student_name, []).append(record)
    return student_records


def detect_contributors(repo_url: str, pat: str) -> list:
    """Fetch all unique commit authors from a GitHub repo.

    Returns a list of dicts: [{"email": "...", "name": "..."}, ...]
    Raises the same exceptions as fetch_and_map_commits.
    """
    if not pat:
        raise AuthenticationError("No Personal Access Token was provided.")

    owner, repo = _parse_repo_url(repo_url)

    # Lightweight: no per-commit detail needed since author info is on the list entry.
    commit_summaries = _fetch_commit_list(owner, repo, pat)

    # dict preserves first-seen order and dedupes by email.
    seen_emails = {}

    for commit in commit_summaries:
        author = commit.get("commit", {}).get("author", {})
        email = author.get("email")
        name = author.get("name")

        if not email:
            continue

        if email not in seen_emails:
            seen_emails[email] = name or email

    return [{"email": email, "name": name}
            for email, name in seen_emails.items()]
