"""
GitHub data connector for FairCollab.

Fetches every commit from a GitHub repository via the REST API v3, maps each
commit to a student using an email -> student-name dictionary, and converts
the matched commits into FairCollab's existing contribution-record schema
(the same shape used by data/students.json and data/research_students.json)
so they can be merged into the rest of the assessment pipeline.
"""

# re is used to parse a GitHub repo URL into its (owner, repo) components.
import re

# logging lets us record non-fatal issues (e.g. a commit whose author email
# isn't in the student mapping) without raising and aborting the whole fetch.
import logging

# requests is the HTTP client used for every call to the GitHub REST API.
import requests

# This module's own logger, namespaced so its messages are identifiable
# separately from any other logger the host application configures.
logger = logging.getLogger(__name__)

# Every GitHub REST API v3 call is made against this fixed base URL.
GITHUB_API_BASE = "https://api.github.com"

# The API version header GitHub's REST API v3 expects; pinning it means
# behavior won't silently change if GitHub rolls out a new default version.
GITHUB_API_VERSION = "2022-11-28"

# How long (in seconds) to wait for any single GitHub API call before giving
# up, so a network hang can't freeze the whole fetch indefinitely.
REQUEST_TIMEOUT_SECONDS = 30

# How many commits to request per page from the list-commits endpoint; 100
# is the maximum GitHub allows, so this minimizes the number of list calls.
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
    # Strip surrounding whitespace so a pasted-with-spaces URL still parses.
    cleaned_url = repo_url.strip()
    # Matches "https://github.com/owner/repo", with or without "www.", a
    # trailing "/" or ".git", or a scheme-less "github.com/owner/repo".
    https_pattern = r"^(?:https?://)?(?:www\.)?github\.com/([A-Za-z0-9_.-]+)/([A-Za-z0-9_.-]+?)(?:\.git)?/?$"
    # Matches the SSH form "git@github.com:owner/repo.git" some users paste instead.
    ssh_pattern = r"^git@github\.com:([A-Za-z0-9_.-]+)/([A-Za-z0-9_.-]+?)(?:\.git)?$"
    # Try the HTTPS form first since it's the overwhelmingly common case.
    match = re.match(https_pattern, cleaned_url) or re.match(ssh_pattern, cleaned_url)
    if not match:
        # Neither pattern matched, so this isn't a URL we can act on.
        raise InvalidRepositoryURLError(
            f"'{repo_url}' doesn't look like a GitHub repository URL. "
            f"Expected something like https://github.com/owner/repo."
        )
    # group(1) is the owner/org, group(2) is the repository name.
    return match.group(1), match.group(2)


def _build_headers(pat: str) -> dict:
    """Build the headers every GitHub API request needs."""
    return {
        # "Bearer <token>" is GitHub's current recommended auth scheme for PATs.
        "Authorization": f"Bearer {pat}",
        # Requests the GitHub-specific JSON media type, per GitHub's own docs.
        "Accept": "application/vnd.github+json",
        # Pins the API version so responses stay in the shape this code expects.
        "X-GitHub-Api-Version": GITHUB_API_VERSION,
    }


def _raise_for_response_errors(response: requests.Response, context: str) -> None:
    """Translate a non-OK GitHub API response into a specific connector error."""
    # 401 always means the token itself was rejected outright.
    if response.status_code == 401:
        raise AuthenticationError(
            f"GitHub rejected the Personal Access Token while {context}. "
            f"Check that it's correct and hasn't expired."
        )
    # 403 is GitHub's overloaded "forbidden" code -- it means either the
    # token lacks permission, or (much more commonly) the rate limit is spent.
    if response.status_code == 403:
        # A remaining count of exactly "0" is GitHub's signal that this 403
        # is a rate-limit rejection rather than a permissions problem.
        remaining = response.headers.get("X-RateLimit-Remaining")
        if remaining == "0":
            # The reset time is a Unix timestamp telling the caller when
            # their quota refills, so surface it directly in the message.
            reset_timestamp = response.headers.get("X-RateLimit-Reset", "unknown")
            raise RateLimitExceededError(
                f"GitHub API rate limit exceeded while {context}. "
                f"The limit resets at Unix timestamp {reset_timestamp} "
                f"(GitHub allows 5000 requests/hour with a PAT)."
            )
        # Otherwise this is a genuine permissions/scope problem, not quota.
        raise AuthenticationError(
            f"GitHub denied access while {context} (HTTP 403). The PAT may be "
            f"missing the required scope, or the repository may be private."
        )
    # 404 covers both "repo truly doesn't exist" and "repo is private and
    # this token can't see it" -- GitHub deliberately doesn't distinguish
    # these to avoid leaking the existence of private repos.
    if response.status_code == 404:
        raise RepositoryNotFoundError(
            f"Repository not found while {context}. Check the URL is correct, "
            f"and that the PAT has access if the repository is private."
        )
    # Anything else non-OK is an error this module doesn't special-case, but
    # it still shouldn't propagate as a raw requests exception to the caller.
    if not response.ok:
        raise GitHubConnectorError(
            f"GitHub API request failed while {context}: "
            f"HTTP {response.status_code} -- {response.text[:200]}"
        )


def _get_json(url: str, headers: dict, params: dict, context: str):
    """Perform one GET request and return its parsed JSON, with unified error handling."""
    try:
        # A single GET call; timeout guards against a hung connection.
        response = requests.get(url, headers=headers, params=params, timeout=REQUEST_TIMEOUT_SECONDS)
    except requests.exceptions.RequestException as exc:
        # Wraps network-level failures (DNS, connection refused, timeout) in
        # our own exception type so callers only ever need to catch one thing.
        raise GitHubConnectorError(f"Network error while {context}: {exc}") from exc
    # Convert any non-OK status into the appropriate specific exception.
    _raise_for_response_errors(response, context)
    # response.ok was True, so this is safe to parse as JSON.
    return response.json()


def _fetch_commit_list(owner: str, repo: str, pat: str) -> list:
    """Fetch every commit's summary (SHA, author, date, message) via pagination."""
    headers = _build_headers(pat)
    all_commits = []
    # GitHub paginates via a "page" query parameter; there's no total-count
    # header to rely on, so we keep requesting pages until one comes back empty.
    page = 1
    while True:
        url = f"{GITHUB_API_BASE}/repos/{owner}/{repo}/commits"
        params = {"per_page": COMMITS_PER_PAGE, "page": page}
        page_commits = _get_json(url, headers, params, context=f"listing commits (page {page})")
        # An empty list means we've paged past the last commit.
        if not page_commits:
            break
        all_commits.extend(page_commits)
        # Move to the next page for the next loop iteration.
        page += 1
    return all_commits


def _fetch_commit_detail(owner: str, repo: str, sha: str, pat: str) -> dict:
    """Fetch one commit's full detail (stats + changed files) by SHA."""
    headers = _build_headers(pat)
    url = f"{GITHUB_API_BASE}/repos/{owner}/{repo}/commits/{sha}"
    # The single-commit endpoint is the only one that includes "stats" and
    # "files" -- the list-commits endpoint used above omits both.
    return _get_json(url, headers, params=None, context=f"fetching details for commit {sha[:7]}")


def _infer_complexity(total_lines_changed: int) -> str:
    """Map a total lines-changed count onto FairCollab's complexity scale."""
    # Boundaries are defined so each integer value falls in exactly one
    # bucket: trivial < 10 <= low < 50 <= medium < 200 <= high.
    if total_lines_changed < 10:
        return "trivial"
    if total_lines_changed < 50:
        return "low"
    if total_lines_changed < 200:
        return "medium"
    return "high"


def _commit_to_record(commit_summary: dict, commit_detail: dict) -> dict:
    """Convert one commit's list-entry + detail into a contribution record."""
    # commit_summary["commit"] holds the raw git commit metadata (as opposed
    # to commit_summary["author"], which is the linked GitHub *account*, if any).
    git_commit = commit_summary.get("commit", {})
    # The commit message's first line is used as the concise description,
    # since commit messages can have long multi-line bodies below the summary.
    full_message = git_commit.get("message", "")
    description = full_message.split("\n", 1)[0].strip()
    # The author date looks like "2026-06-10T14:23:00Z"; the first 10
    # characters are always the "YYYY-MM-DD" part of an ISO 8601 timestamp.
    author_date = git_commit.get("author", {}).get("date", "")
    date_only = author_date[:10]
    # "stats" only exists on the detailed single-commit response.
    stats = commit_detail.get("stats", {})
    additions = stats.get("additions", 0)
    deletions = stats.get("deletions", 0)
    total_lines_changed = additions + deletions
    # "files" is the list of per-file change objects; we only keep the name
    # ("filename") for each, discarding the diff/patch text and per-file stats.
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

    repo_url: a GitHub repository URL, e.g. "https://github.com/owner/repo".
    pat: a GitHub Personal Access Token with at least read access to the repo.
    student_mapping: dict of {commit author email: student name} used to
        attribute each commit to a student.

    Returns a dict of {student name: [contribution record, ...]}. Commits
    whose author email isn't in student_mapping are skipped (and logged),
    not raised as errors, since an unmapped commit is an expected, recoverable
    situation rather than a failure of the fetch itself.
    """
    if not pat:
        # Fail fast with a clear message rather than letting every API call
        # fail one-by-one with a less obvious 401.
        raise AuthenticationError("No Personal Access Token was provided.")
    # Parse and validate the URL before making any network calls at all.
    owner, repo = _parse_repo_url(repo_url)
    # Fetch the lightweight list of every commit first (cheap: 1 call per 100 commits).
    commit_summaries = _fetch_commit_list(owner, repo, pat)
    # This is where matched commits accumulate, keyed by student name.
    student_records = {}
    for commit_summary in commit_summaries:
        # Pull the raw git author email out of the commit metadata.
        author_email = commit_summary.get("commit", {}).get("author", {}).get("email")
        if not author_email or author_email not in student_mapping:
            # Not every commit necessarily belongs to a tracked student (e.g.
            # a bot, a TA, or a contributor not in the mapping) -- skip it
            # rather than failing the whole fetch over one unmatched commit.
            logger.warning(
                "Skipping commit %s: author email %r is not in the student mapping.",
                commit_summary.get("sha", "?")[:7],
                author_email,
            )
            continue
        student_name = student_mapping[author_email]
        # Only now fetch the expensive per-commit detail (stats + files),
        # since there's no point spending an API call on an unmatched commit.
        commit_detail = _fetch_commit_detail(owner, repo, commit_summary["sha"], pat)
        record = _commit_to_record(commit_summary, commit_detail)
        # Append this record to the matched student's growing list, creating
        # the list on first encounter with setdefault.
        student_records.setdefault(student_name, []).append(record)
    return student_records


def detect_contributors(repo_url: str, pat: str) -> list:
    """Fetch all unique commit authors from a GitHub repo.

    Returns a list of dicts: [{"email": "...", "name": "..."}, ...]
    Raises the same exceptions as fetch_and_map_commits.
    """
    # Validate PAT before making any network calls -- fail fast with a clear
    # message rather than letting the first API call fail with a less obvious 401.
    if not pat:
        raise AuthenticationError("No Personal Access Token was provided.")

    # Parse and validate the URL -- raises InvalidRepositoryURLError if it
    # isn't a recognizable GitHub URL, before any network call is made.
    owner, repo = _parse_repo_url(repo_url)

    # Fetch all commit summaries (lightweight, no per-commit detail needed) --
    # raises AuthenticationError, RepositoryNotFoundError, or
    # RateLimitExceededError as appropriate.
    commit_summaries = _fetch_commit_list(owner, repo, pat)

    # Use a dict keyed by email to deduplicate contributors -- a plain dict
    # preserves insertion order, so contributors come out in first-seen order.
    seen_emails = {}

    for commit in commit_summaries:
        # Extract author info from the git commit metadata -- the raw git
        # identity attached to the commit, not necessarily a linked GitHub account.
        author = commit.get("commit", {}).get("author", {})
        # The author's email address -- used as the dict key for deduping.
        email = author.get("email")
        # The author's git-configured display name -- used as the suggested name.
        name = author.get("name")

        # Skip commits with no email address -- nothing to key a contributor on.
        if not email:
            continue

        # Only keep first occurrence of each email (preserves order) -- a
        # later commit with a different name for the same person won't
        # overwrite the first (usually representative) name.
        if email not in seen_emails:
            seen_emails[email] = name or email

    # Return as list of dicts in order of first appearance, matching the
    # documented [{"email": ..., "name": ...}, ...] shape.
    return [{"email": email, "name": name}
            for email, name in seen_emails.items()]
