"""
task_1_extract_tasks.py

For each repo in dataset/test.json, extracts evaluation tasks of three types:
  - bug_fix        : closed bug issues with a linked fix commit
  - feature_impl   : merged PRs with clear descriptions (non-bug, non-chore)
  - code_review    : reverted or explicitly problematic PRs

Each task record includes a belief_cutoff_timestamp (issue/PR open time) so
the belief store can be queried without any look-ahead leakage.

Output: dataset/tasks/<repo_name>.json per repo
        dataset/tasks/all_tasks.json  (merged)

Usage:
    python scripts/randomized_controlled_trial/task_1_extract_tasks.py
    python scripts/randomized_controlled_trial/task_1_extract_tasks.py --repo fmt --limit 3
"""

import os
import re
import json
import time
import argparse
import subprocess
from pathlib import Path
from typing import List, Optional

import requests
from dotenv import load_dotenv
from tqdm import tqdm

from utils.logger import get_logger

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

BASE_DIR  = Path(__file__).resolve().parent.parent.parent
DATASET   = BASE_DIR / "dataset"
TASKS_DIR = DATASET / "tasks"
CLONE_DIR = Path("/tmp/belief_exp_repos")

TASKS_PER_TYPE = 3
SLEEP_BETWEEN  = 0.5

load_dotenv(BASE_DIR / ".env")

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# GitHubAPI — remote HTTP client
# ---------------------------------------------------------------------------

class GitHubAPI:
    _BASE = "https://api.github.com"

    def __init__(self, token: str, sleep_between: float = SLEEP_BETWEEN):
        self._headers = {
            "Authorization": f"token {token}",
            "User-Agent": "belief-experiment-research/1.0",
            "Accept": "application/vnd.github+json",
        }
        self._sleep = sleep_between

    def _get(self, url: str, params: dict = None, accept: str = None) -> requests.Response:
        time.sleep(self._sleep)
        headers = {**self._headers, "Accept": accept} if accept else self._headers
        r = requests.get(url, headers=headers, params=params, timeout=15)
        r.raise_for_status()
        return r

    def get_closed_bug_issues(self, owner: str, repo: str, n: int = 10) -> list:
        """Return closed issues labelled 'bug'. Filters out PRs (GitHub mixes them)."""
        url = f"{self._BASE}/repos/{owner}/{repo}/issues"
        data = self._get(url, params={
            "state": "closed", "labels": "bug",
            "per_page": n, "sort": "updated", "direction": "desc",
        }).json()
        return [i for i in data if "pull_request" not in i]

    def get_merged_prs(self, owner: str, repo: str, n: int = 20) -> list:
        """Return recently closed (potentially merged) PRs."""
        url = f"{self._BASE}/repos/{owner}/{repo}/pulls"
        return self._get(url, params={
            "state": "closed", "per_page": n,
            "sort": "updated", "direction": "desc",
        }).json()

    def get_pr_diff(self, owner: str, repo: str, pr_number: int) -> str:
        """Return the unified diff of a PR as a string."""
        time.sleep(self._sleep)
        r = requests.get(
            f"{self._BASE}/repos/{owner}/{repo}/pulls/{pr_number}",
            headers={**self._headers, "Accept": "application/vnd.github.v3.diff"},
            timeout=30,
        )
        r.raise_for_status()
        return r.text

    def get_issue_details(self, owner: str, repo: str, issue_number: int) -> Optional[dict]:
        """Fetch a single issue. Returns None if it's a PR or not found."""
        try:
            r = self._get(f"{self._BASE}/repos/{owner}/{repo}/issues/{issue_number}")
            d = r.json()
            return None if "pull_request" in d else d
        except Exception:
            return None

    def get_issue_timeline(self, owner: str, repo: str, issue_number: int) -> list:
        """Return timeline events for an issue (used to find linked fix commits)."""
        url = f"{self._BASE}/repos/{owner}/{repo}/issues/{issue_number}/timeline"
        return self._get(url, params={"per_page": 100}).json()


# ---------------------------------------------------------------------------
# GitRepo — local clone + git operations
# ---------------------------------------------------------------------------

class GitRepo:
    def __init__(self, owner: str, repo: str, name: str, clone_dir: Path = CLONE_DIR):
        self.owner = owner
        self.repo  = repo
        self.name  = name
        self.path  = clone_dir / name

    def clone_or_update(self) -> "GitRepo":
        if self.path.exists():
            log.info("git pull %s", self.name)
            self._run("pull", "--quiet")
        else:
            log.info("git clone %s/%s", self.owner, self.repo)
            self.path.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(
                ["git", "clone", "--depth=1000",
                 f"https://github.com/{self.owner}/{self.repo}.git",
                 str(self.path)],
                capture_output=True,
            )
        return self

    def find_fix_commit(self, issue_number: int) -> Optional[str]:
        """Search commit messages for a reference to the given issue number."""
        out = self._run("log", "--oneline", f"--grep=#{issue_number}")
        lines = [l for l in out.splitlines() if l.strip()]
        return lines[0].split()[0] if lines else None

    def find_fix_commits_from_git(self, limit: int = 30) -> List[tuple]:
        """
        Fallback: scan git log for commits with a fix keyword + issue reference.
        Returns list of (commit_hash, issue_number).
        """
        fix_re   = re.compile(r"\b(fix|fixes|fixed|bug|closes|close|resolve)\b", re.IGNORECASE)
        issue_re = re.compile(r"#(\d+)")

        out = self._run("log", "--format=%H %s", "-500")
        results, seen = [], set()
        for line in out.splitlines():
            parts = line.split(" ", 1)
            if len(parts) < 2:
                continue
            commit_hash, subject = parts
            if not fix_re.search(subject):
                continue
            for num_str in issue_re.findall(subject):
                num = int(num_str)
                if num not in seen:
                    seen.add(num)
                    results.append((commit_hash, num))
            if len(results) >= limit:
                break
        return results

    def get_changed_files(self, commit_hash: str) -> List[str]:
        out = self._run("diff-tree", "--no-commit-id", "-r", "--name-only", commit_hash)
        return [f for f in out.splitlines() if f.strip()]

    def get_file_at_commit(self, commit_hash: str, filepath: str) -> str:
        """Return file content at the parent of the given commit (pre-change state)."""
        return self._run("show", f"{commit_hash}^:{filepath}")

    def get_commit_diff(self, commit_hash: str) -> str:
        return self._run("show", commit_hash, "--unified=5")

    def get_commit_timestamp(self, commit_hash: str) -> str:
        return self._run("log", "-1", "--format=%aI", commit_hash)

    def get_parent_commit(self, commit_hash: str) -> str:
        return self._run("rev-parse", f"{commit_hash}^")

    def find_commit_before_timestamp(self, timestamp: str, files: List[str] = None) -> str:
        """
        Return the most recent commit hash strictly before the given ISO timestamp.
        Falls back to the oldest available commit in the shallow clone.
        """
        args = ["log", "-1", "--format=%H", f"--before={timestamp}"]
        if files:
            args += ["--"] + files
        result = self._run(*args)
        if result:
            return result
        oldest = self._run("log", "--format=%H", "--reverse")
        return oldest.splitlines()[0] if oldest else ""

    def symbol_exists(self, symbol: str) -> bool:
        """Return True if the symbol appears anywhere in the repo's tracked files."""
        try:
            result = subprocess.run(
                ["git", "grep", "-rl", "--", symbol],
                cwd=self.path, capture_output=True, text=True, timeout=5,
            )
            return bool(result.stdout.strip())
        except subprocess.TimeoutExpired:
            return False

    def _run(self, *args) -> str:
        result = subprocess.run(
            ["git", "-C", str(self.path)] + list(args),
            capture_output=True, text=True,
        )
        return result.stdout.strip()


# ---------------------------------------------------------------------------
# TaskExtractor — orchestrates API + repo to produce RCT tasks
# ---------------------------------------------------------------------------

class TaskExtractor:
    _SKIP_KEYWORDS = re.compile(
        r"\b(fix|bug|chore|ci|docs|bump|dependabot|typo|lint|format|release)\b",
        re.IGNORECASE,
    )
    _REVERT_RE = re.compile(r"\brevert\b", re.IGNORECASE)

    def __init__(self, api: GitHubAPI, repo: GitRepo, tasks_per_type: int = TASKS_PER_TYPE):
        self._api   = api
        self._repo  = repo
        self._limit = tasks_per_type

    def extract_all(self) -> List[dict]:
        tasks = []
        for label, method in [
            ("bug fix",      self._extract_bug_fix_tasks),
            ("feature impl", self._extract_feature_impl_tasks),
            ("code review",  self._extract_code_review_tasks),
        ]:
            try:
                log.info("extracting %s tasks", label)
                tasks += method()
            except Exception as e:
                log.error("%s extraction failed: %s", label, e)
        log.info("%d tasks extracted from %s", len(tasks), self._repo.name)
        return tasks

    def _extract_bug_fix_tasks(self) -> List[dict]:
        tasks, seen_ids = [], set()
        owner, repo = self._repo.owner, self._repo.repo

        # Primary: GitHub 'bug' label
        for issue in self._api.get_closed_bug_issues(owner, repo, n=30):
            if len(tasks) >= self._limit:
                break
            task = self._build_bug_fix_from_issue(issue, seen_ids)
            if task:
                tasks.append(task)

        # Fallback: fix keywords in git log
        if len(tasks) < self._limit:
            for commit, num in self._repo.find_fix_commits_from_git(limit=60):
                if len(tasks) >= self._limit:
                    break
                task_id = f"{repo}_bug_{num}"
                if task_id in seen_ids:
                    continue
                issue = self._api.get_issue_details(owner, repo, num)
                if not issue or issue.get("state") != "closed":
                    continue
                task = self._build_bug_fix_task(
                    num, issue["title"], issue.get("body") or "",
                    issue["created_at"], commit, source="git_fix_keyword_fallback",
                )
                if task:
                    seen_ids.add(task_id)
                    tasks.append(task)

        return tasks

    def _extract_feature_impl_tasks(self) -> List[dict]:
        tasks, skipped = [], 0
        owner, repo = self._repo.owner, self._repo.repo

        for pr in self._api.get_merged_prs(owner, repo, n=50):
            if len(tasks) >= self._limit:
                break
            if not pr.get("merged_at"):
                continue
            if self._SKIP_KEYWORDS.search(pr["title"]):
                continue

            num    = pr["number"]
            body   = pr.get("body") or ""
            opened = pr["created_at"]

            diff = self._api.get_pr_diff(owner, repo, num)
            if not diff or len(diff) < 100:
                continue

            ok, reason = self._passes_context_filters(diff, body)
            if not ok:
                log.info("  skipping PR #%s: %s", num, reason)
                skipped += 1
                continue

            tasks.append({
                "id": f"{repo}_feature_{num}",
                "repo": f"{owner}/{repo}",
                "task_type": "feature_impl",
                "input": {
                    "pr_number": num,
                    "pr_title": pr["title"],
                    "pr_body": body[:2000],
                },
                "ground_truth": {
                    "merged_at": pr["merged_at"],
                    "diff": diff[:6000],
                },
                "relevant_files": re.findall(r"^\+\+\+ b/(.+)$", diff, re.MULTILINE),
                "snapshot_commit": self._repo.find_commit_before_timestamp(opened),
                "belief_cutoff_timestamp": opened,
                "reference_commit_timestamp": pr["merged_at"],
            })

        log.info("  feature_impl extracted=%d skipped_by_filters=%d", len(tasks), skipped)
        return tasks

    def _extract_code_review_tasks(self) -> List[dict]:
        tasks = []
        owner, repo = self._repo.owner, self._repo.repo

        for pr in self._api.get_merged_prs(owner, repo, n=100):
            if len(tasks) >= self._limit:
                break
            if not self._REVERT_RE.search(pr["title"]):
                continue

            num    = pr["number"]
            body   = pr.get("body") or ""
            opened = pr["created_at"]

            diff = self._api.get_pr_diff(owner, repo, num)
            if not diff or len(diff) < 100:
                continue

            tasks.append({
                "id": f"{repo}_review_{num}",
                "repo": f"{owner}/{repo}",
                "task_type": "code_review",
                "input": {
                    "pr_number": num,
                    "pr_title": pr["title"],
                    "pr_body": body[:2000],
                    "diff": diff[:6000],
                },
                "ground_truth": {
                    "known_issues": "PR was reverted — agent should identify the problematic change.",
                },
                "relevant_files": re.findall(r"^\+\+\+ b/(.+)$", diff, re.MULTILINE),
                "snapshot_commit": self._repo.find_commit_before_timestamp(opened),
                "belief_cutoff_timestamp": opened,
                "reference_commit_timestamp": pr.get("merged_at"),
            })

        return tasks

    def _build_bug_fix_from_issue(self, issue: dict, seen_ids: set) -> Optional[dict]:
        num    = issue["number"]
        task_id = f"{self._repo.repo}_bug_{num}"
        if task_id in seen_ids:
            return None
        commit = self._repo.find_fix_commit(num)
        if not commit:
            return None
        task = self._build_bug_fix_task(
            num, issue["title"], issue.get("body") or "",
            issue["created_at"], commit, source="github_bug_label",
        )
        if task:
            seen_ids.add(task_id)
        return task

    def _build_bug_fix_task(
        self, num: int, title: str, body: str,
        opened: str, commit: str, source: str,
    ) -> Optional[dict]:
        files = self._repo.get_changed_files(commit)
        if not files:
            return None
        owner, repo = self._repo.owner, self._repo.repo
        return {
            "id": f"{repo}_bug_{num}",
            "repo": f"{owner}/{repo}",
            "task_type": "bug_fix",
            "source": source,
            "input": {
                "issue_number": num,
                "issue_title": title,
                "issue_body": body[:2000],
            },
            "ground_truth": {
                "fix_commit": commit,
                "diff": self._repo.get_commit_diff(commit)[:6000],
            },
            "relevant_files": files,
            "snapshot_commit": self._repo.get_parent_commit(commit),
            "belief_cutoff_timestamp": opened,
            "reference_commit_timestamp": self._repo.get_commit_timestamp(commit),
        }

    def _passes_context_filters(self, diff: str, pr_body: str) -> tuple[bool, str]:
        """
        Filters:
          1. Diff touches >= 2 files
          2. Diff has >= 50 lines changed
          3. PR body references at least one existing symbol in the repo
        """
        changed_files = [l for l in diff.splitlines() if l.startswith("diff --git ")]
        if len(changed_files) < 2:
            return False, f"only {len(changed_files)} file(s) changed (need >=2)"

        added   = sum(1 for l in diff.splitlines() if l.startswith("+") and not l.startswith("+++"))
        deleted = sum(1 for l in diff.splitlines() if l.startswith("-") and not l.startswith("---"))
        if added + deleted < 50:
            return False, f"only {added+deleted} lines changed (need >=50)"

        if pr_body:
            candidates = re.findall(r'\b([A-Z][a-zA-Z]{3,}|[a-z][a-z_]{3,}[a-z])\b', pr_body)
            for symbol in list(dict.fromkeys(candidates))[:30]:
                if self._repo.symbol_exists(symbol):
                    return True, ""
            return False, "PR body does not reference any existing symbol in the repo"

        return True, ""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo",  help="Filter to a single repo name (e.g. fmt)")
    parser.add_argument("--limit", type=int, default=TASKS_PER_TYPE,
                        help="Tasks per type per repo (default 3)")
    args = parser.parse_args()

    api = GitHubAPI(token=os.environ["GITHUB_TOKEN"])

    with open(DATASET / "test.json") as f:
        repos = json.load(f)["repositories"]

    if args.repo:
        repos = [r for r in repos if r["name"] == args.repo]
        if not repos:
            log.error("repo '%s' not found in test.json", args.repo)
            return

    TASKS_DIR.mkdir(parents=True, exist_ok=True)

    all_tasks = []
    for entry in tqdm(repos, desc="Repos"):
        match = re.search(r"github\.com/([^/]+)/([^/]+?)(?:\.git)?$", entry["url"])
        if not match:
            log.warning("skipping %s — cannot parse owner/repo from URL", entry["name"])
            continue

        owner, repo_name = match.group(1), match.group(2)
        log.info("=" * 60)
        log.info("repo: %s/%s", owner, repo_name)

        git_repo  = GitRepo(owner, repo_name, entry["name"]).clone_or_update()
        extractor = TaskExtractor(api, git_repo, tasks_per_type=args.limit)
        tasks     = extractor.extract_all()

        if tasks:
            out = TASKS_DIR / f"{entry['name']}.json"
            with open(out, "w") as f:
                json.dump(tasks, f, indent=2)
            all_tasks.extend(tasks)

    with open(TASKS_DIR / "all_tasks.json", "w") as f:
        json.dump(all_tasks, f, indent=2)

    log.info("done. %d total tasks across %d repos", len(all_tasks), len(repos))
    by_type: dict[str, int] = {}
    for t in all_tasks:
        by_type[t["task_type"]] = by_type.get(t["task_type"], 0) + 1
    for k, v in by_type.items():
        log.info("  %s: %d", k, v)


if __name__ == "__main__":
    main()
