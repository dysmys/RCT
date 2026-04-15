"""
task_3b_run_tests.py
====================

For each completed run that has an agent_diff but no test_results entry:
  1. Create a fresh worktree at the snapshot commit
  2. Run baseline tests
  3. Apply the agent's diff
  4. Run post-patch tests
  5. Record pass_rate_delta in test_results table

Run this after task_3_run_experiment.py completes for a repo.

Usage:
  python scripts/task_3b_run_tests.py --repo fmt [--repo-dir /mnt/repos]

Environment:
  DB_PATH   -- optional, defaults to database/results.db
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

_SCRIPTS_DIR  = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPTS_DIR.parent
sys.path.insert(0, str(_SCRIPTS_DIR))
sys.path.insert(0, str(_SCRIPTS_DIR / "tools"))

import database.db as db
from utils.logger import get_logger
from test_runner import run_test_suite
from task_3_run_experiment import (
    create_worktree, remove_worktree,
)

log = get_logger(__name__)

DEFAULT_REPO_DIR  = os.environ.get("REPO_DIR", "/mnt/repos")
TEST_TIMEOUT_SECS = 300


def apply_diff(diff_text: str, repo_path: Path) -> bool:
    """Apply unified diff via git apply. Returns True on success."""
    if not diff_text.strip():
        return False
    result = subprocess.run(
        ["git", "apply", "--whitespace=fix", "-"],
        cwd=str(repo_path),
        input=diff_text,
        capture_output=True, text=True
    )
    if result.returncode != 0:
        log.debug("git apply failed: %s", result.stderr[:200])
    return result.returncode == 0


def run_baseline(repo_path: Path, test_cmd: str) -> tuple[int, int]:
    """Run tests at snapshot (before applying diff)."""
    return run_test_suite(repo_path, test_cmd, timeout=TEST_TIMEOUT_SECS)


def score_run(run: dict, base_clone: Path, test_cmd: str, database, repo_name: str):
    """Apply agent's diff to a fresh worktree and record test results."""
    run_id   = run["run_id"]
    snapshot = run["snapshot_commit"]
    diff     = run["agent_diff"]
    wt_path  = base_clone.parent / f"wt_test_{run_id}"

    try:
        create_worktree(base_clone, wt_path, snapshot)

        # Baseline (no diff applied)
        baseline_pass, baseline_fail = run_baseline(wt_path, test_cmd)
        baseline_total = baseline_pass + baseline_fail

        if baseline_total == 0:
            log.warning("run_id=%d: empty test suite", run_id)
            database.insert_test_result(run_id,
                                        baseline_pass, baseline_fail, 0,
                                        None, None, None,
                                        "empty_suite", test_cmd.split()[0])
            return

        # Apply diff
        if not apply_diff(diff, wt_path):
            log.warning("run_id=%d: diff apply failed", run_id)
            database.insert_test_result(run_id,
                                        baseline_pass, baseline_fail, baseline_total,
                                        None, None, None,
                                        "apply_failed", test_cmd.split()[0])
            return

        # Post-patch tests
        post_pass, post_fail = run_test_suite(wt_path, test_cmd, TEST_TIMEOUT_SECS)
        if post_pass == -1:
            database.insert_test_result(run_id,
                                        baseline_pass, baseline_fail, baseline_total,
                                        None, None, None,
                                        "suite_error", test_cmd.split()[0])
            return

        delta = (post_pass - baseline_pass) / baseline_total
        database.insert_test_result(run_id,
                                    baseline_pass, baseline_fail, baseline_total,
                                    post_pass, post_fail, round(delta, 4),
                                    "clean", test_cmd.split()[0])

        log.info("run_id=%d %s/%s: baseline=%d/%d post=%d/%d delta=%+.3f",
                 run_id, run["arm"], run["agent"],
                 baseline_pass, baseline_total,
                 post_pass, baseline_total, delta)

    finally:
        remove_worktree(base_clone, wt_path)


def main():
    parser = argparse.ArgumentParser(
        description="Apply agent diffs and run test suites for a repo's completed runs."
    )
    parser.add_argument("--repo",      required=True)
    parser.add_argument("--repo-dir",  default=DEFAULT_REPO_DIR)
    args = parser.parse_args()

    database = db.get_db()

    # Fetch test_cmd for this repo
    repo_row = database.conn.execute(
        "SELECT test_cmd, test_runner FROM repos WHERE name = ?", (args.repo,)
    ).fetchone()
    if not repo_row or not repo_row["test_cmd"]:
        log.error("No test_cmd stored for repo '%s'. Run task_2_setup_repos.py first.", args.repo)
        sys.exit(1)

    test_cmd = repo_row["test_cmd"]
    log.info("repo=%s  test_cmd=%s", args.repo, test_cmd)

    runs = database.get_completed_runs_with_diff(args.repo)
    log.info("Found %d runs to test", len(runs))

    base_clone = Path(args.repo_dir) / args.repo / "control"

    scored, failed_count = 0, 0
    for run in runs:
        try:
            score_run(dict(run), base_clone, test_cmd, database, args.repo)
            scored += 1
        except Exception as e:
            log.error("run_id=%d: unexpected error: %s", run["run_id"], e)
            failed_count += 1

    log.info("=" * 60)
    log.info("Test scoring complete: scored=%d  failed=%d", scored, failed_count)


if __name__ == "__main__":
    main()
