"""
task_1b_pilot_filter.py
=======================

Pilot pass: runs the control arm (Claude only, no beliefs) on all candidate
tasks for a repo, applies each diff to a fresh worktree, runs the test suite,
and keeps only mid-range tasks (where control partially succeeds).

Buckets:
  floor    pass_rate_delta <= -0.1 or no diff produced  -> drop
  mid      -0.1 < pass_rate_delta < 0.9                 -> KEEP
  ceiling  pass_rate_delta >= 0.9                        -> drop

Writes kept tasks back to dataset/tasks/<repo>.json (replaces in-place).
Creates dataset/tasks/<repo>_pilot_results.json with full results.

Usage:
  python scripts/task_1b_pilot_filter.py --repo fmt [--repo-dir /mnt/repos]

Environment:
  ANTHROPIC_API_KEY  -- required
  DB_PATH            -- optional
"""

import argparse
import json
import os
import subprocess
import sys
import re
from pathlib import Path

_SCRIPTS_DIR  = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPTS_DIR.parent
sys.path.insert(0, str(_SCRIPTS_DIR))
sys.path.insert(0, str(_SCRIPTS_DIR / "tools"))
from test_runner import detect_test_runner, run_test_suite

import database.db as db
from utils.logger import get_logger
from task_3_run_experiment import (
    create_worktree, remove_worktree, run_claude,
    build_task_prompt, resolve_snapshot_commit,
    CLAUDE_MD_CONTROL,
)

log = get_logger(__name__)

FLOOR_THRESHOLD   = -0.1
CEILING_THRESHOLD =  0.9
DEFAULT_REPO_DIR  = os.environ.get("REPO_DIR", "/mnt/repos")
DEFAULT_MODEL     = "claude-sonnet-4-6"
DEFAULT_TIMEOUT   = 360


def apply_diff(diff_text: str, repo_path: Path) -> bool:
    """Apply a unified diff to repo_path. Returns True on success."""
    if not diff_text.strip():
        return False
    result = subprocess.run(
        ["git", "apply", "--whitespace=fix", "-"],
        cwd=str(repo_path),
        input=diff_text, capture_output=True, text=True
    )
    return result.returncode == 0


def pilot_run_task(task: dict, base_clone: Path, model: str, timeout: int) -> dict:
    """
    Run control arm on one task, apply diff, run tests.
    Returns result dict with keys: task_id, pilot_delta, bucket, passed, failed, diff
    """
    task_id  = task["id"]
    snapshot = resolve_snapshot_commit(base_clone, task)
    wt_path  = base_clone.parent / f"wt_pilot_{task_id}"

    try:
        create_worktree(base_clone, wt_path, snapshot)

        # Write control CLAUDE.md (no beliefs, no file injection)
        (wt_path / "CLAUDE.md").write_text(CLAUDE_MD_CONTROL)

        # Build prompt (control arm: no beliefs, no file pre-injection)
        prompt = build_task_prompt(task, "control")

        # Run Claude
        try:
            output, _ = run_claude(wt_path, prompt, model, timeout)
        except subprocess.TimeoutExpired:
            return {"task_id": task_id, "bucket": "floor", "pilot_delta": -1.0,
                    "reason": "timeout", "diff": ""}

        # Capture diff
        diff_result = subprocess.run(
            ["git", "-C", str(wt_path), "diff", snapshot],
            capture_output=True, text=True
        )
        diff_text = diff_result.stdout.strip()

        if not diff_text:
            return {"task_id": task_id, "bucket": "floor", "pilot_delta": 0.0,
                    "reason": "no_diff", "diff": ""}

        # Detect and run test suite (baseline first, then with patch)
        test_cmd, runner = detect_test_runner(wt_path)
        if not test_cmd:
            # No test suite detected -- keep task (can't filter by execution)
            return {"task_id": task_id, "bucket": "mid", "pilot_delta": None,
                    "reason": "no_test_suite", "diff": diff_text}

        baseline_pass, baseline_fail = run_test_suite(wt_path, test_cmd)
        baseline_total = baseline_pass + baseline_fail
        if baseline_total == 0:
            return {"task_id": task_id, "bucket": "mid", "pilot_delta": None,
                    "reason": "empty_test_suite", "diff": diff_text}

        # Reset worktree, apply diff, re-run
        subprocess.run(["git", "-C", str(wt_path), "checkout", "."], capture_output=True)
        if not apply_diff(diff_text, wt_path):
            return {"task_id": task_id, "bucket": "floor", "pilot_delta": None,
                    "reason": "apply_failed", "diff": diff_text}

        post_pass, post_fail = run_test_suite(wt_path, test_cmd)
        delta = (post_pass - baseline_pass) / baseline_total

        if delta >= CEILING_THRESHOLD:
            bucket = "ceiling"
        elif delta <= FLOOR_THRESHOLD:
            bucket = "floor"
        else:
            bucket = "mid"

        return {
            "task_id":        task_id,
            "bucket":         bucket,
            "pilot_delta":    round(delta, 3),
            "baseline_pass":  baseline_pass,
            "baseline_total": baseline_total,
            "post_pass":      post_pass,
            "test_runner":    runner,
            "diff":           diff_text,
        }

    finally:
        remove_worktree(base_clone, wt_path)


def main():
    parser = argparse.ArgumentParser(
        description="Pilot filter: keep only mid-range tasks for the experiment."
    )
    parser.add_argument("--repo",      required=True)
    parser.add_argument("--repo-dir",  default=DEFAULT_REPO_DIR)
    parser.add_argument("--model",     default=DEFAULT_MODEL)
    parser.add_argument("--timeout",   type=int, default=DEFAULT_TIMEOUT)
    parser.add_argument("--dry-run",   action="store_true",
                        help="Print filter results without modifying tasks file")
    args = parser.parse_args()

    tasks_file = _PROJECT_ROOT / "dataset" / "tasks" / f"{args.repo}.json"
    if not tasks_file.exists():
        log.error("Tasks file not found: %s", tasks_file)
        sys.exit(1)

    tasks = json.loads(tasks_file.read_text())
    base_clone = Path(args.repo_dir) / args.repo / "control"
    if not (base_clone / ".git").exists():
        log.error("Control clone not found at %s -- run task_2_setup_repos.py first", base_clone)
        sys.exit(1)

    db.get_db()  # ensure schema + migrations applied

    results = []
    for task in tasks:
        if task["task_type"] not in ("feature_impl", "bug_fix"):
            # Keep code_review tasks as-is (can't run test suite meaningfully)
            results.append({"task_id": task["id"], "bucket": "mid", "pilot_delta": None,
                            "reason": "code_review_kept"})
            continue

        log.info("Pilot run: %s", task["id"])
        result = pilot_run_task(task, base_clone, args.model, args.timeout)
        results.append(result)
        log.info("  bucket=%-8s  delta=%s  reason=%s",
                 result["bucket"],
                 str(result.get("pilot_delta", "n/a")),
                 result.get("reason", ""))

    # Summarize
    kept     = [r for r in results if r["bucket"] == "mid"]
    dropped  = [r for r in results if r["bucket"] != "mid"]
    log.info("=" * 60)
    log.info("Pilot complete: %d kept (mid), %d dropped (%d floor, %d ceiling)",
             len(kept),
             len(dropped),
             sum(1 for r in dropped if r["bucket"] == "floor"),
             sum(1 for r in dropped if r["bucket"] == "ceiling"))

    # Save results
    results_file = _PROJECT_ROOT / "dataset" / "tasks" / f"{args.repo}_pilot_results.json"
    results_file.write_text(json.dumps(results, indent=2))
    log.info("Pilot results saved to %s", results_file)

    if not args.dry_run:
        kept_ids = {r["task_id"] for r in kept}
        kept_tasks = [t for t in tasks if t["id"] in kept_ids]
        tasks_file.write_text(json.dumps(kept_tasks, indent=2))
        log.info("Updated %s: %d tasks (was %d)", tasks_file, len(kept_tasks), len(tasks))


if __name__ == "__main__":
    main()
