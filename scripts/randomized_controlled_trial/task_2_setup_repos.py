"""
task_2_setup_repos.py
=====================

Sets up the experiment workspace for all repos in dataset/test.json.

For each repo this script creates two fully independent git clones:
  <REPO_DIR>/<name>/control/    — full clone, CLAUDE.md (no beliefs)
  <REPO_DIR>/<name>/treatment/  — full clone, CLAUDE.md + Belief Management section

Each arm is a completely independent git repository. Claude Code runs inside
each clone and can read files, run git commands, and use Bash freely without
any cross-contamination between arms.

Directory layout after running:
  <REPO_DIR>/
    fmt/
      control/     ← full git clone, CLAUDE.md (no beliefs)
      treatment/   ← full git clone, CLAUDE.md + Belief Management
    mockito/
      ...

Usage:
  python scripts/task_2_setup_repos.py [--repo fmt] [--repo-dir /mnt/repos]
"""

import os
import json
import argparse
import subprocess
import sys
from pathlib import Path

from dotenv import load_dotenv
from utils.logger import get_logger

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

BASE_DIR     = Path(__file__).resolve().parent.parent
_SCRIPTS_DIR = Path(__file__).resolve().parent
DATASET      = BASE_DIR / "dataset"

load_dotenv(BASE_DIR / ".env")

sys.path.insert(0, str(_SCRIPTS_DIR))
sys.path.insert(0, str(_SCRIPTS_DIR / "tools"))
import database.db as db
from test_runner import detect_test_runner

# ---------------------------------------------------------------------------
# CLAUDE.md / AGENTS.md templates
# ---------------------------------------------------------------------------

CLAUDE_MD_CONTROL = """\
# Task Instructions

You are a software engineering agent with full access to this repository.

## Tools Available
- Read files directly (use the Read tool or Bash cat/grep/find)
- Write and modify files (use the Edit or Write tool)
- Run: grep, find, git log, git show, git diff (read-only git commands)

## Rules
- Do NOT run: git commit, git push, git add, pytest, mvn, npm test, or any build command
- Do NOT install packages or make network requests
- Write your changes directly to files in this repository

Implement the task described in the prompt below.
"""

CLAUDE_MD_TREATMENT = """\
# Task Instructions

You are a software engineering agent with full access to this repository.

## Tools Available
- Read files directly (use the Read tool or Bash cat/grep/find)
- Write and modify files (use the Edit or Write tool)
- Run: grep, find, git log, git show, git diff (read-only git commands)
- **query_belief_store** -- retrieves pre-processed development beliefs relevant to your task

## Using query_belief_store
Call it with a natural-language description of your task and the relevant files:

    python .seng/query_belief_store.py \
        --repo <repo_name> \
        --query "<task description and relevant file paths>" \
        --cutoff <belief_cutoff_timestamp>

Example:
    python .seng/query_belief_store.py \
        --repo django \
        --query "authentication middleware session token handling" \
        --cutoff 2025-06-15T10:00:00Z \
        --top-k 5

The .seng/.env file contains REPO_NAME, BELIEF_CUTOFF, BELIEF_DB, and BELIEF_CHROMA_DIR
pre-set for this task. You may source it or pass the values directly.

The tool returns beliefs extracted from the commit history up to the cutoff date,
along with supporting evidence chains. Use them to align your implementation with
established patterns in this codebase.

## Rules
- Do NOT run: git commit, git push, git add, pytest, mvn, npm test, or any build command
- Do NOT install packages or make network requests
- Write your changes directly to files in this repository

Implement the task described in the prompt below.
"""

AGENTS_MD_CONTROL = """\
# Task Instructions

You are a software engineering agent with full access to this repository.
You may read any file, write changes directly, and run grep/find/git log/git show.
Do NOT run tests, commit, push, or install packages.
Implement the task described in the prompt.
"""

AGENTS_MD_TREATMENT = """\
# Task Instructions

You are a software engineering agent with full access to this repository.
You may read any file, write changes directly, and run grep/find/git log/git show.
Do NOT run tests, commit, push, or install packages.

You have access to query_belief_store:
    python .seng/query_belief_store.py \
        --repo <repo_name> \
        --query "<task description and relevant file paths>" \
        --cutoff <belief_cutoff_timestamp> \
        --top-k 5

The .seng/.env file contains REPO_NAME, BELIEF_CUTOFF, BELIEF_DB, and BELIEF_CHROMA_DIR
pre-set for this task. Use them to construct the call. You may call it multiple times
with different queries. Use the returned beliefs to align your implementation with
established patterns in this codebase.

Implement the task described in the prompt.
"""

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def run(cmd, cwd=None, check=True):
    result = subprocess.run(
        cmd, cwd=cwd, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT
    )
    if check and result.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(str(c) for c in cmd)}\n{result.stdout}")
    return result.stdout.strip()


def clone_repo(git_url: str, dest: Path):
    """Clone repo into dest/ if not already present."""
    if (dest / ".git").exists():
        log.info("already cloned: %s", dest)
        return
    dest.mkdir(parents=True, exist_ok=True)
    log.info("cloning %s → %s", git_url, dest)
    run(["git", "clone", git_url, str(dest)])


def write_claude_md(dest: Path, content: str, arm: str):
    path = dest / "CLAUDE.md"
    path.write_text(content)
    log.info("CLAUDE.md written for %s arm", arm)


# ---------------------------------------------------------------------------
# Per-repo setup
# ---------------------------------------------------------------------------

def setup_repo(entry: dict, repo_dir: Path) -> bool:
    git_url = entry["url"]
    name    = entry["name"]
    lang    = entry["language"]

    log.info("=" * 60)
    log.info("%s (%s)  %s", name, lang, git_url)

    # Control — independent full clone
    control_path = repo_dir / name / "control"
    try:
        clone_repo(git_url, control_path)
        write_claude_md(control_path, CLAUDE_MD_CONTROL, "control")
        (control_path / "AGENTS.md").write_text(AGENTS_MD_CONTROL)
    except Exception as e:
        log.error("control setup failed for %s: %s", name, e)
        return False

    # Treatment — independent full clone
    treatment_path = repo_dir / name / "treatment"
    try:
        clone_repo(git_url, treatment_path)
        write_claude_md(treatment_path, CLAUDE_MD_TREATMENT, "treatment")
        (treatment_path / "AGENTS.md").write_text(AGENTS_MD_TREATMENT)
    except Exception as e:
        log.error("treatment setup failed for %s: %s", name, e)
        return False

    # Detect and store test runner from the control clone
    test_cmd, test_runner = detect_test_runner(control_path)
    if test_cmd:
        log.info("  test runner: %s (%s)", test_runner, test_cmd)
        db.get_db().upsert_repo_test_cmd(name, test_cmd, test_runner)
    else:
        log.warning("  no test runner detected for %s", name)

    log.info("%s ready", name)
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Clone test repos and set up control/treatment directories."
    )
    parser.add_argument("--repo", metavar="NAME",
                        help="Set up only this repo (e.g. fmt). Omit to run all.")
    parser.add_argument("--repo-dir", metavar="PATH",
                        default=os.environ.get("REPO_DIR", "/mnt/repos"),
                        help="Root directory for clones (default: /mnt/repos or $REPO_DIR).")
    args = parser.parse_args()

    repo_dir = Path(args.repo_dir)
    repo_dir.mkdir(parents=True, exist_ok=True)

    with open(DATASET / "test.json") as f:
        repos = json.load(f)["repositories"]

    if args.repo:
        repos = [r for r in repos if r["name"] == args.repo]
        if not repos:
            log.error("repo '%s' not found in test.json", args.repo)
            return

    log.info("setting up %d repo(s) in %s", len(repos), repo_dir)

    ok, failed = 0, []
    for entry in repos:
        if setup_repo(entry, repo_dir):
            ok += 1
        else:
            failed.append(entry["name"])

    log.info("=" * 60)
    log.info("done. %d/%d repos set up successfully", ok, len(repos))
    if failed:
        log.warning("failed: %s", ", ".join(failed))


if __name__ == "__main__":
    main()
