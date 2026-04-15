#!/usr/bin/env python3
"""
get_beliefs.py
==============
SENG narrative builder + HuggingFace endpoint caller.

Used in two ways:
  1. As a bash CLI (treatment arm agents call it directly):
       python .seng/get_beliefs.py --commits hash1 hash2 hash3 hash4

  2. As an importable module (used by task_1b_pilot_filter.py and tests):
       from tools.get_beliefs import get_beliefs

Appends each call to $REPO_ROOT/.seng/belief_calls.jsonl so the
experiment runner can read the log after the agent session ends.

Environment variables:
  HF_ENDPOINT  -- HuggingFace endpoint URL (required)
  HF_TOKEN     -- HuggingFace API bearer token (required)
"""

import argparse
import json
import os
import re
import subprocess
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


# ---------------------------------------------------------------------------
# Pure helper functions (testable without git or network)
# ---------------------------------------------------------------------------

def parse_refs(refs_str: str) -> str:
    """
    Convert git %D output to (branch:main,tag:v1.0) format.

    Input examples:
      "HEAD -> main"
      "tag: v1.0, HEAD -> main"
      "origin/main"
      ""
    """
    if not refs_str.strip():
        return ""
    parts = []
    for ref in refs_str.split(","):
        ref = ref.strip()
        if ref.startswith("tag: "):
            parts.append(f"tag:{ref[5:].strip()}")
        elif " -> " in ref:
            branch = ref.split(" -> ", 1)[1].strip()
            parts.append(f"branch:{branch}")
        elif ref == "HEAD" or ref.startswith("HEAD"):
            continue
        elif ref:
            parts.append(f"branch:{ref}")
    return f"({','.join(parts)})" if parts else ""


def build_file_change_str(files: list) -> str:
    """
    Build the file-changes part of a SENG narrative line.

    Each file dict: {"filename": str, "added": int, "deleted": int, "funcs": list[str]}
    Output: "~src/foo.py+10/-3@{bar} ~src/baz.py+2/-0"
    """
    parts = []
    for f in files:
        s = f"~{f['filename']}+{f['added']}/-{f['deleted']}"
        if f.get("funcs"):
            funcs = ",".join(sorted(set(f["funcs"])))
            s += f"@{{{funcs}}}"
        parts.append(s)
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Git helpers (require a real git repo)
# ---------------------------------------------------------------------------

def _git(cmd: list, repo_path: str, timeout: int = 30) -> str:
    result = subprocess.run(
        ["git", "-C", repo_path] + cmd,
        capture_output=True, text=True, timeout=timeout
    )
    if result.returncode != 0:
        raise RuntimeError(f"git {' '.join(cmd)}: {result.stderr.strip()}")
    return result.stdout.strip()


def _get_commit_meta(commit_hash: str, repo_path: str) -> dict:
    """
    Return dict with keys: timestamp, hash8, refs, author, committer, subject
    """
    line = _git(
        ["log", "--format=%aI\t%h\t%D\t%aN\t%cN\t%s", "-1", commit_hash],
        repo_path
    )
    parts = line.split("\t", 5)
    if len(parts) < 6:
        raise ValueError(f"Unexpected log output for {commit_hash}: {line!r}")
    timestamp, hash8, refs_raw, author, committer, subject = parts
    return {
        "timestamp":  timestamp,
        "hash8":      hash8,
        "refs":       parse_refs(refs_raw),
        "author":     author,
        "committer":  committer,
        "subject":    subject,
    }


def _get_file_changes(commit_hash: str, repo_path: str) -> list:
    """
    Return list of file change dicts for one commit.
    Each dict: {"filename", "added", "deleted", "funcs"}
    """
    # numstat: added<tab>deleted<tab>filename
    numstat = _git(["show", "--numstat", "--format=", commit_hash], repo_path)
    file_stats = {}
    for line in numstat.splitlines():
        parts = line.split("\t")
        if len(parts) == 3:
            added_s, deleted_s, filename = parts
            file_stats[filename] = {
                "filename": filename,
                "added":    int(added_s)   if added_s   != "-" else 0,
                "deleted":  int(deleted_s) if deleted_s != "-" else 0,
                "funcs":    [],
            }

    # Extract function names from hunk headers in the patch
    patch = _git(["show", "-p", "--format=", commit_hash], repo_path)
    current_file = None
    for line in patch.splitlines():
        if line.startswith("+++ b/"):
            current_file = line[6:]
        elif line.startswith("@@ ") and current_file and current_file in file_stats:
            # e.g. "@@ -45,12 +45,12 @@ validate_token("
            match = re.search(r"@@[^@]*@@\s*(.+)", line)
            if match:
                ctx = match.group(1).strip()
                # Take first identifier token
                token = re.split(r"[\s({<]", ctx)[0].strip()
                if token and re.match(r"^[A-Za-z_]", token):
                    file_stats[current_file]["funcs"].append(token)

    return list(file_stats.values())


def build_narrative_line(commit_hash: str, repo_path: str) -> str:
    """
    Build one SENG narrative line for a commit.

    Format:
      {ISO8601} {hash8} ({refs}) {author}(a) & {committer}(c) :: {subject} | {file_changes}
    """
    meta = _get_commit_meta(commit_hash, repo_path)
    files = _get_file_changes(commit_hash, repo_path)

    actors = f"{meta['author']}(a) & {meta['committer']}(c)"

    line = meta["timestamp"] + " " + meta["hash8"]
    if meta["refs"]:
        line += " " + meta["refs"]
    line += f" {actors} :: {meta['subject']}"
    if files:
        line += " | " + build_file_change_str(files)
    return line


# ---------------------------------------------------------------------------
# HF endpoint call
# ---------------------------------------------------------------------------

def call_hf_endpoint(narrative: str, endpoint_url: str, token: str,
                     timeout: int = 120) -> str:
    """POST narrative to HF endpoint, return generated_text."""
    data = json.dumps({"inputs": narrative}).encode()
    req = urllib.request.Request(
        endpoint_url,
        data=data,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type":  "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        result = json.loads(resp.read())

    if isinstance(result, list):
        return result[0].get("generated_text", "")
    return result.get("generated_text", "")


# ---------------------------------------------------------------------------
# Log helper
# ---------------------------------------------------------------------------

# NOTE: seq is best-effort (no file locking); downstream uses called_at for ordering.
def _log_call(repo_path: str, commits: list, narrative: str, beliefs: str):
    """Append call record to .seng/belief_calls.jsonl (no run_id -- runner assigns it)."""
    seng_dir = Path(repo_path) / ".seng"
    seng_dir.mkdir(exist_ok=True)
    log_file = seng_dir / "belief_calls.jsonl"

    if log_file.exists():
        with log_file.open() as fh:
            seq = sum(1 for _ in fh)
    else:
        seq = 0
    record = {
        "seq":       seq + 1,
        "commits":   commits,
        "narrative": narrative,
        "beliefs":   beliefs,
        "called_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    with log_file.open("a") as f:
        f.write(json.dumps(record) + "\n")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_beliefs(
    commit_hashes: list,
    repo_path=None,
    endpoint_url=None,
    token=None,
) -> str:
    """
    Build SENG narrative from exactly 4 commits, call HF endpoint, return beliefs.

    Args:
        commit_hashes: list of exactly 4 commit hashes
        repo_path:     git repo path (default: cwd)
        endpoint_url:  HF endpoint URL (default: $HF_ENDPOINT)
        token:         HF bearer token (default: $HF_TOKEN)

    Returns:
        Belief text from the SENG model, or "" on failure.
    """
    if len(commit_hashes) != 4:
        raise ValueError(f"Expected exactly 4 commits, got {len(commit_hashes)}")

    repo_path    = repo_path    or os.getcwd()
    endpoint_url = endpoint_url or os.environ.get("HF_ENDPOINT", "")
    token        = token        or os.environ.get("HF_TOKEN", "")

    if not endpoint_url:
        print("Warning: HF_ENDPOINT not set", file=sys.stderr)
        return ""

    if not token:
        print("Warning: HF_TOKEN not set", file=sys.stderr)
        return ""

    # Build narrative lines (skip commits that fail)
    lines = []
    for h in commit_hashes:
        try:
            lines.append(build_narrative_line(h, repo_path))
        except Exception as e:
            print(f"Warning: skipping commit {h}: {e}", file=sys.stderr)

    if not lines:
        return ""

    narrative = "\n".join(lines)

    try:
        beliefs = call_hf_endpoint(narrative, endpoint_url, token)
    except Exception as e:
        print(f"Warning: HF endpoint error: {e}", file=sys.stderr)
        return ""

    _log_call(repo_path, commit_hashes, narrative, beliefs)
    return beliefs


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Extract SENG beliefs from 4 git commits"
    )
    parser.add_argument("--commits", nargs=4, required=True, metavar="HASH",
                        help="Exactly 4 commit hashes")
    parser.add_argument("--repo", default=None,
                        help="Path to git repo (default: cwd)")
    args = parser.parse_args()

    try:
        beliefs = get_beliefs(args.commits, repo_path=args.repo)
        print(beliefs)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
