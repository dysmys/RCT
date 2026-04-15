"""
get_beliefs.py
==============

Retrieves codebase beliefs for a set of files up to a given timestamp.

Called by the treatment agent from within a repo worktree directory:

    python /opt/experiment/get_beliefs.py \
        --files include/fmt/base.h,src/fmt.cc \
        --cutoff 2026-01-03T21:27:32Z

How it works:
  1. Reads git history for the specified files up to --cutoff
  2. Generates a compact narrative in SENG format (same format the model
     was trained on)
  3. Chunks the narrative into 4-line windows
  4. Sends each chunk to the SENG inference server
  5. Collects, deduplicates, and prints the returned beliefs to stdout

The script must be run from inside a git repository (the worktree).

Environment:
  SENG_INFERENCE_URL  — HuggingFace inference endpoint URL (required)
  HF_TOKEN            — HuggingFace API bearer token (required)
"""

import os
import re
import sys
import json
import argparse
import subprocess
import time
from pathlib import Path

import requests
from dotenv import load_dotenv
from utils.logger import get_logger

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

_BASE_URL  = os.environ.get("SENG_INFERENCE_URL", "").rstrip("/")
_HF_TOKEN  = os.environ.get("HF_TOKEN", "")
CHUNK_SIZE = 4      # narrative lines per inference call (matches training)


# ---------------------------------------------------------------------------
# Git helpers
# ---------------------------------------------------------------------------

def git(*args, cwd=None):
    result = subprocess.run(
        ["git"] + list(args),
        cwd=cwd or Path.cwd(),
        capture_output=True, text=True
    )
    return result.stdout.strip()


MAX_COMMITS = 40   # cap narrative to avoid excessive inference calls


def get_commits(files: list[str], cutoff: str) -> list[dict]:
    """
    Return up to MAX_COMMITS most-recent commits touching any of the given
    files, up to (not including) the cutoff timestamp. Ordered newest-first.
    Capping at MAX_COMMITS keeps inference time bounded: 40/4=10 chunks max.
    """
    fmt = "%aI\t%h\t%D\t%an\t%cn\t%s"
    args = [
        "log",
        f"--format={fmt}",
        f"--before={cutoff}",
        f"-n{MAX_COMMITS}",
        "--",
    ] + files

    raw = git(*args)
    if not raw:
        return []

    commits = []
    for line in raw.splitlines():
        parts = line.split("\t", 5)
        if len(parts) < 6:
            continue
        timestamp, hash8, refs, author, committer, subject = parts
        commits.append({
            "timestamp": timestamp,
            "hash8": hash8,
            "refs": refs.strip(),
            "author": author,
            "committer": committer,
            "subject": subject,
        })
    return commits


def get_file_stats(commit_hash: str, files: list[str]) -> list[str]:
    """
    Return numstat lines (added/deleted/filepath) for the given commit,
    filtered to the relevant files.
    """
    raw = git("diff-tree", "--no-commit-id", "-r", "--numstat", commit_hash)
    stats = []
    for line in raw.splitlines():
        parts = line.split("\t")
        if len(parts) != 3:
            continue
        added, deleted, filepath = parts
        if not files or any(f in filepath for f in files):
            stats.append((added, deleted, filepath))
    return stats


# ---------------------------------------------------------------------------
# Narrative generation (SENG format)
# ---------------------------------------------------------------------------

def format_refs(refs: str) -> str:
    """Format git decoration string into (tag:x,branch:y) notation."""
    if not refs:
        return ""
    parts = []
    for token in refs.split(","):
        token = token.strip()
        if token.startswith("tag:"):
            parts.append(token)
        elif "->" in token:
            continue
        elif token:
            parts.append(f"branch:{token.replace('HEAD -> ', '')}")
    return f"({','.join(parts)})" if parts else ""


def format_actors(author: str, committer: str) -> str:
    if author == committer:
        return f"{author}(a)"
    return f"{author}(a) & {committer}(c)"


def format_file_change(added: str, deleted: str, filepath: str) -> str:
    name = Path(filepath).name
    a = f"+{added}" if added != "-" else ""
    d = f"/-{deleted}" if deleted != "-" else ""
    return f"~{name}{a}{d}"


def build_narrative(commits: list[dict], files: list[str]) -> list[str]:
    """
    Build SENG-format narrative lines from a list of commit dicts.
    One line per (commit, file) combination.
    """
    lines = []
    for commit in commits:
        stats = get_file_stats(commit["hash8"], files)
        if not stats:
            # No numstat match — emit one line with just commit info
            stats = [("-", "-", ",".join(files))]

        refs_str   = format_refs(commit["refs"])
        actors_str = format_actors(commit["author"], commit["committer"])
        prefix     = f"{commit['timestamp']} {commit['hash8']}"
        if refs_str:
            prefix += f" {refs_str}"
        prefix += f" {actors_str} :: {commit['subject']}"

        file_parts = " ".join(
            format_file_change(a, d, fp) for a, d, fp in stats
        )
        lines.append(f"{prefix} | {file_parts}")

    return lines


def chunk(lines: list[str], size: int) -> list[str]:
    """Split narrative lines into fixed-size chunks joined by newline."""
    return [
        "\n".join(lines[i:i + size])
        for i in range(0, len(lines), size)
    ]


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def _parse_belief_text(text: str) -> list[str]:
    """Split a generated text block into individual Belief: entries."""
    beliefs = re.split(r"\n\n(?=Belief:)", text.strip())
    return [b.strip() for b in beliefs if b.strip()]


def _hf_headers() -> dict:
    """Authorization headers for HuggingFace Inference Endpoints."""
    return {"Authorization": f"Bearer {_HF_TOKEN}", "Content-Type": "application/json"}


def _call_hf_endpoint(chunk: str, timeout: int = 120) -> str:
    """
    Call HuggingFace Inference Endpoint with a single chunk.
    HF format: POST to base URL with {"inputs": text}
    Returns: {"generated_text": "..."} or [{"generated_text": "..."}]
    """
    resp = requests.post(
        _BASE_URL,
        headers=_hf_headers(),
        json={"inputs": chunk},
        timeout=timeout,
    )
    resp.raise_for_status()
    data = resp.json()
    # Handle both single-object and array response formats
    if isinstance(data, list):
        return data[0].get("generated_text", "") if data else ""
    return data.get("generated_text", "")


def extract_beliefs_batch(narrative_lines: list[str], top_k: int) -> list[str]:
    """Extract beliefs by sending one 4-line chunk at a time to the HF endpoint."""
    return _extract_beliefs_hf(narrative_lines, top_k)


def _extract_beliefs_hf(narrative_lines: list[str], top_k: int) -> list[str]:
    """Send one 4-line chunk at a time to the HF Inference Endpoint."""
    chunks = [
        "\n".join(narrative_lines[i:i + CHUNK_SIZE])
        for i in range(0, len(narrative_lines), CHUNK_SIZE)
    ]
    all_beliefs: list[str] = []
    for c in chunks:
        try:
            text = _call_hf_endpoint(c, timeout=60)
            all_beliefs.extend(_parse_belief_text(text))
            if len(all_beliefs) >= top_k * 2:
                break
        except Exception as e:
            log.warning("HF inference call failed: %s", e)
    return all_beliefs


def deduplicate(beliefs: list[str]) -> list[str]:
    """Remove exact duplicates while preserving order."""
    seen, out = set(), []
    for b in beliefs:
        key = re.sub(r"\s+", " ", b.lower())
        if key not in seen:
            seen.add(key)
            out.append(b)
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Retrieve codebase beliefs for specific files up to a cutoff timestamp."
    )
    parser.add_argument(
        "--files",
        required=True,
        help="Comma-separated list of file paths relevant to the current task.",
    )
    parser.add_argument(
        "--cutoff",
        required=True,
        help="ISO 8601 timestamp. Only commits before this time are considered.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Maximum number of beliefs to return (default: 10).",
    )
    args = parser.parse_args()

    files   = [f.strip() for f in args.files.split(",") if f.strip()]
    cutoff  = args.cutoff
    top_k   = args.top_k

    # 1. Get commits
    commits = get_commits(files, cutoff)
    if not commits:
        log.warning("no commits found for files=%s before cutoff=%s", files, cutoff)
        sys.exit(0)

    # 2. Build narrative
    narrative_lines = build_narrative(commits, files)

    # 3. Run inference via /batch (one HTTP call, server chunks internally)
    all_beliefs = extract_beliefs_batch(narrative_lines, top_k)

    # 4. Deduplicate and trim
    beliefs = deduplicate(all_beliefs)[:top_k]

    # 5. Print to stdout — agent reads this as context
    log.info("extracted %d beliefs for files=%s cutoff=%s", len(beliefs), files, cutoff)

    if not beliefs:
        log.warning("no beliefs extracted for files=%s", files)
        sys.exit(0)

    print(f"# Codebase Beliefs ({len(beliefs)} retrieved, cutoff: {cutoff})\n")
    for i, belief in enumerate(beliefs, 1):
        print(f"## Belief {i}\n{belief}\n")

    # 6. Append call record to beliefs_log.json in cwd (worktree root).
    #    task_3_run_experiment.py reads this after the Claude Code session ends
    #    to record which beliefs the treatment agent actually retrieved.
    log_path = Path.cwd() / "beliefs_log.json"
    try:
        existing = json.loads(log_path.read_text()) if log_path.exists() else []
        existing.append({"files": files, "cutoff": cutoff, "beliefs": beliefs})
        log_path.write_text(json.dumps(existing, indent=2))
    except Exception as e:
        log.warning("could not write beliefs_log.json: %s", e)


if __name__ == "__main__":
    main()
