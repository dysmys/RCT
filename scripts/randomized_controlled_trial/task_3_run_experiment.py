"""
task_3_run_experiment.py
========================

Runs the experiment for a single repo using Claude Code, Codex, or OpenCode CLI as the agent.

Architecture
------------
  For each (task, arm) pair the script creates a temporary git WORKTREE from
  the shared base clone (no disk duplication).  Multiple tasks run in parallel
  via ThreadPoolExecutor.  Each worktree is deleted after its Claude session
  finishes.

  Base clones (created by task_2_setup_repos.py):
    <REPO_DIR>/<name>/control/     — CLAUDE.md with general guidelines
    <REPO_DIR>/<name>/treatment/   — CLAUDE.md with Belief Management section

  Per-run worktrees (ephemeral):
    <REPO_DIR>/<name>/wt_control_<task_id>/
    <REPO_DIR>/<name>/wt_treatment_<task_id>/

The only difference between arms is CLAUDE.md (inherited from the base clone
and written to the worktree before the session starts).

Parallelism
-----------
  --max-workers N   max concurrent Claude sessions (default 4)

  Control and treatment arms for the same task can run in parallel (different
  base clones, different worktrees).  Tasks within the same arm also run in
  parallel because each gets its own worktree.

Belief tracking (treatment arm)
--------------------------------
  query_belief_store.py appends each invocation to .seng/belief_calls.jsonl in
  the worktree.  After the agent session ends, task_3 reads that file and stores
  the full call log in results.db (beliefs_used column).

Usage:
  python scripts/task_3_run_experiment.py --repo fmt [options]

Options:
  --repo NAME          Repo name (required)
  --repo-dir PATH      Root dir for clones  (default: /mnt/repos or $REPO_DIR)
  --agent CHOICE       claude | codex | opencode | both | all  (default: claude)
  --model MODEL        Model name for the chosen agent
                         claude   default: claude-sonnet-4-6
                         codex    default: gpt-5.4
                         opencode default: glm-5.1
  --arm CHOICE         both | control | treatment  (default: both)
  --max-workers N      Max parallel agent sessions  (default: 4)
  --timeout SECS       Max seconds per session  (default: 240)
  --dry-run            Print tasks without running

Environment:
  ANTHROPIC_API_KEY  — required for claude agent
  OPENAI_API_KEY     — required for codex agent
  OPENAI_API_KEY     — used by opencode agent (or configure via .opencode.json)
"""

import json
import os
import re
import shutil
import signal
import sys
import argparse
import subprocess
import textwrap
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Path bootstrap
# ---------------------------------------------------------------------------

_SCRIPTS_DIR  = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPTS_DIR.parent
sys.path.insert(0, str(_SCRIPTS_DIR))

import database.db as db
from utils.logger import get_logger

log = get_logger(__name__)

load_dotenv(_PROJECT_ROOT / ".env")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DEFAULT_CLAUDE_MODEL   = "claude-sonnet-4-6"
DEFAULT_CODEX_MODEL    = "gpt-5.4"
DEFAULT_OPENCODE_MODEL = "opencode/minimax-m2.5-free"
DEFAULT_MODEL          = DEFAULT_CLAUDE_MODEL   # kept for backwards compat
DEFAULT_REPO_DIR     = os.environ.get("REPO_DIR", "/mnt/repos")
DEFAULT_TIMEOUT      = 360    # seconds per agent session
DEFAULT_MAX_WORKERS  = 12
RUNS_DIR           = _PROJECT_ROOT / "runs"

# Thread-safe timing accumulator
_task_times: list[float] = []
_task_times_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Git helpers
# ---------------------------------------------------------------------------

def git(repo_path: Path, *args) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo_path)] + list(args),
        capture_output=True, text=True
    )
    return result.stdout.strip()


def resolve_snapshot_commit(base_clone: Path, task: dict) -> str:
    """
    Return the git commit hash to use as the worktree snapshot.
    Uses task['snapshot_commit'] if present; otherwise resolves the
    most recent commit before belief_cutoff_timestamp from the clone.
    """
    if "snapshot_commit" in task:
        return task["snapshot_commit"]
    cutoff = task["belief_cutoff_timestamp"]
    commit = git(base_clone, "log", f"--before={cutoff}", "--format=%H", "-1")
    if not commit:
        raise RuntimeError(
            f"No commit found before {cutoff} in {base_clone}"
        )
    return commit


def create_worktree(base_clone: Path, worktree_path: Path, commit: str):
    """Create a detached worktree at the given commit."""
    if worktree_path.exists():
        shutil.rmtree(worktree_path)
    result = subprocess.run(
        ["git", "-C", str(base_clone), "worktree", "add", "--detach",
         str(worktree_path), commit],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"git worktree add failed (rc={result.returncode}): {result.stderr.strip()}"
        )


def remove_worktree(base_clone: Path, worktree_path: Path):
    """Remove a worktree and prune the reference."""
    try:
        git(base_clone, "worktree", "remove", "--force", str(worktree_path))
    except Exception:
        pass
    if worktree_path.exists():
        shutil.rmtree(worktree_path, ignore_errors=True)
    git(base_clone, "worktree", "prune")


# ---------------------------------------------------------------------------
# CLAUDE.md / AGENTS.md content — imported from task_2_setup_repos
# ---------------------------------------------------------------------------

from task_2_setup_repos import (
    CLAUDE_MD_CONTROL, CLAUDE_MD_TREATMENT,
    AGENTS_MD_CONTROL, AGENTS_MD_TREATMENT,
)


# ---------------------------------------------------------------------------
# Per-run log writer
# ---------------------------------------------------------------------------

def run_log_dir(task: dict, arm: str, agent: str = "claude") -> Path:
    d = RUNS_DIR / task["repo"].split("/")[-1] / task["task_type"] / task["id"] / f"{arm}_{agent}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _build_task_md(task: dict) -> str:
    inp      = task["input"]
    gt       = task["ground_truth"]
    relevant = task["relevant_files"]
    cutoff   = task["belief_cutoff_timestamp"]
    ref_ts   = task.get("reference_commit_timestamp", "")
    t        = task["task_type"]

    lines = [
        f"# Task: {task['id']}",
        f"**Type:** {t}  |  **Repo:** {task['repo']}",
        f"**Snapshot commit:** `{task.get('snapshot_commit', 'HEAD')}`",
        f"**Belief cutoff:** {cutoff}",
        f"**Reference timestamp:** {ref_ts}",
        "\n**Relevant files:**",
    ]
    for f in relevant:
        lines.append(f"- `{f}`")

    if t == "bug_fix":
        lines += [
            f"\n## Issue #{inp.get('issue_number','?')}: {inp.get('issue_title','')}",
            f"\n{inp.get('issue_body','').strip()}",
            f"\n## Ground Truth Fix\n```diff\n{gt.get('diff','').strip()[:3000]}\n```",
        ]
    elif t == "feature_impl":
        lines += [
            f"\n## PR #{inp.get('pr_number','?')}: {inp.get('pr_title','')}",
            f"\n{inp.get('pr_body','').strip()}",
            f"\n**Merged at:** {gt.get('merged_at','')}",
            f"\n## Ground Truth Diff\n```diff\n{gt.get('diff','').strip()[:3000]}\n```",
        ]
    elif t == "code_review":
        lines += [
            f"\n## PR #{inp.get('pr_number','?')}: {inp.get('pr_title','')}",
            f"\n{inp.get('pr_body','').strip()}",
            f"\n## Diff to Review\n```diff\n{inp.get('diff','').strip()[:3000]}\n```",
            f"\n## Ground Truth\n{gt.get('known_issues','')}",
        ]

    return "\n".join(lines) + "\n"


def write_run_logs(task: dict, arm: str, agent: str, response: str,
                   run_id: int, status: str, agent_diff: str = "",
                   belief_calls: list[dict] | None = None):
    d = run_log_dir(task, arm, agent)
    (d / "task.md").write_text(_build_task_md(task))
    if response:
        (d / "response.md").write_text(response)
    (d / "diff.patch").write_text(agent_diff or "(no changes)\n")
    if belief_calls:
        lines = [f"# Beliefs Called -- {task['id']} (treatment)\n"]
        for call in belief_calls:
            lines.append(f"## Call {call.get('seq', '?')}\n")
            lines.append(f"**Query:** {call.get('query', '')}\n")
            beliefs_val = call.get('beliefs', [])
            if isinstance(beliefs_val, list):
                for b in beliefs_val:
                    lines.append(f"- {b.get('statement', '')}\n")
            else:
                lines.append(f"{beliefs_val}\n")
        (d / "beliefs.md").write_text("\n".join(lines))
    meta = {"run_id": run_id, "task_id": task["id"], "arm": arm,
            "agent": agent, "status": status,
            "belief_calls": len(belief_calls) if belief_calls else 0}
    (d / "meta.json").write_text(json.dumps(meta, indent=2))


# ---------------------------------------------------------------------------
# File content pre-loading
# ---------------------------------------------------------------------------

def read_file_at_commit(worktree: Path, filepath: str, commit: str) -> str:
    """Read a file's content at a specific git commit (before any agent changes)."""
    result = subprocess.run(
        ["git", "-C", str(worktree), "show", f"{commit}:{filepath}"],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        return f"(file not found at commit {commit[:8]}: {filepath})"
    return result.stdout


def load_file_contents(worktree: Path, files: list[str], snapshot: str,
                       max_chars_per_file: int = 8000) -> str:
    """
    Read each relevant file at the snapshot commit and format for the prompt.
    Truncates large files to keep the prompt manageable.
    """
    sections = []
    for filepath in files:
        content = read_file_at_commit(worktree, filepath, snapshot)
        if len(content) > max_chars_per_file:
            content = content[:max_chars_per_file] + f"\n\n... (truncated at {max_chars_per_file} chars)"
        ext = filepath.rsplit(".", 1)[-1] if "." in filepath else ""
        sections.append(f"### `{filepath}`\n```{ext}\n{content}\n```")
    return "\n\n".join(sections)




# ---------------------------------------------------------------------------
# Task prompt builders
# ---------------------------------------------------------------------------

def build_task_prompt(task: dict, arm: str, wt_path: Path | None = None) -> str:
    """
    Build the task prompt for agentic mode.
    No file contents injected -- agents read files themselves.
    Treatment arm: prompt explicitly instructs agent to query belief store first.
    wt_path: absolute path to the worktree (used so belief_calls.jsonl goes there).
    """
    task_type = task["task_type"]
    inp       = task["input"]

    belief_preamble = ""
    if arm == "treatment":
        # Build a concrete, ready-to-run query (no placeholders, no shell vars)
        repo_name = task.get("repo", "").split("/")[-1]
        cutoff    = task.get("belief_cutoff_timestamp", "")
        seng_dir  = str(wt_path / ".seng") if wt_path else ".seng"
        if task_type == "bug_fix":
            q = f"bug fix: {inp.get('issue_title', '')} {inp.get('issue_body', '')[:200]}"
        elif task_type == "feature_impl":
            q = f"feature: {inp.get('pr_title', '')} {inp.get('pr_body', '')[:200]}"
        else:
            q = f"code review: {inp.get('pr_title', '')} {inp.get('pr_body', '')[:200]}"
        query_text = q.replace('"', "'").replace("\n", " ").replace("\r", "")[:400]

        belief_preamble = textwrap.dedent(f"""\
            **MANDATORY FIRST STEP — run this exact command before doing anything else:**

            python3 {seng_dir}/query_belief_store.py \\
                --repo {repo_name} \\
                --query "{query_text}" \\
                --cutoff {cutoff} \\
                --db /opt/experiment/belief_results.db \\
                --seng-dir {seng_dir} \\
                --top-k 5

            Do NOT skip this step. Read all returned beliefs before exploring the repo.
            The beliefs describe coding patterns established in this codebase.

            ---

        """)

    if task_type == "bug_fix":
        return belief_preamble + textwrap.dedent(f"""\
            ## Task: Bug Fix

            **Issue #{inp.get('issue_number','?')}: {inp.get('issue_title','')}**

            {inp.get('issue_body','').strip()}

            Explore the repository to understand the codebase, identify the root cause,
            and apply a fix. Write your changes directly to the relevant files.
        """)

    elif task_type == "feature_impl":
        return belief_preamble + textwrap.dedent(f"""\
            ## Task: Feature Implementation

            **PR #{inp.get('pr_number','?')}: {inp.get('pr_title','')}**

            {inp.get('pr_body','').strip()}

            Explore the repository to understand the existing architecture, then implement
            this feature following the existing patterns. Write your changes to files.
        """)

    elif task_type == "code_review":
        diff_snippet = inp.get("diff", "")[:3000]
        return belief_preamble + textwrap.dedent(f"""\
            ## Task: Code Review

            **PR #{inp.get('pr_number','?')}: {inp.get('pr_title','')}**

            {inp.get('pr_body','').strip()}

            ## Diff to Review
            ```diff
            {diff_snippet}
            ```

            Review this pull request. Explore the repository for additional context.
        """)

    raise ValueError(f"Unknown task_type: {task_type}")


# ---------------------------------------------------------------------------
# Claude Code CLI runner
# ---------------------------------------------------------------------------

def run_claude(worktree: Path, task_prompt: str, model: str, timeout: int) -> tuple[str, int, dict]:
    """
    Run claude --print inside worktree with --output-format json for token tracking.
    Returns (output_text, exit_code, usage_dict).
    usage_dict has keys: input_tokens, output_tokens (may be None on parse failure).
    """
    oauth_token = os.environ.get("CLAUDE_CODE_OAUTH_TOKEN", "")
    env = {
        **os.environ,
        "PYTHONPATH": str(_SCRIPTS_DIR),
    }
    if oauth_token:
        env["CLAUDE_CODE_OAUTH_TOKEN"] = oauth_token
        env.pop("ANTHROPIC_API_KEY", None)
    else:
        env["ANTHROPIC_API_KEY"] = os.environ.get("ANTHROPIC_API_KEY", "")

    proc = subprocess.Popen(
        ["claude", "--print", "--dangerously-skip-permissions", "--model", model,
         "--output-format", "json"],
        cwd=str(worktree),
        env=env,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        preexec_fn=os.setsid,
    )

    try:
        stdout, stderr = proc.communicate(input=task_prompt, timeout=timeout)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except ProcessLookupError:
            pass
        proc.communicate()
        raise

    if proc.returncode != 0:
        log.warning("claude exited %d in %s — stderr: %s",
                    proc.returncode, worktree.name, stderr[:300])

    # Parse JSON output for result text and token usage
    usage = {"input_tokens": None, "output_tokens": None}
    output_text = stdout.strip()
    try:
        data = json.loads(stdout)
        output_text = data.get("result", stdout.strip())
        u = data.get("usage", {})
        usage["input_tokens"] = u.get("input_tokens")
        usage["output_tokens"] = u.get("output_tokens")
    except (json.JSONDecodeError, AttributeError):
        pass  # fallback: treat stdout as plain text

    return output_text, proc.returncode, usage


def run_codex(worktree: Path, task_prompt: str, model: str, timeout: int) -> tuple[str, int, dict]:
    """
    Run codex non-interactively inside worktree.
    Returns (output_text, exit_code, usage_dict).
    Token usage is parsed from stderr where Codex logs it.
    """
    env = {
        **os.environ,
        "OPENAI_API_KEY": os.environ.get("OPENAI_API_KEY", ""),
        "PYTHONPATH":     str(_SCRIPTS_DIR),
    }

    proc = subprocess.Popen(
        ["codex", "exec", "--full-auto", "--skip-git-repo-check", "--model", model, "-"],
        cwd=str(worktree),
        env=env,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        preexec_fn=os.setsid,
    )

    try:
        stdout, stderr = proc.communicate(input=task_prompt, timeout=timeout)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except ProcessLookupError:
            pass
        proc.communicate()
        raise

    if proc.returncode != 0:
        log.warning("codex exited %d in %s — stderr: %s",
                    proc.returncode, worktree.name, stderr[:300])

    # Parse token usage from stderr (Codex logs "input_tokens: N, output_tokens: N")
    usage = {"input_tokens": None, "output_tokens": None}
    for line in (stderr or "").splitlines():
        m = re.search(r'input_tokens\D*(\d+)', line)
        if m:
            usage["input_tokens"] = int(m.group(1))
        m = re.search(r'output_tokens\D*(\d+)', line)
        if m:
            usage["output_tokens"] = int(m.group(1))

    return stdout.strip(), proc.returncode, usage


def run_opencode(worktree: Path, task_prompt: str, model: str, timeout: int) -> tuple[str, int, dict]:
    """
    Run OpenCode non-interactively inside worktree.
    Uses 'opencode run' subcommand with -m for model selection.
    Returns (output_text, exit_code, usage_dict).
    Token usage captured via 'opencode stats' after the run.
    """
    env = {
        **os.environ,
        "PYTHONPATH": str(_SCRIPTS_DIR),
    }

    proc = subprocess.Popen(
        ["opencode", "run", task_prompt, "-m", model],
        cwd=str(worktree),
        env=env,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        preexec_fn=os.setsid,
    )

    try:
        stdout, stderr = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except ProcessLookupError:
            pass
        proc.communicate()
        raise

    if proc.returncode != 0:
        log.warning("opencode exited %d in %s — stderr: %s",
                    proc.returncode, worktree.name, stderr[:300])

    # Try to extract token usage from opencode stats
    usage = {"input_tokens": None, "output_tokens": None}
    try:
        stats_result = subprocess.run(
            ["opencode", "stats"],
            cwd=str(worktree), capture_output=True, text=True, timeout=10,
        )
        for line in stats_result.stdout.splitlines():
            m = re.search(r'input\D*(\d[\d,]*)', line, re.IGNORECASE)
            if m:
                usage["input_tokens"] = int(m.group(1).replace(",", ""))
            m = re.search(r'output\D*(\d[\d,]*)', line, re.IGNORECASE)
            if m:
                usage["output_tokens"] = int(m.group(1).replace(",", ""))
    except Exception:
        pass

    return stdout.strip(), proc.returncode, usage


def read_beliefs_log(worktree: Path) -> list[dict]:
    log_path = worktree / "beliefs_log.json"
    if not log_path.exists():
        return []
    try:
        return json.loads(log_path.read_text())
    except Exception as e:
        log.warning("could not read beliefs_log.json in %s: %s", worktree.name, e)
        return []


# ---------------------------------------------------------------------------
# Treatment arm worktree setup + belief call log reader
# ---------------------------------------------------------------------------

_QUERY_BELIEF_SCRIPT = Path(__file__).resolve().parent.parent / "tools" / "query_belief_store.py"

_DEFAULT_BELIEF_DB    = os.environ.get("BELIEF_DB",         "/opt/experiment/belief_results.db")
_DEFAULT_CHROMA_DIR   = os.environ.get("BELIEF_CHROMA_DIR", "/opt/experiment/chroma_db")


def setup_treatment_worktree(wt_path: Path, task: dict):
    """
    Copy query_belief_store.py into .seng/ in the worktree.
    Writes REPO_NAME, BELIEF_CUTOFF, BELIEF_DB, BELIEF_CHROMA_DIR into .seng/.env
    so the agent can call the tool without hard-coding paths.
    """
    seng_dir = wt_path / ".seng"
    seng_dir.mkdir(exist_ok=True)
    dest = seng_dir / "query_belief_store.py"
    shutil.copy2(str(_QUERY_BELIEF_SCRIPT), str(dest))
    dest.chmod(0o755)

    repo_name = task.get("repo", "").split("/")[-1]
    cutoff    = task.get("belief_cutoff_timestamp", "")
    (seng_dir / ".env").write_text(
        f"REPO_NAME={repo_name}\n"
        f"BELIEF_CUTOFF={cutoff}\n"
        f"BELIEF_DB={_DEFAULT_BELIEF_DB}\n"
        f"BELIEF_CHROMA_DIR={_DEFAULT_CHROMA_DIR}\n"
    )


def _read_belief_calls_jsonl(wt_path: Path) -> list[dict]:
    """Read .seng/belief_calls.jsonl written by query_belief_store.py during the session."""
    log_file = wt_path / ".seng" / "belief_calls.jsonl"
    if not log_file.exists():
        return []
    calls = []
    try:
        for line in log_file.read_text().splitlines():
            if line.strip():
                calls.append(json.loads(line))
    except Exception as e:
        log.warning("Could not read belief_calls.jsonl: %s", e)
    return calls


# ---------------------------------------------------------------------------
# Per-(task, arm) worker
# ---------------------------------------------------------------------------

def run_arm(task: dict, arm: str, base_clone: Path, model: str,
            timeout: int, database, conn_lock: threading.Lock,
            agent: str = "claude") -> str:
    """
    Run one (task, arm, agent) triple.  Returns a status string for logging.
    Creates its own git worktree, runs the agent, cleans up.
    """
    task_id  = task["id"]
    snapshot = resolve_snapshot_commit(base_clone, task)
    wt_name  = f"wt_{arm}_{agent}_{task_id}"
    wt_path  = base_clone.parent / wt_name
    _arm_start = time.monotonic()

    # ---- Idempotency check (thread-safe) ------------------------------------
    with conn_lock:
        existing = database.conn.execute(
            "SELECT run_id, status FROM runs WHERE task_id = ? AND arm = ? AND agent = ?",
            (task_id, arm, agent)
        ).fetchone()
        if existing and existing["status"] == "completed":
            log.info("  %s/%s/%s  run_id=%d already completed — skipping",
                     task_id, arm, agent, existing["run_id"])
            return "skipped"

    # ---- Create worktree ----------------------------------------------------
    try:
        create_worktree(base_clone, wt_path, snapshot)
    except Exception as e:
        log.error("  %s/%s/%s  worktree create failed: %s", task_id, arm, agent, e)
        return "worktree_error"

    try:
        # Set up treatment arm worktree with query_belief_store tool
        if arm == "treatment":
            setup_treatment_worktree(wt_path, task)

        # Write the agent's instruction file into the worktree (per-arm, per-agent).
        if agent == "codex":
            md = AGENTS_MD_TREATMENT if arm == "treatment" else AGENTS_MD_CONTROL
            (wt_path / "AGENTS.md").write_text(md)
        else:
            md = CLAUDE_MD_TREATMENT if arm == "treatment" else CLAUDE_MD_CONTROL
            (wt_path / "CLAUDE.md").write_text(md)

        # ---- Build prompt ---------------------------------------------------
        task_prompt = build_task_prompt(task, arm, wt_path=wt_path)

        # ---- Insert run record (thread-safe) --------------------------------
        with conn_lock:
            run_id = database.insert_run(task_id, arm, model, agent=agent)

        log.info("  %s/%s/%s  run_id=%d  starting agent...",
                 task_id, arm, agent, run_id)

        # ---- Run agent (long operation, no lock held) -----------------------
        try:
            if agent == "codex":
                output, exit_code, usage = run_codex(wt_path, task_prompt, model, timeout)
            elif agent == "opencode":
                output, exit_code, usage = run_opencode(wt_path, task_prompt, model, timeout)
            else:
                output, exit_code, usage = run_claude(wt_path, task_prompt, model, timeout)
        except subprocess.TimeoutExpired:
            with conn_lock:
                database.fail_run(run_id, f"timeout after {timeout}s")
            log.error("  %s/%s/%s  run_id=%d  timed out after %ds",
                      task_id, arm, agent, run_id, timeout)
            return "timeout"
        except Exception as e:
            with conn_lock:
                database.fail_run(run_id, str(e))
            log.error("  %s/%s/%s  run_id=%d  agent error: %s", task_id, arm, agent, run_id, e)
            return "error"

        # ---- Capture git diff from worktree ---------------------------------
        diff_result = subprocess.run(
            ["git", "-C", str(wt_path), "diff", snapshot],
            capture_output=True, text=True
        )
        agent_diff = diff_result.stdout.strip()

        # Parse modified files from diff headers
        files_modified = re.findall(r"^diff --git a/.+ b/(.+)$",
                                    agent_diff, re.MULTILINE)

        # Read belief tool call log (treatment arm)
        belief_calls = _read_belief_calls_jsonl(wt_path)
        belief_calls_total = len(belief_calls)

        # ---- Collect results ------------------------------------------------
        elapsed = time.monotonic() - _arm_start
        input_tokens = usage.get("input_tokens")
        output_tokens = usage.get("output_tokens")

        if exit_code == 0 and output:
            with conn_lock:
                database.complete_run_agentic(
                    run_id,
                    output_text=output,
                    agent_diff=agent_diff,
                    files_modified=files_modified,
                    tool_calls_total=None,
                    belief_calls_total=belief_calls_total,
                    beliefs_used=[
                        json.dumps(c.get("beliefs", [])) if isinstance(c.get("beliefs"), list)
                        else c.get("beliefs", "")
                        for c in belief_calls
                    ] if belief_calls else None,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    duration_seconds=round(elapsed, 2),
                )
                for call in belief_calls:
                    database.insert_belief_tool_call(
                        run_id,
                        call_sequence=call["seq"],
                        commits_used=[call.get("query", "")],
                        beliefs_returned=json.dumps(call.get("beliefs", [])),
                    )

            write_run_logs(task, arm, agent, output,
                           run_id=run_id, status="completed",
                           agent_diff=agent_diff, belief_calls=belief_calls or None)

            with _task_times_lock:
                _task_times.append(elapsed)
            total_tokens = (input_tokens or 0) + (output_tokens or 0) or None
            log.info("  %s/%s/%s  run_id=%d  completed  output_len=%d  belief_calls=%d  "
                     "time=%.1fs  tokens=%s (in=%s out=%s)",
                     task_id, arm, agent, run_id, len(output), belief_calls_total,
                     elapsed, total_tokens, input_tokens, output_tokens)
            return "completed"
        else:
            msg = f"{agent} exited {exit_code} with {'no output' if not output else 'output'}"
            with conn_lock:
                database.fail_run(run_id, msg)
            write_run_logs(task, arm, agent, output,
                           run_id=run_id, status="failed",
                           agent_diff=agent_diff, belief_calls=belief_calls or None)
            log.error("  %s/%s/%s  run_id=%d  failed: %s", task_id, arm, agent, run_id, msg)
            return "failed"

    finally:
        remove_worktree(base_clone, wt_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Run experiment for a repo using Claude Code or Codex CLI."
    )
    parser.add_argument("--repo",        required=True)
    parser.add_argument("--repo-dir",    default=DEFAULT_REPO_DIR)
    parser.add_argument("--agent",       choices=["claude", "codex", "opencode", "both", "all"], default="claude",
                        help="Agent to use: claude | codex | opencode | both | all (default: claude)")
    parser.add_argument("--model",       default=None,
                        help="Model for chosen agent. Defaults: claude=claude-sonnet-4-6, codex=gpt-5.4, opencode=glm-5.1")
    parser.add_argument("--arm",         choices=["both", "control", "treatment"], default="both")
    parser.add_argument("--max-workers", type=int, default=DEFAULT_MAX_WORKERS,
                        help="Max concurrent agent sessions (default: 4)")
    parser.add_argument("--timeout",     type=int, default=DEFAULT_TIMEOUT,
                        help="Max seconds per agent session (default: 240)")
    parser.add_argument("--dry-run",     action="store_true")
    args = parser.parse_args()

    if args.agent == "all":
        agents = ["claude", "codex", "opencode"]
    elif args.agent == "both":
        agents = ["claude", "codex"]
    else:
        agents = [args.agent]
    arms     = ["control", "treatment"] if args.arm == "both" else [args.arm]
    repo_dir = Path(args.repo_dir)
    base_clones = {
        "control":   repo_dir / args.repo / "control",
        "treatment": repo_dir / args.repo / "treatment",
    }

    tasks_file = _PROJECT_ROOT / "dataset" / "tasks" / f"{args.repo}.json"
    if not tasks_file.exists():
        log.error("task file not found: %s", tasks_file)
        sys.exit(1)

    tasks = json.loads(tasks_file.read_text())
    tasks.sort(key=lambda t: (t["task_type"], t["id"]))

    # Resolve model defaults per agent (when --agent both/all, use per-agent defaults)
    model_for: dict[str, str] = {}
    for ag in agents:
        if args.model:
            model_for[ag] = args.model
        elif ag == "codex":
            model_for[ag] = DEFAULT_CODEX_MODEL
        elif ag == "opencode":
            model_for[ag] = DEFAULT_OPENCODE_MODEL
        else:
            model_for[ag] = DEFAULT_CLAUDE_MODEL

    log.info("repo=%s  tasks=%d  agents=%s  arms=%s  models=%s  workers=%d  timeout=%ds",
             args.repo, len(tasks), agents, arms, model_for, args.max_workers, args.timeout)

    if args.dry_run:
        for t in tasks:
            print(f"{t['id']}  type={t['task_type']}  "
                  f"snapshot={t.get('snapshot_commit', '(pending)')[:8]}  "
                  f"files={len(t['relevant_files'])}")
        return

    # Validate base clones exist
    for arm in arms:
        d = base_clones[arm]
        if not (d / ".git").exists():
            log.error("%s clone not found at %s — run task_2_setup_repos.py first", arm, d)
            sys.exit(1)

    # Verify agent CLIs are present
    for ag in agents:
        cli = {"claude": "claude", "codex": "codex", "opencode": "opencode"}[ag]
        result = subprocess.run([cli, "--version"], capture_output=True, text=True)
        if result.returncode != 0:
            log.error("%s CLI not found — install it before running with --agent %s", cli, ag)
            sys.exit(1)
        log.info("%s version: %s", cli, result.stdout.strip())

    # Verify belief store is reachable before starting
    _belief_db = Path(_DEFAULT_BELIEF_DB)
    _chroma_dir = Path(_DEFAULT_CHROMA_DIR)
    if not _belief_db.exists():
        log.warning("belief DB not found at %s — treatment arms will get no beliefs", _belief_db)
    elif not _chroma_dir.exists():
        log.warning("Chroma dir not found at %s — treatment arms will get no beliefs", _chroma_dir)
    else:
        log.info("belief store ready: db=%s chroma=%s", _belief_db, _chroma_dir)

    database  = db.get_db()
    conn_lock = threading.Lock()

    # Resolve snapshot commits for tasks that don't have them pre-computed.
    # This also ensures the field is populated before inserting into DB
    # (DB schema requires snapshot_commit NOT NULL).
    # Use the control clone as the canonical source.
    _control_clone = base_clones.get("control", list(base_clones.values())[0])
    for task in tasks:
        if "snapshot_commit" not in task or not task.get("snapshot_commit"):
            try:
                task["snapshot_commit"] = resolve_snapshot_commit(_control_clone, task)
                log.debug("resolved snapshot for %s: %s", task["id"], task["snapshot_commit"][:12])
            except Exception as _e:
                log.error("could not resolve snapshot for %s: %s", task["id"], _e)

    # Ensure repo record exists in DB (tasks.repo_name FKs to repos.name)
    _test_json = _PROJECT_ROOT / "dataset" / "test.json"
    _repo_entry: dict = {"name": args.repo, "url": "", "language": "unknown"}
    if _test_json.exists():
        _test_data = json.loads(_test_json.read_text())
        for _r in _test_data.get("repositories", []):
            if _r["name"] == args.repo:
                _repo_entry = _r
                break
    database.upsert_repo(_repo_entry)

    # Ensure all tasks exist in DB (idempotent — INSERT OR IGNORE)
    # Pass repo_name=args.repo explicitly because task['repo'] may have a different
    # last component than the directory name (e.g., 'home-assistant/core' → 'core'
    # but the repos table was inserted as 'home-assistant').
    # snapshot_commit is resolved above and injected into each task dict before this call.
    for task in tasks:
        database.insert_task(task, repo_name=args.repo)

    # Build all (task, arm, agent) work items
    work_items = [
        (task, arm, agent)
        for task in tasks
        for arm in arms
        for agent in agents
    ]

    # Run in parallel
    outcomes = {}
    with ThreadPoolExecutor(max_workers=args.max_workers) as pool:
        futures = {
            pool.submit(
                run_arm, task, arm, base_clones[arm],
                model_for[agent], args.timeout, database, conn_lock,
                agent,
            ): (task["id"], arm, agent)
            for task, arm, agent in work_items
        }

        for future in as_completed(futures):
            task_id, arm, agent = futures[future]
            try:
                status = future.result()
            except Exception as exc:
                log.error("%s/%s/%s  unexpected exception: %s", task_id, arm, agent, exc, exc_info=True)
                status = "exception"
            outcomes[(task_id, arm, agent)] = status

    # Summary
    log.info("=" * 60)
    log.info("Run summary for '%s':", args.repo)
    from collections import Counter
    counts = Counter(outcomes.values())
    for status, n in sorted(counts.items()):
        log.info("  %-15s  n=%d", status, n)
    if _task_times:
        avg = sum(_task_times) / len(_task_times)
        mn  = min(_task_times)
        mx  = max(_task_times)
        log.info("Timing (completed runs): n=%d  avg=%.1fs  min=%.1fs  max=%.1fs",
                 len(_task_times), avg, mn, mx)

    task_ids     = tuple(t["id"] for t in tasks)
    placeholders = ",".join("?" * len(task_ids))
    rows = database.conn.execute(f"""
        SELECT agent, arm, status, COUNT(*) AS n FROM runs
        WHERE task_id IN ({placeholders})
        GROUP BY agent, arm, status ORDER BY agent, arm, status
    """, task_ids).fetchall()
    log.info("DB summary:")
    for row in rows:
        log.info("  agent=%-8s  arm=%-12s  status=%-12s  n=%d",
                 row["agent"], row["arm"], row["status"], row["n"])


if __name__ == "__main__":
    main()
