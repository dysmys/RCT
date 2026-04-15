"""
judge.py
========

Single-file judge module.  All judging logic lives in the Judge class.

Usage (CLI):
  python scripts/judge.py --repo fmt --submit
  python scripts/judge.py --repo fmt --status
  python scripts/judge.py --repo fmt --collect
  python scripts/judge.py --repo fmt --sync
  python scripts/judge.py --repo fmt --sync --rescore

Environment:
  OPENAI_API_KEY  — required
  DB_PATH         — optional, defaults to <project_root>/database/results.db
  REPO_DIR        — optional, defaults to /mnt/repos
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import openai
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Path bootstrap
# ---------------------------------------------------------------------------

_SCRIPTS_DIR  = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPTS_DIR.parent

sys.path.insert(0, str(_SCRIPTS_DIR))

import database.db as db
from utils.logger import get_logger

load_dotenv(_PROJECT_ROOT / ".env")

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Rubrics (one per task type)
# ---------------------------------------------------------------------------

_RUBRIC_BUG_FIX = """
Score using a decomposed checklist. For each criterion assign 0 (No), 1 (Partial), or 2 (Yes).
Final score = sum of its 5 criteria (range 0–10).

CORRECTNESS — Did the agent identify the real root cause and fix it correctly?
C1. Root cause identified (0=missed entirely, 1=partially, 2=precise)
C2. Fix technically correct (0=broken/wrong, 1=correct direction with gap, 2=correct)
C3. Completeness vs ground truth (0=major gaps >50%, 1=partial, 2=substantially complete)
C4. Code runs correctly as written (0=broken, 1=needs non-trivial fixes, 2=runnable)
C5. Ground truth alignment (0=fundamentally different, 1=same direction with gaps, 2=matches/improves)
correctness = C1+C2+C3+C4+C5

CONVENTION ADHERENCE — Does the fix follow the codebase's style?
A1. Naming conventions (0=wrong style, 1=mostly right with deviations, 2=matches)
A2. Code structure/organization (0=wrong placement, 1=mostly correct, 2=matches)
A3. Error handling style (0=wrong approach, 1=partial, 2=matches codebase)
A4. Documentation level (0=missing required docs, 1=partial, 2=matches surrounding code)
A5. Imports/dependencies (0=wrong imports or unnecessary deps, 1=minor issues, 2=correct)
convention_adherence = A1+A2+A3+A4+A5

API CORRECTNESS — Does the implementation use actual APIs and patterns from this codebase?
P1. No hallucinated functions (0=multiple non-existent calls, 1=one minor hallucination, 2=all calls real)
P2. Correct API usage (0=wrong args/call site, 1=one meaningful misuse, 2=all calls correct)
P3. Imports match codebase (0=wrong paths/non-existent modules, 1=minor style diff, 2=matches exactly)
P4. File placement (0=wrong files/locations, 1=mostly correct with one questionable, 2=appropriate per structure)
P5. Integration points (0=would not integrate cleanly, 1=mostly correct with minor gap, 2=clean and consistent)
api_correctness = P1+P2+P3+P4+P5

RELEVANCE — Is the response focused on the reported bug?
R1. On-task (0=addresses different problem, 1=partial, 2=directly addresses bug)
R2. No scope creep (0=substantial unrelated changes, 1=minor additions, 2=scoped to bug)
R3. No boilerplate/filler (0=mostly filler, 1=some verbosity, 2=direct and focused)
R4. Bug-specific reasoning (0=generic advice, 1=partially specific, 2=tied to this bug)
R5. Accurate references (0=references non-existent code, 1=minor inaccuracies, 2=all accurate)
relevance = R1+R2+R3+R4+R5
"""

_RUBRIC_FEATURE_IMPL = """
Score using a decomposed checklist. For each criterion assign 0 (No), 1 (Partial), or 2 (Yes).
Final score = sum of its 5 criteria (range 0–10).

CORRECTNESS — Does the implementation correctly fulfill the feature requirement?
C1. Requirement understood (0=wrong interpretation, 1=partial, 2=correct)
C2. Technical approach sound (0=broken/wrong, 1=correct direction with gap, 2=correct)
C3. Completeness vs merged diff (0=major gaps >50% missing, 1=partial, 2=substantially complete)
C4. Code runs correctly as written (0=broken, 1=needs non-trivial fixes, 2=runnable)
C5. Ground truth alignment (0=fundamentally different, 1=same direction with gaps, 2=matches/improves)
correctness = C1+C2+C3+C4+C5

CONVENTION ADHERENCE — Does the implementation follow the codebase's patterns?
A1. Naming conventions (0=wrong style, 1=mostly right with deviations, 2=matches)
A2. Code structure/organization (0=wrong placement, 1=mostly correct, 2=matches)
A3. Error handling style (0=wrong approach, 1=partial, 2=matches codebase)
A4. Documentation level (0=missing required docs, 1=partial, 2=matches surrounding code)
A5. Imports/dependencies (0=wrong imports or unnecessary deps, 1=minor issues, 2=correct)
convention_adherence = A1+A2+A3+A4+A5

API CORRECTNESS — Does the implementation use actual APIs and patterns from this codebase?
P1. No hallucinated functions (0=multiple non-existent calls, 1=one minor hallucination, 2=all calls real)
P2. Correct API usage (0=wrong args/call site, 1=one meaningful misuse, 2=all calls correct)
P3. Imports match codebase (0=wrong paths/non-existent modules, 1=minor style diff, 2=matches exactly)
P4. File placement (0=wrong files/locations, 1=mostly correct with one questionable, 2=appropriate per structure)
P5. Integration points (0=would not integrate cleanly, 1=mostly correct with minor gap, 2=clean and consistent)
api_correctness = P1+P2+P3+P4+P5

RELEVANCE — Does the response focus on what was actually requested?
R1. On-task (0=addresses different problem, 1=partial, 2=directly addresses PR description)
R2. No scope creep (0=substantial unrelated changes, 1=minor additions, 2=scoped to request)
R3. No boilerplate/filler (0=mostly filler, 1=some verbosity, 2=direct and focused)
R4. PR-specific reasoning (0=generic advice, 1=partially specific, 2=tied to this PR)
R5. Accurate references (0=references non-existent code, 1=minor inaccuracies, 2=all accurate)
relevance = R1+R2+R3+R4+R5
"""

_RUBRIC_CODE_REVIEW = """
Score using a decomposed checklist. For each criterion assign 0 (No), 1 (Partial), or 2 (Yes).
Final score = sum of its 5 criteria (range 0–10).

CORRECTNESS — Did the agent identify the real problems that caused this PR to be reverted?
C1. Root cause identified (0=missed entirely/only symptoms, 1=partially, 2=precise root cause)
C2. Technical accuracy (0=wrong reasoning, 1=partially correct, 2=technically sound)
C3. Severity correctly assessed (0=wrong severity/would not prompt revert, 1=somewhat correct, 2=correctly flags as revert-worthy)
C4. Completeness vs ground truth (0=misses >50% of known issues, 1=some but not all, 2=identifies all/nearly all)
C5. Actionability (0=too vague to act on, 1=partially actionable, 2=clear guidance to fix the problem)
correctness = C1+C2+C3+C4+C5

RELEVANCE — Is the review focused on the diff with specific, actionable feedback?
R1. Diff-specific references (0=no specific references, 1=some mixed with generic, 2=tightly focused on diff)
R2. No filler (0=mostly filler/generic praise, 1=some filler, 2=all content substantive)
R3. Correctness-focused (0=mostly style/formatting issues, 1=mixed, 2=focused on correctness/logic)
R4. PR-specific reasoning (0=generic advice only, 1=partially specific, 2=all reasoning specific to this code)
R5. Conciseness (0=severely repetitive/verbose, 1=somewhat verbose, 2=concise and direct)
relevance = R1+R2+R3+R4+R5
"""


class Judge:
    """
    Judges completed agent runs for a single repo using GPT-4o.

    Supports both async (OpenAI Batch API) and synchronous scoring paths.

    Example:
        judge = Judge(repo="fmt", repo_dir="/mnt/repos")
        judge.sync()          # score immediately
        judge.submit()        # submit async batch job
        judge.status()        # check batch status
        judge.collect()       # download and store batch results
    """

    JUDGE_MODEL = "gpt-4o"
    _BATCHES_DIR = _PROJECT_ROOT / "runs" / "judge_batches"
    _TASKS_DIR   = _PROJECT_ROOT / "dataset" / "tasks"

    def __init__(self, repo: str, repo_dir: str,
                 judge_model: str = JUDGE_MODEL):
        self._repo       = repo
        self._repo_dir   = Path(repo_dir)
        self._judge_model = judge_model
        self._client     = self._init_openai_client()
        self._db         = db.get_db()
        self._tasks_by_id: dict[str, dict] = self._load_tasks()

    # -------------------------------------------------------------------------
    # Initialisation (private)
    # -------------------------------------------------------------------------

    def _init_openai_client(self) -> openai.OpenAI:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY not set in environment or .env")
        return openai.OpenAI(api_key=api_key)


    def _load_tasks(self) -> dict[str, dict]:
        path = self._TASKS_DIR / f"{self._repo}.json"
        if not path.exists():
            raise FileNotFoundError(
                f"No task file: {path} — run task_1_extract_tasks.py first"
            )
        tasks = json.loads(path.read_text())
        return {t["id"]: t for t in tasks}

    # -------------------------------------------------------------------------
    # Prompt builders (private)
    # -------------------------------------------------------------------------

    def _build_bug_fix_prompt(self, task: dict, output_text: str,
                               source_files_section: str = "") -> str:
        inp = task["input"]
        gt  = task["ground_truth"]
        source_block = (
            f"\n## Source Files (at snapshot commit)\n\n{source_files_section}\n"
            if source_files_section else ""
        )
        return (
            f"You are scoring an agent's response to a software bug fix task.\n\n"
            f"## Bug Report\n"
            f"Issue #{inp.get('issue_number','?')}: {inp.get('issue_title','')}\n\n"
            f"{inp.get('issue_body','').strip()}\n\n"
            f"## Ground Truth Fix (actual commit diff)\n"
            f"```diff\n{gt.get('diff','').strip()[:4000]}\n```\n"
            f"{source_block}"
            f"## Agent's Response\n{output_text.strip()}\n\n---\n\n"
            f"{_RUBRIC_BUG_FIX}\n\n"
            f'Respond with exactly:\n'
            f'{{"correctness_breakdown": [C1,C2,C3,C4,C5], "correctness": <sum 0-10>, '
            f'"convention_breakdown": [A1,A2,A3,A4,A5], "convention_adherence": <sum 0-10>, '
            f'"api_correctness_breakdown": [P1,P2,P3,P4,P5], "api_correctness": <sum 0-10>, '
            f'"relevance_breakdown": [R1,R2,R3,R4,R5], "relevance": <sum 0-10>, '
            f'"rationale": "<3-5 sentences>"}}'
        )

    def _build_feature_impl_prompt(self, task: dict, output_text: str,
                                    source_files_section: str = "") -> str:
        inp = task["input"]
        gt  = task["ground_truth"]
        source_block = (
            f"\n## Source Files (at snapshot commit)\n\n{source_files_section}\n"
            if source_files_section else ""
        )
        merged_line = f"Merged: {gt.get('merged_at','')}\n" if gt.get("merged_at") else ""
        return (
            f"You are scoring an agent's response to a feature implementation task.\n\n"
            f"## Feature Request\n"
            f"PR #{inp.get('pr_number','?')}: {inp.get('pr_title','')}\n"
            f"{merged_line}\n"
            f"{inp.get('pr_body','').strip()}\n\n"
            f"## Ground Truth Implementation (merged PR diff)\n"
            f"```diff\n{gt.get('diff','').strip()[:4000]}\n```\n"
            f"{source_block}"
            f"## Agent's Response\n{output_text.strip()}\n\n---\n\n"
            f"{_RUBRIC_FEATURE_IMPL}\n\n"
            f'Respond with exactly:\n'
            f'{{"correctness_breakdown": [C1,C2,C3,C4,C5], "correctness": <sum 0-10>, '
            f'"convention_breakdown": [A1,A2,A3,A4,A5], "convention_adherence": <sum 0-10>, '
            f'"api_correctness_breakdown": [P1,P2,P3,P4,P5], "api_correctness": <sum 0-10>, '
            f'"relevance_breakdown": [R1,R2,R3,R4,R5], "relevance": <sum 0-10>, '
            f'"rationale": "<3-5 sentences>"}}'
        )

    def _build_code_review_prompt(self, task: dict, output_text: str) -> str:
        inp = task["input"]
        gt  = task["ground_truth"]
        return (
            f"You are scoring an agent's code review of a pull request.\n\n"
            f"Important context: this PR was subsequently reverted, meaning it introduced "
            f"a real problem. A high-quality review should identify what was wrong.\n\n"
            f"## Pull Request\n"
            f"PR #{inp.get('pr_number','?')}: {inp.get('pr_title','')}\n\n"
            f"{inp.get('pr_body','').strip()}\n\n"
            f"## Diff\n```diff\n{inp.get('diff','').strip()[:4000]}\n```\n\n"
            f"## Ground Truth Context (why it was reverted)\n"
            f"{gt.get('known_issues','').strip()}\n\n"
            f"## Agent's Review\n{output_text.strip()}\n\n---\n\n"
            f"{_RUBRIC_CODE_REVIEW}\n\n"
            f'Respond with exactly:\n'
            f'{{"correctness_breakdown": [C1,C2,C3,C4,C5], "correctness": <sum 0-10>, '
            f'"relevance_breakdown": [R1,R2,R3,R4,R5], "relevance": <sum 0-10>, '
            f'"rationale": "<3-5 sentences>"}}'
        )

    def _build_prompt(self, task: dict, output_text: str,
                      source_files_section: str = "") -> str:
        task_type = task["task_type"]
        if task_type == "bug_fix":
            return self._build_bug_fix_prompt(task, output_text, source_files_section)
        if task_type == "feature_impl":
            return self._build_feature_impl_prompt(task, output_text, source_files_section)
        if task_type == "code_review":
            return self._build_code_review_prompt(task, output_text)
        raise ValueError(f"Unknown task_type: {task_type!r}")

    # -------------------------------------------------------------------------
    # Source file loading (private)
    # -------------------------------------------------------------------------

    def _load_source_files(self, repo_name: str, snapshot_commit: str,
                            relevant_files: list[str],
                            max_chars: int = 3000) -> str:
        repo_path = self._repo_dir / repo_name / "control"
        if not repo_path.exists() or not relevant_files:
            return ""
        sections = []
        for filepath in relevant_files[:5]:
            result = subprocess.run(
                ["git", "show", f"{snapshot_commit}:{filepath}"],
                cwd=str(repo_path), capture_output=True, text=True,
            )
            if result.returncode != 0:
                continue
            content = result.stdout[:max_chars]
            ext = filepath.rsplit(".", 1)[-1] if "." in filepath else ""
            sections.append(f"### `{filepath}`\n```{ext}\n{content}\n```")
        return "\n\n".join(sections)

    # -------------------------------------------------------------------------
    # Scoring helpers (private)
    # -------------------------------------------------------------------------

    def _parse_score(self, raw: dict, key: str,
                     allow_null: bool = False) -> float | None:
        val = raw.get(key)
        if val is None:
            return None if allow_null else 5.5
        return max(0.0, min(10.0, float(val)))

    def _call_judge_api(self, prompt: str) -> dict:
        response = self._client.chat.completions.create(
            model=self._judge_model,
            response_format={"type": "json_object"},
            temperature=0,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an expert software engineering evaluator. "
                        "Return only valid JSON with the exact keys requested."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
        )
        return json.loads(response.choices[0].message.content)

    def _already_scored(self, run_id: int) -> bool:
        row = self._db.conn.execute(
            "SELECT 1 FROM scores WHERE run_id = ?", (run_id,)
        ).fetchone()
        return row is not None

    def _get_unscored_runs(self) -> list:
        return self._db.conn.execute("""
            SELECT r.run_id, r.task_id, r.arm, r.output_text
            FROM runs r
            JOIN tasks t ON r.task_id = t.task_id
            WHERE t.repo_name = ?
              AND r.status = 'completed'
              AND r.output_text IS NOT NULL
              AND NOT EXISTS (SELECT 1 FROM scores s WHERE s.run_id = r.run_id)
        """, (self._repo,)).fetchall()

    def _store_score(self, run_id: int, raw: dict, task_type: str):
        correctness     = self._parse_score(raw, "correctness")
        adherence       = self._parse_score(raw, "convention_adherence",
                                             allow_null=(task_type == "code_review"))
        relevance       = self._parse_score(raw, "relevance")
        api_correctness = self._parse_score(raw, "api_correctness", allow_null=True)
        rationale       = raw.get("rationale", "")
        self._db.insert_score(
            run_id, self._judge_model,
            correctness, adherence, relevance, rationale,
            api_correctness=api_correctness,
        )
        return correctness, adherence, relevance, api_correctness

    def _latest_batch_meta(self) -> tuple[dict, Path]:
        metas = sorted(self._BATCHES_DIR.glob(f"{self._repo}_*.meta.json"))
        if not metas:
            raise FileNotFoundError(
                f"No batch metadata found for repo '{self._repo}' in {self._BATCHES_DIR}"
            )
        meta_path = metas[-1]
        return json.loads(meta_path.read_text()), meta_path

    def _source_files_section_for_run(self, task: dict) -> str:
        if task["task_type"] not in ("feature_impl", "bug_fix"):
            return ""
        snapshot = task.get("snapshot_commit", "HEAD")
        relevant = json.loads(task.get("relevant_files_json", "[]"))
        repo_name = task.get("repo_name") or task.get("repo", "").split("/")[-1]
        return self._load_source_files(repo_name, snapshot, relevant)

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def submit(self, rescore: bool = False) -> None:
        """Build prompts and submit an OpenAI Batch API job (async, up to 24 h)."""
        if rescore:
            log.info("--rescore: deleting existing scores for repo '%s'", self._repo)
            self._db.delete_scores_for_repo(self._repo)

        rows = self._get_unscored_runs()
        if not rows:
            log.info("no un-scored completed runs found for repo '%s'", self._repo)
            return

        log.info("building prompts for %d runs", len(rows))
        self._BATCHES_DIR.mkdir(parents=True, exist_ok=True)

        ts   = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        stem = f"{self._repo}_{ts}"
        jsonl_path = self._BATCHES_DIR / f"{stem}.jsonl"
        meta_path  = self._BATCHES_DIR / f"{stem}.meta.json"

        requests, skipped = [], []
        for row in rows:
            run_id, task_id, arm, output_text = row
            task = self._tasks_by_id.get(task_id)
            if not task:
                log.warning("task %s not in tasks file — skipping run_id=%d", task_id, run_id)
                skipped.append(run_id)
                continue

            source_section = self._source_files_section_for_run(task)
            prompt = self._build_prompt(task, output_text or "", source_section)

            requests.append({
                "custom_id": f"run-{run_id}",
                "method":    "POST",
                "url":       "/v1/chat/completions",
                "body": {
                    "model":           self._judge_model,
                    "response_format": {"type": "json_object"},
                    "temperature":     0,
                    "messages": [
                        {
                            "role":    "system",
                            "content": (
                                "You are an expert software engineering evaluator. "
                                "Return only valid JSON with the exact keys requested."
                            ),
                        },
                        {"role": "user", "content": prompt},
                    ],
                },
            })

        if not requests:
            log.warning("no valid requests to submit (skipped=%d)", len(skipped))
            return

        jsonl_path.write_text("\n".join(json.dumps(r) for r in requests) + "\n")
        log.info("wrote %d requests to %s", len(requests), jsonl_path)

        log.info("uploading batch file to OpenAI...")
        with open(jsonl_path, "rb") as f:
            file_obj = self._client.files.create(file=f, purpose="batch")
        log.info("file uploaded: %s", file_obj.id)

        batch = self._client.batches.create(
            input_file_id=file_obj.id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
        )
        log.info("batch created: %s  status=%s", batch.id, batch.status)

        meta_path.write_text(json.dumps({
            "batch_id":     batch.id,
            "file_id":      file_obj.id,
            "repo":         self._repo,
            "run_ids":      [r["custom_id"].split("-")[1] for r in requests],
            "submitted_at": ts,
            "jsonl_path":   str(jsonl_path),
        }, indent=2))
        log.info("metadata saved: %s", meta_path)
        log.info(
            "check status: python scripts/judge.py --repo %s --status",
            self._repo,
        )

    def status(self) -> None:
        """Print the status of the latest batch job for this repo."""
        meta_data, _ = self._latest_batch_meta()
        batch = self._client.batches.retrieve(meta_data["batch_id"])
        log.info(
            "batch_id=%s  status=%s  completed=%s/%s  failed=%s",
            batch.id, batch.status,
            batch.request_counts.completed,
            batch.request_counts.total,
            batch.request_counts.failed,
        )
        if batch.status == "completed":
            log.info(
                "ready to collect — run: python scripts/judge.py --repo %s --collect",
                self._repo,
            )

    def collect(self) -> None:
        """Download batch results and store scores in the DB."""
        meta_data, _ = self._latest_batch_meta()
        batch_id     = meta_data["batch_id"]

        while True:
            batch = self._client.batches.retrieve(batch_id)
            log.info(
                "batch status: %s  (%s/%s completed)",
                batch.status,
                batch.request_counts.completed,
                batch.request_counts.total,
            )
            if batch.status in ("completed", "failed", "expired", "cancelled"):
                break
            log.info("waiting 30s...")
            time.sleep(30)

        if batch.status != "completed":
            log.error("batch ended with status '%s' — cannot collect", batch.status)
            return

        content = self._client.files.content(batch.output_file_id)
        lines   = content.text.strip().splitlines()
        log.info("downloaded %d result lines", len(lines))

        ts          = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        results_out = self._BATCHES_DIR / f"{self._repo}_{ts}.results.json"

        scored, failed = 0, 0
        all_results    = []

        for line in lines:
            obj       = json.loads(line)
            custom_id = obj.get("custom_id", "")
            run_id    = int(custom_id.split("-")[1]) if "-" in custom_id else None

            if obj.get("error"):
                log.warning("run_id=%s  error=%s", run_id, obj["error"])
                failed += 1
                continue

            try:
                raw_content = obj["response"]["body"]["choices"][0]["message"]["content"]
                raw         = json.loads(raw_content)
            except Exception as e:
                log.warning("run_id=%s  parse error: %s", run_id, e)
                failed += 1
                continue

            if self._already_scored(run_id):
                log.info("run_id=%d already scored — skipping", run_id)
                continue

            task = self._tasks_by_id.get(
                self._db.conn.execute(
                    "SELECT task_id FROM runs WHERE run_id = ?", (run_id,)
                ).fetchone()["task_id"]
            )
            task_type = task["task_type"] if task else "unknown"

            correctness, adherence, relevance, api_correctness = \
                self._store_score(run_id, raw, task_type)

            log.info(
                "run_id=%d  correctness=%.1f  adherence=%s  relevance=%.1f  api_correctness=%s",
                run_id, correctness,
                f"{adherence:.1f}" if adherence is not None else "N/A",
                relevance,
                f"{api_correctness:.1f}" if api_correctness is not None else "N/A",
            )
            scored += 1
            all_results.append({
                "run_id":               run_id,
                "correctness":          correctness,
                "convention_adherence": adherence,
                "relevance":            relevance,
                "api_correctness":      api_correctness,
                "rationale":            raw.get("rationale", ""),
            })

        results_out.write_text(json.dumps(all_results, indent=2))
        log.info("results saved: %s", results_out)
        log.info("done — scored=%d  failed=%d", scored, failed)

    def sync(self, rescore: bool = False) -> None:
        """Score all un-scored completed runs immediately via sequential API calls."""
        if rescore:
            log.info("--rescore: deleting existing scores for repo '%s'", self._repo)
            self._db.delete_scores_for_repo(self._repo)

        rows = self._get_unscored_runs()
        if not rows:
            log.info("no un-scored completed runs found for repo '%s'", self._repo)
            return

        log.info("scoring %d runs synchronously", len(rows))
        self._BATCHES_DIR.mkdir(parents=True, exist_ok=True)

        ts          = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        results_out = self._BATCHES_DIR / f"{self._repo}_{ts}.sync_results.json"

        scored, failed = 0, 0
        all_results    = []

        for row in rows:
            run_id, task_id, arm, output_text = row
            task = self._tasks_by_id.get(task_id)
            if not task:
                log.warning("task %s not found — skipping run_id=%d", task_id, run_id)
                failed += 1
                continue

            source_section = self._source_files_section_for_run(task)
            prompt = self._build_prompt(task, output_text or "", source_section)

            try:
                raw = self._call_judge_api(prompt)
            except Exception as e:
                log.error("run_id=%d  judge call failed: %s", run_id, e)
                failed += 1
                continue

            correctness, adherence, relevance, api_correctness = \
                self._store_score(run_id, raw, task["task_type"])

            log.info(
                "run_id=%d  %s/%s  correctness=%.1f  adherence=%s  "
                "relevance=%.1f  api_correctness=%s",
                run_id, task_id, arm, correctness,
                f"{adherence:.1f}" if adherence is not None else "N/A",
                relevance,
                f"{api_correctness:.1f}" if api_correctness is not None else "N/A",
            )
            scored += 1
            all_results.append({
                "run_id":               run_id,
                "task_id":              task_id,
                "arm":                  arm,
                "correctness":          correctness,
                "convention_adherence": adherence,
                "relevance":            relevance,
                "api_correctness":      api_correctness,
                "rationale":            raw.get("rationale", ""),
            })

        results_out.write_text(json.dumps(all_results, indent=2))
        log.info("results saved: %s", results_out)
        log.info("done — scored=%d  failed=%d", scored, failed)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Judge completed runs for a repo (OpenAI Batch API or sync)."
    )
    parser.add_argument("--repo",     required=True,
                        help="Repo name (e.g. fmt)")
    parser.add_argument("--repo-dir", default=os.environ.get("REPO_DIR", "/mnt/repos"),
                        help="Root dir for clones (default: $REPO_DIR or /mnt/repos)")
    parser.add_argument("--model",    default=Judge.JUDGE_MODEL,
                        help=f"Judge model (default: {Judge.JUDGE_MODEL})")
    parser.add_argument("--rescore",  action="store_true",
                        help="Delete existing scores before scoring (use with --submit or --sync)")

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--submit",  action="store_true",
                       help="Build prompts and submit OpenAI Batch job (async, up to 24h)")
    group.add_argument("--sync",    action="store_true",
                       help="Score immediately with sequential API calls")
    group.add_argument("--status",  action="store_true",
                       help="Check status of latest Batch job")
    group.add_argument("--collect", action="store_true",
                       help="Download Batch results and store in DB")

    args   = parser.parse_args()
    judge  = Judge(repo=args.repo, repo_dir=args.repo_dir, judge_model=args.model)

    if args.submit:
        judge.submit(rescore=args.rescore)
    elif args.sync:
        judge.sync(rescore=args.rescore)
    elif args.status:
        judge.status()
    elif args.collect:
        judge.collect()


if __name__ == "__main__":
    main()
