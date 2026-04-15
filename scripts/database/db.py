"""
db.py
=====

SQLite3 database layer for the belief validation experiment.

Schema
------
  repos   — registry of test repositories
  tasks   — extracted evaluation tasks (one row per task)
  runs    — one agent run per (task, arm, agent) triple
  scores  — judge scores for each run

Usage
-----
  from database.db import get_db

  database = get_db()               # opens/creates results.db, runs schema + migrations
  database.insert_task({...})       # idempotent — skips if task_id exists
  database.insert_run(...)          # returns run_id
  database.insert_score(...)        # attaches scores to a run
  database.conn.execute("SELECT …") # raw access when needed

Database location
-----------------
  Resolved in order:
    1. db_path argument to get_db() / Database()
    2. $DB_PATH environment variable
    3. <project_root>/database/results.db
"""

import os
import json
import sqlite3
from pathlib import Path
from datetime import datetime, timezone

from database.schema import SCHEMA, MIGRATION_RUNS_REBUILD_DDL, MIGRATION_LATE_TABLES_DDL

# ---------------------------------------------------------------------------
# Module-level path constant (used by callers that need the raw path)
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DB_PATH = Path(os.environ.get("DB_PATH", _PROJECT_ROOT / "database" / "results.db"))


# ---------------------------------------------------------------------------
# Database class
# ---------------------------------------------------------------------------

class Database:
    """
    Single connection wrapper for the experiment SQLite database.

    Opened, schema-initialised, and migrated on construction.
    Thread-safe for concurrent reads; callers that issue concurrent writes
    should serialise with their own lock (WAL mode allows concurrent readers).
    """

    def __init__(self, db_path: str | Path | None = None):
        self._path = Path(db_path) if db_path else DB_PATH
        self._conn = self._connect()
        self._init_schema()
        self._migrate()

    # ------------------------------------------------------------------
    # Private — connection and schema
    # ------------------------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self._path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")   # safe for concurrent writers
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def _init_schema(self):
        self._conn.executescript(SCHEMA)
        self._conn.commit()

    def _migrate(self):
        """Forward migrations for existing databases. Safe to call on a fresh DB."""
        run_cols = {r[1] for r in self._conn.execute("PRAGMA table_info(runs)").fetchall()}
        for col, defn in [
            ("agent",               "TEXT NOT NULL DEFAULT 'claude'"),
            ("agent_diff",          "TEXT"),
            ("files_modified_json", "TEXT"),
            ("tool_calls_total",    "INTEGER"),
            ("belief_calls_total",  "INTEGER"),
        ]:
            if col not in run_cols:
                self._conn.execute(f"ALTER TABLE runs ADD COLUMN {col} {defn}")

        repo_cols = {r[1] for r in self._conn.execute("PRAGMA table_info(repos)").fetchall()}
        for col, defn in [
            ("test_cmd",    "TEXT"),
            ("test_runner", "TEXT"),
        ]:
            if col not in repo_cols:
                self._conn.execute(f"ALTER TABLE repos ADD COLUMN {col} {defn}")

        score_cols = {r[1] for r in self._conn.execute("PRAGMA table_info(scores)").fetchall()}
        if "api_correctness" not in score_cols:
            self._conn.execute("ALTER TABLE scores ADD COLUMN api_correctness REAL")

        # Fix UNIQUE constraint: old DBs had UNIQUE(task_id, arm);
        # the 2×2 design requires UNIQUE(task_id, arm, agent).
        runs_ddl = self._conn.execute(
            "SELECT sql FROM sqlite_master WHERE name='runs'"
        ).fetchone()
        if (runs_ddl
                and "UNIQUE(task_id, arm)" in runs_ddl[0]
                and "UNIQUE(task_id, arm, agent)" not in runs_ddl[0]):
            self._conn.executescript(MIGRATION_RUNS_REBUILD_DDL)

        # Create tables that predate the current schema version
        self._conn.executescript(MIGRATION_LATE_TABLES_DDL)
        self._conn.commit()

    @staticmethod
    def _now_iso() -> str:
        return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    # ------------------------------------------------------------------
    # Public — raw connection access
    # ------------------------------------------------------------------

    @property
    def conn(self) -> sqlite3.Connection:
        """Exposes the underlying connection for callers that need raw SQL."""
        return self._conn

    # ------------------------------------------------------------------
    # Public — repos
    # ------------------------------------------------------------------

    def upsert_repo(self, repo: dict):
        """Insert repo into registry; silently ignored if it already exists."""
        self._conn.execute("""
            INSERT OR IGNORE INTO repos (name, url, language, category)
            VALUES (:name, :url, :language, :category)
        """, {
            "name":     repo["name"],
            "url":      repo["url"],
            "language": repo["language"],
            "category": repo.get("category"),
        })
        self._conn.commit()

    def update_repo_status(self, name: str, status: str):
        self._conn.execute(
            "UPDATE repos SET status = ? WHERE name = ?", (status, name)
        )
        self._conn.commit()

    def upsert_repo_test_cmd(self, name: str, test_cmd: str, test_runner: str):
        self._conn.execute(
            "UPDATE repos SET test_cmd = ?, test_runner = ? WHERE name = ?",
            (test_cmd, test_runner, name)
        )
        self._conn.commit()

    # ------------------------------------------------------------------
    # Public — tasks
    # ------------------------------------------------------------------

    def insert_task(self, task: dict, repo_name: str | None = None):
        """Insert a task record; silently skipped if task_id already exists.

        repo_name: override the repo name stored in DB. If omitted, derived from
                   task['repo'] by taking the last path component. Pass explicitly
                   when the directory name differs from the GitHub repo name
                   (e.g., task['repo']='home-assistant/core' but dir='home-assistant').
        """
        _repo_name = repo_name or task["repo"].split("/")[-1]
        self._conn.execute("""
            INSERT OR IGNORE INTO tasks (
                task_id, repo_name, task_type, source,
                input_json, ground_truth_json, relevant_files_json,
                snapshot_commit, belief_cutoff_timestamp, reference_commit_timestamp
            ) VALUES (
                :task_id, :repo_name, :task_type, :source,
                :input_json, :ground_truth_json, :relevant_files_json,
                :snapshot_commit, :belief_cutoff_timestamp, :reference_commit_timestamp
            )
        """, {
            "task_id":                    task["id"],
            "repo_name":                  _repo_name,
            "task_type":                  task["task_type"],
            "source":                     task.get("source"),
            "input_json":                 json.dumps(task["input"]),
            "ground_truth_json":          json.dumps(task["ground_truth"]),
            "relevant_files_json":        json.dumps(task["relevant_files"]),
            "snapshot_commit":            task.get("snapshot_commit"),
            "belief_cutoff_timestamp":    task["belief_cutoff_timestamp"],
            "reference_commit_timestamp": task.get("reference_commit_timestamp"),
        })
        self._conn.commit()

    # ------------------------------------------------------------------
    # Public — runs
    # ------------------------------------------------------------------

    def insert_run(self, task_id: str, arm: str,
                   agent_model: str, agent: str = "claude") -> int:
        """Create a run record in 'running' state. Returns run_id.
        If a run already exists for (task_id, arm, agent), returns its existing id.
        """
        existing = self._conn.execute(
            "SELECT run_id FROM runs WHERE task_id = ? AND arm = ? AND agent = ?",
            (task_id, arm, agent)
        ).fetchone()
        if existing:
            return existing["run_id"]

        cur = self._conn.execute("""
            INSERT INTO runs (task_id, arm, agent, agent_model, status, started_at)
            VALUES (?, ?, ?, ?, 'running', ?)
        """, (task_id, arm, agent, agent_model, self._now_iso()))
        self._conn.commit()
        return cur.lastrowid

    def complete_run(self, run_id: int, output_text: str,
                     beliefs_used: list | None = None, tokens_used: int | None = None):
        self._conn.execute("""
            UPDATE runs
            SET status = 'completed',
                output_text  = ?,
                beliefs_used = ?,
                tokens_used  = ?,
                completed_at = ?
            WHERE run_id = ?
        """, (
            output_text,
            json.dumps(beliefs_used) if beliefs_used else None,
            tokens_used,
            self._now_iso(),
            run_id,
        ))
        self._conn.commit()

    def complete_run_agentic(self, run_id: int, output_text: str,
                             agent_diff: str, files_modified: list[str],
                             tool_calls_total: int | None, belief_calls_total: int,
                             beliefs_used: list | None = None,
                             tokens_used: int | None = None):
        """complete_run variant that also stores diff and agentic metadata."""
        self._conn.execute("""
            UPDATE runs
            SET status = 'completed',
                output_text          = ?,
                agent_diff           = ?,
                files_modified_json  = ?,
                tool_calls_total     = ?,
                belief_calls_total   = ?,
                beliefs_used         = ?,
                tokens_used          = ?,
                completed_at         = ?
            WHERE run_id = ?
        """, (
            output_text,
            agent_diff,
            json.dumps(files_modified),
            tool_calls_total,
            belief_calls_total,
            json.dumps(beliefs_used) if beliefs_used else None,
            tokens_used,
            self._now_iso(),
            run_id,
        ))
        self._conn.commit()

    def fail_run(self, run_id: int, error: str):
        self._conn.execute("""
            UPDATE runs SET status = 'failed', error = ?, completed_at = ?
            WHERE run_id = ?
        """, (error, self._now_iso(), run_id))
        self._conn.commit()

    # ------------------------------------------------------------------
    # Public — scores
    # ------------------------------------------------------------------

    def insert_score(self, run_id: int, judge_model: str,
                     correctness: float, convention_adherence: float | None,
                     relevance: float, rationale: str,
                     api_correctness: float | None = None):
        self._conn.execute("""
            INSERT INTO scores
                (run_id, judge_model, correctness, convention_adherence, relevance,
                 rationale, api_correctness)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (run_id, judge_model, correctness, convention_adherence, relevance,
              rationale, api_correctness))
        self._conn.commit()

    def delete_scores_for_repo(self, repo_name: str):
        """Delete all scores for runs belonging to a repo (used when re-scoring)."""
        self._conn.execute("""
            DELETE FROM scores WHERE run_id IN (
                SELECT r.run_id FROM runs r
                JOIN tasks t ON t.task_id = r.task_id
                WHERE t.repo_name = ?
            )
        """, (repo_name,))
        self._conn.commit()

    # ------------------------------------------------------------------
    # Public — test results
    # ------------------------------------------------------------------

    def insert_test_result(self, run_id: int,
                           baseline_passed: int, baseline_failed: int,
                           baseline_total: int, post_passed: int | None,
                           post_failed: int | None, pass_rate_delta: float | None,
                           apply_status: str, test_runner: str):
        self._conn.execute("""
            INSERT INTO test_results
                (run_id, baseline_passed, baseline_failed, baseline_total,
                 post_passed, post_failed, pass_rate_delta, apply_status, test_runner)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (run_id, baseline_passed, baseline_failed, baseline_total,
              post_passed, post_failed, pass_rate_delta, apply_status, test_runner))
        self._conn.commit()

    # ------------------------------------------------------------------
    # Public — belief tool calls
    # ------------------------------------------------------------------

    def insert_belief_tool_call(self, run_id: int, call_sequence: int,
                                commits_used: list[str],
                                beliefs_returned: str | None):
        self._conn.execute("""
            INSERT INTO belief_tool_calls
                (run_id, call_sequence, commits_used, beliefs_returned)
            VALUES (?, ?, ?, ?)
        """, (run_id, call_sequence, json.dumps(commits_used), beliefs_returned))
        self._conn.commit()

    # ------------------------------------------------------------------
    # Public — queries
    # ------------------------------------------------------------------

    def get_completed_runs_with_diff(self, repo_name: str) -> list[sqlite3.Row]:
        """Return completed runs that have an agent_diff and no test_results yet."""
        return self._conn.execute("""
            SELECT r.run_id, r.task_id, r.arm, r.agent, r.agent_diff,
                   t.snapshot_commit, t.repo_name
            FROM runs r
            JOIN tasks t ON t.task_id = r.task_id
            WHERE t.repo_name = ?
              AND r.status = 'completed'
              AND r.agent_diff IS NOT NULL
              AND r.agent_diff != ''
              AND NOT EXISTS (
                  SELECT 1 FROM test_results tr WHERE tr.run_id = r.run_id
              )
            ORDER BY r.run_id
        """, (repo_name,)).fetchall()

    def get_task_summary(self) -> list[sqlite3.Row]:
        return self._conn.execute("""
            SELECT
                t.repo_name,
                t.task_type,
                r.agent,
                COUNT(DISTINCT t.task_id) AS total_tasks,
                SUM(CASE WHEN r.arm = 'control'   AND r.status = 'completed' THEN 1 ELSE 0 END) AS control_done,
                SUM(CASE WHEN r.arm = 'treatment' AND r.status = 'completed' THEN 1 ELSE 0 END) AS treatment_done
            FROM tasks t
            LEFT JOIN runs r ON r.task_id = t.task_id
            GROUP BY t.repo_name, t.task_type, r.agent
            ORDER BY t.repo_name, t.task_type, r.agent
        """).fetchall()

    def get_score_summary(self) -> list[sqlite3.Row]:
        return self._conn.execute("""
            SELECT
                t.task_type,
                r.arm,
                r.agent,
                ROUND(AVG(s.correctness), 3)           AS avg_correctness,
                ROUND(AVG(s.convention_adherence), 3)  AS avg_adherence,
                ROUND(AVG(s.relevance), 3)             AS avg_relevance,
                COUNT(*) AS n
            FROM scores s
            JOIN runs r  ON r.run_id  = s.run_id
            JOIN tasks t ON t.task_id = r.task_id
            GROUP BY t.task_type, r.arm, r.agent
            ORDER BY t.task_type, r.agent, r.arm
        """).fetchall()


# ---------------------------------------------------------------------------
# Module-level factory
# ---------------------------------------------------------------------------

def get_db(path: str | Path | None = None) -> Database:
    """Open (or create) the experiment database and return a Database instance."""
    return Database(db_path=path)


# ---------------------------------------------------------------------------
# CLI: init or inspect
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    database = get_db()
    print(f"Database ready: {database._path}")

    if "--summary" in sys.argv:
        print("\n--- Task summary ---")
        for row in database.get_task_summary():
            print(dict(row))
        print("\n--- Score summary ---")
        for row in database.get_score_summary():
            print(dict(row))
