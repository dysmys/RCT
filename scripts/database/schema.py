"""
schema.py
=========

All SQLite DDL for the belief validation experiment database.

Tables
------
  repos            — registry of test repositories
  tasks            — extracted evaluation tasks (one row per task)
  runs             — one agent run per (task, arm, agent)
  scores           — GPT-4o judge scores per run
  test_results     — test suite pass/fail counts per run
  belief_tool_calls — per-call log for get_beliefs tool (treatment arm)
"""

# ---------------------------------------------------------------------------
# Base schema — applied on fresh databases via init_db()
# ---------------------------------------------------------------------------

SCHEMA = """
-- Repos under evaluation
CREATE TABLE IF NOT EXISTS repos (
    name        TEXT PRIMARY KEY,
    url         TEXT NOT NULL,
    language    TEXT NOT NULL,
    category    TEXT,
    status      TEXT NOT NULL DEFAULT 'pending',
                -- pending | tasks_extracted | setup_done | running | done | failed
    created_at  TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
    test_cmd    TEXT,
    test_runner TEXT
);

-- Evaluation tasks extracted per repo
CREATE TABLE IF NOT EXISTS tasks (
    task_id                    TEXT PRIMARY KEY,
    repo_name                  TEXT NOT NULL REFERENCES repos(name),
    task_type                  TEXT NOT NULL,   -- bug_fix | feature_impl | code_review
    source                     TEXT,            -- github_bug_label | git_fix_keyword_fallback | ...
    input_json                 TEXT NOT NULL,   -- JSON: issue/PR title, body, diff
    ground_truth_json          TEXT NOT NULL,   -- JSON: fix commit, merged diff, known issues
    relevant_files_json        TEXT NOT NULL,   -- JSON array of file paths
    snapshot_commit            TEXT NOT NULL,   -- git commit to checkout before running agent
    belief_cutoff_timestamp    TEXT NOT NULL,   -- ISO8601: no beliefs after this time
    reference_commit_timestamp TEXT,            -- ISO8601: when the fix/merge landed
    created_at                 TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
);

-- One agent run per (task, arm, agent)
CREATE TABLE IF NOT EXISTS runs (
    run_id               INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id              TEXT NOT NULL REFERENCES tasks(task_id),
    arm                  TEXT NOT NULL,
    agent                TEXT NOT NULL DEFAULT 'claude',
    agent_model          TEXT NOT NULL,
    status               TEXT NOT NULL DEFAULT 'pending',
                         -- pending | running | completed | failed
    output_text          TEXT,
    beliefs_used         TEXT,
    tokens_used          INTEGER,
    error                TEXT,
    agent_diff           TEXT,
    files_modified_json  TEXT,
    tool_calls_total     INTEGER,
    belief_calls_total   INTEGER,
    started_at           TEXT,
    completed_at         TEXT,
    UNIQUE(task_id, arm, agent)
);

-- Judge scores per run
CREATE TABLE IF NOT EXISTS scores (
    score_id              INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id                INTEGER NOT NULL REFERENCES runs(run_id),
    judge_model           TEXT NOT NULL,
    correctness           REAL,     -- 0-10
    convention_adherence  REAL,     -- 0-10 (null for code_review tasks)
    relevance             REAL,     -- 0-10
    api_correctness       REAL,     -- 0-10 (null for code_review tasks)
    rationale             TEXT,     -- judge's chain-of-thought
    scored_at             TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
);

-- Indexes for common query patterns
CREATE INDEX IF NOT EXISTS idx_tasks_repo      ON tasks(repo_name);
CREATE INDEX IF NOT EXISTS idx_tasks_type      ON tasks(task_type);
CREATE INDEX IF NOT EXISTS idx_runs_task       ON runs(task_id);
CREATE INDEX IF NOT EXISTS idx_runs_arm        ON runs(arm);
CREATE INDEX IF NOT EXISTS idx_runs_status     ON runs(status);
CREATE INDEX IF NOT EXISTS idx_scores_run      ON scores(run_id);

-- Execution scoring: test suite results per run
CREATE TABLE IF NOT EXISTS test_results (
    result_id        INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id           INTEGER NOT NULL REFERENCES runs(run_id),
    baseline_passed  INTEGER NOT NULL,
    baseline_failed  INTEGER NOT NULL,
    baseline_total   INTEGER NOT NULL,
    post_passed      INTEGER,
    post_failed      INTEGER,
    pass_rate_delta  REAL,
    apply_status     TEXT NOT NULL DEFAULT 'pending',
                     -- pending | clean | apply_failed | no_changes | suite_error
    test_runner      TEXT,
    ran_at           TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
);
CREATE INDEX IF NOT EXISTS idx_test_results_run ON test_results(run_id);

-- Per-call log for get_beliefs tool (treatment arm)
CREATE TABLE IF NOT EXISTS belief_tool_calls (
    call_id          INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id           INTEGER NOT NULL REFERENCES runs(run_id),
    call_sequence    INTEGER NOT NULL,
    commits_used     TEXT NOT NULL,
    beliefs_returned TEXT,
    called_at        TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
);
CREATE INDEX IF NOT EXISTS idx_belief_calls_run ON belief_tool_calls(run_id);
"""

# ---------------------------------------------------------------------------
# Migration DDL — applied by migrate_db() on existing databases
# ---------------------------------------------------------------------------

# Recreates the runs table with the correct UNIQUE(task_id, arm, agent) constraint,
# replacing the old UNIQUE(task_id, arm) from the original single-agent design.
MIGRATION_RUNS_REBUILD_DDL = """
    PRAGMA foreign_keys=OFF;
    CREATE TABLE runs_new AS SELECT * FROM runs WHERE 0;
    INSERT INTO runs_new SELECT * FROM runs;
    DROP TABLE runs;
    CREATE TABLE runs (
        run_id               INTEGER PRIMARY KEY AUTOINCREMENT,
        task_id              TEXT NOT NULL REFERENCES tasks(task_id),
        arm                  TEXT NOT NULL,
        agent_model          TEXT NOT NULL,
        status               TEXT NOT NULL DEFAULT 'pending',
        output_text          TEXT,
        beliefs_used         TEXT,
        tokens_used          INTEGER,
        error                TEXT,
        started_at           TEXT,
        completed_at         TEXT,
        agent                TEXT NOT NULL DEFAULT 'claude',
        agent_diff           TEXT,
        files_modified_json  TEXT,
        tool_calls_total     INTEGER,
        belief_calls_total   INTEGER,
        UNIQUE(task_id, arm, agent)
    );
    INSERT INTO runs SELECT * FROM runs_new;
    DROP TABLE runs_new;
    CREATE INDEX IF NOT EXISTS idx_runs_task   ON runs(task_id);
    CREATE INDEX IF NOT EXISTS idx_runs_arm    ON runs(arm);
    CREATE INDEX IF NOT EXISTS idx_runs_status ON runs(status);
    CREATE INDEX IF NOT EXISTS idx_runs_agent  ON runs(agent);
    PRAGMA foreign_keys=ON;
"""

# Adds test_results and belief_tool_calls for databases that predate those tables.
MIGRATION_LATE_TABLES_DDL = """
    CREATE TABLE IF NOT EXISTS test_results (
        result_id        INTEGER PRIMARY KEY AUTOINCREMENT,
        run_id           INTEGER NOT NULL REFERENCES runs(run_id),
        baseline_passed  INTEGER NOT NULL,
        baseline_failed  INTEGER NOT NULL,
        baseline_total   INTEGER NOT NULL,
        post_passed      INTEGER,
        post_failed      INTEGER,
        pass_rate_delta  REAL,
        apply_status     TEXT NOT NULL DEFAULT 'pending',
        test_runner      TEXT,
        ran_at           TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
    );
    CREATE INDEX IF NOT EXISTS idx_test_results_run ON test_results(run_id);
    CREATE TABLE IF NOT EXISTS belief_tool_calls (
        call_id          INTEGER PRIMARY KEY AUTOINCREMENT,
        run_id           INTEGER NOT NULL REFERENCES runs(run_id),
        call_sequence    INTEGER NOT NULL,
        commits_used     TEXT NOT NULL,
        beliefs_returned TEXT,
        called_at        TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
    );
    CREATE INDEX IF NOT EXISTS idx_belief_calls_run ON belief_tool_calls(run_id);
"""
