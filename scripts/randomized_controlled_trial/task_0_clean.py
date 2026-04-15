"""
task_0_clean.py
===============

Clears experiment results so you can start a fresh run.

What it clears (each controlled by a flag):

  --runs      Delete the runs/ directory (per-run log files: response.md,
              diff.patch, beliefs.md, meta.json) and clear the runs and
              scores tables in results.db.

  --logs      Delete all Python logger files in logs/ (task_N_*.log files).

  --db        Clear only the runs and scores DB tables (keep runs/ files).

  --all       Equivalent to --runs --logs (full reset).

  --repo NAME Scope --runs and --db to a single repo (leaves others intact).

Usage:
  python scripts/task_0_clean.py --all
  python scripts/task_0_clean.py --runs
  python scripts/task_0_clean.py --runs --repo fmt
  python scripts/task_0_clean.py --logs

Always asks for confirmation before deleting anything.
"""

import argparse
import shutil
import sys
from pathlib import Path

_SCRIPTS_DIR  = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPTS_DIR.parent
sys.path.insert(0, str(_SCRIPTS_DIR))

import database.db as db
from utils.logger import get_logger

log = get_logger(__name__)

RUNS_DIR = _PROJECT_ROOT / "runs"
LOGS_DIR = _PROJECT_ROOT / "logs"
DB_PATH  = db.DB_PATH


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def confirm(message: str) -> bool:
    answer = input(f"\n{message} [y/N] ").strip().lower()
    return answer == "y"


def clean_db_runs(repo: str | None = None):
    """Delete rows from runs and scores tables, optionally scoped to one repo."""
    database = db.get_db()
    conn = database.conn

    if repo:
        task_ids = [
            r[0] for r in conn.execute(
                "SELECT task_id FROM tasks WHERE repo_name = ?", (repo,)
            ).fetchall()
        ]
        if not task_ids:
            log.info("no tasks found for repo '%s' in DB — nothing to clear", repo)
            return

        placeholders = ",".join("?" * len(task_ids))
        run_ids = [
            r[0] for r in conn.execute(
                f"SELECT run_id FROM runs WHERE task_id IN ({placeholders})", task_ids
            ).fetchall()
        ]
        if run_ids:
            score_ph = ",".join("?" * len(run_ids))
            conn.execute(f"DELETE FROM scores WHERE run_id IN ({score_ph})", run_ids)
            conn.execute(f"DELETE FROM runs WHERE run_id IN ({score_ph})", run_ids)
        conn.commit()
        log.info("cleared %d run(s) and their scores for repo '%s'", len(run_ids), repo)
    else:
        conn.execute("DELETE FROM scores")
        conn.execute("DELETE FROM runs")
        conn.commit()
        counts = conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0]
        log.info("cleared all runs and scores from DB (remaining rows: %d)", counts)


def clean_runs_dir(repo: str | None = None):
    """Delete the runs/ directory or a repo subdirectory within it."""
    target = RUNS_DIR / repo if repo else RUNS_DIR
    if not target.exists():
        log.info("runs directory not found: %s — nothing to delete", target)
        return
    shutil.rmtree(target)
    log.info("deleted: %s", target)


def clean_logs_dir():
    """Delete all .log files in logs/."""
    if not LOGS_DIR.exists():
        log.info("logs directory not found — nothing to delete")
        return
    log_files = list(LOGS_DIR.glob("*.log"))
    for f in log_files:
        f.unlink()
    log.info("deleted %d log file(s) from %s", len(log_files), LOGS_DIR)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Clean experiment results for a fresh run."
    )
    parser.add_argument("--all",  action="store_true",
                        help="Full reset: delete runs/ logs, clear DB runs/scores, delete logs/")
    parser.add_argument("--runs", action="store_true",
                        help="Delete runs/ log files and clear DB runs/scores tables")
    parser.add_argument("--db",   action="store_true",
                        help="Clear only DB runs/scores tables (keep runs/ files)")
    parser.add_argument("--logs", action="store_true",
                        help="Delete Python logger files in logs/")
    parser.add_argument("--repo", metavar="NAME",
                        help="Scope --runs and --db to this repo only (e.g. fmt)")
    parser.add_argument("--yes",  action="store_true",
                        help="Skip confirmation prompt")
    args = parser.parse_args()

    if args.all:
        args.runs = True
        args.logs = True

    if not any([args.runs, args.db, args.logs]):
        parser.print_help()
        sys.exit(0)

    # Build summary of what will be deleted
    actions = []
    if args.runs:
        target = f"runs/{args.repo}" if args.repo else "runs/"
        actions.append(f"  • Delete {target} (log files)")
        actions.append(f"  • Clear DB: runs + scores tables"
                       + (f" for repo '{args.repo}'" if args.repo else " (all)"))
    if args.db and not args.runs:
        actions.append(f"  • Clear DB: runs + scores tables"
                       + (f" for repo '{args.repo}'" if args.repo else " (all)"))
    if args.logs:
        actions.append(f"  • Delete all .log files in logs/")

    print("\nThis will:")
    for a in actions:
        print(a)

    if not args.yes and not confirm("Proceed?"):
        print("Aborted.")
        sys.exit(0)

    if args.runs:
        clean_runs_dir(args.repo)
        clean_db_runs(args.repo)
    elif args.db:
        clean_db_runs(args.repo)

    if args.logs:
        clean_logs_dir()

    print("\nDone. Ready for a fresh run.")


if __name__ == "__main__":
    main()
