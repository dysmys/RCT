#!/bin/bash
# run_all_repos.sh
# ----------------
# Runs experiments for all repos that have task files.
# Processes repos in parallel batches to keep total concurrent
# agent sessions under control on the VM.
#
# Usage: bash scripts/run_all_repos.sh [--agent both|claude|codex]
#
# Strategy:
#   BATCH_SIZE repos run in parallel, each with MAX_WORKERS sessions.
#   Total concurrent sessions = BATCH_SIZE * MAX_WORKERS.
#   With BATCH_SIZE=6 and MAX_WORKERS=4 → 24 concurrent sessions (safe on e2-standard-16).

set -euo pipefail

AGENT="${1:-both}"
MAX_WORKERS=4       # per-repo concurrency
BATCH_SIZE=6        # repos running in parallel at once
TIMEOUT=900         # seconds per agent session
REPO_DIR=/mnt/repos
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
VENV=/opt/experiment_venv

set -a && source /opt/experiment/.env && set +a
export PATH="$HOME/.local/bin:$PATH"
export PYTHONPATH="$SCRIPT_DIR"

# Find all repos that have a task file
TASK_DIR="$PROJECT_DIR/dataset/tasks"
REPOS=()
for f in "$TASK_DIR"/*.json; do
    name=$(basename "$f" .json)
    [[ "$name" == "all_tasks" ]] && continue
    count=$(python3 -c "import json; d=json.load(open('$f')); print(len(d) if isinstance(d,list) else 0)" 2>/dev/null || echo 0)
    if [[ "$count" -gt 0 ]]; then
        REPOS+=("$name")
    fi
done

echo "[run_all] Found ${#REPOS[@]} repos with tasks"
echo "[run_all] agent=$AGENT  max_workers=$MAX_WORKERS  batch_size=$BATCH_SIZE  timeout=${TIMEOUT}s"

# Step 1: Clone all repos that aren't already cloned
echo "[run_all] === Setting up repos ==="
setup_pids=()
for repo in "${REPOS[@]}"; do
    if [[ ! -d "$REPO_DIR/$repo/control/.git" ]] || [[ ! -d "$REPO_DIR/$repo/treatment/.git" ]]; then
        "$VENV/bin/python" "$SCRIPT_DIR/task_2_setup_repos.py" \
            --repo "$repo" --repo-dir "$REPO_DIR" \
            > "/tmp/setup_${repo}.log" 2>&1 &
        setup_pids+=($!)
        echo "  cloning $repo (PID $!)"
    else
        echo "  $repo already cloned — skipping"
    fi
done
for pid in "${setup_pids[@]}"; do
    wait "$pid" && echo "  setup PID $pid done" || echo "  setup PID $pid FAILED"
done
echo "[run_all] === All repos ready ==="

# Step 2: Run experiments in batches
echo "[run_all] === Running experiments ==="
total=${#REPOS[@]}
i=0

while [[ $i -lt $total ]]; do
    batch=("${REPOS[@]:$i:$BATCH_SIZE}")
    echo "[run_all] Batch $((i/BATCH_SIZE + 1)): ${batch[*]}"

    batch_pids=()
    for repo in "${batch[@]}"; do
        LOG="/tmp/task3_${repo}.log"
        "$VENV/bin/python" "$SCRIPT_DIR/task_3_run_experiment.py" \
            --repo "$repo" \
            --agent "$AGENT" \
            --max-workers "$MAX_WORKERS" \
            --timeout "$TIMEOUT" \
            --repo-dir "$REPO_DIR" \
            > "$LOG" 2>&1 &
        batch_pids+=($!)
        echo "  started $repo (PID $!)"
        sleep 8   # stagger to avoid hammering HF endpoint
    done

    # Wait for entire batch before starting next (|| true so a non-zero exit doesn't abort the pipeline)
    for pid in "${batch_pids[@]}"; do
        wait "$pid" || true
    done
    echo "[run_all] Batch done"
    i=$((i + BATCH_SIZE))
done

echo "[run_all] === All experiments complete ==="

# Quick DB summary
"$VENV/bin/python" - << 'PYEOF'
import sys; sys.path.insert(0, '/opt/experiment/scripts')
import database.db as db
database = db.get_db()
rows = database.conn.execute("""
    SELECT agent, arm, status, COUNT(*) AS n FROM runs
    GROUP BY agent, arm, status ORDER BY agent, arm, status
""").fetchall()
print("\n=== DB Summary ===")
for r in rows:
    print(f"  agent={r['agent']:<8}  arm={r['arm']:<12}  status={r['status']:<12}  n={r['n']}")
PYEOF
