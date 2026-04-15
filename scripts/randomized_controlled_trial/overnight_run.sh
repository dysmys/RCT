#!/bin/bash
# ---------------------------------------------------------------------------
# overnight_run.sh
# Full pipeline for 100-repo Claude-only overnight run.
# Usage: bash scripts/overnight_run.sh [--repo-dir /tmp/repos_overnight] 2>&1 | tee logs/overnight_run.log
# ---------------------------------------------------------------------------
set -euo pipefail

REPO_DIR="${1:-/tmp/repos_overnight}"
LOG_DIR="$(cd "$(dirname "$0")/.." && pwd)/logs"
START=$(date +%s)

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

log "=== OVERNIGHT RUN START ==="
log "repo-dir: $REPO_DIR"
log "log-dir:  $LOG_DIR"
mkdir -p "$REPO_DIR" "$LOG_DIR"

# All 100 repos from test.json
ALL_REPOS=$(python3 -c "
import json
repos = json.load(open('dataset/test.json'))['repositories']
print(' '.join(r['name'] for r in repos))
")

# Already-completed repos (have task files + both arms cloned)
PILOT_REPOS="django fmt home-assistant jax langchain mitmproxy twisted"
NEW_REPOS=$(python3 -c "
import json
pilot = set('django fmt home-assistant jax langchain mitmproxy twisted'.split())
repos = json.load(open('dataset/test.json'))['repositories']
print(' '.join(r['name'] for r in repos if r['name'] not in pilot))
")

log "Pilot repos (already run, skip setup): $PILOT_REPOS"
log "New repos to set up: $(echo $NEW_REPOS | wc -w)"

# ---------------------------------------------------------------------------
# STEP 1: Extract tasks for new repos (GitHub API)
# ---------------------------------------------------------------------------
log ""
log "=== STEP 1: Extract tasks (task_1) ==="
for repo in $NEW_REPOS; do
    log "  extracting tasks: $repo"
    python3 scripts/task_1_extract_tasks.py --repo "$repo" --limit 8 \
        2>&1 | tail -3 || log "  WARNING: task extraction failed for $repo"
done
log "STEP 1 done"

# ---------------------------------------------------------------------------
# STEP 2: Clone + setup repos (control + treatment arms)
# ---------------------------------------------------------------------------
log ""
log "=== STEP 2: Clone + setup repos (task_2) ==="
for repo in $NEW_REPOS; do
    if [ -d "$REPO_DIR/$repo/control" ] && [ -d "$REPO_DIR/$repo/treatment" ]; then
        log "  $repo already set up — skipping"
        continue
    fi
    log "  setting up: $repo"
    python3 scripts/task_2_setup_repos.py --repo "$repo" --repo-dir "$REPO_DIR" \
        2>&1 | tail -3 || log "  WARNING: setup failed for $repo"
done
log "STEP 2 done"

# ---------------------------------------------------------------------------
# STEP 3: Run experiment — Claude only, both arms, all repos
# ---------------------------------------------------------------------------
log ""
log "=== STEP 3: Run experiment (Claude-only, both arms) ==="

# Run pilot repos against new repo-dir (skip already-completed runs via DB)
for repo in $PILOT_REPOS; do
    # Pilot repos are in /tmp/repos_test — use that path
    if [ ! -d "/tmp/repos_test/$repo/control" ]; then
        log "  WARNING: pilot repo /tmp/repos_test/$repo not found, skipping"
        continue
    fi
    # Nothing to re-run for pilot — all done
    log "  $repo (pilot): already completed — skipping"
done

# Run new repos
for repo in $NEW_REPOS; do
    if [ ! -d "$REPO_DIR/$repo/control" ]; then
        log "  WARNING: $repo not set up, skipping experiment"
        continue
    fi
    task_count=$(python3 -c "
import json
from pathlib import Path
f = Path('dataset/tasks/${repo}.json')
print(len(json.loads(f.read_text())) if f.exists() else 0)
")
    if [ "$task_count" = "0" ]; then
        log "  $repo: no tasks extracted — skipping"
        continue
    fi
    log "  running: $repo ($task_count tasks)"
    python3 scripts/task_3_run_experiment.py \
        --repo "$repo" \
        --repo-dir "$REPO_DIR" \
        --agent both \
        --arm both \
        --max-workers 8 \
        --timeout 900 \
        2>&1 | tail -5
done
log "STEP 3 done"

# ---------------------------------------------------------------------------
# STEP 4: Judge all scored repos
# ---------------------------------------------------------------------------
log ""
log "=== STEP 4: Submit judge batches ==="
REPOS_TO_JUDGE=""
for repo in $NEW_REPOS; do
    completed=$(python3 -c "
import sqlite3
try:
    conn = sqlite3.connect('database/results.db')
    n = conn.execute(\"SELECT COUNT(*) FROM runs r JOIN tasks t ON r.task_id=t.task_id WHERE t.repo_name='${repo}' AND r.status='completed'\").fetchone()[0]
    print(n)
    conn.close()
except: print(0)
")
    if [ "$completed" -gt 0 ]; then
        log "  submitting judge batch: $repo ($completed completed runs)"
        python3 scripts/judge/judge_all.py --repo "$repo" --submit \
            2>&1 | tail -3 || log "  WARNING: judge submit failed for $repo"
        REPOS_TO_JUDGE="$REPOS_TO_JUDGE $repo"
    else
        log "  $repo: no completed runs — skipping judge"
    fi
done
log "STEP 4 done"

# ---------------------------------------------------------------------------
# STEP 5: Wait for judge batches to complete, then collect
# ---------------------------------------------------------------------------
log ""
log "=== STEP 5: Collect judge results ==="
if [ -n "$REPOS_TO_JUDGE" ]; then
    log "Waiting 300s for batch API to process..."
    sleep 300
    for attempt in 1 2 3 4 5 6 8 10; do
        all_done=true
        for repo in $REPOS_TO_JUDGE; do
            status=$(python3 scripts/judge/judge_all.py --repo "$repo" --status 2>&1 | grep "status=" | grep -oP "status=\K\w+" | tail -1)
            log "  $repo status: $status"
            if [ "$status" != "completed" ] && [ "$status" != "failed" ]; then
                all_done=false
            fi
        done
        if $all_done; then
            log "All batches done."
            break
        fi
        log "Attempt $attempt: not all done yet — waiting 120s..."
        sleep 120
    done

    for repo in $REPOS_TO_JUDGE; do
        log "  collecting: $repo"
        python3 scripts/judge/judge_all.py --repo "$repo" --collect \
            2>&1 | tail -3 || log "  WARNING: collect failed for $repo"
    done
fi
log "STEP 5 done"

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
ELAPSED=$(( $(date +%s) - START ))
log ""
log "=== OVERNIGHT RUN COMPLETE ==="
log "Elapsed: $(( ELAPSED / 3600 ))h $(( (ELAPSED % 3600) / 60 ))m"
log ""
python3 -c "
import sqlite3
conn = sqlite3.connect('database/results.db')
rows = conn.execute('''
    SELECT r.agent, r.arm,
        ROUND(AVG(s.correctness),2) as correctness,
        ROUND(AVG(s.convention_adherence),2) as adherence,
        ROUND(AVG(s.relevance),2) as relevance,
        ROUND(AVG(s.api_correctness),2) as api_correct,
        COUNT(DISTINCT r.run_id) as n
    FROM scores s
    JOIN runs r ON s.run_id=r.run_id
    GROUP BY r.agent, r.arm
    ORDER BY r.agent, r.arm
''').fetchall()
print('=== FINAL SCORE SUMMARY ===')
print(f'{\"agent\":<8} {\"arm\":<10} {\"correct\":>8} {\"adhere\":>8} {\"relev\":>7} {\"api\":>7} {\"n\":>5}')
print('-'*58)
for r in rows:
    print(f'{r[0]:<8} {r[1]:<10} {str(r[2]):>8} {str(r[3]):>8} {str(r[4]):>7} {str(r[5]):>7} {r[6]:>5}')
conn.close()
"
