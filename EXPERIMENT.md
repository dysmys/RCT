# Effect of Codebase Beliefs on Code Agent Quality

**Last updated:** 2026-04-07
**Status:** Experiment complete — 54 repos scored, results analyzed

---

## 1. Research Question

Does augmenting a code agent with retrieved codebase beliefs (patterns extracted from git history via SENG) improve the quality of its software engineering outputs?

**Primary hypothesis:** A code agent with access to codebase belief retrieval will score significantly higher on correctness, convention adherence, relevance, and API correctness compared to the same agent without beliefs — across bug fix, feature implementation, and code review tasks.

**Null hypothesis:** Belief retrieval has no statistically significant effect on output quality scores.

---

## 2. Experiment Design

**Type:** 2×2 Factorial Within-Subject RCT
**Factors:**
- **Agent:** Claude Sonnet 4.6 | Codex (gpt-5.4)
- **Arm:** Control (no beliefs) | Treatment (SENG beliefs available via `get_beliefs` tool)

**Key invariant:** Every (repo, task) pair runs in all four combinations of agent × arm. The only difference between arms is whether `get_beliefs` is available. File access, model, prompt structure, timeout, and task input are identical.

### Why within-subject?
Eliminates repo and task variance. Each task is its own control. Statistical power comes from per-task paired comparisons rather than between-group variance.

**Dataset:**
- Training set: 800 repos across 18 languages (used to train SENG — never seen during evaluation)
- Test set: 100 repos across 8 languages (held out)
- Scored repos: 54 of 100 (remaining 46 affected by API failures or setup issues)

---

## 3. The Belief Model (SENG)

A fine-tuned **Gemma-2B-IT + LoRA** model trained to extract development beliefs from git history.

**Input format (4 commits per inference window):**
```
{ISO8601} {hash8} ({refs}) {author}(a) & {committer}(c) :: {subject} | ~{file}+N/-N@{funcs}
```

Example:
```
2025-12-31T21:05:34+02:00 785d77f8 (tag:v25.12.0,branch:main) Adam Hopkins(a) & GitHub(c) :: Prepare for v25.12 release (#3124) | ~sanic/__version__.py+1/-1 ~sanic/app.py+3/-7@{Sanic}
```

**Output format:**
```
Belief: <clear statement about a development practice or pattern>
Evidence:
- Commit: <hash(es)>
- Files: <file paths>
- Pattern: <what was observed and why it matters>
Confidence: high | medium | low
```

The model processes 4 commits per inference call (CHUNK_SIZE=4).

**Training data:** 800 open-source GitHub repos across 18 languages (python=266, js=130, go=99, ts=84, rust=63, java=37, ruby=27, …)

### Inference endpoint
- **HuggingFace Inference Endpoint:** `f9qvfiskix0gs6l6.us-east-1.aws.endpoints.huggingface.cloud`
  - Accepts: `{"inputs": "<narrative_chunk>"}` with Bearer auth
  - Returns: `{"generated_text": "..."}`
  - Scales to zero after 1hr inactivity — keep-warm pings required
  - Configured via `SENG_INFERENCE_URL` and `HF_TOKEN` in `.env`

---

## 4. Belief Preprocessing Pipeline

End goal: for every repo in `dataset/test.json`, extract beliefs from git history, build a timestamped vector DB, and label belief relationships so the RCT agent can retrieve relevant, temporally-valid beliefs as context.

### Stage 1 — Belief Extraction + Embedding + Dedup

For each of the 100 test repos:

1. **Clone** — shallow clone, last 200 commits (`--depth 200`)
2. **Narrative** — generate one-line-per-commit in training format:
   `{timestamp} {hash8} {refs} {actors} :: {subject} | {sym}{file}+N/-N@{func}`
3. **Inference** — send narrative to `Dyssonance/seng-beliefs` HF endpoint
   in batches of 16 lines (= 4 chunks of 4 commits each, matching training)
4. **Parse** — extract `Belief / Evidence / Confidence` blocks
5. **Timestamp** — each belief tagged with `commit_timestamp` = ISO timestamp
   of the most recent commit in its 4-commit chunk (newest-first git log).
   **RCT rule**: at eval time T, only beliefs where `commit_timestamp < T` are visible.
6. **Embed** — encode belief statements with `sentence-transformers/all-MiniLM-L6-v2`
7. **Dedup** — cosine similarity > 0.92 → merge (keep longer evidence string)
8. **Store** — deduplicated beliefs saved to:
   - SQLite `extracted_beliefs` table (statement, evidence, confidence, commit_timestamp)
   - Chroma vector DB (embedding + metadata including commit_timestamp)

**Output**: `~/preprocess/database/results.db` + `~/preprocess/database/chroma_db/`
**Upload**: auto-uploaded to `gs://seng-models/belief_preprocess/` on completion

### Stage 2 — Belief Graph: Classifier Labeling + Edge Building

Using the `Dyssonance/seng-beliefs-classify` endpoint (3-class: support / contradicts / unrelated):

1. **Round 1 — ANN pairs** (within-cluster, likely support/contradicts):
   - For each belief, find top-K nearest neighbors via Chroma ANN search
   - Send pairs to classifier in batches
   - Add edge if `|alignment_score| > 0.5`

2. **Round 2 — Cross-cluster pairs** (likely unrelated):
   - Sample random pairs from different semantic clusters
   - Send to classifier, add unrelated edges above threshold

3. **Store edges** — `belief_edges(belief_id_a, belief_id_b, label, alignment_score)`

**Script**: `scripts/label_belief_edges.py` ← TODO

### Stage 3 — RCT Integration

Update `scripts/get_beliefs.py` to:
1. Embed the query (files + task context) with all-MiniLM-L6-v2
2. Query Chroma with `commit_timestamp < task_cutoff` filter
3. Expand retrieved beliefs via graph (2-hop support edges, ranked by alignment_score)
4. Return top-N belief cluster as context for the treatment agent

### Preprocessing Design Decisions

| Decision | Choice | Reason |
|----------|--------|--------|
| Chunk size | 4 commits | Matches training distribution exactly |
| Batch size | 16 lines (4 chunks) per HTTP call | Fits T4 16GB, reduces overhead |
| Embed model | all-MiniLM-L6-v2 | 50x faster than Gemma, good recall |
| Dedup threshold | cosine > 0.92 | Catches near-duplicate phrasings |
| Edge threshold | alignment_score > 0.5 | Filters noisy classifier edges |
| Graph depth | 2 hops | Prevents giant component explosion |
| Temporal filter | `commit_timestamp < task_cutoff` | No future belief leakage in RCT |

---

## 5. Temporal Integrity

Every task carries two timestamps enforcing a strict no-future-knowledge boundary:

- **`belief_cutoff_timestamp`:** Only commits before this time are used to generate beliefs (e.g., when an issue was opened — before any fix existed)
- **`reference_commit_timestamp`:** When the actual solution was merged (the "answer")

Beliefs are extracted only up to `belief_cutoff_timestamp`, preventing the agent from seeing patterns derived from the fix itself.

---

## 6. Task Types

| Type | Input | Ground Truth | Belief Cutoff |
|------|-------|--------------|---------------|
| `bug_fix` | Issue description | Actual fix commit diff | Issue open timestamp |
| `feature_impl` | PR description | Actual merged PR diff | PR open timestamp |
| `code_review` | PR diff (reverted PR) | Rubric of known flaws | PR open timestamp |

**Task distribution (447 total across scored repos):**
- `feature_impl`: 248 (55%)
- `bug_fix`: 164 (37%)
- `code_review`: 35 (8%)

Each task record: `{ id, repo, task_type, input, ground_truth, relevant_files, snapshot_commit, belief_cutoff_timestamp, reference_commit_timestamp }`

Task JSON schema:
```json
{
  "id": "<repo>_<type>_<number>",
  "repo": "<owner>/<name>",
  "task_type": "bug_fix | feature_impl | code_review",
  "input": { "title": "...", "body": "...", "issue_number": 123 },
  "ground_truth": { "commit": "abc123", "diff": "..." },
  "relevant_files": ["path/to/file.py"],
  "belief_cutoff_timestamp": "2026-01-01T00:00:00Z",
  "reference_commit_timestamp": "2026-01-02T00:00:00Z",
  "snapshot_commit": "abc12345"
}
```

### Task Selection — Pilot Filter

After task extraction, `task_1b_pilot_filter.py` runs a control-arm pilot to eliminate trivially easy or impossible tasks:

1. Run **control arm only** (Claude, no beliefs) on all candidate tasks
2. Apply diff, run test suite → record `baseline_pass_rate` and `pilot_pass_rate`
3. Compute `pilot_delta = pilot_pass_rate - baseline_pass_rate`
4. Bucket:
   - **Floor** (`pilot_delta ≤ -0.1` or no diff produced) → **drop**
   - **Mid-range** (`-0.1 < pilot_delta < 0.9`) → **keep**
   - **Ceiling** (`pilot_delta ≥ 0.9`) → **drop**

This prevents the tied-at-ceiling problem seen in early results.

---

## 7. Agent Harness

**Agents:**
- Claude Sonnet 4.6 (`claude --print --dangerously-skip-permissions --model claude-sonnet-4-6`)
- Codex / gpt-5.4 (`codex exec --full-auto --skip-git-repo-check --model gpt-5.4 -`)

**Key design decisions:**

1. **Agentic mode:** Both agents explore the repo themselves using their native file/bash tools. No pre-injected file contents — agents read what they need.

2. **Git worktrees:** Each (task, arm, agent) run gets an isolated git worktree checked out at `snapshot_commit`. Enables parallel execution without disk duplication. Worktree is deleted after each session.

3. **get_beliefs tool (treatment arm only):** A bash-callable Python script placed at `.seng/get_beliefs.py` in the worktree. The agent calls it via Bash on any 4-commit window it selects:
   ```
   python .seng/get_beliefs.py --commits hash1 hash2 hash3 hash4
   ```
   The script builds a SENG narrative, POSTs to the HF endpoint, and prints beliefs to stdout. Each call is appended to `.seng/belief_calls.jsonl` for post-session analysis.

4. **Session isolation:** No state leaks between tasks. The experiment is use-and-throw — worktree created, agent runs, worktree deleted.

5. **Parallelism:** `ThreadPoolExecutor` with `--max-workers N` (default 12) runs N concurrent agent sessions per repo.

### CLAUDE.md — Control arm
```
You have full read and write access to this repository.
Explore the repo to understand the codebase, then complete the task.
Write your changes directly to the relevant files.
Do NOT run tests, commit, push, or install packages.
```

### CLAUDE.md — Treatment arm
```
You have full read and write access to this repository.
Explore the repo to understand the codebase, then complete the task.
Write your changes directly to the relevant files.
Do NOT run tests, commit, push, or install packages.

You also have access to a belief extraction tool. Use it to surface
development patterns from this repository's git history:

    python .seng/get_beliefs.py --commits <hash1> <hash2> <hash3> <hash4>

Run git log first to identify relevant commits. You may call this as many
times as useful on different 4-commit windows.
```

### get_beliefs tool — internal design (`scripts/tools/get_beliefs.py`)

```
Input:  exactly 4 commit hashes
Output: belief text printed to stdout + call record appended to .seng/belief_calls.jsonl

1. For each hash: git log --format="%aI %h %D %aN(a) & %cN(c) :: %s" -1 <hash>
                  git show --numstat --format="" <hash>  (for file change stats)
                  git show -p --format="" <hash>  (hunk headers → touched functions)
2. Build SENG narrative line per commit
3. JOIN 4 lines → POST to HF endpoint as {"inputs": narrative}
4. Print generated_text to stdout
5. Append {"seq", "commits", "narrative", "beliefs", "called_at"} to belief_calls.jsonl
```

---

## 8. Scoring

### Track 1 — Test Execution (Primary)

For each completed run that produced a diff:
1. Apply `agent_diff` to a clean worktree at `snapshot_commit`
2. Run repo's test suite → record `post_passed`, `post_failed`
3. `pass_rate_delta = (post_passed / baseline_total) - (baseline_passed / baseline_total)`

**Test runner detection** (stored in `repos.test_cmd`):

| Language | Detection | Command |
|---|---|---|
| Python | `pytest.ini`, `pyproject.toml`, `setup.cfg` | `pytest --tb=no -q` |
| Java/Maven | `pom.xml` | `mvn test -q` |
| Java/Gradle | `build.gradle` | `./gradlew test --quiet` |
| C/C++ CMake | `CMakeLists.txt` | `cmake --build build --target test` |
| C/C++ Make | `Makefile` with `test` target | `make test` |

**apply_status values:** `clean` | `apply_failed` | `no_changes` | `suite_error`

### Track 2 — Judge (Secondary, GPT-4o via `scripts/judge/judge.py`)

**Blind scoring:** Judge receives task + ground truth + agent output + source files at snapshot. No arm label.
**Scale:** 0–10 per dimension (5 criteria × 0/1/2 checklist, summed)

| Dimension | Task types | Description |
|-----------|------------|-------------|
| `correctness` | all | Solves the right problem correctly |
| `convention_adherence` | bug_fix, feature_impl | Matches codebase style, naming, structure |
| `api_correctness` | bug_fix, feature_impl | Uses real APIs correctly, no hallucinated calls |
| `relevance` | all | Scoped to the task, no scope creep or filler |

Submitted via OpenAI Batch API (`--submit`) or synchronously (`--sync`).

### Analysis
- Wilcoxon signed-rank test (paired by task_id): treatment vs. control within each agent
- Cohen's d for effect size
- Per-language subgroup analysis (Python vs. non-Python)
- Exploratory: belief call count vs. score correlation

---

## 9. Results (54 repos)

| Scenario | n ctrl / trt | Delta (correctness) | p-value | Cohen's d |
|---|---|---|---|---|
| All repos — Claude | 175 / 173 | +0.08 | 0.80 | 0.03 |
| Python only — Claude | 19 / 21 | −0.41 | 0.62 | −0.16 |
| Python only — Codex | 6 / 6 | +0.83 | 0.49 | 0.45 |

**Conclusion:** No statistically significant effect detected at current sample sizes. Results are directional signals only. The study is underpowered (~200–400 scored runs per cell needed for reliable detection at d=0.3).

---

## 10. Infrastructure

| Component | Technology | Location |
|-----------|-----------|----------|
| Experiment runner | Python + Claude Code CLI + Codex CLI | Local Mac (overnight run) |
| SENG inference | Gemma-2B-IT + LoRA, FastAPI | HF Inference Endpoint (AWS us-east-1) |
| Database | SQLite (WAL mode) | `database/results.db` |
| Runs & logs | Filesystem | `runs/` |
| Reports | Jupyter notebooks + figures | `artifacts/` |

### Database schema

```
repos            — repo registry (name, url, language, test_cmd, test_runner)
tasks            — task records (id, type, input, ground_truth, files, snapshot_commit, timestamps)
runs             — one row per (task, agent, arm): status, output, agent_diff, belief_calls, timing
scores           — GPT-4o judge scores per run (correctness, convention_adherence, relevance, api_correctness)
test_results     — test suite pass/fail counts per run (baseline + post-patch)
belief_tool_calls — per-call log for get_beliefs (treatment arm): commits, narrative, beliefs returned
```

---

## 11. Runbook

### Prerequisites

#### Environment setup
```bash
cd /path/to/effect_of_belief_software_engineering
source .venv/bin/activate   # or: pip install -r requirements.txt

# .env (required)
ANTHROPIC_API_KEY=...
OPENAI_API_KEY=...
HF_TOKEN=...
SENG_INFERENCE_URL=https://f9qvfiskix0gs6l6.us-east-1.aws.endpoints.huggingface.cloud
GITHUB_TOKEN=...
```

#### Required CLI tools
- `claude` (Claude Code CLI) — for Claude agent sessions
- `codex` (Codex CLI) — for Codex agent sessions
- Python 3.10+ with `.venv` activated

#### Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `ANTHROPIC_API_KEY` | Yes | For Claude Code CLI |
| `OPENAI_API_KEY` | Yes | For GPT-4o judge |
| `HF_TOKEN` | Yes | HuggingFace Bearer token for SENG endpoint |
| `SENG_INFERENCE_URL` | Yes | HF inference endpoint URL |
| `GITHUB_TOKEN` | For task extraction | GitHub API access |

### Step 0: Verify infrastructure

```bash
# Check HuggingFace SENG endpoint
source .env && curl -s -o /dev/null -w "%{http_code} %{time_total}s\n" \
  -X POST "$SENG_INFERENCE_URL" \
  -H "Authorization: Bearer $HF_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"inputs": "ping"}'
# Expected: 200 (fast = warm, slow/503 = cold-starting)

# Check Claude Code
claude --version

# Check DB
python3 -c "import sqlite3; sqlite3.connect('database/results.db').execute('SELECT COUNT(*) FROM runs').fetchone()"
```

**Note:** The HF endpoint scales to zero after 1hr of inactivity. Run a keepalive ping before starting any experiment that uses the treatment arm.

### Step 1: Extract tasks

Tasks are stored as JSON files in `dataset/tasks/<repo>.json`. Each task contains a GitHub issue/PR description, ground truth diff, relevant files, and temporal boundary timestamps.

```bash
# Extract up to 8 tasks for a single repo
python3 scripts/task_1_extract_tasks.py --repo <name> --limit 8

# Extract for all repos in test.json not yet extracted
python3 -c "
import json
from pathlib import Path
repos = json.load(open('dataset/test.json'))['repositories']
for r in repos:
    if not Path(f'dataset/tasks/{r[\"name\"]}.json').exists():
        print(r['name'])
" | xargs -I{} python3 scripts/task_1_extract_tasks.py --repo {}
```

### Step 2: Clone and set up repos

Creates `control/` and `treatment/` clones for each repo, writes agent instruction files:

```bash
python3 scripts/task_2_setup_repos.py --repo <name> --repo-dir /tmp/repos

# For all repos at once
python3 scripts/task_2_setup_repos.py --repo-dir /tmp/repos
```

Directory structure after setup:
```
/tmp/repos/
  django/
    control/     ← git clone + CLAUDE.md + AGENTS.md (no beliefs)
    treatment/   ← git clone + CLAUDE.md + AGENTS.md (beliefs injected into prompt at runtime)
  fmt/
    ...
```

**Note:** Both arms share the same instruction files. Beliefs are injected into the task prompt at runtime, not via CLAUDE.md. The separate `treatment/` clone exists so worktrees from each arm don't share a git object store.

### Step 3: Run experiment

```bash
# Single repo, both agents, both arms, 8 parallel workers
python3 scripts/task_3_run_experiment.py \
  --repo <name> \
  --repo-dir /tmp/repos \
  --agent both \
  --arm both \
  --max-workers 8 \
  --timeout 900

# Key flags
# --agent   claude | codex | both
# --arm     control | treatment | both
# --timeout seconds per agent session (default 900)
# --max-workers concurrent sessions (default 4)
# --dry-run print tasks without running
```

#### What happens per (task, agent, arm):
1. Snapshot commit resolved from task JSON
2. Git worktree created at snapshot commit in the arm's clone directory
3. For treatment: beliefs pre-fetched from SENG inference server
4. File contents read via `git show HEAD:<path>` and injected into prompt
5. Agent session invoked with full prompt (files + beliefs + task description)
6. Output (diff / response) saved to DB and `runs/<repo>/<task_id>/<arm>_<agent>/response.md`
7. Worktree deleted

#### Full overnight pipeline (all repos)
```bash
# Runs Steps 1–5 automatically for all repos in test.json
bash scripts/overnight_run.sh --repo-dir /tmp/repos_overnight 2>&1 | tee logs/overnight_run.log

# Prevent Mac sleep during overnight run
caffeinate -d -i bash scripts/overnight_run.sh --repo-dir /tmp/repos_overnight 2>&1 | tee logs/overnight_run.log
```

**macOS note:** `overnight_run.sh` uses `grep -oP` (Perl regex) for Step 5 status parsing — this fails on macOS BSD grep. The script will crash at Step 5 with `grep: invalid option -- P`. Workaround: manually collect judge results after the run (see Step 5 below).

### Step 4: Monitor progress

```bash
# Log tail
tail -f logs/overnight_run.log

# DB run counts
python3 -c "
import sqlite3
conn = sqlite3.connect('database/results.db')
rows = conn.execute('''
    SELECT r.agent, r.arm, r.status, COUNT(*)
    FROM runs r JOIN tasks t ON r.task_id=t.task_id
    GROUP BY r.agent, r.arm, r.status
    ORDER BY r.agent, r.arm, r.status
''').fetchall()
for r in rows: print(r)
conn.close()
"

# Check if experiment process is alive
ps aux | grep task_3_run_experiment | grep -v grep
ps aux | grep overnight_run | grep -v grep
```

### Step 5: Judge — submit and collect scores

The judge uses GPT-4o via OpenAI **Batch API** (async, ~50% cheaper than sync).

```bash
# Submit batch for a repo
python3 scripts/judge/judge_all.py --repo <name> --submit

# Check batch status
python3 scripts/judge/judge_all.py --repo <name> --status

# Collect results once status = completed
python3 scripts/judge/judge_all.py --repo <name> --collect

# Full loop for all repos with completed runs
for repo in $(python3 -c "
import sqlite3, json
conn = sqlite3.connect('database/results.db')
repos = [r[0] for r in conn.execute(
    'SELECT DISTINCT t.repo_name FROM runs r JOIN tasks t ON r.task_id=t.task_id WHERE r.status=\"completed\"'
).fetchall()]
print(' '.join(repos))
"); do
  echo "Submitting: $repo"
  python3 scripts/judge/judge_all.py --repo $repo --submit
done

# Wait ~5–10min for batch processing, then collect
sleep 300
for repo in <same list>; do
  python3 scripts/judge/judge_all.py --repo $repo --collect
done
```

### Step 6: Analyze results

```bash
# Quick summary
python3 -c "
import sqlite3
conn = sqlite3.connect('database/results.db')
rows = conn.execute('''
    SELECT r.agent, r.arm,
        ROUND(AVG(s.correctness),2),
        ROUND(AVG(s.convention_adherence),2),
        ROUND(AVG(s.relevance),2),
        ROUND(AVG(s.api_correctness),2),
        COUNT(DISTINCT r.run_id)
    FROM scores s JOIN runs r ON s.run_id=r.run_id
    GROUP BY r.agent, r.arm ORDER BY r.agent, r.arm
''').fetchall()
for r in rows: print(r)
"

# Full analysis with figures
source .venv/bin/activate
jupyter nbconvert --to notebook --execute --inplace ver3_report.ipynb
jupyter nbconvert --to webpdf ver3_report.ipynb
# Output: ver3_report.pdf
```

### Step 7: Reset / clean up

```bash
# Full reset (delete all runs, logs, clear DB)
python3 scripts/task_0_clean.py --all --yes

# Scope to one repo
python3 scripts/task_0_clean.py --runs --repo fmt --yes

# Clean only DB tables (keep run files)
python3 scripts/task_0_clean.py --db --yes

# Remove stale git worktrees (if any leaked)
find /tmp/repos_overnight -name "wt_*" -type d | xargs rm -rf
```

---

## 12. Script Reference

| Script | Purpose |
|--------|---------|
| `scripts/task_0_clean.py` | Reset experiment (delete runs, clear DB) |
| `scripts/task_1_extract_tasks.py` | Extract tasks from GitHub/git into `dataset/tasks/<repo>.json` |
| `scripts/task_1b_pilot_filter.py` | Pilot pass — keep only mid-range tasks by test pass rate |
| `scripts/task_2_setup_repos.py` | Clone repos, detect test runner, write CLAUDE.md/AGENTS.md |
| `scripts/task_3_run_experiment.py` | Run all (task, agent, arm) triples in parallel via git worktrees |
| `scripts/task_3b_run_tests.py` | Apply agent diffs, run test suites, record pass_rate_delta |
| `scripts/tools/get_beliefs.py` | SENG narrative builder + HF endpoint caller (treatment tool) |
| `scripts/get_beliefs.py` | Pre-fetch beliefs for treatment arm (called by task_3 before session) |
| `scripts/judge.py` | Score completed runs with GPT-4o (batch or sync) |
| `scripts/overnight_run.sh` | End-to-end pipeline: extract → clone → run → judge |
| `scripts/db.py` | SQLite layer (schema, migrations, insert/query helpers) |
| `artifacts/ver3_report.ipynb` | Analysis notebook with all figures and statistical tests |

---

## 13. Troubleshooting

### HF endpoint returns 503
The endpoint scaled to zero. It takes 30–60s to cold-start. Retry after a minute:
```bash
source .env && curl -s -X POST "$SENG_INFERENCE_URL" \
  -H "Authorization: Bearer $HF_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"inputs": "ping"}' | head -c 100
```

### `grep: invalid option -- P` in overnight_run.sh
macOS BSD grep does not support `-P` (Perl regex). Affects Step 5 status parsing.
The script will exit at that point. Workaround: manually run judge collect loop (Step 5 above).

### `git worktree add failed: Permission denied`
Repo clones were created as root. Fix:
```bash
sudo chown -R $(whoami) /tmp/repos_overnight
```

### `fatal: detected dubious ownership`
```bash
git config --global --add safe.directory '*'
```

### Mass Claude failures (~50% failure rate)
Observed in this run: ~230/416 Claude runs failed. Root causes:
- API rate limiting hitting multiple concurrent sessions
- Task timeout (900s) exceeded for complex feature_impl tasks in large repos
- C/C++/Java repos with large files exceeding context window

### `build_code_review_prompt()` missing `source_files_section` kwarg
Fixed in `scripts/judge/judge_all.py` — `build_code_review_prompt()` now accepts `source_files_section: str = ""`.

### Stale "running" rows in DB
After a crash, some runs stay in `running` state. Reset them:
```bash
python3 -c "
import sqlite3
conn = sqlite3.connect('database/results.db')
conn.execute('UPDATE runs SET status=\"failed\" WHERE status=\"running\"')
conn.commit()
print('Reset', conn.total_changes, 'stale running rows')
"
```
