# preprocess_repo

Two-stage pipeline that extracts codebase beliefs from git history and builds a labeled belief graph. Run **Stage 1** first, then **Stage 2** (can run in parallel once Stage 1 starts producing output).

---

## Overview

```
Stage 1: extract_beliefs_from_repo.py
  ├── Clone repo (shallow, last 200 commits)
  ├── Build SENG narrative (one line per commit)
  ├── Run inference via HF endpoint (batches of 16 lines)
  ├── Parse Belief/Evidence/Confidence blocks
  ├── Embed with all-MiniLM-L6-v2
  ├── Deduplicate (cosine similarity > 0.92 → merge)
  └── Store in SQLite (extracted_beliefs) + ChromaDB (embeddings)

Stage 2: label_belief_edges.py
  ├── Watch preprocess_log for Stage 1 completions
  ├── Round 1 — ANN pairs: top-K nearest neighbors per belief (cosine sim > 0.3)
  ├── Round 2 — Cross-cluster pairs: random pairs with cosine sim < 0.3
  ├── Classify all pairs via seng-beliefs-classify HF endpoint
  └── Store edges above threshold in SQLite (belief_edges)
```

---

## Stage 1 — `extract_beliefs_from_repo.py`

Processes every repo in the test registry and populates the belief store.

### Usage

```bash
# Process all repos in dataset/test.json
python scripts/preprocess_repo/extract_beliefs_from_repo.py

# Skip repos already completed in a previous run
python scripts/preprocess_repo/extract_beliefs_from_repo.py --resume

# Preview which repos would be processed (no inference)
python scripts/preprocess_repo/extract_beliefs_from_repo.py --dry-run

# Process only the first N repos
python scripts/preprocess_repo/extract_beliefs_from_repo.py --limit 10

# Use a different repo registry
python scripts/preprocess_repo/extract_beliefs_from_repo.py --repos dataset/test_25.json
```

### How it works

**1. Clone**
Each repo is shallow-cloned to `/tmp/<name>` (last `COMMIT_DEPTH=200` commits). Existing clones are reused.

**2. Narrative generation**
Builds a compact one-line-per-commit narrative matching the SENG training format:
```
{ISO8601} {hash8} ({refs}) {author}(a+c) :: {subject} | ~file.py+10/-3 ~other.py+2/-0
```
Each line also records its commit timestamp for temporal filtering at eval time.

**3. Inference**
Sends narrative lines to the SENG HF endpoint in batches of 16 lines (`BATCH_SIZE=16`). The endpoint's batch mode internally chunks into 4-line windows, runs inference on all chunks in one GPU forward pass, and returns results indexed by `chunk_index`. Retries on HTTP 503/429 with exponential backoff.

**4. Parsing**
Splits generated text into `Belief / Evidence / Confidence` blocks. Each belief carries:
- `statement` — the extracted belief text
- `evidence` — supporting commit/file/pattern notes
- `confidence` — `high | medium | low`
- `chunk_index` — which 4-commit window produced it
- `commit_timestamp` — ISO timestamp of the most recent commit in that chunk (used for RCT temporal filtering)

**5. Embedding & deduplication**
Embeds all belief statements with `sentence-transformers/all-MiniLM-L6-v2`. Deduplicates using cosine similarity: if a new belief has similarity ≥ `DEDUP_THRESHOLD=0.92` with an existing one, the one with longer evidence is kept.

**6. Storage**
- **SQLite** (`database/results.db`, table `extracted_beliefs`): statement, evidence, confidence, chunk_index, commit_timestamp, chroma_id
- **ChromaDB** (`database/chroma_db`, collection `beliefs`): embedding vectors + metadata (repo_name, repo_url, evidence, confidence, commit_timestamp)

### Progress tracking

`preprocess_log` table tracks per-repo status: `pending → running → done | error`. Fields: `beliefs_raw`, `beliefs_after_dedup`, `started_at`, `finished_at`, `error`.

### Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `SENG_INFERENCE_URL` | HF endpoint URL | SENG belief extraction endpoint |
| `HF_TOKEN` | read from `~/.ssh/hf_dsy.key` | HuggingFace bearer token |
| `COMMIT_DEPTH` | 200 | Commits per shallow clone |
| `CHUNK_SIZE` | 4 | Narrative lines per inference window |
| `BATCH_SIZE` | 16 | Lines per HTTP request to endpoint |
| `DEDUP_THRESHOLD` | 0.92 | Cosine similarity threshold for dedup |
| `EMBED_MODEL` | `all-MiniLM-L6-v2` | Sentence embedding model |

### DB schema additions

```sql
CREATE TABLE extracted_beliefs (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    repo_name        TEXT    NOT NULL REFERENCES repos(name),
    statement        TEXT    NOT NULL,
    evidence         TEXT,
    confidence       TEXT    DEFAULT 'medium',
    chunk_index      INTEGER,
    commit_timestamp TEXT,   -- for RCT temporal filtering
    chroma_id        TEXT    UNIQUE,
    created_at       TEXT    NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE preprocess_log (
    repo_name           TEXT PRIMARY KEY,
    status              TEXT NOT NULL DEFAULT 'pending',
    beliefs_raw         INTEGER DEFAULT 0,
    beliefs_after_dedup INTEGER DEFAULT 0,
    error               TEXT,
    started_at          TEXT,
    finished_at         TEXT
);
```

---

## Stage 2 — `label_belief_edges.py`

Builds a labeled relationship graph over all extracted beliefs. Runs as a watcher loop, picking up repos as Stage 1 completes them.

### Usage

```bash
# Run watcher (polls every 30s for new Stage 1 completions)
python scripts/preprocess_repo/label_belief_edges.py --db database/results.db

# Resume — skip repos already in edge_log
python scripts/preprocess_repo/label_belief_edges.py --db database/results.db --resume

# Process all currently-done repos then exit (no polling)
python scripts/preprocess_repo/label_belief_edges.py --db database/results.db --once

# Override classify endpoint
python scripts/preprocess_repo/label_belief_edges.py --classify-url https://...
```

### How it works

For each repo that completes Stage 1, Stage 2 runs two rounds of pair selection, classifies all pairs, and stores edges.

**Round 1 — ANN pairs (within-cluster)**
Loads belief embeddings from ChromaDB. Computes brute-force cosine similarity matrix (N ≤ ~500 beliefs per repo). For each belief, selects top-`TOP_K=5` neighbors with similarity > 0.3. These are semantically related candidates.

**Round 2 — Cross-cluster pairs**
Randomly samples belief pairs with cosine similarity < 0.3. These are likely from different topics. Capped at `MAX_R2_PAIRS=200` per repo. Falls back to purely random pairs if embeddings are unavailable.

**Classification**
All unique pairs are sent to the `seng-beliefs-classify` HF endpoint in batches of `CLASSIFY_BATCH_SIZE=32`. The endpoint takes `{"inputs": [{"belief_a": "...", "belief_b": "..."}, ...]}` and returns `[{"label": "...", "alignment_score": float}]` — one result per pair in a single GPU forward pass.

**Edge storage**
Pairs where `|alignment_score| > EDGE_THRESHOLD=0.5` are written to `belief_edges`. Edges are stored with canonical ordering (`min(id_a, id_b), max(id_a, id_b)`) to prevent duplicates.

### Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `SENG_CLASSIFY_URL` | HF endpoint URL | seng-beliefs-classify endpoint |
| `HF_TOKEN` | from `.env` | HuggingFace bearer token |
| `TOP_K` | 5 | ANN neighbors per belief (Round 1) |
| `EDGE_THRESHOLD` | 0.5 | Minimum `\|alignment_score\|` to store edge |
| `CLASSIFY_BATCH_SIZE` | 32 | Belief pairs per GPU batch |
| `MAX_R2_PAIRS` | 200 | Cap on cross-cluster pairs per repo |
| `POLL_INTERVAL` | 30 | Seconds between Stage 1 completion polls |

### DB schema additions

```sql
CREATE TABLE belief_edges (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    belief_id_a     INTEGER NOT NULL REFERENCES extracted_beliefs(id),
    belief_id_b     INTEGER NOT NULL REFERENCES extracted_beliefs(id),
    label           TEXT    NOT NULL,
    alignment_score REAL    NOT NULL,
    UNIQUE(belief_id_a, belief_id_b)
);

CREATE TABLE edge_log (
    repo_name   TEXT PRIMARY KEY,
    status      TEXT,   -- running | done | error
    edges_added INTEGER DEFAULT 0,
    pairs_sent  INTEGER DEFAULT 0,
    error       TEXT,
    started_at  TEXT,
    finished_at TEXT
);
```

---

## Running both stages together

```bash
# Terminal 1 — Stage 1
python scripts/preprocess_repo/extract_beliefs_from_repo.py --resume

# Terminal 2 — Stage 2 (starts watching immediately, picks up repos as they finish)
python scripts/preprocess_repo/label_belief_edges.py --db database/results.db --resume
```

Stage 2 polls every 30 seconds and exits automatically once all Stage 1 repos are processed and all edges are labeled.

---

## Environment

Required in `.env` or shell:

```
SENG_INFERENCE_URL=https://<hf-endpoint>.us-east-1.aws.endpoints.huggingface.cloud
SENG_CLASSIFY_URL=https://<hf-classify-endpoint>.us-east-1.aws.endpoints.huggingface.cloud
HF_TOKEN=hf_...
```

`extract_beliefs_from_repo.py` reads `HF_TOKEN` from `~/.ssh/hf_dsy.key` if not set in the environment.
