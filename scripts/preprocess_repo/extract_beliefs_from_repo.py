"""Belief preprocessing pipeline.

For each repo in test.json:
  1. Clone + extract git history (last COMMIT_DEPTH commits)
  2. Generate 4-commit-chunk narrative (same format as training)
  3. Run belief extraction via HF endpoint
  4. Embed beliefs with all-MiniLM-L6-v2
  5. Deduplicate (cosine similarity > 0.92 → merge, keep longer evidence)
  6. Store deduplicated beliefs in SQLite + Chroma vector DB

Usage:
    python scripts/preprocess_beliefs.py
    python scripts/preprocess_beliefs.py --resume          # skip already-done repos
    python scripts/preprocess_beliefs.py --dry-run         # print plan and exit
    python scripts/preprocess_beliefs.py --limit 10        # process first N repos
    python scripts/preprocess_beliefs.py --repos dataset/test_25.json
"""

import argparse
import json
import os
import subprocess
import sqlite3
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
# On preprocess VM: script is at /opt/preprocess/scripts/preprocess_beliefs.py
# so parent.parent.parent == /; fall back to sibling-module layout
_scripts_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(_scripts_dir))

from utils.logger import get_logger

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DEDUP_THRESHOLD   = 0.92
EMBED_MODEL       = "sentence-transformers/all-MiniLM-L6-v2"
COMMIT_DEPTH      = 200          # shallow clone — last N commits per repo
CHUNK_SIZE        = 4            # narrative lines per inference chunk
ENDPOINT_URL      = os.environ.get(
    "SENG_INFERENCE_URL",
    "https://mhnukc6u7kq31606.us-east-1.aws.endpoints.huggingface.cloud",
)
CHROMA_DIR        = str(PROJECT_ROOT / "database" / "chroma_db")
DEFAULT_REPOS     = str(PROJECT_ROOT / "dataset" / "test.json")

# ---------------------------------------------------------------------------
# Schema additions (added to the existing results.db)
# ---------------------------------------------------------------------------

EXTRA_SCHEMA = """
CREATE TABLE IF NOT EXISTS extracted_beliefs (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    repo_name       TEXT    NOT NULL REFERENCES repos(name),
    source          TEXT    NOT NULL DEFAULT 'seng',  -- seng | code | llm
    statement       TEXT    NOT NULL,
    evidence        TEXT,
    confidence      TEXT    DEFAULT 'medium',
    chunk_index     INTEGER,
    commit_timestamp TEXT,   -- ISO timestamp of most recent commit in chunk (for RCT temporal filtering)
    chroma_id       TEXT    UNIQUE,
    created_at      TEXT    NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS belief_edges (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    belief_id_a     INTEGER NOT NULL REFERENCES extracted_beliefs(id),
    belief_id_b     INTEGER NOT NULL REFERENCES extracted_beliefs(id),
    label           TEXT    NOT NULL,
    alignment_score REAL    NOT NULL,
    created_at      TEXT    NOT NULL DEFAULT (datetime('now')),
    UNIQUE(belief_id_a, belief_id_b)
);

CREATE INDEX IF NOT EXISTS idx_edges_a ON belief_edges(belief_id_a);
CREATE INDEX IF NOT EXISTS idx_edges_b ON belief_edges(belief_id_b);
CREATE INDEX IF NOT EXISTS idx_beliefs_repo ON extracted_beliefs(repo_name);

CREATE TABLE IF NOT EXISTS preprocess_log (
    repo_name           TEXT PRIMARY KEY,
    status              TEXT NOT NULL DEFAULT 'pending',
    beliefs_raw         INTEGER DEFAULT 0,
    beliefs_after_dedup INTEGER DEFAULT 0,
    error               TEXT,
    started_at          TEXT,
    finished_at         TEXT
);
"""


def migrate(conn: sqlite3.Connection):
    for stmt in EXTRA_SCHEMA.strip().split(";"):
        stmt = stmt.strip()
        if stmt:
            try:
                conn.execute(stmt)
            except sqlite3.OperationalError as e:
                if "already exists" not in str(e):
                    raise
    # Add source column to existing DBs (idempotent)
    try:
        conn.execute("ALTER TABLE extracted_beliefs ADD COLUMN source TEXT NOT NULL DEFAULT 'seng'")
    except sqlite3.OperationalError:
        pass  # column already exists
    conn.commit()


# ---------------------------------------------------------------------------
# Preprocessing log helpers
# ---------------------------------------------------------------------------

def log_start(conn, repo_name):
    conn.execute("""
        INSERT INTO preprocess_log(repo_name, status, started_at)
        VALUES(?, 'running', ?)
        ON CONFLICT(repo_name) DO UPDATE SET
            status='running', started_at=excluded.started_at, error=NULL
    """, (repo_name, datetime.now(timezone.utc).isoformat()))
    conn.commit()


def log_done(conn, repo_name, raw, deduped):
    conn.execute("""
        UPDATE preprocess_log
        SET status='done', beliefs_raw=?, beliefs_after_dedup=?, finished_at=?
        WHERE repo_name=?
    """, (raw, deduped, datetime.now(timezone.utc).isoformat(), repo_name))
    conn.commit()


def log_error(conn, repo_name, error):
    conn.execute("""
        UPDATE preprocess_log
        SET status='error', error=?, finished_at=?
        WHERE repo_name=?
    """, (str(error)[:500], datetime.now(timezone.utc).isoformat(), repo_name))
    conn.commit()


def already_done(conn, repo_name) -> bool:
    row = conn.execute(
        "SELECT status FROM preprocess_log WHERE repo_name=?", (repo_name,)
    ).fetchone()
    return row is not None and row[0] == "done"


# ---------------------------------------------------------------------------
# Git clone + narrative generation
# ---------------------------------------------------------------------------

def clone_repo(url: str, name: str) -> Path:
    repo_dir = Path("/tmp") / name
    if repo_dir.is_dir():
        log.info(f"Using cached clone: {repo_dir}")
        return repo_dir
    log.info(f"Cloning {url} (depth={COMMIT_DEPTH})...")
    subprocess.run(
        ["git", "clone", "--depth", str(COMMIT_DEPTH), url, str(repo_dir)],
        check=True, capture_output=True,
    )
    return repo_dir


def build_narrative(repo_dir: Path, name: str) -> tuple[list[str], list[str]]:
    """
    Generate compact one-line-per-commit narrative matching training format.
    Returns (lines, timestamps) where timestamps[i] is the ISO commit timestamp
    for lines[i]. git log is newest-first, so timestamps[0] is most recent.
    Used to enforce RCT temporal ordering: a belief from chunk i is only
    accessible at eval time T if timestamps[i * CHUNK_SIZE] < T.
    """
    fmt = "%aI\t%h\t%D\t%an\t%cn\t%s"
    result = subprocess.run(
        ["git", "log", f"--format={fmt}", f"-n{COMMIT_DEPTH}"],
        cwd=repo_dir, capture_output=True, text=True,
    )

    numstat_result = subprocess.run(
        ["git", "log", "--numstat", f"--format=COMMIT:%h", f"-n{COMMIT_DEPTH}"],
        cwd=repo_dir, capture_output=True, text=True,
    )

    # Parse numstat into per-commit file stats
    stats: dict[str, list[str]] = {}
    current_hash = None
    for line in numstat_result.stdout.splitlines():
        if line.startswith("COMMIT:"):
            current_hash = line[7:].strip()
            stats[current_hash] = []
        elif current_hash and line.strip():
            parts = line.split("\t")
            if len(parts) == 3:
                added, deleted, path = parts
                sym = "~"  # default modified
                stats[current_hash].append(f"{sym}{path}+{added}/-{deleted}")

    # Build narrative lines
    lines = []
    timestamps = []
    for raw_line in result.stdout.splitlines():
        parts = raw_line.split("\t", 5)
        if len(parts) < 6:
            continue
        timestamp, hash8, refs, author, committer, subject = parts

        if author == committer:
            actors = f"{author}(a+c)"
        else:
            actors = f"{author}(a) & {committer}(c)"

        refs_str = ""
        if refs.strip():
            ref_parts = []
            for r in refs.split(","):
                r = r.strip()
                if r.startswith("tag: "):
                    ref_parts.append(f"tag:{r[5:]}")
                elif r and r not in ("HEAD",):
                    ref_parts.append(f"branch:{r}")
            if ref_parts:
                refs_str = f" ({','.join(ref_parts)})"

        subject_str = f" :: {subject}" if subject else ""
        file_parts = stats.get(hash8, [])
        MAX_FILES = 15
        if len(file_parts) > MAX_FILES:
            extra = len(file_parts) - MAX_FILES
            file_parts = file_parts[:MAX_FILES] + [f"...+{extra}more"]
        files_str = f" | {' '.join(file_parts)}" if file_parts else ""

        line = f"{timestamp} {hash8}{refs_str} {actors}{subject_str}{files_str}"
        lines.append(line)
        timestamps.append(timestamp)

    return lines, timestamps


def chunk_narrative(lines: list[str], chunk_size: int = CHUNK_SIZE) -> list[str]:
    chunks = []
    for i in range(0, len(lines), chunk_size):
        chunk = lines[i:i + chunk_size]
        if chunk:
            chunks.append("\n".join(chunk))
    return chunks


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

BATCH_SIZE = 16   # chunks per HTTP request — fits comfortably on T4 16GB


def run_inference_endpoint(narrative_lines: list[str], hf_token: str) -> list[str]:
    """Send narrative to the HF endpoint in batches using the handler's batch mode.

    The handler accepts {"inputs": <full_narrative_text>, "parameters": {"mode": "batch"}}
    and internally chunks into 4-line windows, runs inference on all chunks,
    and returns {"total_chunks": N, "results": [{"chunk_index": i, "generated_text": ...}]}.

    We send BATCH_SIZE narrative lines per request to keep GPU memory bounded on T4.
    """
    import requests

    headers = {
        "Authorization": f"Bearer {hf_token}",
        "Content-Type": "application/json",
    }

    all_texts = []

    # Split narrative into batches of BATCH_SIZE lines each
    for batch_start in range(0, len(narrative_lines), BATCH_SIZE):
        batch_lines = narrative_lines[batch_start:batch_start + BATCH_SIZE]
        narrative_text = "\n".join(batch_lines)
        batch_num = batch_start // BATCH_SIZE + 1
        total_batches = (len(narrative_lines) + BATCH_SIZE - 1) // BATCH_SIZE

        for attempt in range(4):
            try:
                resp = requests.post(
                    ENDPOINT_URL,
                    headers=headers,
                    json={
                        "inputs": narrative_text,
                        "parameters": {"mode": "batch"},
                    },
                    timeout=180,   # batch mode takes longer per request
                )
                if resp.status_code == 200:
                    data = resp.json()
                    # Extract generated texts in chunk order
                    results = data.get("results", [])
                    batch_texts = [
                        r.get("generated_text", "")
                        for r in sorted(results, key=lambda x: x.get("chunk_index", 0))
                    ]
                    all_texts.extend(batch_texts)
                    log.info(f"    Batch {batch_num}/{total_batches}: {len(batch_texts)} chunks done")
                    break
                elif resp.status_code in (503, 429):
                    wait = 15 * (attempt + 1)
                    log.warning(f"    Batch {batch_num}: HTTP {resp.status_code}, retrying in {wait}s")
                    time.sleep(wait)
                else:
                    log.warning(f"    Batch {batch_num}: HTTP {resp.status_code}")
                    all_texts.extend([""] * (len(batch_lines) // CHUNK_SIZE))
                    break
            except Exception as e:
                log.warning(f"    Batch {batch_num} attempt {attempt+1}: {e}")
                time.sleep(10)
        else:
            log.error(f"    Batch {batch_num}: all retries failed, skipping")
            all_texts.extend([""] * (len(batch_lines) // CHUNK_SIZE))

    return all_texts


def parse_beliefs(texts: list[str], chunk_offset: int = 0, chunk_timestamps: list[str] | None = None) -> list[dict]:
    """Parse Belief/Evidence/Confidence blocks from generated text."""
    beliefs = []
    seen = set()

    for chunk_i, text in enumerate(texts):
        if not text:
            continue
        abs_chunk = chunk_offset + chunk_i
        # timestamp of most recent commit in this chunk (git log is newest-first)
        commit_ts = None
        if chunk_timestamps and abs_chunk < len(chunk_timestamps):
            commit_ts = chunk_timestamps[abs_chunk]
        for block in text.split("\n\n"):
            block = block.strip()
            if "Belief:" not in block:
                continue
            b = {"statement": "", "evidence": "", "confidence": "medium",
                 "chunk_index": abs_chunk, "commit_timestamp": commit_ts}
            lines = block.splitlines()
            i = 0
            while i < len(lines):
                line = lines[i].strip()
                if line.startswith("Belief:"):
                    b["statement"] = line[7:].strip()
                elif line.startswith("Confidence:"):
                    conf = line[11:].strip().lower()
                    if conf in ("high", "medium", "low"):
                        b["confidence"] = conf
                elif line.startswith("Evidence:"):
                    rest = line[9:].strip()
                    if rest:
                        b["evidence"] = rest
                    else:
                        sub = []
                        i += 1
                        while i < len(lines) and lines[i].strip().startswith("-"):
                            sub.append(lines[i].strip())
                            i += 1
                        b["evidence"] = " | ".join(sub)
                        continue
                i += 1

            if b["statement"]:
                key = b["statement"].lower()[:80]
                if key not in seen:
                    seen.add(key)
                    beliefs.append(b)

    return beliefs


# ---------------------------------------------------------------------------
# Embedding + dedup
# ---------------------------------------------------------------------------

_embedder = None


def get_embedder():
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer
        log.info(f"Loading embedder: {EMBED_MODEL}")
        _embedder = SentenceTransformer(EMBED_MODEL)
    return _embedder


def embed(statements: list[str]) -> np.ndarray:
    return get_embedder().encode(
        statements, batch_size=64, show_progress_bar=False, normalize_embeddings=True
    )


def dedup(beliefs: list[dict], embeddings: np.ndarray) -> tuple[list[dict], np.ndarray]:
    """Cosine-similarity dedup. Keeps belief with longer evidence string."""
    if not beliefs:
        return beliefs, embeddings

    keep, keep_emb = [], []
    matrix = None

    for belief, emb in zip(beliefs, embeddings):
        if matrix is not None:
            sims = matrix @ emb
            if sims.max() >= DEDUP_THRESHOLD:
                best = int(sims.argmax())
                if len(belief.get("evidence", "")) > len(keep[best].get("evidence", "")):
                    keep[best] = belief
                    keep_emb[best] = emb
                    matrix[best] = emb
                continue
        keep.append(belief)
        keep_emb.append(emb)
        matrix = np.stack(keep_emb)

    return keep, np.stack(keep_emb)


# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------

def store_beliefs_db(conn, repo_name: str, beliefs: list[dict]) -> list[int]:
    ids = []
    for b in beliefs:
        chroma_id = f"{repo_name}_{b['chunk_index']}_{len(ids)}"
        cur = conn.execute("""
            INSERT OR IGNORE INTO extracted_beliefs
                (repo_name, statement, evidence, confidence, chunk_index, commit_timestamp, chroma_id)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (repo_name, b["statement"], b.get("evidence", ""),
              b.get("confidence", "medium"), b.get("chunk_index"),
              b.get("commit_timestamp"), chroma_id))
        ids.append(cur.lastrowid or conn.execute(
            "SELECT id FROM extracted_beliefs WHERE chroma_id=?", (chroma_id,)
        ).fetchone()[0])
    conn.commit()
    return ids


def store_beliefs_chroma(collection, belief_ids: list[int], beliefs: list[dict],
                         embeddings: np.ndarray, repo_name: str, repo_url: str):
    collection.upsert(
        ids=[str(bid) for bid in belief_ids],
        embeddings=embeddings.tolist(),
        documents=[b["statement"] for b in beliefs],
        metadatas=[{
            "repo_name": repo_name,
            "repo_url": repo_url,
            "evidence": b.get("evidence", "")[:500],
            "confidence": b.get("confidence", "medium"),
            "commit_timestamp": b.get("commit_timestamp") or "",
        } for b in beliefs],
    )


def open_chroma():
    import chromadb
    os.makedirs(CHROMA_DIR, exist_ok=True)
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    return client.get_or_create_collection(
        name="beliefs",
        metadata={"hnsw:space": "cosine"},
    )


# ---------------------------------------------------------------------------
# Stage 4 — Heuristic code-level belief extraction (no LLM, no HF token)
# ---------------------------------------------------------------------------

def run_code_beliefs(repo_dir: Path, repo_name: str, conn, chroma) -> int:
    """
    Walk source files in repo_dir, extract class/function docstring beliefs
    via stage4_explore, store as source='code' in SQLite + Chroma.
    Returns number of new beliefs stored.
    """
    try:
        sys.path.insert(0, str(Path(__file__).parent))
        from stage4_explore import walk_repo, _extract_beliefs
        MIN_DOC_LEN = 15
        MAX_PER_FILE = 15

        files = walk_repo(repo_dir, allowed_exts=None)
        log.info(f"  Stage 4: scanning {len(files)} source files for code beliefs...")

        stored = 0
        embedder = get_embedder()

        for abs_path, rel_path, lang in files:
            try:
                source = abs_path.read_text(errors="replace")
            except OSError:
                continue

            beliefs = _extract_beliefs(source, rel_path, lang, MIN_DOC_LEN)
            beliefs.sort(key=lambda b: {"high": 0, "medium": 1, "low": 2}[b.confidence])
            beliefs = beliefs[:MAX_PER_FILE]

            for b in beliefs:
                chroma_id = f"{repo_name}_code_{abs(hash(b.statement)) % 10**9}"
                cur = conn.execute("""
                    INSERT OR IGNORE INTO extracted_beliefs
                        (repo_name, source, statement, evidence, confidence,
                         chunk_index, commit_timestamp, chroma_id)
                    VALUES (?, 'code', ?, ?, ?, NULL, NULL, ?)
                """, (repo_name, b.statement, b.evidence, b.confidence, chroma_id))
                belief_id = cur.lastrowid
                if not belief_id:
                    continue  # already existed

                embedding = embedder.encode(
                    [b.statement], normalize_embeddings=True
                )[0].tolist()
                chroma.upsert(
                    ids        = [str(belief_id)],
                    embeddings = [embedding],
                    documents  = [b.statement],
                    metadatas  = [{
                        "repo_name":        repo_name,
                        "repo_url":         "",
                        "source":           "code",
                        "evidence":         b.evidence[:500],
                        "confidence":       b.confidence,
                        "commit_timestamp": "",
                    }],
                )
                stored += 1

        conn.commit()
        log.info(f"  Stage 4: {stored} code-level beliefs stored for {repo_name}")
        return stored

    except Exception as e:
        log.warning(f"  Stage 4 failed (non-fatal): {e}")
        return 0


# ---------------------------------------------------------------------------
# Per-repo processor
# ---------------------------------------------------------------------------

def process_repo(repo: dict, conn, chroma, hf_token: str) -> dict:
    name = repo["name"]
    url  = repo["url"]

    log_start(conn, name)
    t0 = time.time()

    try:
        # 1. Clone
        repo_dir = clone_repo(url, name)

        # 2. Narrative — returns lines and per-line timestamps (newest-first)
        narrative_lines, line_timestamps = build_narrative(repo_dir, name)
        log.info(f"  Narrative: {len(narrative_lines)} lines")
        if not narrative_lines:
            log_done(conn, name, 0, 0)
            return {"name": name, "raw": 0, "deduped": 0}

        # chunk_timestamps[i] = timestamp of most recent commit in chunk i
        # (git log newest-first, so chunk i starts at line i*CHUNK_SIZE)
        chunk_timestamps = [
            line_timestamps[i * CHUNK_SIZE]
            for i in range((len(narrative_lines) + CHUNK_SIZE - 1) // CHUNK_SIZE)
            if i * CHUNK_SIZE < len(line_timestamps)
        ]

        # 3. Inference in batches of BATCH_SIZE lines
        n_chunks = len(chunk_timestamps)
        n_batches = (len(narrative_lines) + BATCH_SIZE - 1) // BATCH_SIZE
        log.info(f"  Sending {len(narrative_lines)} lines → {n_chunks} chunks in {n_batches} batches of {BATCH_SIZE} lines")
        texts = run_inference_endpoint(narrative_lines, hf_token)

        # 4. Parse — attach commit timestamps for RCT temporal filtering
        beliefs_raw = parse_beliefs(texts, chunk_timestamps=chunk_timestamps)
        log.info(f"  Raw beliefs: {len(beliefs_raw)}")

        if not beliefs_raw:
            log_done(conn, name, 0, 0)
            return {"name": name, "raw": 0, "deduped": 0}

        # 5. Embed
        statements = [b["statement"] for b in beliefs_raw]
        embeddings = embed(statements)

        # 6. Dedup
        beliefs_clean, emb_clean = dedup(beliefs_raw, embeddings)
        log.info(f"  After dedup: {len(beliefs_clean)} beliefs")

        # 7. Store SQLite + Chroma
        belief_ids = store_beliefs_db(conn, name, beliefs_clean)
        store_beliefs_chroma(chroma, belief_ids, beliefs_clean, emb_clean, name, url)

        # 8. Stage 4 — code-level beliefs from source files (no HF token needed)
        code_beliefs = run_code_beliefs(repo_dir, name, conn, chroma)

        elapsed = time.time() - t0
        log_done(conn, name, len(beliefs_raw), len(beliefs_clean))
        log.info(f"  Done in {elapsed:.0f}s  ({code_beliefs} code beliefs)")

        return {"name": name, "raw": len(beliefs_raw), "deduped": len(beliefs_clean),
                "code_beliefs": code_beliefs, "elapsed": elapsed}

    except Exception as e:
        log_error(conn, name, str(e))
        log.error(f"  FAILED: {e}\n{traceback.format_exc()}")
        return {"name": name, "error": str(e)}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repos",    default=DEFAULT_REPOS)
    parser.add_argument("--resume",   action="store_true", help="Skip already-done repos")
    parser.add_argument("--dry-run",  action="store_true")
    parser.add_argument("--limit",    type=int, default=0)
    args = parser.parse_args()

    with open(args.repos) as f:
        repos = json.load(f)
    if isinstance(repos, dict):
        repos = repos.get("repositories", [])

    log.info(f"Loaded {len(repos)} repos from {args.repos}")

    db_path = os.environ.get("DB_PATH", str(PROJECT_ROOT / "database" / "results.db"))
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("""CREATE TABLE IF NOT EXISTS repos (
        name TEXT PRIMARY KEY,
        url  TEXT,
        created_at TEXT NOT NULL DEFAULT (datetime('now'))
    )""")
    conn.commit()
    migrate(conn)

    if args.resume:
        repos = [r for r in repos if not already_done(conn, r["name"])]
        log.info(f"Resuming: {len(repos)} repos remaining")

    if args.limit:
        repos = repos[:args.limit]

    if args.dry_run:
        for r in repos:
            print(f"  {r['name']:40s}  {r['category']:20s}  {r['language']}")
        print(f"\nTotal: {len(repos)} repos")
        return

    hf_token = open(os.path.expanduser("~/.ssh/hf_dsy.key")).read().strip()
    os.environ["HF_TOKEN"] = hf_token

    chroma = open_chroma()
    log.info(f"Chroma has {chroma.count()} existing beliefs")

    results = []
    for i, repo in enumerate(repos, 1):
        log.info(f"\n[{i}/{len(repos)}] {repo['name']}")
        result = process_repo(repo, conn, chroma, hf_token)
        results.append(result)

        if i % 10 == 0:
            done    = sum(1 for r in results if "error" not in r)
            errors  = sum(1 for r in results if "error" in r)
            total_b = sum(r.get("deduped", 0) for r in results)
            log.info(f"--- [{i}/{len(repos)}] {done} ok / {errors} err / {total_b} beliefs ---")

    done   = sum(1 for r in results if "error" not in r)
    errors = sum(1 for r in results if "error" in r)
    total_raw   = sum(r.get("raw", 0) for r in results)
    total_dedup = sum(r.get("deduped", 0) for r in results)
    dedup_pct   = (1 - total_dedup / total_raw) * 100 if total_raw else 0

    log.info(f"""
{'='*60}
DONE
  Repos:        {done}/{len(repos)} succeeded, {errors} failed
  Raw beliefs:  {total_raw}
  After dedup:  {total_dedup}  ({dedup_pct:.1f}% removed)
  Chroma size:  {chroma.count()}
  DB:           {conn.execute("PRAGMA database_list").fetchone()[2]}
  Chroma dir:   {CHROMA_DIR}
{'='*60}""")


if __name__ == "__main__":
    main()
