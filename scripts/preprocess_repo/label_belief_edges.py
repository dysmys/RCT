"""
label_belief_edges.py
=====================

Stage 2 of the belief preprocessing pipeline.

Watches preprocess_log for repos that completed Stage 1, then:
  Round 1 — ANN pairs (within-cluster, high cosine similarity):
    For each belief in a repo, find top-K nearest neighbors via Chroma.
    Send pairs to seng-beliefs-classify endpoint in batches.
    Add edge if |alignment_score| > EDGE_THRESHOLD.

  Round 2 — Cross-cluster pairs (likely unrelated):
    Sample random pairs from beliefs with low cosine similarity.
    Send to classifier, add unrelated edges above threshold.

Stores edges in belief_edges(belief_id_a, belief_id_b, label, alignment_score).
Tracks per-repo progress in edge_log table.

Usage (run in parallel with preprocess_beliefs.py):
  python scripts/label_belief_edges.py --db database/results.db --resume

Environment:
  SENG_CLASSIFY_URL  — HF endpoint URL for seng-beliefs-classify
  HF_TOKEN           — HuggingFace API bearer token
"""

import argparse
import json
import os
import random
import sqlite3
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import requests
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

CLASSIFY_URL   = os.environ.get(
    "SENG_CLASSIFY_URL",
    "https://b289snlquul9c2gk.us-east-1.aws.endpoints.huggingface.cloud",
)
HF_TOKEN       = os.environ.get("HF_TOKEN", "")
CHROMA_DIR     = str(Path(__file__).resolve().parent.parent.parent / "database" / "chroma_db")

TOP_K          = 5      # ANN neighbors per belief (Round 1)
EDGE_THRESHOLD = 0.5    # |alignment_score| > this → add edge
CLASSIFY_BATCH_SIZE = 32   # pairs per GPU batch (fits T4 16GB at max_length=256)
MAX_R2_PAIRS   = 200    # cap on cross-cluster pairs per repo (Round 2)
POLL_INTERVAL  = 30     # seconds between polls for new completed repos

# ---------------------------------------------------------------------------
# Schema additions
# ---------------------------------------------------------------------------

EDGE_LOG_SCHEMA = """
CREATE TABLE IF NOT EXISTS edge_log (
    repo_name   TEXT PRIMARY KEY,
    status      TEXT,   -- 'running', 'done', 'error'
    edges_added INTEGER DEFAULT 0,
    pairs_sent  INTEGER DEFAULT 0,
    error       TEXT,
    started_at  TEXT,
    finished_at TEXT
);
"""

EDGE_SCHEMA = """
CREATE TABLE IF NOT EXISTS belief_edges (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    belief_id_a     INTEGER NOT NULL REFERENCES extracted_beliefs(id),
    belief_id_b     INTEGER NOT NULL REFERENCES extracted_beliefs(id),
    label           TEXT    NOT NULL,
    alignment_score REAL    NOT NULL,
    UNIQUE(belief_id_a, belief_id_b)
);
"""

# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------

def get_conn(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init_edge_tables(conn: sqlite3.Connection):
    conn.executescript(EDGE_LOG_SCHEMA + EDGE_SCHEMA)
    conn.commit()


def get_pending_repos(conn: sqlite3.Connection) -> list[str]:
    """Repos where Stage 1 is done but Stage 2 hasn't started."""
    rows = conn.execute("""
        SELECT p.repo_name FROM preprocess_log p
        WHERE p.status = 'done'
          AND p.repo_name NOT IN (SELECT repo_name FROM edge_log)
        ORDER BY p.finished_at ASC
    """).fetchall()
    return [r["repo_name"] for r in rows]


def get_beliefs_for_repo(conn: sqlite3.Connection, repo_name: str) -> list[dict]:
    rows = conn.execute("""
        SELECT id, statement, evidence, confidence
        FROM extracted_beliefs
        WHERE repo_name = ?
        ORDER BY id
    """, (repo_name,)).fetchall()
    return [dict(r) for r in rows]


def insert_edge(conn: sqlite3.Connection, id_a: int, id_b: int,
                label: str, score: float):
    try:
        conn.execute("""
            INSERT OR IGNORE INTO belief_edges
                (belief_id_a, belief_id_b, label, alignment_score)
            VALUES (?, ?, ?, ?)
        """, (min(id_a, id_b), max(id_a, id_b), label, score))
    except sqlite3.IntegrityError:
        pass


# ---------------------------------------------------------------------------
# Chroma helpers
# ---------------------------------------------------------------------------

def get_chroma_collection(chroma_dir: str):
    import chromadb
    client = chromadb.PersistentClient(path=chroma_dir)
    try:
        return client.get_collection("beliefs")
    except Exception:
        return None


def get_embeddings_for_repo(collection, repo_name: str, belief_ids: list[int]):
    """Return {belief_id: embedding_array} for all beliefs in this repo."""
    if not belief_ids or collection is None:
        return {}
    where = {"repo_name": {"$eq": repo_name}}
    # Chroma stores IDs as strings
    str_ids = [str(bid) for bid in belief_ids]
    try:
        result = collection.get(ids=str_ids, where=where, include=["embeddings"])
        emb_map = {}
        for i, cid in enumerate(result["ids"]):
            emb_map[int(cid)] = np.array(result["embeddings"][i], dtype=np.float32)
        return emb_map
    except Exception as e:
        print(f"    [warn] Chroma get embeddings failed: {e}")
        return {}


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def ann_pairs(belief_ids: list[int], emb_map: dict, top_k: int) -> list[tuple[int, int]]:
    """Round 1: for each belief, find top-K ANN neighbors. Return unique pairs."""
    pairs = set()
    ids = [bid for bid in belief_ids if bid in emb_map]
    if len(ids) < 2:
        return []
    embs = np.stack([emb_map[bid] for bid in ids])  # (N, D)

    # Brute-force dot product (N ≤ ~500 per repo, fast enough)
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1e-9, norms)
    normed = embs / norms
    sim_matrix = normed @ normed.T  # (N, N)

    for i, bid_a in enumerate(ids):
        sims = sim_matrix[i]
        sims[i] = -1  # exclude self
        top_indices = np.argsort(sims)[::-1][:top_k]
        for j in top_indices:
            if sims[j] > 0.3:  # only semantically related pairs
                a, b = min(bid_a, ids[j]), max(bid_a, ids[j])
                pairs.add((a, b))
    return list(pairs)


def cross_cluster_pairs(belief_ids: list[int], emb_map: dict,
                        max_pairs: int) -> list[tuple[int, int]]:
    """Round 2: random pairs with low cosine similarity (cross-cluster).
    Falls back to random pairs when embeddings are unavailable."""
    ids = [bid for bid in belief_ids if bid in emb_map]
    if len(ids) < 2:
        # No embeddings — sample random pairs from all belief_ids
        if len(belief_ids) < 2:
            return []
        sample_ids = belief_ids
        n = min(max_pairs, len(sample_ids) * (len(sample_ids) - 1) // 2)
        pairs = set()
        attempts = 0
        while len(pairs) < n and attempts < n * 10:
            attempts += 1
            a_id, b_id = random.sample(sample_ids, 2)
            pairs.add((min(a_id, b_id), max(a_id, b_id)))
        return list(pairs)

    random.shuffle(ids)
    pairs = set()
    attempts = 0
    max_attempts = max_pairs * 10

    while len(pairs) < max_pairs and attempts < max_attempts:
        attempts += 1
        a_id, b_id = random.sample(ids, 2)
        pair = (min(a_id, b_id), max(a_id, b_id))
        if pair in pairs:
            continue
        sim = cosine_similarity(emb_map[a_id], emb_map[b_id])
        if sim < 0.3:  # low similarity → likely cross-cluster
            pairs.add(pair)

    return list(pairs)


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------

def _hf_headers() -> dict:
    return {"Authorization": f"Bearer {HF_TOKEN}", "Content-Type": "application/json"}


def classify_pairs(pairs: list[tuple[int, int]],
                   id_to_statement: dict[int, str]) -> list[dict]:
    """
    Classify pairs in batches using the endpoint's native batch mode:
      POST {"inputs": [{"belief_a": "...", "belief_b": "..."}, ...]}
    The GPU processes the whole batch in one forward pass — much more efficient
    than one request per pair.
    Returns list of {id_a, id_b, label, alignment_score}.
    """
    results = []
    for batch_start in range(0, len(pairs), CLASSIFY_BATCH_SIZE):
        batch = pairs[batch_start:batch_start + CLASSIFY_BATCH_SIZE]
        payload = [
            {"belief_a": id_to_statement.get(id_a, ""),
             "belief_b": id_to_statement.get(id_b, "")}
            for (id_a, id_b) in batch
        ]
        try:
            resp = requests.post(
                CLASSIFY_URL,
                headers=_hf_headers(),
                json={"inputs": payload},
                timeout=120,
            )
            resp.raise_for_status()
            data = resp.json()   # list of {label, alignment_score, ...}

            for i, (id_a, id_b) in enumerate(batch):
                if i >= len(data):
                    break
                item = data[i]
                results.append({
                    "id_a": id_a,
                    "id_b": id_b,
                    "label": item.get("label", "unrelated").lower(),
                    "alignment_score": float(item.get("alignment_score", 0.0)),
                })
        except Exception as e:
            n = batch_start // CLASSIFY_BATCH_SIZE + 1
            print(f"    [warn] Classify batch {n} failed: {e}")
            time.sleep(5)

    return results


# ---------------------------------------------------------------------------
# Per-repo Stage 2
# ---------------------------------------------------------------------------

def process_repo(repo_name: str, conn: sqlite3.Connection, collection) -> dict:
    """Run Stage 2 for a single repo. Returns summary dict."""
    print(f"\n  [Stage 2] {repo_name}")
    t0 = time.time()

    beliefs = get_beliefs_for_repo(conn, repo_name)
    if not beliefs:
        print(f"    No beliefs found — skipping")
        return {"edges_added": 0, "pairs_sent": 0}

    belief_ids = [b["id"] for b in beliefs]
    id_to_statement = {b["id"]: b["statement"] for b in beliefs}
    print(f"    {len(beliefs)} beliefs")

    # Get embeddings from Chroma
    emb_map = get_embeddings_for_repo(collection, repo_name, belief_ids)
    print(f"    {len(emb_map)} embeddings loaded from Chroma")

    # Round 1: ANN pairs
    r1_pairs = ann_pairs(belief_ids, emb_map, TOP_K)
    print(f"    Round 1: {len(r1_pairs)} ANN pairs")

    # Round 2: cross-cluster pairs
    r2_pairs = cross_cluster_pairs(belief_ids, emb_map, MAX_R2_PAIRS)
    print(f"    Round 2: {len(r2_pairs)} cross-cluster pairs")

    all_pairs = list(set(r1_pairs + r2_pairs))
    print(f"    Total unique pairs: {len(all_pairs)}")

    if not all_pairs:
        return {"edges_added": 0, "pairs_sent": 0}

    # Classify
    classified = classify_pairs(all_pairs, id_to_statement)
    print(f"    Classified {len(classified)} pairs")

    # Store edges above threshold
    edges_added = 0
    for result in classified:
        if abs(result["alignment_score"]) > EDGE_THRESHOLD:
            insert_edge(
                conn,
                result["id_a"], result["id_b"],
                result["label"], result["alignment_score"],
            )
            edges_added += 1

    conn.commit()
    elapsed = time.time() - t0
    print(f"    Edges added: {edges_added} (threshold={EDGE_THRESHOLD}) — {elapsed:.1f}s")
    return {"edges_added": edges_added, "pairs_sent": len(classified)}


# ---------------------------------------------------------------------------
# Main watcher loop
# ---------------------------------------------------------------------------

def main():
    global CLASSIFY_URL
    parser = argparse.ArgumentParser(description="Stage 2: label belief edges")
    parser.add_argument("--db", default="database/results.db")
    parser.add_argument("--chroma-dir", default=CHROMA_DIR)
    parser.add_argument("--classify-url", default=None)
    parser.add_argument("--resume", action="store_true",
                        help="Skip repos already in edge_log")
    parser.add_argument("--poll", type=int, default=POLL_INTERVAL,
                        help="Seconds between polls (default: 30)")
    parser.add_argument("--once", action="store_true",
                        help="Process all currently done repos, then exit")
    args = parser.parse_args()

    CLASSIFY_URL = args.classify_url or CLASSIFY_URL

    db_path = str(Path(args.db).resolve())
    print(f"Stage 2 starting — DB: {db_path}")
    print(f"Classify URL: {CLASSIFY_URL}")

    conn = get_conn(db_path)
    init_edge_tables(conn)

    # Load Chroma
    try:
        collection = get_chroma_collection(args.chroma_dir)
        if collection is None:
            print("[warn] Chroma collection 'beliefs' not found yet — embeddings will be skipped")
    except Exception as e:
        print(f"[warn] Chroma load failed: {e} — ANN pairs will use empty embeddings")
        collection = None

    total_edges = 0
    processed = set()

    while True:
        pending = get_pending_repos(conn)
        new_pending = [r for r in pending if r not in processed]

        if new_pending:
            print(f"\n[poll] {len(new_pending)} new repo(s) ready for Stage 2")
            for repo_name in new_pending:
                # Mark as running
                now = datetime.now(timezone.utc).isoformat()
                conn.execute("""
                    INSERT OR REPLACE INTO edge_log
                        (repo_name, status, started_at)
                    VALUES (?, 'running', ?)
                """, (repo_name, now))
                conn.commit()

                try:
                    summary = process_repo(repo_name, conn, collection)
                    total_edges += summary["edges_added"]
                    now = datetime.now(timezone.utc).isoformat()
                    conn.execute("""
                        UPDATE edge_log
                        SET status='done', edges_added=?, pairs_sent=?, finished_at=?
                        WHERE repo_name=?
                    """, (summary["edges_added"], summary["pairs_sent"], now, repo_name))
                    conn.commit()
                    processed.add(repo_name)

                except Exception as e:
                    print(f"  [error] {repo_name}: {e}")
                    traceback.print_exc()
                    conn.execute("""
                        UPDATE edge_log SET status='error', error=? WHERE repo_name=?
                    """, (str(e), repo_name))
                    conn.commit()
                    processed.add(repo_name)

        else:
            # Check if Stage 1 is fully done too
            stage1_remaining = conn.execute("""
                SELECT COUNT(*) FROM preprocess_log WHERE status != 'done'
            """).fetchone()[0]

            if stage1_remaining == 0:
                stage2_remaining = get_pending_repos(conn)
                if not stage2_remaining:
                    print(f"\n[done] Stage 2 complete. Total edges added: {total_edges}")
                    break

            if args.once:
                print(f"[once] No new repos ready. Exiting.")
                break

            print(f"[poll] No new repos yet — waiting {args.poll}s ...")
            time.sleep(args.poll)


if __name__ == "__main__":
    main()
