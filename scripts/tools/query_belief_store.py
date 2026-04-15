#!/usr/bin/env python3
"""
query_belief_store.py
=====================

Treatment-arm belief retrieval tool. Queries the pre-processed belief store
(SQLite) using sentence-transformers + numpy cosine similarity — no ChromaDB
dependency, so no Rust-binding segfaults.

Given a repo name, a natural-language query (task description + relevant files),
and a belief cutoff timestamp, returns the top-K most relevant beliefs for that
repo along with their supporting belief chain.

Usage (agent calls this from inside a worktree):
    python3 .seng/query_belief_store.py \\
        --repo django \\
        --query "authentication middleware session token handling" \\
        --cutoff 2025-06-15T10:00:00Z \\
        --top-k 5

Environment variables (or pass as CLI args):
    BELIEF_DB  -- path to belief_results.db  (default: /opt/experiment/belief_results.db)
"""

import argparse
import json
import os
import sqlite3
import sys
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

_LOCAL_DB = Path(".dyssonance/belief_results.db")
DEFAULT_DB = os.environ.get(
    "BELIEF_DB",
    str(_LOCAL_DB) if _LOCAL_DB.exists() else "/opt/experiment/belief_results.db"
)
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EDGE_THRESHOLD = 0.7
MAX_SUPPORT_LINKS = 3


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class Belief:
    id: int
    statement: str
    evidence: str
    confidence: str
    commit_timestamp: str
    similarity: float = 0.0
    support_chain: List["Belief"] = field(default_factory=list)


# ---------------------------------------------------------------------------
# BeliefStore — SQLite + sentence-transformers (no ChromaDB)
# ---------------------------------------------------------------------------

class BeliefStore:
    def __init__(self, db_path: str):
        self._conn = sqlite3.connect(db_path)
        self._conn.row_factory = sqlite3.Row
        self._embed = self._load_embedder()

    def _load_embedder(self):
        try:
            from sentence_transformers import SentenceTransformer
            return SentenceTransformer(EMBED_MODEL)
        except Exception as e:
            print(f"[query_belief_store] WARNING: could not load embedder: {e}", file=sys.stderr)
            return None

    def query(
        self,
        repo_name: str,
        query_text: str,
        cutoff: Optional[str],
        top_k: int = 5,
    ) -> List[Belief]:
        """Return top-K beliefs for repo_name semantically similar to query_text."""
        import numpy as np

        # Fetch candidate beliefs from SQLite
        sql = """
            SELECT id, statement, evidence, confidence, commit_timestamp
            FROM extracted_beliefs
            WHERE repo_name = ?
        """
        params: list = [repo_name]

        if cutoff:
            # Try normalized repo_name matching too (handle org/repo vs repo)
            sql += " AND commit_timestamp <= ?"
            params.append(cutoff)

        rows = self._conn.execute(sql, params).fetchall()

        # Also try short repo name if the full name returned nothing
        if not rows and "/" in repo_name:
            short = repo_name.split("/")[-1]
            params2: list = [short]
            sql2 = "SELECT id, statement, evidence, confidence, commit_timestamp FROM extracted_beliefs WHERE repo_name = ?"
            if cutoff:
                sql2 += " AND commit_timestamp <= ?"
                params2.append(cutoff)
            rows = self._conn.execute(sql2, params2).fetchall()

        if not rows:
            return []

        statements = [r["statement"] for r in rows]

        if self._embed is None:
            # Fallback: return first top_k beliefs without ranking
            return [
                Belief(
                    id=rows[i]["id"],
                    statement=rows[i]["statement"],
                    evidence=rows[i]["evidence"] or "",
                    confidence=rows[i]["confidence"] or "medium",
                    commit_timestamp=rows[i]["commit_timestamp"] or "",
                    similarity=0.5,
                )
                for i in range(min(top_k, len(rows)))
            ]

        # Encode query + all beliefs
        all_texts = [query_text] + statements
        embeddings = self._embed.encode(all_texts, show_progress_bar=False, normalize_embeddings=True)

        query_emb = embeddings[0]
        belief_embs = embeddings[1:]

        # Cosine similarity (already normalized, so just dot product)
        scores = np.dot(belief_embs, query_emb)

        # Rank and return top_k
        top_indices = np.argsort(scores)[::-1][:top_k]

        beliefs = []
        for idx in top_indices:
            row = rows[idx]
            b = Belief(
                id=row["id"],
                statement=row["statement"],
                evidence=row["evidence"] or "",
                confidence=row["confidence"] or "medium",
                commit_timestamp=row["commit_timestamp"] or "",
                similarity=float(scores[idx]),
            )
            b.support_chain = self._get_support_chain(b.id, cutoff)
            beliefs.append(b)

        return beliefs

    def _get_support_chain(self, belief_id: int, cutoff: Optional[str]) -> List[Belief]:
        """Load supporting beliefs via belief_edges table."""
        try:
            sql = """
                SELECT eb.id, eb.statement, eb.confidence, eb.commit_timestamp,
                       be.alignment_score
                FROM belief_edges be
                JOIN extracted_beliefs eb
                     ON (be.belief_id_b = eb.id AND be.belief_id_a = ?)
                     OR (be.belief_id_a = eb.id AND be.belief_id_b = ?)
                WHERE be.label = 'support'
                  AND ABS(be.alignment_score) >= ?
            """
            params: list = [belief_id, belief_id, EDGE_THRESHOLD]
            if cutoff:
                sql += " AND eb.commit_timestamp <= ?"
                params.append(cutoff)
            sql += f" ORDER BY be.alignment_score DESC LIMIT {MAX_SUPPORT_LINKS}"
            rows = self._conn.execute(sql, params).fetchall()
            return [
                Belief(
                    id=r["id"],
                    statement=r["statement"],
                    evidence="",
                    confidence=r["confidence"] or "medium",
                    commit_timestamp=r["commit_timestamp"] or "",
                    similarity=round(float(r["alignment_score"]), 4),
                )
                for r in rows
            ]
        except Exception:
            return []


# ---------------------------------------------------------------------------
# Formatter
# ---------------------------------------------------------------------------

def format_beliefs(beliefs: List[Belief]) -> str:
    if not beliefs:
        return "No relevant beliefs found for this query.\n"

    lines = []
    for i, b in enumerate(beliefs, 1):
        lines.append(f"## Belief {i}  [{b.confidence} confidence | similarity {b.similarity:.3f}]")
        lines.append(b.statement)
        if b.evidence:
            lines.append(f"Evidence: {b.evidence}")
        if b.commit_timestamp:
            lines.append(f"Observed: {b.commit_timestamp[:10]}")
        if b.support_chain:
            lines.append("Supporting beliefs:")
            for s in b.support_chain:
                lines.append(f"  • {s.statement}")
        lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Call logger
# ---------------------------------------------------------------------------

def log_call(repo_name: str, query: str, cutoff: str, beliefs: List[Belief],
             seng_dir_override: Optional[str] = None):
    seng_dir = Path(seng_dir_override) if seng_dir_override else Path.cwd() / ".seng"
    seng_dir.mkdir(parents=True, exist_ok=True)
    log_file = seng_dir / "belief_calls.jsonl"

    seq = sum(1 for _ in log_file.open()) if log_file.exists() else 0
    record = {
        "seq": seq + 1,
        "repo": repo_name,
        "query": query,
        "cutoff": cutoff,
        "beliefs": [
            {
                "id": b.id,
                "statement": b.statement,
                "confidence": b.confidence,
                "similarity": b.similarity,
                "support_chain": [s.statement for s in b.support_chain],
            }
            for b in beliefs
        ],
        "called_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    with log_file.open("a") as f:
        f.write(json.dumps(record) + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Query pre-processed belief store for treatment arm context."
    )
    parser.add_argument("--repo", required=True, help="Repository name (e.g. django)")
    parser.add_argument("--query", required=True, help="Natural-language task description")
    parser.add_argument("--cutoff", default=None,
                        help="ISO 8601 cutoff timestamp (beliefs at or before this date)")
    parser.add_argument("--top-k", type=int, default=5, help="Number of beliefs to return")
    parser.add_argument("--db", default=DEFAULT_DB, help="Path to belief_results.db")
    parser.add_argument("--seng-dir", default=None,
                        help="Absolute path to .seng/ dir for writing belief_calls.jsonl "
                             "(defaults to <cwd>/.seng/)")
    # kept for backward-compat; ignored (no longer needed)
    parser.add_argument("--chroma-dir", default=None, help="(ignored — no longer uses ChromaDB)")
    parser.add_argument("--repo-root", default=None, help="(unused)")
    parser.add_argument("--no-snippets", action="store_true", help="(unused)")
    args = parser.parse_args()

    try:
        store = BeliefStore(db_path=args.db)
    except Exception as e:
        print(f"[query_belief_store] ERROR: could not open belief store: {e}", file=sys.stderr)
        sys.exit(1)

    beliefs = store.query(
        repo_name=args.repo,
        query_text=args.query,
        cutoff=args.cutoff,
        top_k=args.top_k,
    )

    log_call(args.repo, args.query, args.cutoff or "", beliefs,
             seng_dir_override=args.seng_dir)

    print(format_beliefs(beliefs))


if __name__ == "__main__":
    main()
