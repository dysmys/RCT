#!/usr/bin/env python3
"""
stage4_explore.py — Multi-language heuristic code belief extraction.

Walks source files in the repository, extracts class/function definitions,
docstrings/comments, and structural patterns, then stores them as beliefs
in the .dyssonance belief store. No LLM or API key required.

Supported languages:
  Python (.py)              — AST-based (docstrings, type hints, base classes)
  TypeScript (.ts, .tsx)    — regex (JSDoc, class, interface, function)
  JavaScript (.js, .jsx)    — regex (JSDoc, class, function)
  Go (.go)                  — regex (doc comments, func, type, struct)
  Java (.java)              — regex (Javadoc, class, method)
  Rust (.rs)                — regex (/// doc comments, struct, impl, fn)
  Ruby (.rb)                — regex (# comments, class, module, def)
  C# (.cs)                  — regex (XML doc, class, method)
  Swift (.swift)            — regex (/// comments, class, struct, func)
  Kotlin (.kt)              — regex (KDoc, class, fun)

Usage:
    python stage4_explore.py [REPO_DIR] [OPTIONS]
    python .dyssonance/stage4_explore.py --dry-run

Options:
    --repo-dir PATH    Repository root (default: CWD)
    --db PATH          Path to belief_results.db (default: .dyssonance/belief_results.db)
    --chroma-dir PATH  Path to chroma_db (default: .dyssonance/chroma_db)
    --max-per-file N   Max beliefs stored per file (default: 15)
    --min-doc-len N    Minimum docstring/comment length to include (default: 15)
    --dry-run          Print beliefs without storing
    --extensions LIST  Comma-separated extensions to process (e.g. py,ts,go)
"""

import ast
import argparse
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DEFAULT_DB       = os.environ.get("BELIEF_DB",         ".dyssonance/belief_results.db")
DEFAULT_CHROMA   = os.environ.get("BELIEF_CHROMA_DIR", ".dyssonance/chroma_db")

SKIP_DIRS = {
    ".git", ".dyssonance", ".seng", "__pycache__", "node_modules",
    "vendor", "venv", ".venv", "env", "dist", "build", ".next",
    "target", "bin", "obj", ".gradle", "Pods", ".terraform",
}
SKIP_FILE_PATTERNS = [
    r"(^|/)tests?/",                          # test/ or tests/ directories
    r"_test\.(py|go|ts|js|rb|rs|cs|kt|swift)$",
    r"\.test\.(ts|js|tsx|jsx)$",
    r"\.spec\.(ts|js|tsx|jsx)$",
    r"migrations?/",
    r"generated/",
    r"\.pb\.(go|py|ts|js)$",    # protobuf generated
    r"setup\.py$",
    r"conftest\.py$",
]

LANGUAGE_EXTENSIONS = {
    "python":     [".py"],
    "typescript": [".ts", ".tsx"],
    "javascript": [".js", ".jsx"],
    "go":         [".go"],
    "java":       [".java"],
    "rust":       [".rs"],
    "ruby":       [".rb"],
    "csharp":     [".cs"],
    "swift":      [".swift"],
    "kotlin":     [".kt"],
}

EXT_TO_LANG = {
    ext: lang
    for lang, exts in LANGUAGE_EXTENSIONS.items()
    for ext in exts
}


# ---------------------------------------------------------------------------
# Belief data class
# ---------------------------------------------------------------------------

@dataclass
class CodeBelief:
    statement:  str
    evidence:   str          # "rel/path/to/file.py:ClassName.method_name"
    confidence: str          # high | medium | low


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _clean(text: str) -> str:
    """Collapse whitespace and strip."""
    return re.sub(r"\s+", " ", text).strip()


def _preceding_comment(lines: List[str], def_line_idx: int,
                        single_prefix: str, block_start: str, block_end: str,
                        max_lookback: int = 10) -> str:
    """
    Walk backwards from def_line_idx to collect the nearest comment block.
    Works for C-style (// and /* */), Python (#), Ruby (#), etc.
    """
    i = def_line_idx - 1
    comment_lines = []

    # Skip blank lines immediately above definition
    while i >= 0 and not lines[i].strip():
        i -= 1

    if i < 0:
        return ""

    # Block comment (/** ... */ or /* ... */)
    if block_end and lines[i].strip().endswith(block_end.strip()):
        end_i = i
        while i >= 0 and block_start not in lines[i]:
            comment_lines.insert(0, lines[i])
            i -= 1
        if i >= 0:
            comment_lines.insert(0, lines[i])
        raw = " ".join(comment_lines)
        # Strip /** */ markers and * prefixes
        raw = re.sub(r"/\*+\*?|/\*|\*/|\*", " ", raw)
        return _clean(raw)

    # Single-line comments stacked above
    while i >= 0 and i >= def_line_idx - max_lookback:
        stripped = lines[i].strip()
        if stripped.startswith(single_prefix):
            comment_lines.insert(0, stripped.lstrip(single_prefix).strip())
            i -= 1
        else:
            break

    return _clean(" ".join(comment_lines))


# ---------------------------------------------------------------------------
# Python extractor (AST-based)
# ---------------------------------------------------------------------------

def _extract_python(source: str, rel_path: str, min_doc_len: int) -> List[CodeBelief]:
    beliefs: List[CodeBelief] = []

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return beliefs

    module_doc = ast.get_docstring(tree)
    if module_doc and len(module_doc) >= min_doc_len:
        beliefs.append(CodeBelief(
            statement  = f"Module {rel_path}: {_clean(module_doc[:300])}",
            evidence   = rel_path,
            confidence = "high",
        ))

    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            doc = ast.get_docstring(node) or ""
            bases = []
            for b in node.bases:
                if isinstance(b, ast.Name):
                    bases.append(b.id)
                elif isinstance(b, ast.Attribute):
                    bases.append(f"{b.attr}")

            stmt = f"{node.name}"
            if bases:
                stmt += f" (extends {', '.join(bases)})"
            if doc and len(doc) >= min_doc_len:
                stmt += f": {_clean(doc[:250])}"
            elif not doc:
                continue  # skip undocumented classes — too noisy

            beliefs.append(CodeBelief(
                statement  = stmt,
                evidence   = f"{rel_path}:{node.name}",
                confidence = "high" if doc else "medium",
            ))

            # Methods with docstrings
            for item in node.body:
                if not isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue
                if item.name.startswith("__") and item.name != "__init__":
                    continue
                mdoc = ast.get_docstring(item) or ""
                if not mdoc or len(mdoc) < min_doc_len:
                    continue
                args = [a.arg for a in item.args.args if a.arg != "self"]
                ret  = ""
                if item.returns and isinstance(item.returns, ast.Name):
                    ret = f" -> {item.returns.id}"
                elif item.returns and isinstance(item.returns, ast.Constant):
                    ret = f" -> {item.returns.value}"

                beliefs.append(CodeBelief(
                    statement  = f"{node.name}.{item.name}({', '.join(args)}){ret}: {_clean(mdoc[:200])}",
                    evidence   = f"{rel_path}:{node.name}.{item.name}",
                    confidence = "high",
                ))

        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            # Module-level functions only
            if node.col_offset != 0:
                continue
            if node.name.startswith("_"):
                continue
            doc = ast.get_docstring(node) or ""
            if not doc or len(doc) < min_doc_len:
                continue
            args = [a.arg for a in node.args.args]
            beliefs.append(CodeBelief(
                statement  = f"{node.name}({', '.join(args)}): {_clean(doc[:200])}",
                evidence   = f"{rel_path}:{node.name}",
                confidence = "high",
            ))

    return beliefs


# ---------------------------------------------------------------------------
# Generic regex extractor for C-style languages (TS, JS, Java, C#, Swift, Kotlin)
# ---------------------------------------------------------------------------

def _extract_cstyle(
    source: str, rel_path: str, min_doc_len: int,
    class_re: str, func_re: str,
    single_prefix: str = "//",
    block_start: str = "/*", block_end: str = "*/",
) -> List[CodeBelief]:
    beliefs: List[CodeBelief] = []
    lines   = source.splitlines()

    for i, line in enumerate(lines):
        # Class / interface / struct definitions
        cm = re.search(class_re, line)
        if cm:
            name    = cm.group(1)
            comment = _preceding_comment(lines, i, single_prefix, block_start, block_end)
            if comment and len(comment) >= min_doc_len:
                beliefs.append(CodeBelief(
                    statement  = f"{name}: {comment[:300]}",
                    evidence   = f"{rel_path}:{name}",
                    confidence = "high",
                ))
            elif not comment:
                beliefs.append(CodeBelief(
                    statement  = f"{rel_path} defines {name}",
                    evidence   = f"{rel_path}:{name}",
                    confidence = "low",
                ))
            continue

        # Function / method definitions
        fm = re.search(func_re, line)
        if fm:
            name    = fm.group(1)
            comment = _preceding_comment(lines, i, single_prefix, block_start, block_end)
            if not comment or len(comment) < min_doc_len:
                continue
            beliefs.append(CodeBelief(
                statement  = f"{name}(): {comment[:250]}",
                evidence   = f"{rel_path}:{name}",
                confidence = "high",
            ))

    return beliefs


# ---------------------------------------------------------------------------
# Go extractor
# ---------------------------------------------------------------------------

def _extract_go(source: str, rel_path: str, min_doc_len: int) -> List[CodeBelief]:
    beliefs: List[CodeBelief] = []
    lines   = source.splitlines()

    # Package-level doc comment
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("package "):
            comment = _preceding_comment(lines, i, "//", "/*", "*/")
            if comment and len(comment) >= min_doc_len:
                beliefs.append(CodeBelief(
                    statement  = f"Package {rel_path}: {comment[:300]}",
                    evidence   = rel_path,
                    confidence = "high",
                ))
            break

    type_re = re.compile(r"^type\s+(\w+)\s+(struct|interface)")
    func_re = re.compile(r"^func\s+(?:\(\w+\s+\*?\w+\)\s+)?(\w+)\s*\(")

    for i, line in enumerate(lines):
        tm = type_re.match(line)
        if tm:
            name    = tm.group(1)
            kind    = tm.group(2)
            comment = _preceding_comment(lines, i, "//", "/*", "*/")
            stmt    = f"{name} ({kind})"
            if comment and len(comment) >= min_doc_len:
                stmt += f": {comment[:300]}"
                conf = "high"
            else:
                conf = "low"
            beliefs.append(CodeBelief(statement=stmt, evidence=f"{rel_path}:{name}", confidence=conf))
            continue

        fm = func_re.match(line)
        if fm:
            name    = fm.group(1)
            comment = _preceding_comment(lines, i, "//", "/*", "*/")
            if not comment or len(comment) < min_doc_len:
                continue
            beliefs.append(CodeBelief(
                statement  = f"{name}(): {comment[:250]}",
                evidence   = f"{rel_path}:{name}",
                confidence = "high",
            ))

    return beliefs


# ---------------------------------------------------------------------------
# Rust extractor (/// doc comments)
# ---------------------------------------------------------------------------

def _extract_rust(source: str, rel_path: str, min_doc_len: int) -> List[CodeBelief]:
    beliefs: List[CodeBelief] = []
    lines   = source.splitlines()

    struct_re = re.compile(r"^pub\s+(?:struct|enum|trait|type)\s+(\w+)")
    fn_re     = re.compile(r"^pub\s+(?:async\s+)?fn\s+(\w+)")

    for i, line in enumerate(lines):
        for pattern, kind in [(struct_re, "type"), (fn_re, "fn")]:
            m = pattern.match(line)
            if not m:
                continue
            name = m.group(1)
            # Collect /// lines above
            j = i - 1
            doc_lines = []
            while j >= 0 and lines[j].strip().startswith("///"):
                doc_lines.insert(0, lines[j].strip().lstrip("/").strip())
                j -= 1
            doc = _clean(" ".join(doc_lines))
            if not doc or len(doc) < min_doc_len:
                continue
            beliefs.append(CodeBelief(
                statement  = f"{name}: {doc[:300]}",
                evidence   = f"{rel_path}:{name}",
                confidence = "high",
            ))

    return beliefs


# ---------------------------------------------------------------------------
# Ruby extractor
# ---------------------------------------------------------------------------

def _extract_ruby(source: str, rel_path: str, min_doc_len: int) -> List[CodeBelief]:
    return _extract_cstyle(
        source, rel_path, min_doc_len,
        class_re    = r"^\s*(?:class|module)\s+(\w+)",
        func_re     = r"^\s*def\s+(\w+)",
        single_prefix = "#",
        block_start = "=begin", block_end = "=end",
    )


# ---------------------------------------------------------------------------
# Language dispatch table
# ---------------------------------------------------------------------------

def _extract_beliefs(source: str, rel_path: str, lang: str, min_doc_len: int) -> List[CodeBelief]:
    if lang == "python":
        return _extract_python(source, rel_path, min_doc_len)

    if lang == "go":
        return _extract_go(source, rel_path, min_doc_len)

    if lang == "rust":
        return _extract_rust(source, rel_path, min_doc_len)

    if lang == "ruby":
        return _extract_ruby(source, rel_path, min_doc_len)

    if lang in ("typescript", "javascript"):
        return _extract_cstyle(
            source, rel_path, min_doc_len,
            class_re = r"(?:^|\s)(?:export\s+)?(?:abstract\s+)?(?:class|interface)\s+(\w+)",
            func_re  = r"(?:export\s+)?(?:async\s+)?function\s+(\w+)|(?:export\s+const\s+(\w+)\s*=\s*(?:async\s+)?\(?)",
        )

    if lang == "java":
        return _extract_cstyle(
            source, rel_path, min_doc_len,
            class_re = r"(?:public\s+)?(?:abstract\s+)?(?:class|interface|enum)\s+(\w+)",
            func_re  = r"(?:public|protected)\s+(?:static\s+)?(?:\w+(?:<[^>]+>)?)\s+(\w+)\s*\(",
        )

    if lang == "csharp":
        return _extract_cstyle(
            source, rel_path, min_doc_len,
            class_re = r"(?:public|internal|private)?\s*(?:partial\s+)?(?:class|interface|struct|enum)\s+(\w+)",
            func_re  = r"(?:public|protected|private|internal)\s+(?:static\s+)?(?:async\s+)?(?:\w+(?:<[^>]+>)?)\s+(\w+)\s*\(",
            single_prefix = "///",
        )

    if lang == "swift":
        return _extract_cstyle(
            source, rel_path, min_doc_len,
            class_re = r"(?:public\s+|open\s+)?(?:class|struct|protocol|enum)\s+(\w+)",
            func_re  = r"(?:public\s+|open\s+)?(?:override\s+)?func\s+(\w+)",
            single_prefix = "///",
        )

    if lang == "kotlin":
        return _extract_cstyle(
            source, rel_path, min_doc_len,
            class_re = r"(?:data\s+|sealed\s+|open\s+)?(?:class|interface|object)\s+(\w+)",
            func_re  = r"(?:fun\s+)(\w+)\s*\(",
        )

    return []


# ---------------------------------------------------------------------------
# File walker
# ---------------------------------------------------------------------------

def _should_skip(rel_path: str) -> bool:
    for pattern in SKIP_FILE_PATTERNS:
        if re.search(pattern, rel_path):
            return True
    return False


def walk_repo(repo_root: Path, allowed_exts: Optional[set]) -> List[Tuple[Path, str, str]]:
    """Yield (abs_path, rel_path, language) for all source files."""
    results = []
    for dirpath, dirnames, filenames in os.walk(repo_root):
        # Prune skip dirs in-place
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS and not d.startswith(".")]

        for fname in filenames:
            ext  = Path(fname).suffix.lower()
            lang = EXT_TO_LANG.get(ext)
            if not lang:
                continue
            if allowed_exts and ext not in allowed_exts:
                continue
            abs_path = Path(dirpath) / fname
            rel_path = str(abs_path.relative_to(repo_root))
            if _should_skip(rel_path):
                continue
            results.append((abs_path, rel_path, lang))

    return results


# ---------------------------------------------------------------------------
# Store beliefs (reuse store.py logic inline to avoid subprocess overhead)
# ---------------------------------------------------------------------------

def _store_belief(statement: str, evidence: str, confidence: str,
                  db_path: str, chroma_dir: str, repo_name: str,
                  embedder, chroma_collection) -> bool:
    """Insert into SQLite + Chroma. Returns True if newly stored."""
    import sqlite3
    source   = "code"
    chroma_id = f"{repo_name}_{source}_{abs(hash(statement)) % 10**9}"

    conn = sqlite3.connect(db_path, timeout=30)
    conn.execute("PRAGMA journal_mode=WAL")

    conn.execute("INSERT OR IGNORE INTO repos(name) VALUES(?)", (repo_name,))

    existing = conn.execute(
        "SELECT id FROM extracted_beliefs WHERE chroma_id = ?", (chroma_id,)
    ).fetchone()
    if existing:
        conn.close()
        return False

    cur = conn.execute("""
        INSERT INTO extracted_beliefs
            (repo_name, source, statement, evidence, confidence,
             chunk_index, commit_timestamp, chroma_id)
        VALUES (?, ?, ?, ?, ?, NULL, NULL, ?)
    """, (repo_name, source, statement, evidence, confidence, chroma_id))
    belief_id = cur.lastrowid
    conn.commit()
    conn.close()

    embedding = embedder.encode([statement], normalize_embeddings=True)[0].tolist()
    chroma_collection.upsert(
        ids         = [str(belief_id)],
        embeddings  = [embedding],
        documents   = [statement],
        metadatas   = [{
            "repo_name":        repo_name,
            "repo_url":         "",
            "source":           source,
            "evidence":         evidence[:500],
            "confidence":       confidence,
            "commit_timestamp": "",
        }],
    )
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Multi-language heuristic code belief extraction for .dyssonance store."
    )
    parser.add_argument("repo_dir", nargs="?", default=".",
                        help="Repository root directory (default: CWD)")
    parser.add_argument("--db",           default=DEFAULT_DB)
    parser.add_argument("--chroma-dir",   default=DEFAULT_CHROMA)
    parser.add_argument("--max-per-file", type=int, default=15,
                        help="Max beliefs stored per file (default: 15)")
    parser.add_argument("--min-doc-len",  type=int, default=15,
                        help="Minimum docstring/comment length (default: 15)")
    parser.add_argument("--dry-run",      action="store_true",
                        help="Print beliefs without storing")
    parser.add_argument("--extensions",   default=None,
                        help="Comma-separated extensions to process (e.g. py,ts,go)")
    parser.add_argument("--repo-name",    default=None,
                        help="Override repo name (default: directory name)")
    args = parser.parse_args()

    repo_root  = Path(args.repo_dir).resolve()
    repo_name  = args.repo_name if args.repo_name else repo_root.name

    allowed_exts = None
    if args.extensions:
        allowed_exts = {"." + e.lstrip(".") for e in args.extensions.split(",")}

    if not args.dry_run:
        if not Path(args.db).exists():
            print(f"[stage4] ERROR: DB not found at {args.db}. Run build_beliefs.sh first.",
                  file=sys.stderr)
            sys.exit(1)

        print("[stage4] Loading embedding model...", flush=True)
        from sentence_transformers import SentenceTransformer
        embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

        import chromadb
        chroma_client = chromadb.PersistentClient(path=args.chroma_dir)
        collection    = chroma_client.get_or_create_collection(
            name="beliefs", metadata={"hnsw:space": "cosine"}
        )
    else:
        embedder   = None
        collection = None

    files = walk_repo(repo_root, allowed_exts)
    print(f"[stage4] Found {len(files)} source files to scan.", flush=True)

    total_stored = 0
    total_skipped = 0

    for abs_path, rel_path, lang in files:
        try:
            source = abs_path.read_text(errors="replace")
        except OSError:
            continue

        beliefs = _extract_beliefs(source, rel_path, lang, args.min_doc_len)
        # Sort by confidence desc, cap per file
        beliefs.sort(key=lambda b: {"high": 0, "medium": 1, "low": 2}[b.confidence])
        beliefs = beliefs[:args.max_per_file]

        for b in beliefs:
            if args.dry_run:
                print(f"[{b.confidence}] {b.statement[:100]}")
                print(f"  Evidence: {b.evidence}")
                total_stored += 1
            else:
                stored = _store_belief(
                    statement  = b.statement,
                    evidence   = b.evidence,
                    confidence = b.confidence,
                    db_path    = args.db,
                    chroma_dir = args.chroma_dir,
                    repo_name  = repo_name,
                    embedder   = embedder,
                    chroma_collection = collection,
                )
                if stored:
                    total_stored  += 1
                else:
                    total_skipped += 1

        if beliefs:
            print(f"  {rel_path} ({lang}) → {len(beliefs)} beliefs", flush=True)

    print(f"\n[stage4] Done. {total_stored} beliefs stored, {total_skipped} already existed.",
          flush=True)


if __name__ == "__main__":
    main()
