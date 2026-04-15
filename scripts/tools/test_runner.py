"""Shared test runner detection and execution utilities."""

import subprocess
import re
from pathlib import Path


def detect_test_runner(repo_path) -> tuple[str | None, str | None]:
    """
    Detect the test runner for a repository.
    Returns (test_cmd, runner_name) or (None, None).
    """
    repo_path = Path(repo_path)
    checks = [
        (repo_path / "pytest.ini",       "pytest --tb=no -q",              "pytest"),
        (repo_path / "setup.cfg",        "pytest --tb=no -q",              "pytest"),
        (repo_path / "pyproject.toml",   "pytest --tb=no -q",              "pytest"),
        (repo_path / "pom.xml",          "mvn test -q",                    "mvn"),
        (repo_path / "build.gradle",     "./gradlew test --quiet",         "gradle"),
        (repo_path / "build.gradle.kts", "./gradlew test --quiet",         "gradle"),
        (repo_path / "CMakeLists.txt",   "cmake --build build --target test", "cmake"),
    ]
    for marker, cmd, runner in checks:
        if marker.exists():
            return cmd, runner
    if (repo_path / "Makefile").exists():
        r = subprocess.run(["grep", "-q", "^test:", "Makefile"],
                           cwd=str(repo_path), capture_output=True)
        if r.returncode == 0:
            return "make test", "make"
    return None, None


def run_test_suite(repo_path, test_cmd: str,
                   timeout: int = 300) -> tuple[int, int]:
    """
    Run test suite. Returns (passed, failed). Returns (-1, -1) on crash/timeout.
    """
    repo_path = Path(repo_path)
    try:
        result = subprocess.run(
            test_cmd.split(), cwd=str(repo_path),
            capture_output=True, text=True, timeout=timeout
        )
        return parse_test_counts(result.stdout + result.stderr, test_cmd)
    except subprocess.TimeoutExpired:
        return -1, -1
    except Exception:
        return -1, -1


def parse_test_counts(output: str, test_cmd: str) -> tuple[int, int]:
    """Parse (passed, failed) from test runner output."""
    if "pytest" in test_cmd:
        p = re.search(r"(\d+) passed", output)
        f = re.search(r"(\d+) failed", output)
        return int(p.group(1)) if p else 0, int(f.group(1)) if f else 0
    if "mvn" in test_cmd or "gradle" in test_cmd:
        m = re.search(r"Tests run:\s*(\d+),\s*Failures:\s*(\d+)", output)
        if m:
            total, failures = int(m.group(1)), int(m.group(2))
            return total - failures, failures
        return 0, 0
    if "cmake" in test_cmd or "make" in test_cmd:
        m = re.search(r"(\d+)/(\d+) tests passed", output)
        if m:
            passed, total = int(m.group(1)), int(m.group(2))
            return passed, total - passed
        return 0, 0
    return 0, 0
