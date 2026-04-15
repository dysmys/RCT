"""
logger.py
=========

Shared logging setup for all experiment scripts.

Usage in any script:
    from utils.logger import get_logger
    log = get_logger(__name__)

    log.info("Starting task extraction")
    log.warning("No bug label found, using fallback")
    log.error("Clone failed: %s", e)
    log.debug("Raw git output: %s", output)

Log output:
  - Console  : INFO and above
  - File     : DEBUG and above → logs/<script_name>_<timestamp>.log

Log directory:
  Resolved in order:
    1. $LOG_DIR environment variable
    2. <project_root>/logs/
"""

import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_LOG_DIR = Path(os.environ.get("LOG_DIR", _PROJECT_ROOT / "logs"))


def _ensure_log_dir():
    _LOG_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Formatter
# ---------------------------------------------------------------------------

_CONSOLE_FMT = "%(asctime)s  %(levelname)-8s  %(name)s  %(message)s"
_FILE_FMT    = "%(asctime)s  %(levelname)-8s  %(name)s  [%(filename)s:%(lineno)d]  %(message)s"
_DATE_FMT    = "%Y-%m-%d %H:%M:%S"

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

# One log file per process invocation, named after the entry-point script
_session_start = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
_entry_script  = Path(sys.argv[0]).stem if sys.argv[0] else "experiment"
_log_filename  = f"{_entry_script}_{_session_start}.log"

_file_handler_added = False   # ensure we only add the file handler once


def get_logger(name: str) -> logging.Logger:
    """
    Return a logger with the given name.

    First call also attaches:
      - StreamHandler (INFO+) → stdout
      - FileHandler  (DEBUG+) → logs/<script>_<timestamp>.log
    """
    global _file_handler_added

    logger = logging.getLogger(name)

    if not logger.handlers and not logging.getLogger().handlers:
        logger.setLevel(logging.DEBUG)

        # Console handler
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.INFO)
        ch.setFormatter(logging.Formatter(_CONSOLE_FMT, datefmt=_DATE_FMT))
        logger.addHandler(ch)

        # File handler (first call only — shared across all loggers in session)
        if not _file_handler_added:
            _ensure_log_dir()
            log_path = _LOG_DIR / _log_filename
            fh = logging.FileHandler(log_path, encoding="utf-8")
            fh.setLevel(logging.DEBUG)
            fh.setFormatter(logging.Formatter(_FILE_FMT, datefmt=_DATE_FMT))

            # Attach to root logger so all loggers in this process share the file
            root = logging.getLogger()
            root.setLevel(logging.DEBUG)
            root.addHandler(fh)

            _file_handler_added = True
            logger.info("Logging to %s", log_path)

    return logger
