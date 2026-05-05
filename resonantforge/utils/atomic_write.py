"""
Atomic file write utilities for the ResonantForge corpus generator.

All writes use a write-to-temp-then-rename pattern so that a process kill or
disk-full condition mid-write never leaves a truncated file behind.  On POSIX
systems os.replace() is atomic when source and destination are on the same
filesystem (which is always true here since we use path.with_suffix('.tmp')).

Public API
----------
- :func:`atomic_write_text`  — write a string to a file atomically
- :func:`atomic_write_jsonl` — write an iterable of serialised JSON lines atomically
- :func:`append_jsonl_line`  — durably append a single JSON line (fsync on close)
- :func:`read_jsonl_robust`  — read JSONL, skipping corrupt trailing lines
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Iterable

logger = logging.getLogger(__name__)


def atomic_write_text(path: Path, content: str) -> None:
    """
    Write *content* to *path* atomically.

    Writes to a sibling temp file first, then renames.  On exception, the temp
    file is deleted and the original path is left untouched (or absent).

    Args:
        path:    Destination path.
        content: String content to write.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        tmp.write_text(content, encoding="utf-8")
        os.replace(tmp, path)
    except Exception:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def atomic_write_jsonl(path: Path, lines: Iterable[str]) -> None:
    """
    Write *lines* (pre-serialised JSON strings) to *path* as NDJSON atomically.

    Each element of *lines* is written as a separate line.  Hashes must be
    computed AFTER calling this function (i.e. by reading the final path), not
    from the in-memory lines list, so that the hash covers exactly what is on disk.

    Args:
        path:  Destination path.
        lines: Iterable of pre-serialised JSON strings (one object per line).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        with tmp.open("w", encoding="utf-8") as fh:
            for line in lines:
                fh.write(line + "\n")
        os.replace(tmp, path)
    except Exception:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def append_jsonl_line(path: Path, line: str) -> None:
    """
    Append a single pre-serialised JSON line to a JSONL file with fsync durability.

    Uses O_APPEND open mode so each call is safe against concurrent writers on
    POSIX (each write is atomic up to PIPE_BUF).  The fsync on close guarantees
    the line is on-disk before the caller proceeds to the next record, making
    the file recoverable after a mid-run interrupt.

    Args:
        path: Destination JSONL file (created if absent).
        line: Pre-serialised JSON string (no newline).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def read_jsonl_robust(path: Path) -> list[dict]:
    """
    Read a JSONL file, skipping lines that fail to parse.

    A crash mid-write can leave a truncated (invalid) JSON fragment as the
    last line.  This reader tolerates that: it logs a warning for each bad
    line and returns only the records that parsed successfully.

    Args:
        path: Path to the JSONL file.

    Returns:
        List of parsed dicts; empty list if the file does not exist.
    """
    if not path.exists():
        return []
    records: list[dict] = []
    for i, raw in enumerate(path.read_text(encoding="utf-8").splitlines()):
        if not raw.strip():
            continue
        try:
            records.append(json.loads(raw))
        except json.JSONDecodeError:
            logger.warning(
                "read_jsonl_robust: skipping unparseable line %d in %s", i + 1, path
            )
    return records
