"""
Atomic file write utilities for the Confabra corpus generator.

All writes use a write-to-temp-then-rename pattern so that a process kill or
disk-full condition mid-write never leaves a truncated file behind.  On POSIX
systems os.replace() is atomic when source and destination are on the same
filesystem (which is always true here since we use path.with_suffix('.tmp')).

Public API
----------
- :func:`atomic_write_text`  — write a string to a file atomically
- :func:`atomic_write_jsonl` — write an iterable of serialised JSON lines atomically
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable


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
