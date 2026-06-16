"""Memory-safe access to the candidate pool (ARCHITECTURE.md §3).

The pool is ~465 MB / 100K records, so nothing here ever loads the whole file
into memory: every entry point streams ``candidates.jsonl`` one line at a time.

Two layers, intentionally:

- ``stream_raw`` yields raw dicts — the **fast path** for the offline encode
  loop, which only needs ``candidate_to_text`` (no typed object per row).
- ``stream_candidates`` wraps ``stream_raw`` with ``parse_candidate`` and yields
  typed :class:`~caliber.schema.Candidate` objects for the scoring modules.

This is the **canonical** streamer for the project. (``scripts/precompute.py``
still has its own ``stream_records`` from before this module existed; it should
later be refactored to call ``stream_raw`` so there is a single reader — flagged,
not changed here.)

File order is preserved exactly: it is the join key between ``candidate_emb.npy``,
``candidate_ids.npy`` and the FAISS index. No network. Stdlib only.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import IO, Iterator, Union

from . import config
from .schema import Candidate, parse_candidate

PathLike = Union[str, Path]


def _open_text(path: Path) -> IO[str]:
    """Open ``path`` as a UTF-8 text stream, transparently gunzipping ``.gz``.

    The official bundle ships ``candidates.jsonl.gz``; our local working copy is
    plain ``candidates.jsonl``. Detect by suffix so the same code reads both.
    Both ``open`` and ``gzip.open`` here are lazy/streaming — neither pulls the
    whole file into memory.
    """
    if path.suffix == ".gz":
        return gzip.open(path, mode="rt", encoding="utf-8")
    return open(path, mode="r", encoding="utf-8")


def stream_raw(path: PathLike = config.CANDIDATES_PATH) -> Iterator[dict]:
    """Yield raw candidate dicts from a JSONL file, one decoded line at a time.

    Memory-safe: only one line is held at once. Blank lines are skipped so a
    trailing newline never produces an empty record. Transparently reads
    ``.jsonl`` and ``.jsonl.gz``.
    """
    path = Path(path)
    with _open_text(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def stream_candidates(path: PathLike = config.CANDIDATES_PATH) -> Iterator[Candidate]:
    """Yield typed :class:`Candidate` objects, preserving file order.

    Same memory profile as :func:`stream_raw` (one record at a time) — it just
    parses each dict into the canonical typed structure for the scoring code.
    """
    for rec in stream_raw(path):
        yield parse_candidate(rec)


def load_all_ids(path: PathLike = config.CANDIDATES_PATH) -> list[str]:
    """Return every ``candidate_id`` in file order (the canonical join key).

    Streams the file and keeps only the ids, so it is cheap even on the full
    100K pool — it never materialises the records themselves.
    """
    return [rec["candidate_id"] for rec in stream_raw(path)]


def load_sample(
    n: int, path: PathLike = config.CANDIDATES_PATH, typed: bool = False
) -> list:
    """Return the first ``n`` records for quick inspection or tests.

    Raw dicts by default; typed :class:`Candidate` objects when ``typed=True``.
    The underlying stream is lazy and we stop after ``n`` records, so this does
    not scan the whole file.
    """
    source = stream_candidates(path) if typed else stream_raw(path)
    out = []
    for rec in source:
        if len(out) >= n:
            break
        out.append(rec)
    return out
