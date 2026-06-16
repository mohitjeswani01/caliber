"""Tests for the canonical candidate streamer (src/caliber/io_utils.py).

Uses small temp .jsonl / .jsonl.gz files so the tests are fast and
self-contained, plus one check against the real sample to keep parsing honest.
"""

import gzip
import json

import pytest

from caliber import config, io_utils
from caliber.schema import Candidate

# Three minimal-but-schema-valid records (file order matters).
_SAMPLE_PATH = config.DATA_DIR / "challenge" / "sample_candidates.json"


def _records(n=3):
    """Borrow real records from the sample array (which is JSON, not JSONL)."""
    recs = json.loads(_SAMPLE_PATH.read_text(encoding="utf-8"))
    return recs[:n]


def _write_jsonl(path, recs, blank_lines=False):
    lines = []
    for r in recs:
        lines.append(json.dumps(r))
        if blank_lines:
            lines.append("")  # exercise blank-line skipping
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


# --- stream_raw -------------------------------------------------------------

def test_stream_raw_order_and_count(tmp_path):
    recs = _records(3)
    p = _write_jsonl(tmp_path / "c.jsonl", recs, blank_lines=True)
    out = list(io_utils.stream_raw(p))
    assert len(out) == 3                       # blank lines skipped
    assert [r["candidate_id"] for r in out] == [r["candidate_id"] for r in recs]
    assert out == recs                         # full dict fidelity, in order


def test_stream_raw_is_lazy_generator(tmp_path):
    p = _write_jsonl(tmp_path / "c.jsonl", _records(3))
    gen = io_utils.stream_raw(p)
    # A generator yields incrementally rather than returning a materialised list.
    assert iter(gen) is gen
    first = next(gen)
    assert first["candidate_id"] == _records(1)[0]["candidate_id"]


# --- stream_candidates ------------------------------------------------------

def test_stream_candidates_types_and_order(tmp_path):
    recs = _records(3)
    p = _write_jsonl(tmp_path / "c.jsonl", recs)
    cands = list(io_utils.stream_candidates(p))
    assert all(isinstance(c, Candidate) for c in cands)
    assert [c.candidate_id for c in cands] == [r["candidate_id"] for r in recs]
    # Order matches stream_raw exactly.
    raw_ids = [r["candidate_id"] for r in io_utils.stream_raw(p)]
    assert [c.candidate_id for c in cands] == raw_ids


# --- load_all_ids -----------------------------------------------------------

def test_load_all_ids_file_order(tmp_path):
    recs = _records(3)
    p = _write_jsonl(tmp_path / "c.jsonl", recs)
    ids = io_utils.load_all_ids(p)
    assert ids == [r["candidate_id"] for r in recs]
    assert ids == [r["candidate_id"] for r in io_utils.stream_raw(p)]


# --- gzip transparency ------------------------------------------------------

def test_gzip_reads_identically(tmp_path):
    recs = _records(3)
    plain = _write_jsonl(tmp_path / "c.jsonl", recs)
    gz = tmp_path / "c.jsonl.gz"
    with gzip.open(gz, "wt", encoding="utf-8") as fh:
        for r in recs:
            fh.write(json.dumps(r) + "\n")
    assert list(io_utils.stream_raw(gz)) == list(io_utils.stream_raw(plain))
    assert io_utils.load_all_ids(gz) == io_utils.load_all_ids(plain)


# --- load_sample ------------------------------------------------------------

def test_load_sample_raw_and_typed(tmp_path):
    recs = _records(3)
    p = _write_jsonl(tmp_path / "c.jsonl", recs)

    raw = io_utils.load_sample(2, path=p)
    assert len(raw) == 2
    assert all(isinstance(r, dict) for r in raw)
    assert [r["candidate_id"] for r in raw] == [r["candidate_id"] for r in recs[:2]]

    typed = io_utils.load_sample(2, path=p, typed=True)
    assert len(typed) == 2
    assert all(isinstance(c, Candidate) for c in typed)
    assert [c.candidate_id for c in typed] == [r["candidate_id"] for r in recs[:2]]


def test_load_sample_caps_at_available(tmp_path):
    recs = _records(3)
    p = _write_jsonl(tmp_path / "c.jsonl", recs)
    # Asking for more than exist returns only what's there (no error).
    assert len(io_utils.load_sample(10, path=p)) == 3
