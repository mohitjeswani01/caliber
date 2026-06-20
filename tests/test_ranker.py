"""Tests for src/caliber/ranker.py — the top-N selection + DQ-grade invariants.

These pin the make-or-break contract (a malformed value = disqualification): the
ranker selects exactly N, sorted desc with candidate_id tie-break, ranks 1..N
contiguous, no dupes — and the HONEYPOT GUARDRAIL fires (raises) before any CSV
could be written when the would-be top-N is >10% honeypots. We drive it with a
tiny pool of hand-built CandidateScores; no model, no pool I/O.
"""

import pytest

from caliber.ranker import (
    HONEYPOT_MAX_FRACTION,
    SUBMISSION_COLUMNS,
    HoneypotGuardrailError,
    SubmissionRow,
    build_submission_rows,
    select_top,
)
from caliber.scorer import CandidateScore


def _cid(n: int) -> str:
    return f"CAND_{n:07d}"


def _score(cid, final, *, is_honeypot=False, reasons=None):
    return CandidateScore(
        candidate_id=cid,
        final_score=final,
        base_score=max(final, 0.0),
        behavioral_mult=1.0,
        is_honeypot=is_honeypot,
        honeypot_reasons=reasons or (["impossible profile"] if is_honeypot else []),
        feature_dict={
            "role_substance": 0.5, "skill_corroboration": 1.0, "experience_band": 0.5,
            "nlp_ir_signal": 0.5, "product_vs_consulting": 1.0, "production_recency": 0.6,
            "tenure_stability": 0.7, "external_validation": 0.5, "location_fit": 0.8,
            "is_honeypot": 1.0 if is_honeypot else 0.0,
        },
        ce_score=None, ce_used=False, semantic_sim=0.4, rrf_score=0.0,
    )


def _real_pool(n, *, base=0.9, step=0.001):
    """n real (non-honeypot) candidates with strictly DEScending scores so the
    expected order is unambiguous; ids ascending alongside scores descending."""
    return {_cid(i): _score(_cid(i), base - i * step) for i in range(1, n + 1)}


# --------------------------------------------------------------------------- #
# Selection: exactly N, sorted desc, ranks contiguous, no dupes.
# --------------------------------------------------------------------------- #
def test_selects_exactly_n_sorted_desc_contiguous():
    pool = _real_pool(120)
    rows = build_submission_rows(pool, top_n=100)

    assert len(rows) == 100
    assert [r.rank for r in rows] == list(range(1, 101))
    assert all(a.score >= b.score for a, b in zip(rows, rows[1:]))
    assert len({r.candidate_id for r in rows}) == 100
    # Top score is the largest in the pool; the 21 weakest are dropped.
    assert rows[0].score == max(round(cs.final_score, 6) for cs in pool.values())


def test_too_few_candidates_raises():
    with pytest.raises(ValueError):
        build_submission_rows(_real_pool(99), top_n=100)


def test_duplicate_candidate_id_in_input_raises():
    a = _score(_cid(1), 0.5)
    b = _score(_cid(1), 0.4)  # same id
    with pytest.raises(ValueError):
        select_top([a, b] + [_score(_cid(i), 0.3) for i in range(2, 12)], top_n=5)


# --------------------------------------------------------------------------- #
# Tie-break: equal scores ⇒ candidate_id ASCENDING (verified on a forced tie).
# --------------------------------------------------------------------------- #
def test_tie_break_candidate_id_ascending():
    # Three candidates share the SAME score; ids fed in DEScending order.
    tied = [_score(_cid(30), 0.5), _score(_cid(20), 0.5), _score(_cid(10), 0.5)]
    filler = [_score(_cid(100 + i), 0.4) for i in range(10)]
    rows = build_submission_rows(tied + filler, top_n=5)

    top3 = [r.candidate_id for r in rows[:3]]
    assert top3 == [_cid(10), _cid(20), _cid(30)]  # ascending despite input order
    # and the submission rule (equal score ⇒ id ascending) holds across the tie.
    for a, b in zip(rows, rows[1:]):
        if a.score == b.score:
            assert a.candidate_id <= b.candidate_id


# --------------------------------------------------------------------------- #
# Honeypot floored upstream → never appears above a real candidate, end to end.
# --------------------------------------------------------------------------- #
def test_floored_honeypot_never_outranks_real_candidate():
    real = _real_pool(100, base=0.5)
    # A few honeypots floored to -1.0 (as the scorer does) mixed in.
    honeypots = {_cid(900 + i): _score(_cid(900 + i), -1.0, is_honeypot=True)
                 for i in range(5)}
    pool = {**real, **honeypots}
    rows = build_submission_rows(pool, top_n=100)

    selected_ids = {r.candidate_id for r in rows}
    assert not (selected_ids & set(honeypots))      # no honeypot selected at all
    assert all(r.score >= 0.0 for r in rows)        # nothing at the -1.0 floor


# --------------------------------------------------------------------------- #
# Honeypot GUARDRAIL: >10% honeypots in the would-be top-N → raises, no rows.
# --------------------------------------------------------------------------- #
def test_honeypot_guardrail_fires_above_threshold():
    # 10 selected, 2 of them honeypots (20% ≥ 10% limit) with HIGH scores so the
    # floor can't save us — proves the guardrail is independent of the floor.
    pool = {_cid(i): _score(_cid(i), 0.9 - i * 0.001) for i in range(1, 9)}
    pool[_cid(900)] = _score(_cid(900), 0.95, is_honeypot=True)
    pool[_cid(901)] = _score(_cid(901), 0.96, is_honeypot=True)
    assert len(pool) == 10

    with pytest.raises(HoneypotGuardrailError):
        build_submission_rows(pool, top_n=10)


def test_honeypot_guardrail_passes_under_threshold():
    # 1 honeypot in 100 = 1% < 10% → allowed (it's high-scored so it IS selected,
    # but the rate is within bounds and rows build fine).
    pool = _real_pool(100, base=0.5)
    # replace one real with a high-scored honeypot at 9 selected-but-fine rate.
    pool[_cid(900)] = _score(_cid(900), 0.99, is_honeypot=True)
    rows = build_submission_rows(pool, top_n=100)
    assert len(rows) == 100
    assert 1.0 / 100 < HONEYPOT_MAX_FRACTION  # sanity on the threshold semantics


# --------------------------------------------------------------------------- #
# Row structure matches the exact submission columns.
# --------------------------------------------------------------------------- #
def test_row_structure_matches_submission_columns():
    assert SUBMISSION_COLUMNS == ("candidate_id", "rank", "score", "reasoning")
    rows = build_submission_rows(_real_pool(100), top_n=100)
    r = rows[0]
    assert isinstance(r, SubmissionRow)
    assert tuple(r.as_dict().keys()) == SUBMISSION_COLUMNS
    assert isinstance(r.candidate_id, str) and isinstance(r.rank, int)
    assert isinstance(r.score, float) and isinstance(r.reasoning, str)
    assert r.reasoning  # grounded note attached


def test_rows_carry_grounded_reasoning():
    rows = build_submission_rows(_real_pool(100), top_n=100)
    # Every row has a non-empty, single-line reasoning string.
    assert all(row.reasoning and "\n" not in row.reasoning for row in rows)


def test_deterministic_two_runs_identical():
    pool = _real_pool(100)
    r1 = build_submission_rows(pool, top_n=100)
    r2 = build_submission_rows(pool, top_n=100)
    assert [x.as_dict() for x in r1] == [x.as_dict() for x in r2]
