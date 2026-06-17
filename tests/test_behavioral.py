"""Tests for the behavioral multiplier (src/caliber/behavioral.py).

The multiplier modulates the substance score, so its correctness is about three
properties (CLAUDE.md / STRATEGY.md §6, §3, ARCHITECTURE.md §5):

- BOUNDS: across randomized + adversarial signals it is ALWAYS inside the
  envelope [floor, cap]. No single signal (however absurd) can blow past it.
- ORDERING: an available/engaged candidate -> near the cap; an
  inactive/unresponsive one -> near the floor; an average one -> ~1.0.
- NEUTRALITY of sentinels: github_activity_score == -1 ("no GitHub linked") is
  NOT a penalty — ~65% of the pool has none.
- DETERMINISM: recency keys off config.REFERENCE_DATE, never the wall clock — a
  monkeypatched/exploding clock cannot change the result.
"""

import datetime as dt
import random

import pytest

from caliber import behavioral, config
from caliber.behavioral import behavioral_multiplier
from caliber.schema import parse_candidate

FLOOR = config.BEHAVIORAL_MULTIPLIER_FLOOR  # 0.50
CAP = config.BEHAVIORAL_MULTIPLIER_CAP      # 1.15
REF = dt.date.fromisoformat(config.REFERENCE_DATE)  # 2026-06-16


# --------------------------------------------------------------------------- #
# Builders: full schema-valid records so parse_candidate() succeeds and we drive
# the real typed (ranker) path. Only redrob_signals vary between cases.
# --------------------------------------------------------------------------- #
def _signals(**overrides):
    base = {
        "profile_completeness_score": 0.7, "signup_date": "2020-01-01",
        # ~3 months before the 2026-06-16 snapshot == the neutral recency point.
        "last_active_date": "2026-03-16", "open_to_work_flag": False,
        "profile_views_received_30d": 5, "applications_submitted_30d": 1,
        "recruiter_response_rate": 0.44, "avg_response_time_hours": 24.0,
        "skill_assessment_scores": {}, "connection_count": 200,
        "endorsements_received": 20, "notice_period_days": 30,
        "expected_salary_range_inr_lpa": {"min": 20.0, "max": 40.0},
        "preferred_work_mode": "hybrid", "willing_to_relocate": True,
        "github_activity_score": -1.0, "search_appearance_30d": 5,
        "saved_by_recruiters_30d": 0, "interview_completion_rate": 0.50,
        "offer_acceptance_rate": -1.0, "verified_email": False,
        "verified_phone": False, "linkedin_connected": True,
    }
    base.update(overrides)
    return base


def _cand(cid="C", **signal_overrides):
    rec = {
        "candidate_id": cid,
        "profile": {
            "anonymized_name": "T", "headline": "Engineer", "summary": "",
            "location": "Bangalore", "country": "India",
            "years_of_experience": 6.0, "current_title": "Engineer",
            "current_company": "Acme", "current_company_size": "501-1000",
            "current_industry": "Software",
        },
        "career_history": [], "education": [], "skills": [],
        "certifications": [], "languages": [],
        "redrob_signals": _signals(**signal_overrides),
    }
    return parse_candidate(rec)


# --------------------------------------------------------------------------- #
# Three worked examples: available -> ~cap, inactive -> ~floor, average -> ~1.0.
# --------------------------------------------------------------------------- #
def _available_engaged():
    return _cand(
        "AVAIL",
        last_active_date="2026-06-10",       # ~6 days before snapshot -> very recent
        open_to_work_flag=True,
        recruiter_response_rate=0.9,
        saved_by_recruiters_30d=6,
        interview_completion_rate=0.95,
        profile_completeness_score=0.95,
        github_activity_score=80.0,
        verified_email=True, verified_phone=True,
        notice_period_days=15,
    )


def _inactive_unresponsive():
    return _cand(
        "INACT",
        last_active_date="2024-12-16",       # ~18 months before snapshot -> stale
        open_to_work_flag=False,
        recruiter_response_rate=0.05,
        saved_by_recruiters_30d=0,
        interview_completion_rate=0.10,
        profile_completeness_score=0.30,
        github_activity_score=-1.0,           # no GitHub -> must stay neutral
        verified_email=False, verified_phone=False,
        notice_period_days=180,
    )


def _average():
    # Every signal at its neutral point (defaults of _signals): expect ~1.0.
    return _cand("AVG")


def test_available_candidate_lands_near_cap():
    m = behavioral_multiplier(_available_engaged())
    assert 1.10 <= m <= CAP


def test_inactive_candidate_lands_near_floor():
    m = behavioral_multiplier(_inactive_unresponsive())
    assert FLOOR <= m <= 0.60


def test_average_candidate_lands_near_one():
    m = behavioral_multiplier(_average())
    assert 0.97 <= m <= 1.03


def test_ordering_inactive_below_average_below_available():
    lo = behavioral_multiplier(_inactive_unresponsive())
    mid = behavioral_multiplier(_average())
    hi = behavioral_multiplier(_available_engaged())
    assert lo < mid < hi


# --------------------------------------------------------------------------- #
# github_activity_score == -1 ("no GitHub linked") is NEUTRAL, not a penalty.
# --------------------------------------------------------------------------- #
def test_github_minus_one_is_neutral_equals_neutral_score():
    """-1 (no GitHub) must score the SAME as a linked score sitting at the neutral
    point — i.e. it contributes nothing, neither lift nor penalty."""
    no_github = behavioral_multiplier(_cand(github_activity_score=-1.0))
    neutral_github = behavioral_multiplier(
        _cand(github_activity_score=behavioral._GH_NEUTRAL)
    )
    assert no_github == pytest.approx(neutral_github)


def test_github_minus_one_not_penalised_vs_linked_low_score():
    """A linked-but-dead account (score 0) is allowed a small penalty; "no GitHub"
    (-1) must NOT be treated as that penalty — -1 ranks at or above the linked-low
    candidate."""
    no_github = behavioral_multiplier(_cand(github_activity_score=-1.0))
    linked_low = behavioral_multiplier(_cand(github_activity_score=0.0))
    assert no_github >= linked_low
    # and a healthy linked score should beat "no GitHub".
    linked_high = behavioral_multiplier(_cand(github_activity_score=90.0))
    assert linked_high > no_github


# --------------------------------------------------------------------------- #
# BOUNDS property test: never escapes [FLOOR, CAP], over edge cases + randomized.
# --------------------------------------------------------------------------- #
_EDGE_DATES = [
    None, "", "not-a-date", "1900-01-01", "2026-06-16", "2026-06-15",
    "2030-01-01", "2025-06-16", "2024-06-16",
]
_EDGE_RATES = [-1.0, -5.0, 0.0, 0.44, 0.5, 1.0, 2.0, 1000.0]
_EDGE_INTS = [-100, 0, 1, 5, 30, 90, 365, 100000]
_EDGE_GH = [-1.0, 0.0, 25.0, 70.0, 100.0, -50.0, 1e9]


def test_bounds_on_adversarial_edge_cases():
    """Cartesian-ish sweep of pathological values: still always in-envelope."""
    for ld in _EDGE_DATES:
        for rr in _EDGE_RATES:
            for notice in _EDGE_INTS:
                for gh in _EDGE_GH:
                    c = _cand(
                        last_active_date=ld, recruiter_response_rate=rr,
                        interview_completion_rate=rr, notice_period_days=notice,
                        saved_by_recruiters_30d=notice,
                        profile_completeness_score=rr, github_activity_score=gh,
                        open_to_work_flag=bool(notice % 2),
                        verified_email=bool(gh > 0), verified_phone=bool(rr > 0),
                    )
                    m = behavioral_multiplier(c)
                    assert FLOOR <= m <= CAP, (ld, rr, notice, gh, m)


def test_bounds_on_randomized_signals():
    """5000 seeded-random signal sets (deterministic): never escapes the envelope."""
    rng = random.Random(config.SEED)
    years = ["2023", "2024", "2025", "2026", "2027"]
    for _ in range(5000):
        ld = f"{rng.choice(years)}-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}"
        c = _cand(
            last_active_date=ld,
            recruiter_response_rate=rng.uniform(-1.0, 2.0),
            interview_completion_rate=rng.uniform(-1.0, 2.0),
            profile_completeness_score=rng.uniform(-1.0, 2.0),
            notice_period_days=rng.randint(-50, 5000),
            saved_by_recruiters_30d=rng.randint(-10, 1000),
            github_activity_score=rng.choice([-1.0, rng.uniform(-50, 1000)]),
            open_to_work_flag=rng.random() < 0.5,
            verified_email=rng.random() < 0.5,
            verified_phone=rng.random() < 0.5,
        )
        m = behavioral_multiplier(c)
        assert FLOOR <= m <= CAP, (ld, m)


# --------------------------------------------------------------------------- #
# DETERMINISM: recency uses config.REFERENCE_DATE, never the wall clock.
# --------------------------------------------------------------------------- #
def test_recency_uses_reference_date_recent_beats_year_old():
    """A candidate active just before the snapshot scores higher than one active a
    year earlier — recency is measured against REFERENCE_DATE."""
    recent = behavioral_multiplier(_cand(last_active_date="2026-06-15"))
    year_old = behavioral_multiplier(_cand(last_active_date="2025-06-15"))
    assert recent > year_old


def test_multiplier_independent_of_wall_clock(monkeypatch):
    """Make dt.date.today() explode: if recency math touched the wall clock the
    call would error. The result must come purely from REFERENCE_DATE."""
    class _ExplodingDate(dt.date):
        @classmethod
        def today(cls):
            raise AssertionError("behavioral multiplier must not read the wall clock")

    before = behavioral_multiplier(_available_engaged())
    monkeypatch.setattr("caliber.behavioral.dt.date", _ExplodingDate)
    after = behavioral_multiplier(_available_engaged())
    assert before == after


def test_reference_date_drives_recency(monkeypatch):
    """Move REFERENCE_DATE and the same last_active_date is re-scored relative to
    it — proving the pivot is the configured snapshot, not an absolute clock."""
    c = _cand(last_active_date="2026-06-15")
    # Snapshot 2026-06-16: active 1 day before -> very recent -> a recency lift.
    fresh = behavioral_multiplier(c)
    # Advance the snapshot two years: the SAME date is now ~2 years stale.
    monkeypatch.setattr(config, "REFERENCE_DATE", "2028-06-16")
    stale = behavioral_multiplier(c)
    assert fresh > stale
