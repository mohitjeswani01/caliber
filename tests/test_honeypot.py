"""Tests for the canonical honeypot detector (src/caliber/honeypot.py).

Per CLAUDE.md the honeypot detector is required-coverage: ranking >10% honeypots
into our top 100 is a Stage-3 disqualifier, so a regression here is existential.
We verify four things:

- POSITIVES trip and NEGATIVES don't (impossible tenure; >=3 expert-zero skills;
  a clean strong candidate stays clean), with human-readable reasons.
- DETERMINISM: date logic keys off ``config.REFERENCE_DATE``, never the wall
  clock (we make ``date.today`` explode and the verdict is unchanged).
- RECONCILIATION: the eval answer key's ``honeypot_reasons`` and the ranker's
  ``is_honeypot`` return the SAME verdict on crafted candidates — proving there
  is one shared definition, not two that can drift.
"""

import datetime as dt

import pytest

from caliber import config
from caliber.honeypot import honeypot_reasons, is_honeypot
from caliber.schema import parse_candidate
from eval.heuristics import honeypot_reasons as eval_honeypot_reasons

REF = dt.date.fromisoformat(config.REFERENCE_DATE)  # 2026-06-16


# --------------------------------------------------------------------------- #
# Fixtures: full schema-valid records, so they survive parse_candidate() and we
# can exercise the typed (ranker) path as well as the raw-dict (eval) path.
# --------------------------------------------------------------------------- #
def _signals():
    return {
        "profile_completeness_score": 0.9, "signup_date": "2020-01-01",
        "last_active_date": "2026-06-01", "open_to_work_flag": True,
        "profile_views_received_30d": 10, "applications_submitted_30d": 2,
        "recruiter_response_rate": 0.6, "avg_response_time_hours": 12.0,
        "skill_assessment_scores": {}, "connection_count": 300,
        "endorsements_received": 50, "notice_period_days": 30,
        "expected_salary_range_inr_lpa": {"min": 20.0, "max": 40.0},
        "preferred_work_mode": "hybrid", "willing_to_relocate": True,
        "github_activity_score": 40.0, "search_appearance_30d": 5,
        "saved_by_recruiters_30d": 3, "interview_completion_rate": 0.9,
        "offer_acceptance_rate": 0.5, "verified_email": True,
        "verified_phone": True, "linkedin_connected": True,
    }


def _role(company, title, months, desc, current=False,
          start="2018-01-01", end="2020-01-01"):
    return {
        "company": company, "title": title, "start_date": start,
        "end_date": None if current else end, "duration_months": months,
        "is_current": current, "industry": "Software", "company_size": "501-1000",
        "description": desc,
    }


def _sk(name, prof="advanced", months=24):
    return {"name": name, "proficiency": prof, "endorsements": 5,
            "duration_months": months}


def _cand(cid, title, yoe, roles, skills):
    return {
        "candidate_id": cid,
        "profile": {
            "anonymized_name": "T", "headline": title, "summary": "",
            "location": "Bangalore", "country": "India",
            "years_of_experience": yoe, "current_title": title,
            "current_company": "Acme", "current_company_size": "501-1000",
            "current_industry": "Software",
        },
        "career_history": roles, "education": [], "skills": skills,
        "certifications": [], "languages": [],
        "redrob_signals": _signals(),
    }


# --- the crafted profiles -------------------------------------------------- #
def _impossible_tenure():
    # 120-month single role but only 3 years (36mo) of total experience.
    return _cand(
        "T_TENURE", "Machine Learning Engineer", 3.0,
        [_role("Google", "ML Engineer", 120,
               "Built ranking and retrieval systems in production.", current=True)],
        [_sk("NLP"), _sk("Ranking")],
    )


def _expert_zero():
    # Four advanced/expert skills, each with 0 months of use.
    return _cand(
        "T_EXPERT0", "AI Engineer", 6.0,
        [_role("Amazon", "AI Engineer", 24,
               "Built retrieval and ranking systems.", current=True)],
        [_sk("NLP", "expert", 0), _sk("Information Retrieval", "expert", 0),
         _sk("Ranking", "advanced", 0), _sk("Embeddings", "expert", 0)],
    )


def _future_start():
    # A role starting AFTER the reference snapshot date -> impossible.
    return _cand(
        "T_FUTURE", "AI Engineer", 6.0,
        [_role("Acme", "AI Engineer", 12,
               "Builds retrieval systems.", current=True,
               start="2026-09-01", end=None)],
        [_sk("NLP")],
    )


def _clean_strong():
    # A genuine senior fit — long career, real substance, nothing impossible.
    return _cand(
        "T_CLEAN", "Senior AI Engineer", 7.0,
        [
            _role("Flipkart", "Senior AI Engineer", 30,
                  "Built and deployed embeddings-based semantic search and a "
                  "learning-to-rank system in production.", current=True,
                  start="2023-01-01", end=None),
            _role("Myntra", "Machine Learning Engineer", 36,
                  "Built recommendation and IR systems; deployed ranking models.",
                  start="2019-06-01", end="2022-06-01"),
        ],
        [_sk("NLP"), _sk("Information Retrieval"), _sk("Embeddings"), _sk("Ranking")],
    )


# --------------------------------------------------------------------------- #
# Positives & negatives (typed Candidate / ranker path)
# --------------------------------------------------------------------------- #
def test_impossible_tenure_is_honeypot():
    flag, reasons = is_honeypot(parse_candidate(_impossible_tenure()))
    assert flag is True
    assert any("exceeds total experience" in r for r in reasons)


def test_expert_zero_months_is_honeypot():
    flag, reasons = is_honeypot(parse_candidate(_expert_zero()))
    assert flag is True
    assert any("advanced/expert skills with 0 months used" in r for r in reasons)


def test_future_start_date_is_honeypot():
    flag, reasons = is_honeypot(parse_candidate(_future_start()))
    assert flag is True
    assert any("future" in r for r in reasons)


def test_clean_strong_candidate_is_not_honeypot():
    flag, reasons = is_honeypot(parse_candidate(_clean_strong()))
    assert flag is False
    assert reasons == []


def test_reasons_are_human_readable():
    _, reasons = is_honeypot(parse_candidate(_impossible_tenure()))
    assert reasons, "a positive must explain itself"
    for r in reasons:
        assert isinstance(r, str) and len(r) > 10  # a sentence, not a code


def test_two_expert_zero_skills_do_not_trip():
    # The threshold is >=3; two must NOT trip (precision guard for real fits).
    c = _cand(
        "T_TWO", "AI Engineer", 6.0,
        [_role("Amazon", "AI Engineer", 24, "Built systems.", current=True,
               start="2024-01-01", end=None)],
        [_sk("NLP", "expert", 0), _sk("Ranking", "expert", 0), _sk("Python", "advanced", 36)],
    )
    flag, _ = is_honeypot(parse_candidate(c))
    assert flag is False


# --------------------------------------------------------------------------- #
# Mapping path: is_honeypot also accepts a raw record (convenience / parity).
# --------------------------------------------------------------------------- #
def test_is_honeypot_accepts_raw_mapping():
    flag, reasons = is_honeypot(_impossible_tenure())
    assert flag is True and reasons


# --------------------------------------------------------------------------- #
# DETERMINISM: verdict comes from config.REFERENCE_DATE, never the wall clock.
# --------------------------------------------------------------------------- #
def test_future_check_keys_off_reference_date_not_wall_clock(monkeypatch):
    """A role starting 2026-09-01 is 'future' relative to the 2026-06-16 snapshot.

    We make ``date.today`` explode: if detection touched the wall clock the test
    would error. The verdict must come purely from REFERENCE_DATE, so it stands.
    """
    class _ExplodingDate(dt.date):
        @classmethod
        def today(cls):
            raise AssertionError("honeypot detection must not read the wall clock")

    monkeypatch.setattr("caliber.honeypot.dt.date", _ExplodingDate)

    flag, reasons = is_honeypot(parse_candidate(_future_start()))
    assert flag is True
    assert any("future" in r for r in reasons)


def test_reference_date_drives_the_future_verdict(monkeypatch):
    """Move the snapshot date past the role's start and the same profile is no
    longer impossible — proving REFERENCE_DATE (not an absolute clock) is the
    pivot. Confirms the check is genuinely date-relative and deterministic."""
    # Default snapshot 2026-06-16: 2026-09-01 start is in the future -> honeypot.
    assert is_honeypot(parse_candidate(_future_start()))[0] is True
    # Advance the reference date past the start; now it is in the past -> clean.
    monkeypatch.setattr(config, "REFERENCE_DATE", "2027-01-01")
    flag, _ = is_honeypot(parse_candidate(_future_start()))
    assert flag is False


# --------------------------------------------------------------------------- #
# RECONCILIATION: the eval key and the ranker share ONE definition.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("builder", [
    _impossible_tenure, _expert_zero, _future_start, _clean_strong,
])
def test_eval_and_ranker_agree(builder):
    """eval.heuristics.honeypot_reasons (raw dict + today) and
    honeypot.is_honeypot (typed Candidate) must return the SAME verdict and the
    SAME reasons for every crafted candidate — single source of truth."""
    rec = builder()
    eval_reasons = eval_honeypot_reasons(rec, REF)
    rank_flag, rank_reasons = is_honeypot(parse_candidate(rec))
    assert bool(eval_reasons) == rank_flag
    assert eval_reasons == rank_reasons


def test_eval_delegates_to_canonical():
    """The eval entry point is literally the canonical function (proves there is
    not a second copy that could silently drift)."""
    from caliber.honeypot import honeypot_reasons as canonical
    assert eval_honeypot_reasons is canonical
