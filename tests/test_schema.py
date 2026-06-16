"""Tests for the canonical candidate schema (src/caliber/schema.py).

Covers: parse_candidate over the real sample data + spot-checks, tolerance of
missing OPTIONAL fields, a clear error on a missing REQUIRED field, the
Candidate.raw round-trip, and — critically — the regression guard proving that
candidate_to_text's raw-dict path (the embedding caller) is unchanged.
"""

import copy
import json

import pytest

from caliber import config
from caliber.schema import (
    Candidate,
    SchemaError,
    candidate_to_text,
    parse_candidate,
)

SAMPLE = config.DATA_DIR / "challenge" / "sample_candidates.json"


def _load_sample():
    return json.loads(SAMPLE.read_text(encoding="utf-8"))


# --- parse_candidate over the real sample -----------------------------------

def test_parse_all_sample_candidates():
    recs = _load_sample()
    parsed = [parse_candidate(r) for r in recs]
    assert len(parsed) == 50
    assert all(isinstance(c, Candidate) for c in parsed)


def test_parse_spot_check_fields():
    rec = _load_sample()[0]
    c = parse_candidate(rec)

    # candidate_id format + scalar profile fields straight from the raw JSON.
    assert c.candidate_id == "CAND_0000001"
    assert c.candidate_id.startswith("CAND_")
    assert c.profile.years_of_experience == rec["profile"]["years_of_experience"]
    assert c.profile.current_company == "Mindtree"
    assert c.profile.current_company_size == "10001+"

    # A nested role value.
    assert c.career_history[0].duration_months == rec["career_history"][0]["duration_months"]
    assert c.career_history[0].is_current is True
    assert c.career_history[0].end_date is None  # nullable end_date preserved

    # A signals value, including the nested salary range + assessment dict.
    sig = rec["redrob_signals"]
    assert c.redrob_signals.recruiter_response_rate == sig["recruiter_response_rate"]
    assert c.redrob_signals.expected_salary_range_inr_lpa.min == sig["expected_salary_range_inr_lpa"]["min"]
    assert c.redrob_signals.expected_salary_range_inr_lpa.max == sig["expected_salary_range_inr_lpa"]["max"]
    assert c.redrob_signals.skill_assessment_scores == sig["skill_assessment_scores"]


def test_github_activity_score_minus_one_allowed():
    """github_activity_score == -1 (no GitHub) must parse, not be rejected."""
    rec = copy.deepcopy(_load_sample()[0])
    rec["redrob_signals"]["github_activity_score"] = -1
    c = parse_candidate(rec)
    assert c.redrob_signals.github_activity_score == -1


# --- tolerance of missing OPTIONAL fields -----------------------------------

def test_missing_optional_fields_parse_cleanly():
    rec = copy.deepcopy(_load_sample()[0])
    # Empty education + skills (schema allows minItems 0), no certs/langs at all,
    # and a null grade on... well there is no edu, so add one with null grade too.
    rec["education"] = []
    rec["skills"] = []
    rec.pop("certifications", None)
    rec.pop("languages", None)

    c = parse_candidate(rec)
    assert c.education == []
    assert c.skills == []
    assert c.certifications == []
    assert c.languages == []


def test_nullable_grade_and_missing_tier():
    rec = copy.deepcopy(_load_sample()[0])
    rec["education"] = [{
        "institution": "Somewhere",
        "degree": "B.Tech",
        "field_of_study": "CS",
        "start_year": 2015,
        "end_year": 2019,
        "grade": None,           # nullable
        # tier omitted entirely
    }]
    c = parse_candidate(rec)
    assert c.education[0].grade is None
    assert c.education[0].tier is None


def test_skill_without_duration_months():
    rec = copy.deepcopy(_load_sample()[0])
    rec["skills"] = [{"name": "Python", "proficiency": "expert", "endorsements": 5}]
    c = parse_candidate(rec)
    assert c.skills[0].duration_months is None


# --- clear error on a missing REQUIRED field --------------------------------

def test_missing_required_top_level_raises():
    rec = copy.deepcopy(_load_sample()[0])
    del rec["redrob_signals"]
    with pytest.raises(SchemaError) as exc:
        parse_candidate(rec)
    assert "redrob_signals" in str(exc.value)


def test_missing_required_nested_field_raises_with_path():
    rec = copy.deepcopy(_load_sample()[0])
    del rec["profile"]["current_title"]
    with pytest.raises(SchemaError) as exc:
        parse_candidate(rec)
    msg = str(exc.value)
    assert "current_title" in msg
    assert "profile" in msg  # path names the offending object


# --- Candidate.raw round-trips the original dict ----------------------------

def test_raw_round_trips_original_dict():
    rec = _load_sample()[0]
    c = parse_candidate(rec)
    assert c.raw == rec


# --- REGRESSION GUARD: candidate_to_text raw-dict path unchanged -------------

# The exact string candidate_to_text produced for sample[0] BEFORE the schema
# work. Hard-coded so this test fails loudly if the dict path ever drifts.
def _expected_text(rec):
    """Re-derive the expected text from the raw record using the documented
    construction rules, independent of the implementation under test."""
    p = rec["profile"]
    parts = [p["headline"], p["summary"],
             f"Current role: {p['current_title']} at {p['current_company']}."]
    for r in rec["career_history"]:
        parts.append(f"{r['title']} at {r['company']}. {r['description']}")
    skills = [f"{s['name']} ({s['proficiency']})" for s in rec["skills"]]
    parts.append("Skills: " + ", ".join(skills))
    return "\n".join(parts)


def test_candidate_to_text_raw_dict_unchanged():
    rec = _load_sample()[0]
    assert candidate_to_text(rec) == _expected_text(rec)


def test_candidate_to_text_dict_and_candidate_equivalent():
    """A Candidate must produce the SAME text as its raw dict — proves the
    dispatch did not change what the embedding pipeline sees."""
    recs = _load_sample()
    for rec in recs:
        c = parse_candidate(rec)
        assert candidate_to_text(c) == candidate_to_text(rec)


def test_candidate_to_text_still_handles_empty_dict():
    # The embedding test relies on this exact behaviour.
    assert candidate_to_text({}) == ""
