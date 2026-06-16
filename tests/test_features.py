"""Tests for the structured feature scorer (src/caliber/features.py).

The features are the substance backbone of the ranker, so we pin the behaviours
we must be able to defend at Stage-5: a real fit scores high on the §4 drivers, a
keyword-stuffer is GATED to near-zero role_substance (the headline stuffer
defence — the one we most care about), the consulting / experience-band /
nlp-ir / github-neutral rules behave per §4, and recency is deterministic against
config.REFERENCE_DATE rather than the wall clock.
"""

import datetime as dt

import pytest

from caliber import config
from caliber.features import structured_features
from caliber.schema import parse_candidate

# The JD profile shape features reads (mirrors artifacts/jd_profile.json).
JD = {
    "experience_band": {"min": 5, "max": 9, "ideal_min": 6, "ideal_max": 8},
    "consulting_firms": ["TCS", "Infosys", "Wipro", "Accenture", "Cognizant",
                         "Capgemini", "Mindtree", "LTIMindtree", "HCL", "Tech Mahindra"],
    "location_prefs": {"country_priority": "India"},
}

EXPECTED_KEYS = {
    "role_substance", "skill_corroboration", "experience_band", "nlp_ir_signal",
    "product_vs_consulting", "production_recency", "tenure_stability",
    "external_validation", "location_fit", "is_honeypot",
}


# --------------------------------------------------------------------------- #
# Fixtures: full schema-valid records (survive parse_candidate()).
# --------------------------------------------------------------------------- #
def _signals(github=40.0, relocate=True):
    return {
        "profile_completeness_score": 0.9, "signup_date": "2020-01-01",
        "last_active_date": "2026-06-01", "open_to_work_flag": True,
        "profile_views_received_30d": 10, "applications_submitted_30d": 2,
        "recruiter_response_rate": 0.6, "avg_response_time_hours": 12.0,
        "skill_assessment_scores": {}, "connection_count": 300,
        "endorsements_received": 50, "notice_period_days": 30,
        "expected_salary_range_inr_lpa": {"min": 20.0, "max": 40.0},
        "preferred_work_mode": "hybrid", "willing_to_relocate": relocate,
        "github_activity_score": github, "search_appearance_30d": 5,
        "saved_by_recruiters_30d": 3, "interview_completion_rate": 0.9,
        "offer_acceptance_rate": 0.5, "verified_email": True,
        "verified_phone": True, "linkedin_connected": True,
    }


def _role(company, title, months, desc, current=False,
          start="2019-01-01", end="2022-01-01"):
    return {
        "company": company, "title": title, "start_date": start,
        "end_date": None if current else end, "duration_months": months,
        "is_current": current, "industry": "Software", "company_size": "501-1000",
        "description": desc,
    }


def _sk(name, prof="advanced", months=24):
    return {"name": name, "proficiency": prof, "endorsements": 5,
            "duration_months": months}


def _cand(cid, title, yoe, roles, skills, location="Bangalore", country="India",
          company="Acme", summary="", github=40.0, relocate=True):
    return {
        "candidate_id": cid,
        "profile": {
            "anonymized_name": "T", "headline": title, "summary": summary,
            "location": location, "country": country, "years_of_experience": yoe,
            "current_title": title, "current_company": company,
            "current_company_size": "501-1000", "current_industry": "Software",
        },
        "career_history": roles, "education": [], "skills": skills,
        "certifications": [], "languages": [],
        "redrob_signals": _signals(github=github, relocate=relocate),
    }


def _feat(rec):
    return structured_features(parse_candidate(rec), JD)


# --- the crafted profiles -------------------------------------------------- #
def _strong_fit(github=40.0):
    return _cand(
        "T_FIT", "Senior AI Engineer", 7.0,
        [
            _role("Flipkart", "Senior AI Engineer", 30,
                  "Built and deployed embeddings-based semantic search and a "
                  "learning-to-rank system in production serving millions; measured "
                  "NDCG and MRR, ran A/B tests.", current=True, start="2023-06-01"),
            _role("Myntra", "Machine Learning Engineer", 36,
                  "Built recommendation and information-retrieval systems over text; "
                  "deployed ranking models to production.",
                  start="2019-06-01", end="2022-06-01"),
        ],
        [_sk("NLP"), _sk("Information Retrieval"), _sk("Embeddings"), _sk("Ranking")],
        summary="NLP/IR engineer building retrieval and ranking systems in production.",
        github=github,
    )


def _stuffer():
    # Non-tech HR title + many AI skill TAGS, descriptions about HR only.
    return _cand(
        "T_STUFF", "HR Manager", 8.0,
        [_role("Infosys", "HR Manager", 48,
               "Managed recruitment, onboarding, payroll and employee relations.",
               current=True, start="2020-01-01")],
        [_sk("Machine Learning"), _sk("Deep Learning"), _sk("NLP"),
         _sk("Computer Vision"), _sk("LLMs")],
    )


def _consulting_lifer():
    return _cand(
        "T_CONSULT", "Software Engineer", 7.0,
        [
            _role("TCS", "Software Engineer", 40,
                  "Built data pipelines and retrieval features for client systems.",
                  current=True, start="2022-01-01"),
            _role("Infosys", "Software Engineer", 44,
                  "Implemented search and ranking modules for client projects.",
                  start="2018-01-01", end="2021-09-01"),
        ],
        [_sk("Python"), _sk("Information Retrieval")],
    )


def _cv_only():
    return _cand(
        "T_CV", "Machine Learning Engineer", 7.0,
        [_role("Acme", "Machine Learning Engineer", 40,
               "Built computer vision and object detection models; image "
               "classification and segmentation with OpenCV.", current=True,
               start="2022-01-01")],
        [_sk("Computer Vision"), _sk("PyTorch")],
    )


# --------------------------------------------------------------------------- #
# Output shape
# --------------------------------------------------------------------------- #
def test_all_features_present_and_in_range():
    f = _feat(_strong_fit())
    assert set(f.keys()) == EXPECTED_KEYS
    for k, v in f.items():
        assert isinstance(v, float)
        assert 0.0 <= v <= 1.0, f"{k}={v} out of [0,1]"


# --------------------------------------------------------------------------- #
# Strong fit scores high on the headline drivers
# --------------------------------------------------------------------------- #
def test_strong_fit_scores_high():
    f = _feat(_strong_fit())
    assert f["role_substance"] >= 0.8
    assert f["experience_band"] == 1.0       # 7y is dead-centre of 6-8
    assert f["nlp_ir_signal"] == 1.0
    assert f["location_fit"] == 1.0          # India Tier-1 (Bangalore)
    assert f["product_vs_consulting"] == 1.0  # no consulting roles
    assert f["is_honeypot"] == 0.0


# --------------------------------------------------------------------------- #
# THE stuffer defence: near-zero role_substance for a keyword-stuffer.
# --------------------------------------------------------------------------- #
def test_stuffer_gated_to_near_zero_role_substance():
    f = _feat(_stuffer())
    assert f["role_substance"] <= 0.05, (
        f"stuffer must be gated to ~0 role_substance, got {f['role_substance']}"
    )
    # The explicit skill-gate also flags it: many AI tags, zero substance.
    assert f["skill_corroboration"] == 0.0
    # And it is NOT a honeypot — stuffers are gated, not floored (different trap).
    assert f["is_honeypot"] == 0.0


def test_stuffer_role_substance_far_below_strong_fit():
    assert _feat(_stuffer())["role_substance"] < _feat(_strong_fit())["role_substance"]


def test_plain_language_fit_not_punished_by_skill_gate():
    """A real fit who lists FEW AI skill tags must not be penalised by the
    skill-corroboration gate (it targets stuffers, not the quiet Tier-5)."""
    quiet = _cand(
        "T_QUIET", "Data Engineer", 7.0,
        [_role("Swiggy", "Data Engineer", 40,
               "Built dense retrieval and learning-to-rank systems; deployed "
               "ranking models to production with NDCG-based evaluation.",
               current=True, start="2022-01-01")],
        [_sk("Python"), _sk("Spark")],  # only 0 AI-lexicon skills -> nothing to gate
        summary="Information retrieval and ranking over text.",
    )
    assert _feat(quiet)["skill_corroboration"] == 1.0
    assert _feat(quiet)["role_substance"] >= 0.45  # adjacent title + real substance


# --------------------------------------------------------------------------- #
# Consulting
# --------------------------------------------------------------------------- #
def test_consulting_lifer_scores_low_product():
    assert _feat(_consulting_lifer())["product_vs_consulting"] == 0.0


# --------------------------------------------------------------------------- #
# Experience band: junior & senior both below an ideal-band candidate
# --------------------------------------------------------------------------- #
def test_experience_band_junior_and_senior_below_ideal():
    ideal = _feat(_strong_fit())["experience_band"]  # 7y -> 1.0
    junior = _feat(_cand("T_JR", "AI Engineer", 2.0,
                         [_role("Acme", "AI Engineer", 24, "Built retrieval systems.",
                                current=True, start="2024-06-01")],
                         [_sk("NLP")]))["experience_band"]
    senior = _feat(_cand("T_SR", "AI Engineer", 15.0,
                         [_role("Acme", "AI Engineer", 24, "Built retrieval systems.",
                                current=True, start="2024-06-01")],
                         [_sk("NLP")]))["experience_band"]
    assert junior < ideal
    assert senior < ideal


def test_experience_band_is_smooth():
    """Distinct values away from the plateau (proves it is not a hard step)."""
    def band(yoe):
        return _feat(_cand("T", "AI Engineer", yoe,
                           [_role("Acme", "AI Engineer", 24, "Built retrieval.",
                                  current=True, start="2024-06-01")],
                           [_sk("NLP")]))["experience_band"]
    assert band(5.0) != band(4.0)          # not flat below the band
    assert 0.0 < band(5.0) < 1.0
    assert 0.0 < band(10.0) < 1.0


# --------------------------------------------------------------------------- #
# external_validation: -1 github is NEUTRAL, not penalised
# --------------------------------------------------------------------------- #
def test_github_minus_one_is_neutral_not_penalised():
    no_gh = _feat(_strong_fit(github=-1.0))["external_validation"]
    assert no_gh == 0.5, f"missing GitHub must be neutral 0.5, got {no_gh}"
    # An active GitHub scores strictly higher (presence only adds).
    active = _feat(_strong_fit(github=70.0))["external_validation"]
    assert active > no_gh
    # And a candidate with no GitHub is not below one with a linked-but-zero one.
    zero_gh = _feat(_strong_fit(github=0.0))["external_validation"]
    assert no_gh >= zero_gh


# --------------------------------------------------------------------------- #
# nlp_ir: a CV/speech-only candidate scores low
# --------------------------------------------------------------------------- #
def test_cv_only_scores_low_nlp_ir():
    assert _feat(_cv_only())["nlp_ir_signal"] == 0.0


# --------------------------------------------------------------------------- #
# Determinism: recency keys off REFERENCE_DATE, never the wall clock
# --------------------------------------------------------------------------- #
def test_recency_uses_reference_date_not_wall_clock(monkeypatch):
    # A senior in a non-shipping lead role that started >18mo before the snapshot.
    stale = _cand(
        "T_STALE", "Engineering Manager", 12.0,
        [_role("Acme", "Engineering Manager", 40,
               "Responsible for roadmap, strategy and stakeholder alignment across teams.",
               current=True, start="2022-01-01")],  # ~53mo before 2026-06-16
        [_sk("Leadership")],
    )

    class _ExplodingDate(dt.date):
        @classmethod
        def today(cls):
            raise AssertionError("recency math must not read the wall clock")

    monkeypatch.setattr("caliber.features.dt.date", _ExplodingDate)
    f = structured_features(parse_candidate(stale), JD)
    # Non-shipping lead role, 18+ months in -> the deeper 0.1 penalty.
    assert f["production_recency"] == 0.1


def test_recency_reference_date_drives_the_staleness_depth(monkeypatch):
    """A non-shipping lead role that started only recently (relative to the
    snapshot) gets the shallow 0.2, not the deep 0.1 — proving REFERENCE_DATE,
    not an absolute clock, is the pivot."""
    fresh_lead = _cand(
        "T_FRESHLEAD", "Engineering Manager", 12.0,
        [_role("Acme", "Engineering Manager", 6,
               "Responsible for roadmap and stakeholder strategy.",
               current=True, start="2026-03-01")],  # ~3mo before snapshot
        [_sk("Leadership")],
    )
    assert _feat(fresh_lead)["production_recency"] == 0.2
    # Move the snapshot far forward: now the same role is >18mo old -> deep 0.1.
    monkeypatch.setattr(config, "REFERENCE_DATE", "2028-01-01")
    assert _feat(fresh_lead)["production_recency"] == 0.1
