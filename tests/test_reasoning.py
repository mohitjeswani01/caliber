"""Tests for src/caliber/reasoning.py — the grounded, no-LLM reasoning column.

These pin the Stage-4 contract (CLAUDE.md "Reasoning column"): the string is
GROUNDED in the candidate's real signals, never invented; a honeypot's note reuses
its detector reasons verbatim; a strong candidate's note cites real strengths; a
weak one honestly flags real gaps. Everything is deterministic and bounded.

We drive these straight off ``CandidateScore`` (the documented input) with a tiny
factory — no model, no pool, no faiss/torch.
"""

from caliber.reasoning import MAX_REASONING_LEN, reasoning_for
from caliber.scorer import CandidateScore


def _feats(**over):
    """A neutral 10-key feature_dict (mirrors features.structured_features), with
    overrides. Neutral defaults so a test only moves the signals it cares about."""
    base = {
        "role_substance": 0.5,
        "skill_corroboration": 1.0,
        "experience_band": 0.5,
        "nlp_ir_signal": 0.5,
        "product_vs_consulting": 1.0,
        "production_recency": 0.6,
        "tenure_stability": 0.7,
        "external_validation": 0.5,
        "location_fit": 0.8,
        "is_honeypot": 0.0,
    }
    base.update(over)
    return base


def _score(cid="CAND_0000001", *, final=0.5, is_honeypot=False, honeypot_reasons=None,
           feats=None, ce_score=None, ce_used=False, semantic_sim=0.4, behavioral_mult=1.0):
    return CandidateScore(
        candidate_id=cid,
        final_score=final,
        base_score=final,
        behavioral_mult=behavioral_mult,
        is_honeypot=is_honeypot,
        honeypot_reasons=honeypot_reasons or [],
        feature_dict=feats if feats is not None else _feats(),
        ce_score=ce_score,
        ce_used=ce_used,
        semantic_sim=semantic_sim,
        rrf_score=0.0,
    )


# --------------------------------------------------------------------------- #
# Strong candidate: cites real strengths, leads "Strong fit".
# --------------------------------------------------------------------------- #
def test_strong_candidate_cites_real_strengths():
    cs = _score(
        final=0.9,
        feats=_feats(role_substance=1.0, experience_band=1.0, nlp_ir_signal=1.0,
                     product_vs_consulting=1.0, production_recency=1.0,
                     tenure_stability=1.0, external_validation=0.9, location_fit=1.0),
        semantic_sim=0.7, ce_used=True, ce_score=0.8, behavioral_mult=1.1,
    )
    out = reasoning_for(cs)
    assert out
    assert out.startswith("Strong fit")
    # grounded in the actual strong feature values:
    assert "substance" in out
    assert "target band" in out
    assert "NLP/IR" in out
    assert "product-company" in out
    assert "India Tier-1" in out


def test_semantic_and_cross_encoder_clauses_present():
    # A focused candidate (few feature clauses) so the head signals survive the
    # length cap: proves semantic_sim and the cross-encoder are wired in.
    cs = _score(
        final=0.7,
        feats=_feats(role_substance=1.0),  # one strong feature, rest neutral
        semantic_sim=0.75, ce_used=True, ce_score=0.82,
    )
    out = reasoning_for(cs)
    assert "semantic match" in out
    assert "cross-encoder" in out


# --------------------------------------------------------------------------- #
# Honeypot: reuses its detector reasons VERBATIM, does not rank on merit.
# --------------------------------------------------------------------------- #
def test_honeypot_reuses_reasons_verbatim():
    reason = "role tenure 120mo exceeds total experience 36mo"
    cs = _score(
        cid="CAND_0000099", final=-1.0, is_honeypot=True, honeypot_reasons=[reason],
        # Even with maximally strong merit features, the note must NOT praise them.
        feats=_feats(role_substance=1.0, experience_band=1.0),
    )
    out = reasoning_for(cs)
    assert reason in out                      # verbatim reuse
    assert "floor" in out.lower()
    assert "substance" not in out             # merit signals are moot once floored


# --------------------------------------------------------------------------- #
# Down-ranked candidate: honestly flags the real weak signals as gaps.
# --------------------------------------------------------------------------- #
def test_weak_candidate_flags_real_gaps():
    cs = _score(
        final=0.1,
        feats=_feats(role_substance=0.1, experience_band=0.2, nlp_ir_signal=0.0,
                     product_vs_consulting=0.0, location_fit=0.15),
        behavioral_mult=0.8,
    )
    out = reasoning_for(cs)
    assert "Gaps:" in out
    assert "no retrieval/ranking/ML substance" in out
    assert "outside the target band" in out
    assert "no NLP/IR" in out
    assert "services/consulting" in out
    assert "non-India" in out
    assert "engagement" in out  # behavioral drag flagged


def test_stuffer_signal_flagged_as_gap():
    # Many AI skill tags not backed by the career → skill_corroboration low.
    cs = _score(feats=_feats(role_substance=0.0, skill_corroboration=0.0))
    out = reasoning_for(cs)
    assert "keyword-stuffing" in out


# --------------------------------------------------------------------------- #
# Determinism, non-empty, length bound, CSV-safety (single line).
# --------------------------------------------------------------------------- #
def test_deterministic_and_nonempty():
    cs = _score(feats=_feats(role_substance=1.0))
    assert reasoning_for(cs) == reasoning_for(cs)
    assert reasoning_for(cs).strip()


def test_respects_length_bound_and_single_line():
    # Pile on every clause to stress the cap.
    cs = _score(
        feats=_feats(role_substance=1.0, experience_band=1.0, nlp_ir_signal=1.0,
                     product_vs_consulting=1.0, production_recency=1.0,
                     tenure_stability=1.0, external_validation=1.0, location_fit=1.0),
        semantic_sim=0.9, ce_used=True, ce_score=0.9, behavioral_mult=1.1,
    )
    out = reasoning_for(cs)
    assert len(out) <= MAX_REASONING_LEN
    assert "\n" not in out and "\r" not in out


def test_optional_candidate_enrichment_prepends_facts():
    # When the Candidate is threaded, the literal title + years lead the note.
    from caliber.schema import parse_candidate

    rec = {
        "candidate_id": "CAND_0000001",
        "profile": {
            "anonymized_name": "T", "headline": "Senior AI Engineer", "summary": "",
            "location": "Bangalore", "country": "India", "years_of_experience": 7.0,
            "current_title": "Senior AI Engineer", "current_company": "Acme",
            "current_company_size": "501-1000", "current_industry": "Software",
        },
        "career_history": [], "education": [], "skills": [],
        "certifications": [], "languages": [],
        "redrob_signals": {
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
        },
    }
    cand = parse_candidate(rec)
    cs = _score(feats=_feats(role_substance=1.0))
    out = reasoning_for(cs, cand)
    assert out.startswith("Senior AI Engineer, 7 yrs")
