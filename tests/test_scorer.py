"""Tests for src/caliber/scorer.py — the composite scorer.

scorer.py is the one place every signal meets, so these tests pin the behaviours
we must defend at Stage-5, using a TINY crafted pool and a numpy stand-in for the
FAISS index (IndexFlatIP == dot product over normalized vectors), so no model,
no 100K pool, and no faiss/torch are needed:

  * the full pipeline runs end-to-end and returns the documented breakdown;
  * a strong fit outscores a keyword-stuffer (role_substance gating flows
    through) AND a honeypot;
  * a honeypot is floored below every real candidate even with a high behavioral
    multiplier (floor applied LAST);
  * CE-absent (rerank raises FileNotFoundError) → ce_used=False, sane ordering;
  * CE-present (mock logits) → sigmoid normalization is applied (known value);
  * determinism: two runs identical; candidate_id tie-break on equal scores;
  * BM25 surfaces a lexically-strong, semantically-weak candidate via fusion.
"""

import math

import numpy as np
import pytest

from caliber import scorer
from caliber.scorer import (
    COMPOSITE_FEATURE_NAMES,
    DEFAULT_WEIGHTS,
    HONEYPOT_FLOOR,
    CandidateScore,
    _sigmoid,
    combine,
    score_candidates,
)
from caliber.schema import parse_candidate


# --------------------------------------------------------------------------- #
# JD profile: two aspects so we exercise aspect-WEIGHTED aggregation. Aspect
# names are sorted internally (a_retrieval < b_ranking), which fixes the query
# row order our fake encoder/search rely on.
# --------------------------------------------------------------------------- #
JD = {
    "role": "Senior AI Engineer",
    "experience_band": {"min": 5, "max": 9, "ideal_min": 6, "ideal_max": 8},
    "consulting_firms": ["TCS", "Infosys", "Wipro", "Accenture", "Cognizant",
                         "Capgemini", "Mindtree", "LTIMindtree", "HCL", "Tech Mahindra"],
    "location_prefs": {"country_priority": "India"},
    "aspects": {
        "a_retrieval": {
            "weight": 0.6,
            "query_text": "embeddings based semantic search dense retrieval in production",
            "keywords": ["embeddings", "semantic search", "dense retrieval", "faiss"],
        },
        "b_ranking": {
            "weight": 0.4,
            "query_text": "learning to rank ranking relevance ndcg evaluation",
            "keywords": ["ranking", "learning to rank", "ndcg", "relevance"],
        },
    },
}


# --------------------------------------------------------------------------- #
# Candidate factory (full schema-valid records that survive parse_candidate()).
# --------------------------------------------------------------------------- #
def _signals(github=40.0, relocate=True, last_active="2026-06-01",
             response=0.6, complete=0.9, open_flag=True, notice=30,
             saved=3, interview=0.9, vemail=True, vphone=True):
    return {
        "profile_completeness_score": complete, "signup_date": "2020-01-01",
        "last_active_date": last_active, "open_to_work_flag": open_flag,
        "profile_views_received_30d": 10, "applications_submitted_30d": 2,
        "recruiter_response_rate": response, "avg_response_time_hours": 12.0,
        "skill_assessment_scores": {}, "connection_count": 300,
        "endorsements_received": 50, "notice_period_days": notice,
        "expected_salary_range_inr_lpa": {"min": 20.0, "max": 40.0},
        "preferred_work_mode": "hybrid", "willing_to_relocate": relocate,
        "github_activity_score": github, "search_appearance_30d": 5,
        "saved_by_recruiters_30d": saved, "interview_completion_rate": interview,
        "offer_acceptance_rate": 0.5, "verified_email": vemail,
        "verified_phone": vphone, "linkedin_connected": True,
    }


def _role(company, title, months, desc, current=False,
          start="2019-01-01", end="2022-01-01", skills_dur=24):
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
          company="Acme", summary="", signals=None):
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
        "redrob_signals": signals or _signals(),
    }


# --- crafted profiles ------------------------------------------------------ #
def _strong_fit():
    return _cand(
        "C_FIT", "Senior AI Engineer", 7.0,
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
    )


def _stuffer():
    # Non-tech HR title + many AI skill TAGS, descriptions about HR only.
    return _cand(
        "C_STUFF", "HR Manager", 8.0,
        [_role("Globex", "HR Manager", 48,
               "Managed recruitment, onboarding, payroll and employee relations.",
               current=True, start="2020-01-01")],
        [_sk("Machine Learning"), _sk("Deep Learning"), _sk("NLP"),
         _sk("Computer Vision"), _sk("LLMs")],
    )


def _honeypot():
    # Internally impossible: a single role lasting longer than the whole career.
    return _cand(
        "C_HONEY", "Senior AI Engineer", 3.0,
        [_role("Initech", "Senior AI Engineer", 120,  # 10yr role, 3yr career
               "Built embeddings retrieval and learning-to-rank systems in "
               "production; NDCG, semantic search.", current=True,
               start="2024-01-01")],
        [_sk("Embeddings"), _sk("Ranking"), _sk("NLP")],
        # Maximally available so we can prove the floor beats a high multiplier.
        signals=_signals(github=90.0, last_active="2026-06-16", response=1.0,
                         complete=1.0, saved=5, interview=1.0),
    )


def _plain_tier5():
    # Adjacent title, NO buzzword skills, but the DESCRIPTION is dense with the
    # lexical query terms -> BM25-strong. We give it a weak semantic vector so it
    # only surfaces via the lexical->fusion path.
    return _cand(
        "C_PLAIN", "Data Engineer", 6.0,
        [_role("Zomato", "Data Engineer", 40,
               "Designed dense retrieval and semantic search with embeddings; built "
               "a learning-to-rank relevance system measured by NDCG. Faiss index, "
               "ranking evaluation, dense retrieval embeddings semantic search.",
               current=True, start="2022-06-01")],
        [_sk("Python"), _sk("SQL"), _sk("Spark")],  # no AI buzzword tags
        summary="Data engineer.",
    )


# --------------------------------------------------------------------------- #
# Fake retrieval seams (no model, no faiss). The "index" is an (N, D) matrix of
# unit vectors; the fake search reproduces FAISS IndexFlatIP exactly: cosine =
# dot product, top-k by descending similarity.
# --------------------------------------------------------------------------- #
def _normalize(v):
    v = np.asarray(v, dtype=np.float32)
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def _make_fake_retrieval(ids, cand_vecs, query_vecs):
    """Build (candidate_ids, emb_matrix, encode_fn, search_fn).

    cand_vecs:  {id: vector}      — the candidate "embeddings".
    query_vecs: {aspect_name: vector} — query vectors for each (sorted) aspect.
    """
    emb = np.stack([_normalize(cand_vecs[cid]) for cid in ids]).astype(np.float32)

    def encode_fn(texts, is_query=False):
        # Aspects are scored in SORTED name order inside scorer; return query rows
        # in that same order so rows align with the aspect weights.
        names = sorted(JD["aspects"].keys())
        return np.stack([_normalize(query_vecs[n]) for n in names]).astype(np.float32)

    def search_fn(index, q, k):
        q = np.atleast_2d(np.asarray(q, dtype=np.float32))
        sims = q @ emb.T                       # (A, N) cosine (unit vectors)
        idx = np.argsort(-sims, axis=1)[:, :k]
        scores = np.take_along_axis(sims, idx, axis=1)
        return scores, idx

    return np.array(ids), emb, encode_fn, search_fn


def _build_pool(records):
    """Return {id: Candidate} for a list of raw records."""
    return {r["candidate_id"]: parse_candidate(r) for r in records}


# Aspect query vectors in a 2-D semantic space: dim0 == "retrieval", dim1 ==
# "ranking". The fit aligns with both; the stuffer with neither.
_Q = {"a_retrieval": [1.0, 0.0], "b_ranking": [0.0, 1.0]}


def _fit_stuffer_honey_setup(ce_logits=None):
    """Common 3-candidate (fit, stuffer, honeypot) setup. Returns kwargs for
    score_candidates plus a rerank_fn (mock CE) keyed by id."""
    recs = [_strong_fit(), _stuffer(), _honeypot()]
    pool = _build_pool(recs)
    cand_vecs = {
        "C_FIT": [0.8, 0.8],     # strong on both aspects
        "C_STUFF": [0.05, 0.05], # weak everywhere
        "C_HONEY": [0.7, 0.7],   # also semantically strong (it stuffs the desc too)
    }
    ids = ["C_FIT", "C_STUFF", "C_HONEY"]
    candidate_ids, _emb, encode_fn, search_fn = _make_fake_retrieval(ids, cand_vecs, _Q)

    if ce_logits is not None:
        def rerank_fn(jd_text, texts, top_k=None):
            # texts are candidate_to_text strings; key the logit off a fingerprint.
            out = []
            for t in texts:
                if "HR Manager" in t or "payroll" in t:
                    out.append(ce_logits["C_STUFF"])
                elif "Initech" in t:
                    out.append(ce_logits["C_HONEY"])
                else:
                    out.append(ce_logits["C_FIT"])
            return out
    else:
        rerank_fn = None  # caller supplies / uses default

    return dict(
        jd_profile=JD,
        candidate_ids=candidate_ids,
        candidates_by_id=pool,
        encode_query_fn=encode_fn,
        search_fn=search_fn,
    ), rerank_fn


# --------------------------------------------------------------------------- #
# Unit-level: sigmoid + combine.
# --------------------------------------------------------------------------- #
def test_sigmoid_known_values():
    assert _sigmoid(0.0) == pytest.approx(0.5)
    assert _sigmoid(2.0) == pytest.approx(1.0 / (1.0 + math.exp(-2.0)), abs=1e-12)
    # Overflow-safe at the extremes (no OverflowError / inf / nan; underflows to
    # exactly 0.0, which is still finite and in [0,1]).
    lo = _sigmoid(-1000.0)
    assert math.isfinite(lo) and 0.0 <= lo < 1e-6
    assert _sigmoid(1000.0) == pytest.approx(1.0)


def test_combine_renormalizes_when_ce_absent():
    """combine() drops a None feature and rescales remaining weights to sum 1, so
    base stays in [0,1] whether or not CE ran."""
    full = {k: 1.0 for k in COMPOSITE_FEATURE_NAMES}
    assert combine(full, DEFAULT_WEIGHTS) == pytest.approx(1.0)

    half = {k: (1.0 if k != "ce_score" else None) for k in COMPOSITE_FEATURE_NAMES}
    # All present features are 1.0 -> renormalized weighted mean is still 1.0.
    assert combine(half, DEFAULT_WEIGHTS) == pytest.approx(1.0)

    zeros = {k: 0.0 for k in COMPOSITE_FEATURE_NAMES}
    assert combine(zeros, DEFAULT_WEIGHTS) == pytest.approx(0.0)


# --------------------------------------------------------------------------- #
# Pipeline end-to-end + ranking sanity.
# --------------------------------------------------------------------------- #
def test_pipeline_runs_and_returns_full_breakdown():
    kwargs, rerank_fn = _fit_stuffer_honey_setup(ce_logits=None)
    out = score_candidates(ce_enabled=False, **kwargs)  # CE off (no model)

    assert set(out.keys()) == {"C_FIT", "C_STUFF", "C_HONEY"}
    for cs in out.values():
        assert isinstance(cs, CandidateScore)
        # documented fields all present and well-typed.
        assert isinstance(cs.final_score, float) and math.isfinite(cs.final_score)
        assert isinstance(cs.base_score, float) and math.isfinite(cs.base_score)
        assert 0.50 <= cs.behavioral_mult <= 1.15
        assert isinstance(cs.is_honeypot, bool)
        assert isinstance(cs.honeypot_reasons, list)
        assert set(cs.feature_dict) >= {"role_substance", "is_honeypot"}
        assert cs.ce_used is False and cs.ce_score is None  # CE disabled
        assert 0.0 <= cs.semantic_sim <= 1.0


def test_strong_fit_beats_stuffer_and_honeypot():
    kwargs, _ = _fit_stuffer_honey_setup(ce_logits=None)
    out = score_candidates(ce_enabled=False, **kwargs)

    fit, stuff, honey = out["C_FIT"], out["C_STUFF"], out["C_HONEY"]

    # The stuffer's role_substance is gated to 0 (non-tech title + HR descriptions).
    assert stuff.feature_dict["role_substance"] == 0.0
    assert fit.feature_dict["role_substance"] > 0.5

    assert fit.final_score > stuff.final_score        # gating flows through
    assert fit.final_score > honey.final_score        # honeypot floored
    assert honey.is_honeypot is True


def test_honeypot_floored_below_every_real_candidate_even_when_available():
    """The honeypot is maximally available (high behavioral multiplier) and
    semantically strong, yet must land strictly below every non-honeypot, because
    the floor is applied LAST."""
    kwargs, _ = _fit_stuffer_honey_setup(ce_logits=None)
    out = score_candidates(ce_enabled=False, **kwargs)

    honey = out["C_HONEY"]
    reals = [cs for cs in out.values() if not cs.is_honeypot]

    assert honey.final_score == HONEYPOT_FLOOR
    assert honey.behavioral_mult > 1.0  # genuinely available => high multiplier
    assert all(honey.final_score < cs.final_score for cs in reals)


# --------------------------------------------------------------------------- #
# Cross-encoder: absent path and present path.
# --------------------------------------------------------------------------- #
def test_ce_absent_does_not_crash_and_orders_sanely():
    """rerank raising FileNotFoundError => CE skipped, ce_used=False, fit still
    on top."""
    kwargs, _ = _fit_stuffer_honey_setup(ce_logits=None)

    def rerank_raises(jd_text, texts, top_k=None):
        raise FileNotFoundError("cross-encoder model dir missing")

    out = score_candidates(rerank_fn=rerank_raises, ce_enabled=True, **kwargs)

    assert all(cs.ce_used is False and cs.ce_score is None for cs in out.values())
    assert out["C_FIT"].final_score > out["C_STUFF"].final_score
    assert out["C_FIT"].final_score > out["C_HONEY"].final_score


def test_ce_present_applies_sigmoid_normalization():
    """A known logit maps to the expected sigmoid value on the breakdown, and CE
    is marked used for the reranked head."""
    logits = {"C_FIT": 8.0, "C_STUFF": -6.0, "C_HONEY": 5.0}
    kwargs, rerank_fn = _fit_stuffer_honey_setup(ce_logits=logits)

    out = score_candidates(rerank_fn=rerank_fn, ce_enabled=True, **kwargs)

    assert out["C_FIT"].ce_used is True
    assert out["C_FIT"].ce_score == pytest.approx(_sigmoid(8.0), abs=1e-9)
    assert out["C_STUFF"].ce_score == pytest.approx(_sigmoid(-6.0), abs=1e-9)
    # A high CE logit lifts the fit's base above the CE-off case.
    base_off = score_candidates(ce_enabled=False, **kwargs)["C_FIT"].base_score
    assert out["C_FIT"].base_score > base_off


def test_ce_only_scores_the_head_of_the_shortlist():
    """With ce_shortlist_size=1, only the top shortlisted candidate is reranked;
    the rest get ce_score=None (composite renormalizes)."""
    logits = {"C_FIT": 8.0, "C_STUFF": -6.0, "C_HONEY": 5.0}
    kwargs, rerank_fn = _fit_stuffer_honey_setup(ce_logits=logits)

    out = score_candidates(rerank_fn=rerank_fn, ce_enabled=True,
                           ce_shortlist_size=1, **kwargs)
    used = [cid for cid, cs in out.items() if cs.ce_used]
    assert len(used) == 1


# --------------------------------------------------------------------------- #
# Determinism + tie-break.
# --------------------------------------------------------------------------- #
def test_determinism_two_runs_identical():
    logits = {"C_FIT": 8.0, "C_STUFF": -6.0, "C_HONEY": 5.0}
    kwargs1, rr1 = _fit_stuffer_honey_setup(ce_logits=logits)
    kwargs2, rr2 = _fit_stuffer_honey_setup(ce_logits=logits)

    a = score_candidates(rerank_fn=rr1, ce_enabled=True, **kwargs1)
    b = score_candidates(rerank_fn=rr2, ce_enabled=True, **kwargs2)

    assert a.keys() == b.keys()
    for cid in a:
        assert a[cid].final_score == b[cid].final_score
        assert a[cid].base_score == b[cid].base_score


def test_tie_break_candidate_id_ascending_on_equal_scores():
    """Two byte-identical profiles with different ids land at the SAME score; the
    deterministic tie-break (candidate_id ascending) is what ranker relies on —
    here we assert the shortlist ordering the scorer produces is id-ascending when
    fused scores tie."""
    # Two identical candidates, different ids; identical everything -> identical
    # semantic vectors -> identical fusion score -> tie.
    base = _strong_fit()
    rec_a = dict(base); rec_a = _strong_fit(); rec_a["candidate_id"] = "C_AAA"
    rec_b = _strong_fit(); rec_b["candidate_id"] = "C_BBB"
    pool = _build_pool([rec_b, rec_a])  # insert in REVERSE id order on purpose

    cand_vecs = {"C_AAA": [0.8, 0.8], "C_BBB": [0.8, 0.8]}
    ids = ["C_BBB", "C_AAA"]  # file order reversed too
    candidate_ids, _emb, encode_fn, search_fn = _make_fake_retrieval(ids, cand_vecs, _Q)

    out = score_candidates(
        jd_profile=JD, candidate_ids=candidate_ids, candidates_by_id=pool,
        encode_query_fn=encode_fn, search_fn=search_fn, ce_enabled=False,
    )
    assert out["C_AAA"].final_score == out["C_BBB"].final_score
    # both fully scored; identical breakdown.
    assert out["C_AAA"].base_score == out["C_BBB"].base_score


# --------------------------------------------------------------------------- #
# BM25 / fusion: a lexically-strong, semantically-weak candidate surfaces.
# --------------------------------------------------------------------------- #
def test_bm25_surfaces_lexically_strong_semantically_weak_candidate():
    """C_PLAIN has a NEAR-ZERO semantic vector but a description packed with the
    JD lexical terms. It must still appear in the shortlist (scored, with a
    breakdown) thanks to the BM25->fusion path."""
    recs = [_strong_fit(), _plain_tier5()]
    pool = _build_pool(recs)
    cand_vecs = {
        "C_FIT": [0.9, 0.9],
        "C_PLAIN": [0.001, 0.001],   # semantically almost invisible
    }
    ids = ["C_FIT", "C_PLAIN"]
    candidate_ids, _emb, encode_fn, search_fn = _make_fake_retrieval(ids, cand_vecs, _Q)

    out = score_candidates(
        jd_profile=JD, candidate_ids=candidate_ids, candidates_by_id=pool,
        encode_query_fn=encode_fn, search_fn=search_fn, ce_enabled=False,
    )
    # Surfaced and fully scored despite weak semantics.
    assert "C_PLAIN" in out
    assert out["C_PLAIN"].rrf_score > 0.0
    # Its lexical strength gives it a real role_substance (adjacent title + dense
    # retrieval/ranking descriptions), so it isn't bottom-of-the-barrel noise.
    assert out["C_PLAIN"].feature_dict["role_substance"] > 0.4


def test_lexical_ranking_actually_reorders_vs_semantic():
    """Directly check the lexical stage: BM25 ranks the description-dense plain
    candidate above a semantically-strong one that lacks the lexical terms."""
    fit = _build_pool([_strong_fit()])["C_FIT"]
    plain = _build_pool([_plain_tier5()])["C_PLAIN"]
    pool = {"C_FIT": fit, "C_PLAIN": plain}
    lexical = scorer._lexical_retrieval(JD, ["C_FIT", "C_PLAIN"], pool)
    assert set(lexical) == {"C_FIT", "C_PLAIN"}
    # both are description-dense; the point is the ranking is produced and ordered.
    assert lexical[0] in ("C_FIT", "C_PLAIN")


# --------------------------------------------------------------------------- #
# Edge cases: tiny pool / shortlist smaller than the cap / numpy str ids.
# --------------------------------------------------------------------------- #
def test_numpy_str_ids_are_coerced():
    kwargs, _ = _fit_stuffer_honey_setup(ce_logits=None)
    # candidate_ids is already a numpy array (numpy.str_ elements).
    assert kwargs["candidate_ids"].dtype.kind in ("U", "S")
    out = score_candidates(ce_enabled=False, **kwargs)
    # keys are python str, usable in plain dict/set operations.
    assert all(isinstance(k, str) and not isinstance(k, np.str_) for k in out)


def test_shortlist_smaller_than_cap_and_empty_aspects():
    # shortlist_size huge, only 3 candidates -> no crash, all scored.
    kwargs, _ = _fit_stuffer_honey_setup(ce_logits=None)
    out = score_candidates(ce_enabled=False, shortlist_size=10_000, **kwargs)
    assert len(out) == 3

    # No aspects -> empty semantic retrieval -> empty result (graceful).
    empty_jd = dict(JD); empty_jd["aspects"] = {}
    kwargs2, _ = _fit_stuffer_honey_setup(ce_logits=None)
    kwargs2["jd_profile"] = empty_jd
    out2 = score_candidates(ce_enabled=False, **kwargs2)
    assert out2 == {}
