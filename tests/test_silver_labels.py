"""Tests for scripts/make_silver_labels.py — the silver-label factory.

This is our own ground-truth answer key (STRATEGY.md §7); a silent bug here
would quietly corrupt every weight-tuning decision downstream. So the things we
verify are the things we must be able to defend in the Stage-5 interview:

- sampling is DETERMINISTIC (same seed -> same ids, twice),
- the rule grader hits hand-verified extremes (honeypot/stuffer -> 0, a textbook
  fit -> 4),
- the agreement maths are correct on a tiny hand-made pair,
- the whole pipeline runs end-to-end with NO LLM grades (rule-only fallback).
"""

import json

import pytest

import datetime as dt

from eval.agreement import agreement_report, kendall_tau, reconcile, spearman
from eval.anchors import build_anchors
from eval.heuristics import honeypot_reasons, stuffer_reasons
from eval.rubric import grade_rules
from eval.sampling import scan_and_bucket, select_sample
from scripts.make_silver_labels import (
    DEFAULT_JD_PROFILE,
    build_silver_set,
    load_jd_profile,
    load_llm_grades,
)

TODAY = dt.date(2026, 6, 16)
JD = load_jd_profile(DEFAULT_JD_PROFILE)


# --------------------------------------------------------------------------- #
# Fixtures: minimal but schema-valid synthetic candidates.
# --------------------------------------------------------------------------- #
def _role(company, title, months, desc, current=False, start="2018-01-01", end="2020-01-01"):
    return {
        "company": company, "title": title, "start_date": start,
        "end_date": None if current else end, "duration_months": months,
        "is_current": current, "industry": "Software", "company_size": "501-1000",
        "description": desc,
    }


def _cand(cid, title, yoe, roles, skills, location="Bangalore", country="India",
          company="Acme", summary="", github=40, relocate=True):
    return {
        "candidate_id": cid,
        "profile": {
            "anonymized_name": "T", "headline": title, "summary": summary,
            "location": location, "country": country, "years_of_experience": yoe,
            "current_title": title, "current_company": company,
            "current_company_size": "501-1000", "current_industry": "Software",
        },
        "career_history": roles, "education": [], "skills": skills,
        "redrob_signals": {
            "github_activity_score": github, "willing_to_relocate": relocate,
            "last_active_date": "2026-06-01", "recruiter_response_rate": 0.6,
            "open_to_work_flag": True, "interview_completion_rate": 0.9,
            "notice_period_days": 30,
        },
    }


def _sk(name, prof="advanced", months=24):
    return {"name": name, "proficiency": prof, "endorsements": 5, "duration_months": months}


def _perfect_fit():
    return _cand(
        "T_FIT", "Senior AI Engineer", 7.0,
        [
            _role("Flipkart", "Senior AI Engineer", 30,
                  "Built and deployed embeddings-based semantic search and a learning-to-rank "
                  "system in production serving millions; measured NDCG and MRR, ran A/B tests.",
                  current=True),
            _role("Myntra", "Machine Learning Engineer", 36,
                  "Built recommendation and information-retrieval systems over text; deployed "
                  "ranking models to production."),
        ],
        [_sk("NLP"), _sk("Information Retrieval"), _sk("Embeddings"), _sk("Ranking")],
        summary="NLP/IR engineer building retrieval and ranking systems in production.",
    )


def _honeypot():
    return _cand(
        "T_HONEY", "Machine Learning Engineer", 3.0,
        [_role("Google", "ML Engineer", 120,
               "Built ranking and retrieval systems in production.", current=True)],
        [_sk("NLP"), _sk("Ranking")],
    )


def _stuffer():
    return _cand(
        "T_STUFF", "HR Manager", 8.0,
        [_role("Infosys", "HR Manager", 48,
               "Managed recruitment, onboarding, payroll and employee relations.", current=True)],
        [_sk("Machine Learning"), _sk("Deep Learning"), _sk("NLP"),
         _sk("Computer Vision"), _sk("LLMs")],
    )


# --------------------------------------------------------------------------- #
# STEP 1 — deterministic sampling
# --------------------------------------------------------------------------- #
def _toy_pool():
    """A small mixed pool: strong, stuffer, honeypot, and lots of noise."""
    pool = [_perfect_fit(), _honeypot(), _stuffer()]
    pool[0]["candidate_id"] = "T_STRONG_1"
    # extra strong-title candidates so the stratum has something to sample from
    for i in range(5):
        c = _perfect_fit()
        c["candidate_id"] = f"T_STRONG_{i+2}"
        pool.append(c)
    # noise floor
    for i in range(40):
        pool.append(_cand(f"T_NOISE_{i:03d}", "Accountant", 5.0,
                          [_role("Firm", "Accountant", 40, "Tax filings and audits.", current=True)],
                          [_sk("Excel", "intermediate", 20)]))
    return pool


def test_sampling_is_deterministic():
    """Same pool + same seed -> identical chosen ids, twice (ARCHITECTURE §5)."""
    pool = _toy_pool()
    b1, n1 = scan_and_bucket(iter(pool), TODAY)
    b2, n2 = scan_and_bucket(iter(pool), TODAY)
    c1 = select_sample(b1, n1, seed=42, total_target=20)
    c2 = select_sample(b2, n2, seed=42, total_target=20)
    assert c1 == c2
    # and a different seed should (here) change the random-pool draw
    c3 = select_sample(b1, n1, seed=99, total_target=20)
    assert c3["random_pool"] != c1["random_pool"] or len(c1["random_pool"]) == 0


def test_strata_assignment():
    """Honeypot/stuffer/strong land in the right buckets; noise stays out."""
    pool = _toy_pool()
    buckets, non_special = scan_and_bucket(iter(pool), TODAY)
    assert "T_HONEY" in buckets["suspected_honeypot"]
    assert "T_STUFF" in buckets["suspected_stuffer"]
    assert any(cid.startswith("T_STRONG") for cid in buckets["strong_ml"])
    # 40 accountants are the noise floor, not a special stratum
    assert sum(1 for cid in non_special if cid.startswith("T_NOISE")) == 40


# --------------------------------------------------------------------------- #
# STEP 2 — rule grader hits hand-verified extremes
# --------------------------------------------------------------------------- #
def test_rule_grader_perfect_fit_is_4():
    g, bd = grade_rules(_perfect_fit(), JD, TODAY)
    assert g == 4, bd


def test_rule_grader_honeypot_is_0():
    c = _honeypot()
    assert honeypot_reasons(c, TODAY), "fixture should trip the honeypot detector"
    g, bd = grade_rules(c, JD, TODAY)
    assert g == 0
    assert bd["forced_zero"] and bd["forced_zero"][0]["type"] == "honeypot"


def test_rule_grader_stuffer_is_0():
    c = _stuffer()
    assert stuffer_reasons(c), "fixture should trip the stuffer detector"
    g, bd = grade_rules(c, JD, TODAY)
    assert g == 0
    assert any(f["type"] == "keyword_stuffer" for f in bd["forced_zero"])


def test_breakdown_is_transparent():
    """Every grade ships its full, inspectable rationale."""
    _, bd = grade_rules(_perfect_fit(), JD, TODAY)
    for key in ("components", "weights", "contributions", "base_score", "final"):
        assert key in bd
    # contributions are weight * component
    for k in bd["components"]:
        assert bd["contributions"][k] == pytest.approx(bd["weights"][k] * bd["components"][k], abs=1e-3)


# --------------------------------------------------------------------------- #
# STEP 4 — agreement maths on hand-made pairs
# --------------------------------------------------------------------------- #
def test_agreement_exact_and_within1():
    # rules vs llm
    pairs = [(4, 4), (3, 3), (2, 3), (1, 0), (0, 2)]
    # exact: (4,4),(3,3) -> 2/5 = 0.4
    # within1: all but (0,2) -> 4/5 = 0.8
    rep = agreement_report(pairs)
    assert rep["n"] == 5
    assert rep["exact_match"] == pytest.approx(0.4)
    assert rep["within_1"] == pytest.approx(0.8)


def test_spearman_perfect_monotonic():
    x = [0, 1, 2, 3, 4]
    y = [1, 2, 3, 4, 5]  # strictly increasing -> rho = 1
    assert spearman(x, y) == pytest.approx(1.0)
    assert spearman(x, list(reversed(y))) == pytest.approx(-1.0)


def test_kendall_known_value():
    # one discordant swap in an otherwise sorted list
    x = [1, 2, 3, 4]
    y = [1, 2, 4, 3]
    # pairs: (1,2)(1,3)(1,4)(2,3)(2,4) concordant=5? recount below
    # all 6 pairs: (1,2)C (1,4)... compute tau directly
    # concordant=5, discordant=1 -> tau = (5-1)/6 = 0.6667
    assert kendall_tau(x, y) == pytest.approx(4 / 6)


def test_reconcile_policy():
    # honeypot/stuffer forced zero wins regardless
    assert reconcile(4, 4, forced_zero=True) == (0, False)
    # llm absent -> rule-only fallback
    assert reconcile(3, None, forced_zero=False) == (3, False)
    # agree within 1 -> rounded average (3.5 -> 4)
    assert reconcile(4, 3, forced_zero=False) == (4, False)
    assert reconcile(2, 3, forced_zero=False) == (3, False)
    # disagree by >=2 -> null + needs_review
    assert reconcile(4, 1, forced_zero=False) == (None, True)


# --------------------------------------------------------------------------- #
# STEP 5 — anchors are internally consistent with the grader
# --------------------------------------------------------------------------- #
def test_anchors_pass_the_grader():
    """The sacred anchors must agree with the rule grader (a tripwire)."""
    failures = []
    for a in build_anchors():
        g, _ = grade_rules(a["candidate"], JD, TODAY)
        if "expected_grade" in a:
            ok = g == a["expected_grade"]
        else:
            ok = a.get("expected_min", 0) <= g <= a.get("expected_max", 4)
        if not ok:
            failures.append((a["name"], g))
    assert not failures, f"anchor mismatches: {failures}"


# --------------------------------------------------------------------------- #
# End-to-end: rule-only fallback (no LLM grades present)
# --------------------------------------------------------------------------- #
def test_end_to_end_rule_only(tmp_path):
    """Pipeline runs with NO llm grades -> grade_llm null, final == grade_rules."""
    pool = _toy_pool()
    cpath = tmp_path / "cands.jsonl"
    with cpath.open("w", encoding="utf-8") as f:
        for c in pool:
            f.write(json.dumps(c) + "\n")

    # no llm_grades file -> empty dict
    llm = load_llm_grades(tmp_path / "does_not_exist.jsonl")
    assert llm == {}

    labels, chosen = build_silver_set(str(cpath), JD, TODAY, seed=42, total_target=20, llm_grades=llm)
    assert labels
    for x in labels:
        assert x["grade_llm"] is None
        assert x["needs_review"] is False
        # rule-only fallback: final mirrors the rule grade
        assert x["grade_final"] == x["grade_rules"]
        assert x["stratum"] in {"strong_ml", "adjacent_gem", "suspected_stuffer",
                                "suspected_honeypot", "random_pool"}
        assert "components" in x["rule_breakdown"]
