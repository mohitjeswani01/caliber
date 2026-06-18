"""Tests for eval/evaluate.py — the silver-set evaluation harness.

These pin the three properties the harness must have to be trustworthy:

  * METRICS WIRING: on a tiny crafted ranking with KNOWN grades and a KNOWN
    scorer ordering, the composite + components equal a hand-computed expectation
    (so we know the number is real, not an artifact of glue bugs).
  * NO LEAKAGE: the object handed to the scorer carries no relevance grade — the
    labels physically cannot enter the scoring path; the eval only compares the
    produced ranking to the grades afterward.
  * END-TO-END: the whole harness (silver-only index build -> real
    score_candidates -> real metrics) runs on a small sample with NO model, NO
    100K artifacts, and NO faiss/torch, via the same injected-seam pattern the
    scorer's own tests use.

We import metrics directly to hand-compute the expectation rather than trusting
evaluate's own call.
"""

import json
import math

import numpy as np
import pytest

from eval import evaluate
from eval.metrics import (
    average_precision,
    ndcg_at_k,
    precision_at_k,
    composite_score,
)
from caliber.schema import parse_candidate

# Reuse the scorer test's candidate factories + fake retrieval seams (no model,
# no faiss): they are the canonical way to drive score_candidates offline.
from tests.test_scorer import (
    JD,
    _cand,
    _role,
    _sk,
    _strong_fit,
    _stuffer,
    _plain_tier5,
    _make_fake_retrieval,
    _Q,
)


def _eye_encoder(cands):
    """Trivial fake candidate encoder: one orthonormal row per candidate. Lets the
    on-the-fly index build run without a model when the scorer is also faked."""
    return np.eye(len(cands), dtype=np.float32)


def _write_jsonl(path, records):
    """Write raw candidate records to a JSONL file (one per line)."""
    path.write_text(
        "\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8"
    )
    return path


# --------------------------------------------------------------------------- #
# load_grades — skips None / needs_review.
# --------------------------------------------------------------------------- #
def test_load_grades_skips_null_and_needs_review(tmp_path):
    labels = [
        {"candidate_id": "A", "grade_final": 4, "needs_review": False},
        {"candidate_id": "B", "grade_final": 0, "needs_review": False},
        {"candidate_id": "C", "grade_final": None, "needs_review": False},  # ungraded
        {"candidate_id": "D", "grade_final": 3, "needs_review": True},      # flagged
    ]
    p = tmp_path / "silver.json"
    p.write_text(__import__("json").dumps(labels), encoding="utf-8")

    grades = evaluate.load_grades(p)
    assert grades == {"A": 4.0, "B": 0.0}        # C dropped (None), D dropped (review)
    assert all(isinstance(v, float) for v in grades.values())


# --------------------------------------------------------------------------- #
# Manual anchor overlay — merged on top of the silver labels, manual wins.
# --------------------------------------------------------------------------- #
def test_load_manual_grades_reads_anchors_and_ignores_doc_fields(tmp_path):
    overlay = [
        {"candidate_id": "X", "grade": 4, "rationale": "doc", "source": "manual"},
        {"candidate_id": "Y", "grade": 3},
        {"candidate_id": "Z"},                # no grade -> skipped
        {"grade": 2},                          # no id -> skipped
    ]
    p = tmp_path / "manual.json"
    p.write_text(json.dumps(overlay), encoding="utf-8")

    out = evaluate.load_manual_grades(p)
    assert out == {"X": 4.0, "Y": 3.0}
    assert all(isinstance(v, float) for v in out.values())


def test_load_manual_grades_missing_file_is_empty(tmp_path):
    assert evaluate.load_manual_grades(tmp_path / "nope.json") == {}


def test_resolve_grades_merges_overlay_on_top_of_silver(tmp_path):
    silver = [
        {"candidate_id": "A", "grade_final": 0, "needs_review": False},  # overridden by anchor
        {"candidate_id": "B", "grade_final": 2, "needs_review": False},
    ]
    overlay = [
        {"candidate_id": "A", "grade": 4},     # anchor corrects the silver 0 -> 4
        {"candidate_id": "NEW", "grade": 3},   # id the silver set never sampled
    ]
    sp = tmp_path / "silver.json"; sp.write_text(json.dumps(silver), encoding="utf-8")
    mp = tmp_path / "manual.json"; mp.write_text(json.dumps(overlay), encoding="utf-8")

    merged, n_manual = evaluate.resolve_grades(sp, mp)
    assert merged == {"A": 4.0, "B": 2.0, "NEW": 3.0}   # A overridden, NEW added
    assert n_manual == 2


def test_resolve_grades_no_overlay_equals_silver(tmp_path):
    silver = [{"candidate_id": "A", "grade_final": 1, "needs_review": False}]
    sp = tmp_path / "silver.json"; sp.write_text(json.dumps(silver), encoding="utf-8")
    merged, n_manual = evaluate.resolve_grades(sp, tmp_path / "absent.json")
    assert merged == {"A": 1.0} and n_manual == 0


def test_shipped_manual_grades_file_has_the_two_anchors():
    """The checked-in overlay grades the two genuine fits the realistic pool
    surfaced: CAND_0002025 -> 4, CAND_0075439 -> 3."""
    anchors = evaluate.load_manual_grades(evaluate.DEFAULT_MANUAL_GRADES)
    assert anchors == {"CAND_0002025": 4.0, "CAND_0075439": 3.0}


# --------------------------------------------------------------------------- #
# Metrics wiring — KNOWN ordering + KNOWN grades => hand-computed composite.
# --------------------------------------------------------------------------- #
def test_composite_matches_hand_computation_with_fake_scorer():
    """Inject a fake score_fn that returns a KNOWN final_score per candidate, and a
    KNOWN grade map. The eval must rank by score desc and produce exactly the
    composite we compute by hand from metrics.py primitives."""

    # 5 candidates, grades 0-4. Scorer orders them imperfectly on purpose: it puts
    # a grade-2 above a grade-4, so NDCG < 1 and the number is non-trivial.
    grades = {"c4": 4.0, "c3": 3.0, "c2": 2.0, "c1": 1.0, "c0": 0.0}
    final_scores = {"c2": 0.9, "c4": 0.8, "c3": 0.7, "c1": 0.6, "c0": 0.5}

    class _FakeScore:
        def __init__(self, cid, score):
            self.candidate_id = cid
            self.final_score = score
            self.is_honeypot = False
            self.ce_used = False

    def fake_score_fn(**kwargs):
        # The scorer is handed candidates_by_id; it returns a score per id. It is
        # given NO grades (asserted separately) — here we just emit known scores.
        ids = list(kwargs["candidates_by_id"])
        return {cid: _FakeScore(cid, final_scores[cid]) for cid in ids}

    # Candidate objects are irrelevant to the fake scorer, but the index build still
    # runs — inject a trivial fake encoder so no model is needed.
    pool = {cid: object() for cid in grades}

    def fake_encode_candidates(cands):
        # one unit vector per candidate; contents don't matter to the fake scorer.
        return np.eye(len(cands), dtype=np.float32)

    out = evaluate.evaluate_silver(
        grades=grades,
        candidates_by_id=pool,
        jd_profile=JD,
        score_fn=fake_score_fn,
        encode_candidates_fn=fake_encode_candidates,
    )

    # Expected ranking: by score desc -> c2, c4, c3, c1, c0
    assert out["ranked_ids"] == ["c2", "c4", "c3", "c1", "c0"]

    # Hand-compute from the SAME relevance sequence the metrics see.
    rels = [grades[cid] for cid in out["ranked_ids"]]   # [2,4,3,1,0]
    exp_ndcg10 = ndcg_at_k(rels, 10)
    exp_ndcg50 = ndcg_at_k(rels, 50)
    exp_map = average_precision(rels, threshold=3.0)    # official tier>=3
    exp_p10 = precision_at_k(rels, 10, threshold=3.0)
    exp_comp = composite_score(exp_ndcg10, exp_ndcg50, exp_map, exp_p10)

    m = out["metrics"]
    assert m["ndcg@10"] == pytest.approx(exp_ndcg10)
    assert m["ndcg@50"] == pytest.approx(exp_ndcg50)
    assert m["map"] == pytest.approx(exp_map)
    assert m["p@10"] == pytest.approx(exp_p10)
    assert m["composite"] == pytest.approx(exp_comp)

    # And independently sanity-check the AP/P@10: only c4 and c3 are relevant
    # (tier>=3), at ranks 2 and 3 -> AP = mean(1/2, 2/3). P@10 clamps k to the 5
    # items present, so it is 2/5 (not 2/10) on this tiny set.
    assert exp_map == pytest.approx((1 / 2 + 2 / 3) / 2)
    assert exp_p10 == pytest.approx(2 / 5)


# --------------------------------------------------------------------------- #
# Leakage guard — the scorer never receives a grade.
# --------------------------------------------------------------------------- #
def test_scoring_path_receives_no_grades():
    """Prove labels cannot leak: capture exactly what evaluate hands the scorer and
    assert (a) no ``grades`` kwarg is passed, and (b) no candidate object (nor its
    raw dict) exposes any grade/relevance field."""
    grades = {"C_FIT": 4.0, "C_STUFF": 0.0}
    recs = [_strong_fit(), _stuffer()]
    recs[0]["candidate_id"] = "C_FIT"
    recs[1]["candidate_id"] = "C_STUFF"
    pool = {r["candidate_id"]: parse_candidate(r) for r in recs}

    captured = {}

    class _FakeScore:
        def __init__(self, cid):
            self.candidate_id = cid
            self.final_score = 1.0 if cid == "C_FIT" else 0.0
            self.is_honeypot = False
            self.ce_used = False

    def spy_score_fn(**kwargs):
        captured["kwargs"] = kwargs
        return {cid: _FakeScore(cid) for cid in kwargs["candidates_by_id"]}

    def fake_encode_candidates(cands):
        return np.eye(len(cands), dtype=np.float32)

    evaluate.evaluate_silver(
        grades=grades,
        candidates_by_id=pool,
        jd_profile=JD,
        score_fn=spy_score_fn,
        encode_candidates_fn=fake_encode_candidates,
    )

    kwargs = captured["kwargs"]
    GRADE_KEYS = {"grade", "grade_final", "grade_rules", "grade_llm",
                  "relevance", "label", "needs_review", "grades"}

    # (a) no grade-bearing kwarg reached the scorer.
    assert not (set(kwargs) & GRADE_KEYS)

    # (b) no candidate object — nor its preserved raw dict — exposes a grade.
    for cand in kwargs["candidates_by_id"].values():
        assert not (set(vars(cand)) & GRADE_KEYS)
        raw = getattr(cand, "raw", {}) or {}
        assert not (set(raw) & GRADE_KEYS)


# --------------------------------------------------------------------------- #
# End-to-end — real score_candidates + real metrics, no model / no 100K / no faiss.
# --------------------------------------------------------------------------- #
def test_end_to_end_with_real_scorer_no_model():
    """Drive the FULL harness through the REAL score_candidates and REAL metrics,
    using the scorer-test fake retrieval seams (numpy stand-in for faiss, fixed
    query vectors). A strong fit must outrank a keyword-stuffer, and the composite
    must be a finite number in [0,1]."""
    recs = [_strong_fit(), _stuffer(), _plain_tier5()]
    for r in recs:
        pass
    pool = {r["candidate_id"]: parse_candidate(r) for r in recs}
    grades = {"C_FIT": 4.0, "C_STUFF": 0.0, "C_PLAIN": 3.0}

    # Candidate + query vectors in the scorer test's 2-D semantic space.
    cand_vecs = {"C_FIT": [0.9, 0.9], "C_STUFF": [0.05, 0.05], "C_PLAIN": [0.4, 0.4]}
    ordered = sorted(pool)  # evaluate builds the index in sorted-id order
    candidate_ids, emb, encode_q, search = _make_fake_retrieval(ordered, cand_vecs, _Q)

    # evaluate builds the index itself; feed it a fake candidate encoder that
    # returns the SAME matrix rows in the SAME order so query/search line up.
    def fake_encode_candidates(cands):
        # cands are the Candidate objects in `ordered` order; return their vectors.
        return emb

    out = evaluate.evaluate_silver(
        grades=grades,
        candidates_by_id=pool,
        jd_profile=JD,
        ce_enabled=False,                       # CE off (no model)
        encode_candidates_fn=fake_encode_candidates,
        encode_query_fn=encode_q,
        search_fn=search,
    )

    # All three silver candidates scored and ranked.
    assert set(out["results"]) == {"C_FIT", "C_STUFF", "C_PLAIN"}
    assert len(out["ranked_ids"]) == 3
    # The strong fit beats the stuffer (substance gating flows through the real scorer).
    rank = out["ranked_ids"]
    assert rank.index("C_FIT") < rank.index("C_STUFF")

    m = out["metrics"]
    assert math.isfinite(m["composite"]) and 0.0 <= m["composite"] <= 1.0
    # Report is a non-empty string mentioning the composite.
    assert "COMPOSITE" in out["report"]
    assert out["n_found"] == 3


def test_honeypot_id_appears_last_when_floored():
    """A floored honeypot (final_score = -1) sorts below every real candidate in
    the ranking the eval produces."""
    grades = {"C_FIT": 4.0, "C_HONEY": 0.0}

    class _FakeScore:
        def __init__(self, cid, score, hp):
            self.candidate_id = cid
            self.final_score = score
            self.is_honeypot = hp
            self.ce_used = False

    def fake_score_fn(**kwargs):
        return {
            "C_FIT": _FakeScore("C_FIT", 0.7, False),
            "C_HONEY": _FakeScore("C_HONEY", -1.0, True),
        }

    pool = {"C_FIT": object(), "C_HONEY": object()}

    out = evaluate.evaluate_silver(
        grades=grades, candidates_by_id=pool, jd_profile=JD,
        score_fn=fake_score_fn,
        encode_candidates_fn=lambda cands: np.eye(len(cands), dtype=np.float32),
    )
    assert out["ranked_ids"] == ["C_FIT", "C_HONEY"]


# =========================================================================== #
# REALISTIC POOL MODE — silver + a deterministic random draw of UNLABELED
# candidates (treated as relevance 0).
# =========================================================================== #
def _unlabeled(cid):
    """A plausible non-silver candidate record (adjacent title, no grade). The
    facts don't matter to these tests; only that it parses and has a unique id."""
    return _cand(
        cid, "Backend Engineer", 5.0,
        [_role("SomeCo", "Backend Engineer", 36,
               "Built REST services and data pipelines.", current=True)],
        [_sk("Python"), _sk("SQL")],
        summary="Backend engineer.",
    )


def test_sample_realistic_pool_is_deterministic_and_excludes_silver(tmp_path):
    """Same SEED -> same random sample across two runs, and the silver ids are
    always kept while exactly pool_size non-silver records are drawn."""
    silver = [_strong_fit(), _stuffer(), _plain_tier5()]          # 3 silver
    unlabeled = [_unlabeled(f"U_{i:03d}") for i in range(20)]     # 20 candidates
    path = _write_jsonl(tmp_path / "candidates.jsonl", silver + unlabeled)
    silver_ids = {"C_FIT", "C_STUFF", "C_PLAIN"}

    pool1, n_found1, n_un1 = evaluate.sample_realistic_pool(
        path, silver_ids, pool_size=6, seed=evaluate.config.SEED
    )
    pool2, n_found2, n_un2 = evaluate.sample_realistic_pool(
        path, silver_ids, pool_size=6, seed=evaluate.config.SEED
    )

    # Deterministic: identical membership across runs.
    assert set(pool1) == set(pool2)
    assert (n_found1, n_un1) == (n_found2, n_un2) == (3, 6)
    # Every silver id is present; exactly pool_size unlabeled were added.
    assert silver_ids <= set(pool1)
    drawn = set(pool1) - silver_ids
    assert len(drawn) == 6 and drawn <= {f"U_{i:03d}" for i in range(20)}

    # A different seed draws a (generally) different subset — sampling is real.
    pool3, _, _ = evaluate.sample_realistic_pool(
        path, silver_ids, pool_size=6, seed=evaluate.config.SEED + 1
    )
    assert (set(pool3) - silver_ids) != drawn


def test_realistic_mode_end_to_end_real_scorer_no_model():
    """Full realistic harness through the REAL score_candidates + REAL metrics on a
    small combined pool (silver + unlabeled), with injected retrieval seams — no
    model, no 100K, no faiss. Reports pool composition + timing."""
    silver_recs = [_strong_fit(), _stuffer()]            # C_FIT (4), C_STUFF (0)
    unlabeled_recs = [_unlabeled("U_A"), _unlabeled("U_B")]
    pool = {r["candidate_id"]: parse_candidate(r) for r in silver_recs + unlabeled_recs}
    grades = {"C_FIT": 4.0, "C_STUFF": 0.0}              # silver grades ONLY

    cand_vecs = {
        "C_FIT": [0.9, 0.9], "C_STUFF": [0.05, 0.05],
        "U_A": [0.3, 0.2], "U_B": [0.2, 0.3],
    }
    ordered = sorted(pool)                                # evaluate indexes in sorted-id order
    _ids, emb, encode_q, search = _make_fake_retrieval(ordered, cand_vecs, _Q)

    out = evaluate.evaluate_realistic(
        grades=grades,
        candidates_by_id=pool,
        jd_profile=JD,
        ce_enabled=False,
        encode_candidates_fn=lambda cands: emb,
        encode_query_fn=encode_q,
        search_fn=search,
    )

    # Pool composition: 2 silver found, 2 unlabeled distractors.
    assert out["n_graded"] == 2
    assert out["n_found"] == 2
    assert out["n_unlabeled"] == 2
    # Metric is a finite composite in [0,1]; report is the REALISTIC variant.
    m = out["metrics"]
    assert math.isfinite(m["composite"]) and 0.0 <= m["composite"] <= 1.0
    assert "REALISTIC POOL" in out["report"]
    # Timing fields are present and non-negative.
    assert out["encode_seconds"] >= 0.0 and out["total_seconds"] >= 0.0
    # The strong fit still beats the keyword stuffer.
    rank = out["ranked_ids"]
    assert rank.index("C_FIT") < rank.index("C_STUFF")


def test_unlabeled_candidate_is_scored_but_counts_as_relevance_zero():
    """An unlabeled candidate is scored (appears in results/ranking) yet the metric
    treats it as relevance 0 — proven by hand-computing NDCG with grade 0 for it."""
    grades = {"C_FIT": 4.0}                               # only the silver fit is graded
    pool = {"C_FIT": object(), "U_HI": object()}          # U_HI is unlabeled

    class _FakeScore:
        def __init__(self, cid, score):
            self.candidate_id = cid
            self.final_score = score
            self.is_honeypot = False
            self.ce_used = False

    def fake_score_fn(**kwargs):
        # U_HI scores ABOVE the grade-4 fit on purpose, so a relevance-0 item leads
        # the ranking and NDCG must drop below 1.0 if it is correctly counted as 0.
        scores = {"U_HI": 0.9, "C_FIT": 0.8}
        return {cid: _FakeScore(cid, scores[cid]) for cid in kwargs["candidates_by_id"]}

    out = evaluate.evaluate_realistic(
        grades=grades,
        candidates_by_id=pool,
        jd_profile=JD,
        score_fn=fake_score_fn,
        encode_candidates_fn=_eye_encoder,
    )

    assert out["ranked_ids"] == ["U_HI", "C_FIT"]
    assert out["n_unlabeled"] == 1
    # U_HI was scored (it is in results) but is NOT a graded ground-truth id.
    assert "U_HI" in out["results"] and "U_HI" not in grades

    # Metric sees relevances [grade(U_HI)=0, grade(C_FIT)=4]; hand-compute it.
    exp_ndcg10 = ndcg_at_k([0.0, 4.0], 10)
    assert out["metrics"]["ndcg@10"] == pytest.approx(exp_ndcg10)
    assert exp_ndcg10 < 1.0                                # the 0-relevance leader hurt us


def test_realistic_mode_scoring_path_receives_no_grades():
    """Leakage guard holds in realistic mode too: neither a grade kwarg nor any
    candidate object (silver or unlabeled) carries a relevance field to the scorer."""
    grades = {"C_FIT": 4.0, "C_STUFF": 0.0}
    recs = [_strong_fit(), _stuffer(), _unlabeled("U_A")]
    pool = {r["candidate_id"]: parse_candidate(r) for r in recs}

    captured = {}

    class _FakeScore:
        def __init__(self, cid):
            self.candidate_id = cid
            self.final_score = 1.0 if cid == "C_FIT" else 0.0
            self.is_honeypot = False
            self.ce_used = False

    def spy_score_fn(**kwargs):
        captured["kwargs"] = kwargs
        return {cid: _FakeScore(cid) for cid in kwargs["candidates_by_id"]}

    evaluate.evaluate_realistic(
        grades=grades, candidates_by_id=pool, jd_profile=JD,
        score_fn=spy_score_fn, encode_candidates_fn=_eye_encoder,
    )

    kwargs = captured["kwargs"]
    GRADE_KEYS = {"grade", "grade_final", "grade_rules", "grade_llm",
                  "relevance", "label", "needs_review", "grades"}
    assert not (set(kwargs) & GRADE_KEYS)
    for cand in kwargs["candidates_by_id"].values():
        assert not (set(vars(cand)) & GRADE_KEYS)
        raw = getattr(cand, "raw", {}) or {}
        assert not (set(raw) & GRADE_KEYS)
