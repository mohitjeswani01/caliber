"""Tests for eval/sweep.py — the weight-sweep machinery.

These pin the methodology guarantees, all without a model / encode / 100K:

  * ENCODE ONCE: build_cache runs the (faked) scorer exactly once; the sweep then
    re-ranks from the cache and never calls it again.
  * DETERMINISTIC, DISJOINT, STRATIFIED train/val split.
  * RENORMALISATION: every grid weight set sums to 1.0; infeasible levers drop;
    a known weight set yields the hand-computed combined score.
  * ANCHORS SACRED: anchor ids never enter the tuning split / objective.
"""

import pytest

from eval import sweep
from caliber.scorer import COMPOSITE_FEATURE_NAMES, DEFAULT_WEIGHTS, HONEYPOT_FLOOR


# --------------------------------------------------------------------------- #
# Fakes — a CandidateScore stand-in and a minimal cache entry.
# --------------------------------------------------------------------------- #
class _FakeCS:
    def __init__(self, cid, role=0.5, ce=None, sem=0.5, beh=1.0, hp=False, final=0.0):
        self.candidate_id = cid
        self.feature_dict = {k: 0.5 for k in sweep.FEATURE_KEYS}
        self.feature_dict["role_substance"] = role
        self.ce_score = ce
        self.semantic_sim = sem
        self.behavioral_mult = beh
        self.is_honeypot = hp
        self.final_score = final


def _entry(role=0.5, ce=None, sem=0.5, beh=1.0, hp=False):
    feats = {k: 0.5 for k in sweep.FEATURE_KEYS}
    feats["role_substance"] = role
    return {"features": feats, "ce_score": ce, "semantic_sim": sem,
            "behavioral_mult": beh, "is_honeypot": hp}


# --------------------------------------------------------------------------- #
# Weight renormalisation + the known-score recompute.
# --------------------------------------------------------------------------- #
def test_make_weights_sums_to_one_and_sets_levers():
    w = sweep.make_weights(ce=0.30, role=0.30, sem=0.20)
    assert set(w) == set(COMPOSITE_FEATURE_NAMES)
    assert w["ce_score"] == 0.30 and w["role_substance"] == 0.30 and w["semantic_sim"] == 0.20
    assert abs(sum(w.values()) - 1.0) < 1e-9


def test_make_weights_infeasible_returns_none():
    assert sweep.make_weights(ce=0.40, role=0.40, sem=0.30) is None   # levers sum 1.1 > 1


def test_weight_grid_all_feasible_and_normalised():
    grid = sweep.weight_grid()
    assert 10 < len(grid) < 60                       # a few dozen, not thousands
    for w in grid:
        assert abs(sum(w.values()) - 1.0) < 1e-9
        assert set(w) == set(COMPOSITE_FEATURE_NAMES)


def test_recompute_final_matches_hand_computation():
    # weights: all mass on role_substance => combine renormalises to the role value.
    w = {k: 0.0 for k in COMPOSITE_FEATURE_NAMES}
    w["role_substance"] = 1.0
    e = _entry(role=0.8, ce=None, sem=0.123, beh=0.9)
    # base = 0.8 (only weighted feature), final = base * behavioural = 0.72
    assert sweep.recompute_final(e, w) == pytest.approx(0.8 * 0.9)


def test_recompute_final_honeypot_is_floored():
    w = sweep.make_weights(ce=0.2, role=0.3, sem=0.2)
    e = _entry(role=0.99, beh=1.15, hp=True)
    assert sweep.recompute_final(e, w) == HONEYPOT_FLOOR


def test_rerank_orders_by_recomputed_score_then_id_and_appends_tail():
    cache = {
        "all_pool_ids": ["A", "B", "C", "TAIL"],     # TAIL is un-scored
        "scored": {
            "A": _entry(role=0.2),
            "B": _entry(role=0.9),
            "C": _entry(role=0.9),                    # ties B -> id break A<B<C
        },
    }
    w = {k: 0.0 for k in COMPOSITE_FEATURE_NAMES}; w["role_substance"] = 1.0
    ranked = sweep.rerank(cache, w)
    assert ranked == ["B", "C", "A", "TAIL"]          # B,C (0.9) before A (0.2); tail last


# --------------------------------------------------------------------------- #
# Train / validation split.
# --------------------------------------------------------------------------- #
def test_split_is_deterministic_disjoint_and_stratified():
    grades = {f"g{i}": (i % 5) for i in range(100)}   # 20 per grade 0..4
    tr1, va1 = sweep.stratified_split(grades, seed=sweep.config.SEED)
    tr2, va2 = sweep.stratified_split(grades, seed=sweep.config.SEED)

    assert (tr1, va1) == (tr2, va2)                   # deterministic
    assert tr1.isdisjoint(va1)                        # disjoint
    assert tr1 | va1 == set(grades)                   # covers everything
    # stratified: each grade roughly half/half in each split.
    for g in range(5):
        ids = {cid for cid, gg in grades.items() if gg == g}
        assert 8 <= len(ids & tr1) <= 12

    # a different seed yields a different partition (the shuffle is real).
    tr3, _ = sweep.stratified_split(grades, seed=sweep.config.SEED + 1)
    assert tr3 != tr1


# --------------------------------------------------------------------------- #
# ENCODE ONCE — build_cache calls the scorer once; the sweep never re-encodes.
# --------------------------------------------------------------------------- #
def test_build_cache_encodes_once_and_sweep_reuses_it(tmp_path):
    results = {cid: _FakeCS(cid, role=r)
               for cid, r in [("A", 0.2), ("B", 0.9), ("C", 0.5), ("D", 0.7)]}
    # Make the fake scorer's ranked_ids self-consistent with the cache + DEFAULT
    # weights, so build_cache's faithfulness self-check passes.
    consistent = sweep.rerank(sweep.cache_from_results(results, list(results)), DEFAULT_WEIGHTS)

    calls = {"n": 0}

    def fake_eval(**kwargs):
        calls["n"] += 1
        return {"results": results, "ranked_ids": consistent,
                "encode_seconds": 12.3, "total_seconds": 45.6, "report": "fake"}

    out = sweep.build_cache(
        eval_fn=fake_eval, pool_size=4,
        cache_path=tmp_path / "cache.json", ranking_csv_path=tmp_path / "rank.csv",
    )
    assert calls["n"] == 1                             # the ONE expensive call
    assert out["encode_seconds"] == 12.3
    assert (tmp_path / "cache.json").exists() and (tmp_path / "rank.csv").exists()

    # Sweep over the cache with a multi-config grid — the scorer is NOT called again.
    grid = [sweep.make_weights(0.0, 0.23, 0.20), sweep.make_weights(0.30, 0.30, 0.20),
            sweep.make_weights(0.40, 0.15, 0.10)]
    grid = [w for w in grid if w is not None]
    res = sweep.run_sweep(
        out["cache"],
        tuning_grades={"A": 4.0, "B": 0.0, "C": 3.0, "D": 0.0},
        report_grades={"A": 4.0, "B": 0.0, "C": 3.0, "D": 0.0},
        anchor_ids=set(),
        grid=grid,
    )
    assert calls["n"] == 1                             # STILL one — sweep re-ranked from cache
    assert res["n_grid"] == len(grid)
    assert "VALIDATION" in res["report"]


# --------------------------------------------------------------------------- #
# ANCHORS SACRED — never in the split / objective.
# --------------------------------------------------------------------------- #
def test_anchors_excluded_from_tuning_split():
    # stratified_split only ever sees the silver grades it is handed.
    silver = {"s1": 4.0, "s2": 0.0, "s3": 3.0, "s4": 0.0}
    train, val = sweep.stratified_split(silver, seed=sweep.config.SEED)
    assert "a1" not in (train | val) and "a2" not in (train | val)

    # And run_sweep strips any anchor id that sneaks into the tuning grades.
    cache = {
        "all_pool_ids": ["s1", "s2", "s3", "s4", "a1", "x1", "x2"],
        "scored": {cid: _entry(role=0.5) for cid in ["s1", "s2", "s3", "s4", "a1", "x1", "x2"]},
    }
    res = sweep.run_sweep(
        cache,
        tuning_grades={"a1": 4.0, "s1": 4.0, "s2": 0.0, "s3": 3.0, "s4": 0.0},  # a1 is an anchor
        report_grades={"a1": 4.0, "s1": 4.0, "s2": 0.0, "s3": 3.0, "s4": 0.0},
        anchor_ids={"a1"},
        grid=[sweep.make_weights(0.2, 0.23, 0.2)],
    )
    # a1 stripped from the objective: only the 4 silver ids are split.
    assert res["train_size"] + res["val_size"] == 4
    assert res["n_anchors_excluded"] == 1
