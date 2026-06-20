"""Tests for src/caliber/ltr.py + scripts/train_ltr.py — the learned combiner.

LTR is an ENHANCEMENT over the tuned hand-weights: a LightGBM lambdarank booster
that replaces ONLY the base-relevance ``combine`` step, over the SAME ordered
``COMPOSITE_FEATURE_NAMES`` vector. These tests pin the contract we must defend:

  * PREDICT INTERFACE — ``predict(values, weights)`` is a drop-in ``combine_fn``:
    same signature scorer calls, returns one float, orders features deterministically.
  * GRACEFUL FALLBACK — with NO model artifact (or lightgbm missing) ``predict``
    is byte-identical to ``scorer.combine``, so rank.py is never broken.
  * DETERMINISM — same vector → same score; a re-loaded booster matches the
    in-memory one.
  * TRAIN → PREDICT ROUND-TRIP — on tiny mock data the trainer learns a monotone
    combiner, the floor + behavioural multiplier are still applied, and the
    held-out head-to-head report is produced.

The round-trip / training tests need lightgbm; they ``importorskip`` so a checkout
without it still runs the interface + fallback tests (which need no model at all).
"""

import functools
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

from caliber import ltr
from caliber.scorer import (
    COMPOSITE_FEATURE_NAMES,
    DEFAULT_WEIGHTS,
    HONEYPOT_FLOOR,
    CandidateScore,
    combine,
    score_candidates,
)

# scripts/ is importable as a namespace package (pytest puts ROOT on sys.path).
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts import train_ltr  # noqa: E402

# The 9 structured-feature names the cache stores (composite minus the two
# description-level signals) — mirrors eval.sweep.FEATURE_KEYS.
_STRUCT_KEYS = tuple(n for n in COMPOSITE_FEATURE_NAMES if n not in ("ce_score", "semantic_sim"))


# --------------------------------------------------------------------------- #
# Helpers — a composite value dict and a mock sweep cache.
# --------------------------------------------------------------------------- #
def _values(level: float, ce=None, sem=None):
    """A composite value dict with every structured feature at ``level`` (and
    ce/semantic defaulting to ``level`` when not given) — the mapping scorer hands
    to its combine_fn."""
    v = {k: level for k in _STRUCT_KEYS}
    v["ce_score"] = level if ce is None else ce
    v["semantic_sim"] = level if sem is None else sem
    return v


def _entry(level: float, *, beh=1.0, hp=False, ce=None, sem=None):
    """A cache 'scored' entry for one candidate at feature ``level``."""
    feats = {k: level for k in _STRUCT_KEYS}
    return {
        "features": feats,
        "ce_score": level if ce is None else ce,
        "semantic_sim": level if sem is None else sem,
        "behavioral_mult": beh,
        "is_honeypot": hp,
    }


def _mock_cache():
    """A separable mock pool: 5 graded silver ids per grade 0..4 (features ∝ grade),
    plus distractors (grade 0, low features) and one strong-but-honeypot id.

    Features scale with grade so a correct learner orders high grades on top; the
    head-to-head then has real positives in both train and val halves.
    """
    scored = {}
    grades = {}
    # graded silver: 5 per grade so stratified_split yields train+val positives.
    for g in range(5):
        for j in range(5):
            cid = f"S{g}_{j}"
            scored[cid] = _entry(g / 4.0)
            grades[cid] = float(g)
    # distractors: ungraded, low features, relevance 0.
    for j in range(40):
        scored[f"D{j}"] = _entry(0.05)
    # a strong-on-paper honeypot — must be floored regardless of features.
    scored["HP"] = _entry(1.0, hp=True)
    cache = {"all_pool_ids": sorted(scored), "scored": scored}
    return cache, grades


# --------------------------------------------------------------------------- #
# 1. PREDICT INTERFACE — vectorization + drop-in signature.
# --------------------------------------------------------------------------- #
def test_composite_vector_orders_features_and_maps_none_to_nan():
    vals = {name: float(i) for i, name in enumerate(COMPOSITE_FEATURE_NAMES)}
    vals["ce_score"] = None  # CE skipped → NaN
    vec = ltr.composite_vector(vals)
    assert vec.shape == (len(COMPOSITE_FEATURE_NAMES),)
    ce_idx = COMPOSITE_FEATURE_NAMES.index("ce_score")
    assert np.isnan(vec[ce_idx])
    # every other slot matches its position-encoded value, in order.
    for i, name in enumerate(COMPOSITE_FEATURE_NAMES):
        if name == "ce_score":
            continue
        assert vec[i] == float(i)


def test_composite_matrix_stacks_rows():
    rows = [_values(0.1), _values(0.9)]
    M = ltr.composite_matrix(rows)
    assert M.shape == (2, len(COMPOSITE_FEATURE_NAMES))
    assert ltr.composite_matrix([]).shape == (0, len(COMPOSITE_FEATURE_NAMES))


# --------------------------------------------------------------------------- #
# 2. GRACEFUL FALLBACK — no model ⇒ identical to scorer.combine.
# --------------------------------------------------------------------------- #
def test_predict_falls_back_to_handweights_when_model_absent(tmp_path):
    ltr.clear_cache()
    missing = tmp_path / "no_such_model.txt"
    vals = _values(0.5, ce=0.7, sem=0.3)
    got = ltr.predict(vals, DEFAULT_WEIGHTS, model_path=missing)
    assert got == pytest.approx(combine(vals, DEFAULT_WEIGHTS))
    assert ltr.is_available(missing) is False


def test_predict_fallback_renormalizes_like_combine_when_ce_none(tmp_path):
    ltr.clear_cache()
    missing = tmp_path / "absent.txt"
    vals = _values(0.4, sem=0.6)
    vals["ce_score"] = None  # CE skipped — combine drops the weight & renormalizes
    assert ltr.predict(vals, DEFAULT_WEIGHTS, model_path=missing) == pytest.approx(
        combine(vals, DEFAULT_WEIGHTS)
    )


def test_predict_batch_fallback_matches_combine(tmp_path):
    ltr.clear_cache()
    missing = tmp_path / "absent.txt"
    rows = [_values(0.2), _values(0.8, ce=None)]
    got = ltr.predict_batch(rows, DEFAULT_WEIGHTS, model_path=missing)
    assert got == pytest.approx([combine(r, DEFAULT_WEIGHTS) for r in rows])


def test_load_model_returns_and_caches_none_for_missing(tmp_path):
    ltr.clear_cache()
    missing = tmp_path / "nope.txt"
    assert ltr.load_model(missing) is None
    # cached: a second call (still missing) is served from cache, still None.
    assert ltr.load_model(missing) is None


# --------------------------------------------------------------------------- #
# 3. TRAIN → PREDICT (needs lightgbm).
# --------------------------------------------------------------------------- #
# The DEFAULT_PARAMS leaf size (min_data_in_leaf=20) is tuned for the ~3000-row
# real pool; on the 50-row mock the positive leaf (8 rows) can't form, so the
# booster makes no splits. Shrink the leaf/regularisation for the tiny fixture so
# the trainer is genuinely exercised (the params are a pass-through to lightgbm).
_TEST_PARAMS = {**train_ltr.DEFAULT_PARAMS, "min_data_in_leaf": 1,
                "num_leaves": 7, "max_depth": 3, "lambda_l2": 0.0}


def _trained_model():
    cache, grades = _mock_cache()
    train_ids, val_ids = train_ltr.sweep.stratified_split(grades, seed=42)
    X, y, used = train_ltr.build_training_data(
        cache, grades, train_ids, val_ids, anchor_ids=set()
    )
    model = train_ltr.train_ranker(X, y, params=_TEST_PARAMS, num_boost_round=60)
    return cache, grades, train_ids, val_ids, model, X, y, used


def test_train_predict_roundtrip_and_determinism(tmp_path):
    pytest.importorskip("lightgbm")
    cache, grades, train_ids, val_ids, model, X, y, used = _trained_model()

    # held-out positives never entered the training matrix (overfit guard).
    assert val_ids and not (set(used) & val_ids)
    # distractors were included as relevance-0 rows.
    assert any(cid.startswith("D") for cid in used)

    # predict returns one float, deterministically (same vector → same score).
    strong = ltr.predict(_values(1.0), model=model)
    weak = ltr.predict(_values(0.0), model=model)
    assert isinstance(strong, float)
    assert ltr.predict(_values(1.0), model=model) == strong   # deterministic
    # learned a monotone combiner on the separable mock data.
    assert strong > weak

    # save → reload → identical predictions (no network, file round-trip).
    out = tmp_path / "ltr.txt"
    train_ltr.save_model(model, out)
    ltr.clear_cache()
    reloaded = ltr.load_model(out)
    assert reloaded is not None
    assert ltr.is_available(out) is True
    assert ltr.predict(_values(0.7), model=reloaded) == pytest.approx(
        ltr.predict(_values(0.7), model=model), abs=1e-9
    )


def test_predict_batch_matches_per_candidate_predict():
    pytest.importorskip("lightgbm")
    _, _, _, _, model, _, _, _ = _trained_model()
    rows = [_values(0.1), _values(0.5), _values(0.9)]
    batch = ltr.predict_batch(rows, model=model)
    one_by_one = [ltr.predict(r, model=model) for r in rows]
    assert batch == pytest.approx(one_by_one, abs=1e-9)


def test_rerank_with_model_applies_floor_and_behavioral():
    pytest.importorskip("lightgbm")
    cache, _, _, _, model, _, _, _ = _trained_model()

    # honeypot floored LAST → strictly below every real candidate, at the bottom
    # of the scored block (its great features cannot lift it).
    assert train_ltr.model_final(cache["scored"]["HP"], model) == HONEYPOT_FLOOR
    ranked = train_ltr.rerank_with_model(cache, model)
    assert ranked[-1] == "HP"

    # behavioural multiplier still scales the base (LTR replaces ONLY the combine).
    e = cache["scored"]["S4_0"]
    base = ltr.predict(train_ltr.cache_values(e), model=model)
    boosted = dict(e); boosted["behavioral_mult"] = 1.15
    assert train_ltr.model_final(boosted, model) == pytest.approx(base * 1.15)


def test_train_ranker_rejects_empty():
    pytest.importorskip("lightgbm")
    with pytest.raises(ValueError):
        train_ltr.train_ranker(np.empty((0, len(COMPOSITE_FEATURE_NAMES))), [])


# --------------------------------------------------------------------------- #
# 4. END-TO-END run() — cache + silver + anchors on disk → model, meta, report.
# --------------------------------------------------------------------------- #
def test_run_end_to_end_reports_head_to_head(tmp_path):
    pytest.importorskip("lightgbm")
    cache, grades = _mock_cache()
    cache_path = tmp_path / "pool_cache.json"
    cache_path.write_text(json.dumps(cache), encoding="utf-8")

    # silver_labels.json in the make_silver_labels record shape.
    silver = [{"candidate_id": cid, "grade_final": g} for cid, g in grades.items()]
    silver_path = tmp_path / "silver_labels.json"
    silver_path.write_text(json.dumps(silver), encoding="utf-8")
    # one manual anchor that exists in the cache — must be held out of tuning.
    anchors = [{"candidate_id": "S4_4", "grade": 4}]
    anchor_path = tmp_path / "manual_grades.json"
    anchor_path.write_text(json.dumps(anchors), encoding="utf-8")

    out = train_ltr.run(
        cache_path=cache_path,
        silver_labels_path=silver_path,
        manual_grades_path=anchor_path,
        model_out=tmp_path / "ltr.txt",
        meta_out=tmp_path / "ltr_meta.json",
        num_boost_round=60,
        params=_TEST_PARAMS,
    )

    assert (tmp_path / "ltr.txt").exists()
    meta = json.loads((tmp_path / "ltr_meta.json").read_text(encoding="utf-8"))
    assert meta["feature_names"] == list(COMPOSITE_FEATURE_NAMES)
    assert meta["n_anchors_excluded"] == 1
    assert "VALIDATION" in out["report"]
    assert isinstance(out["adopt"], bool)
    # the anchor never entered training (held out).
    res = out["result"]
    assert "val" in res["ltr"] and "val" in res["baseline"]
    # honeypot stays out of the head for BOTH rankers.
    assert res["honeypots_ltr"]["top10"] == 0
    assert res["honeypots_baseline"]["top10"] == 0


# --------------------------------------------------------------------------- #
# 5. SCORER INTEGRATION — ltr.predict is a true drop-in combine_fn (no change to
#    scorer's feature assembly). Reuses the real scorer + test_scorer's fakes.
# --------------------------------------------------------------------------- #
def test_predict_is_dropin_combine_fn_fallback_matches_default(tmp_path):
    """With NO model, score_candidates(combine_fn=ltr.predict) is byte-identical to
    the default hand-weighted scorer — proves the swap is safe for rank.py."""
    from tests.test_scorer import _fit_stuffer_honey_setup

    ltr.clear_cache()
    missing = str(tmp_path / "absent.txt")
    fallback = functools.partial(ltr.predict, model_path=missing)  # forces no-model path

    kwargs, _ = _fit_stuffer_honey_setup(ce_logits=None)
    base = score_candidates(ce_enabled=False, **kwargs)
    kwargs2, _ = _fit_stuffer_honey_setup(ce_logits=None)
    via_ltr = score_candidates(ce_enabled=False, combine_fn=fallback, **kwargs2)

    assert base.keys() == via_ltr.keys()
    for cid in base:
        assert via_ltr[cid].final_score == pytest.approx(base[cid].final_score)


def test_scorer_runs_end_to_end_with_ltr_model_and_floors_honeypot():
    """score_candidates with a REAL trained booster as combine_fn — the whole
    pipeline runs, returns the documented breakdown, the honeypot is still floored
    (LTR replaces ONLY the base combine), and the behavioural multiplier still
    applies (final == base_combine × multiplier within (0,1))."""
    pytest.importorskip("lightgbm")
    from tests.test_scorer import _fit_stuffer_honey_setup

    _, _, _, _, model, _, _, _ = _trained_model()
    combine_fn = functools.partial(ltr.predict, model=model)

    kwargs, _ = _fit_stuffer_honey_setup(ce_logits=None)
    out = score_candidates(ce_enabled=False, combine_fn=combine_fn, **kwargs)

    assert set(out.keys()) == {"C_FIT", "C_STUFF", "C_HONEY"}
    for cs in out.values():
        assert isinstance(cs, CandidateScore)
        assert math.isfinite(cs.final_score)
    # honeypot floor still wins over everything (applied AFTER the LTR combine).
    assert out["C_HONEY"].final_score == HONEYPOT_FLOOR
    assert out["C_FIT"].final_score > out["C_HONEY"].final_score
    # real (non-honeypot) base came from the booster, squashed to (0,1), and the
    # behavioural multiplier scaled it — final == base × mult, base in (0,1).
    fit = out["C_FIT"]
    assert 0.0 < fit.base_score < 1.0
    assert fit.final_score == pytest.approx(fit.base_score * fit.behavioral_mult)
