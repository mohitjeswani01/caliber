"""Tests for eval/metrics.py — the ranking metrics.

Every expected value below is computed BY HAND in the comments first, then
asserted. These metrics are the measuring stick for all weight tuning, so a
silent bug here would quietly corrupt every downstream decision.
"""

import math

import pytest

from eval.metrics import (
    OFFICIAL_RELEVANCE_THRESHOLD,
    average_precision,
    composite_score,
    dcg_at_k,
    evaluate_ranking,
    ndcg_at_k,
    precision_at_k,
)

ABS = 1e-9


# ---------------------------------------------------------------------------
# NDCG — main hand-verified fixture
# ---------------------------------------------------------------------------
def test_ndcg_hand_computed():
    """relevances = [3, 2, 3, 0, 1, 2], NDCG@6 verified by hand.

    gain(rel) = 2**rel - 1; discount at 1-based rank r = 1 / log2(r + 1).

    DCG@6:
      r1: (2^3-1)/log2(2) = 7 / 1            = 7.000000000
      r2: (2^2-1)/log2(3) = 3 / 1.5849625    = 1.892789261
      r3: (2^3-1)/log2(4) = 7 / 2            = 3.500000000
      r4: (2^0-1)/log2(5) = 0 / 2.3219281    = 0.000000000
      r5: (2^1-1)/log2(6) = 1 / 2.5849625    = 0.386852807
      r6: (2^2-1)/log2(7) = 3 / 2.8073549    = 1.068621561
      DCG@6 = 13.848263629

    Ideal order = [3, 3, 2, 2, 1, 0] -> gains [7, 7, 3, 3, 1, 0]
    IDCG@6:
      7/1 + 7/1.5849625 + 3/2 + 3/2.3219281 + 1/2.5849625 + 0
      = 7 + 4.416508275 + 1.5 + 1.292029674 + 0.386852807 + 0
      = 14.595390756

    NDCG@6 = 13.848263629 / 14.595390756 = 0.9488107485678985
    """
    rels = [3, 2, 3, 0, 1, 2]
    assert dcg_at_k(rels, 6) == pytest.approx(13.848263629272981, abs=ABS)
    assert dcg_at_k(sorted(rels, reverse=True), 6) == pytest.approx(
        14.595390756454924, abs=ABS
    )
    assert ndcg_at_k(rels, 6) == pytest.approx(0.9488107485678985, abs=1e-6)


def test_ndcg_perfect_ranking_is_one():
    # Already in ideal (descending) order -> DCG == IDCG -> NDCG == 1.0 exactly.
    rels = [3, 3, 2, 2, 1, 0]
    assert ndcg_at_k(rels, 6) == 1.0
    assert ndcg_at_k(rels, 3) == 1.0  # top-k of a perfect prefix is still perfect


def test_ndcg_reversed_is_worse_than_perfect():
    perfect = [3, 3, 2, 2, 1, 0]
    reversed_worst = [0, 1, 2, 2, 3, 3]
    assert ndcg_at_k(reversed_worst, 6) < ndcg_at_k(perfect, 6)
    # And strictly less than 1.0.
    assert ndcg_at_k(reversed_worst, 6) < 1.0


def test_ndcg_all_zero_is_zero_no_exception():
    # IDCG == 0 -> guarded, returns 0.0 rather than dividing by zero.
    assert ndcg_at_k([0, 0, 0, 0], 10) == 0.0
    assert dcg_at_k([0, 0, 0], 3) == 0.0


def test_ndcg_k_larger_than_list_no_crash():
    rels = [3, 2, 1]
    # k=100 on a length-3 list just uses all 3 items; equals NDCG@3.
    assert ndcg_at_k(rels, 100) == pytest.approx(ndcg_at_k(rels, 3), abs=ABS)
    assert dcg_at_k(rels, 100) == pytest.approx(dcg_at_k(rels, 3), abs=ABS)


def test_empty_inputs():
    assert dcg_at_k([], 10) == 0.0
    assert ndcg_at_k([], 10) == 0.0
    assert precision_at_k([], 10) == 0.0
    assert average_precision([]) == 0.0


# ---------------------------------------------------------------------------
# Average precision & precision@k — binary fixture
# ---------------------------------------------------------------------------
def test_average_precision_binary_hand_computed():
    """relevances = [1, 0, 1, 1, 0, 1] (binary, threshold = 1.0).

    Relevant items sit at 1-based ranks 1, 3, 4, 6.
      precision@1 = 1/1 = 1.000000
      precision@3 = 2/3 = 0.666667
      precision@4 = 3/4 = 0.750000
      precision@6 = 4/6 = 0.666667
    AP = (1.0 + 0.666667 + 0.75 + 0.666667) / 4 = 3.083333 / 4 = 0.770833333
    """
    rels = [1, 0, 1, 1, 0, 1]
    expected = (1.0 + 2 / 3 + 3 / 4 + 4 / 6) / 4
    assert average_precision(rels, threshold=1.0) == pytest.approx(expected, abs=ABS)
    assert average_precision(rels, threshold=1.0) == pytest.approx(0.7708333333, abs=1e-9)


def test_average_precision_no_relevant_is_zero():
    assert average_precision([0, 0, 0], threshold=1.0) == 0.0
    # threshold gates: tier-2 items aren't "relevant" at threshold 3.
    assert average_precision([2, 2, 2], threshold=3.0) == 0.0


def test_average_precision_graded_threshold():
    """Graded labels binarized at threshold=3.0 (the official 'tier 3+').

    relevances = [3, 1, 4, 0, 3] -> relevant (rel>=3) at ranks 1, 3, 5.
      precision@1 = 1/1 = 1.000000
      precision@3 = 2/3 = 0.666667
      precision@5 = 3/5 = 0.600000
    AP = (1.0 + 0.666667 + 0.6) / 3 = 2.266667 / 3 = 0.755555556
    """
    rels = [3, 1, 4, 0, 3]
    expected = (1.0 + 2 / 3 + 3 / 5) / 3
    assert average_precision(rels, threshold=3.0) == pytest.approx(expected, abs=ABS)


def test_precision_at_k_hand_computed():
    rels = [1, 0, 1, 1, 0, 1]
    # top-4 = [1,0,1,1] -> 3 relevant / 4 = 0.75
    assert precision_at_k(rels, 4, threshold=1.0) == pytest.approx(0.75, abs=ABS)
    # top-3 = [1,0,1] -> 2 relevant / 3 = 0.666667
    assert precision_at_k(rels, 3, threshold=1.0) == pytest.approx(2 / 3, abs=ABS)
    # k beyond length clamps to 6: 4 relevant / 6
    assert precision_at_k(rels, 100, threshold=1.0) == pytest.approx(4 / 6, abs=ABS)


def test_precision_at_k_graded_threshold():
    # rel >= 3 only: [3,2,3,0,4] -> relevant at positions 1,3,5 -> 3/5 in top5.
    assert precision_at_k([3, 2, 3, 0, 4], 5, threshold=3.0) == pytest.approx(
        3 / 5, abs=ABS
    )


# ---------------------------------------------------------------------------
# Composite
# ---------------------------------------------------------------------------
def test_composite_score_hand_computed():
    # 0.50*0.9 + 0.30*0.8 + 0.15*0.7 + 0.05*0.6
    # = 0.45 + 0.24 + 0.105 + 0.03 = 0.825
    assert composite_score(0.9, 0.8, 0.7, 0.6) == pytest.approx(0.825, abs=ABS)


def test_composite_weights_sum_to_one():
    from eval.metrics import COMPOSITE_WEIGHTS

    assert sum(COMPOSITE_WEIGHTS.values()) == pytest.approx(1.0, abs=ABS)
    # A perfect ranking on every component scores exactly 1.0.
    assert composite_score(1.0, 1.0, 1.0, 1.0) == pytest.approx(1.0, abs=ABS)


# ---------------------------------------------------------------------------
# evaluate_ranking — end to end, including a missing id
# ---------------------------------------------------------------------------
def test_evaluate_ranking_end_to_end_with_missing_id():
    """ranked_ids includes 'missing' which is absent from the dict -> rel 0.0.

    Effective relevances (in ranked order) = [3, 0, 2, 1, 0].
    threshold defaults to 1.0.

    P@10: k clamps to 5; relevant (rel>=1) at ranks 1,3,4 -> 3/5 = 0.6
    MAP : relevant at ranks 1,3,4
          precision@1 = 1/1 = 1.0
          precision@3 = 2/3 = 0.666667
          precision@4 = 3/4 = 0.75
          AP = (1.0 + 0.666667 + 0.75) / 3 = 2.416667 / 3 = 0.805555556
    """
    ranked_ids = ["a", "b", "c", "d", "missing"]
    relevance_by_id = {"a": 3.0, "b": 0.0, "c": 2.0, "d": 1.0}
    out = evaluate_ranking(ranked_ids, relevance_by_id, threshold=1.0)

    assert set(out) == {"ndcg@10", "ndcg@50", "map", "p@10", "composite"}
    assert out["p@10"] == pytest.approx(0.6, abs=ABS)
    assert out["map"] == pytest.approx((1.0 + 2 / 3 + 3 / 4) / 3, abs=ABS)

    # NDCG@10 over [3,0,2,1,0] cross-checked against the bare function.
    assert out["ndcg@10"] == pytest.approx(
        ndcg_at_k([3.0, 0.0, 2.0, 1.0, 0.0], 10), abs=ABS
    )
    # Composite must equal the weighted sum of the reported components.
    assert out["composite"] == pytest.approx(
        composite_score(out["ndcg@10"], out["ndcg@50"], out["map"], out["p@10"]),
        abs=ABS,
    )


def test_evaluate_ranking_perfect_order():
    # Ground truth descending and ranked in that order -> NDCG components == 1.0.
    ranked_ids = ["a", "b", "c"]
    rel = {"a": 3.0, "b": 2.0, "c": 1.0}
    out = evaluate_ranking(ranked_ids, rel, threshold=1.0)
    assert out["ndcg@10"] == 1.0
    assert out["ndcg@50"] == 1.0
    assert out["p@10"] == pytest.approx(1.0, abs=ABS)  # all 3 relevant, k clamps to 3


def test_evaluate_ranking_official_threshold_tier3():
    # With the official threshold=3.0, only tier-3+ count for MAP/P@10.
    ranked_ids = ["a", "b", "c", "d"]
    rel = {"a": 3.0, "b": 2.0, "c": 4.0, "d": 1.0}  # relevant: a, c
    out = evaluate_ranking(ranked_ids, rel, threshold=3.0)
    # relevant at ranks 1 and 3: AP = (1/1 + 2/3)/2 = 0.833333
    assert out["map"] == pytest.approx((1.0 + 2 / 3) / 2, abs=ABS)
    # p@10 clamps to 4: 2 relevant / 4 = 0.5
    assert out["p@10"] == pytest.approx(0.5, abs=ABS)


def test_official_threshold_constant_is_tier3():
    assert OFFICIAL_RELEVANCE_THRESHOLD == 3.0


def test_evaluate_ranking_default_uses_official_threshold():
    """The DEFAULT (no threshold arg) must apply the official tier-3+ cutoff.

    relevances in ranked order = [3, 2, 4, 1] -> relevant (rel>=3) at ranks 1,3.
      MAP  = (1/1 + 2/3) / 2 = 0.833333
      P@10 = 2 relevant / 4 (k clamps) = 0.5
    A tier-2 / tier-1 item must NOT count as relevant by default — that's the
    whole point of this regression guard.
    """
    ranked_ids = ["a", "b", "c", "d"]
    rel = {"a": 3.0, "b": 2.0, "c": 4.0, "d": 1.0}
    default_out = evaluate_ranking(ranked_ids, rel)  # no threshold passed
    explicit_out = evaluate_ranking(ranked_ids, rel, threshold=3.0)
    assert default_out == explicit_out
    assert default_out["map"] == pytest.approx((1.0 + 2 / 3) / 2, abs=ABS)
    assert default_out["p@10"] == pytest.approx(0.5, abs=ABS)
    # And it must differ from the loose threshold=1.0 reading (sanity that the
    # default isn't silently 1.0): at threshold 1.0 all four are relevant.
    loose_out = evaluate_ranking(ranked_ids, rel, threshold=1.0)
    assert loose_out["p@10"] == pytest.approx(1.0, abs=ABS)
    assert loose_out["map"] != pytest.approx(default_out["map"], abs=ABS)
