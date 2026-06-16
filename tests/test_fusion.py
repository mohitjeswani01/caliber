"""Tests for src/caliber/fusion.py — Reciprocal Rank Fusion.

Every expected fused score is computed BY HAND in the comments first, then
asserted to 6 decimals. RRF is what fuses the retrieval rankings into the
shortlist, so a silent bug here would distort the whole candidate funnel.

RRF: score(id) = sum over each ranking containing id of 1 / (k + rank), rank
1-based, default k = 60.
"""

import pytest

from caliber.fusion import reciprocal_rank_fusion

PLACES = 6

# Reference reciprocals at k = 60 (rank 1, 2, 3):
#   1/61 = 0.016393442622950...
#   1/62 = 0.016129032258064...
#   1/63 = 0.015873015873015...


def test_two_rankings_hand_verified():
    """Two lists, k=60, every id verified by hand.

    A = ["x", "y", "z"]   -> ranks x:1, y:2, z:3
    B = ["y", "x", "w"]   -> ranks y:1, x:2, w:3

    x = 1/61 + 1/62 = 0.016393442623 + 0.016129032258 = 0.032522474881
    y = 1/62 + 1/61 = 0.016129032258 + 0.016393442623 = 0.032522474881
    z = 1/63                                            = 0.015873015873
    w = 1/63                                            = 0.015873015873
    """
    a = ["x", "y", "z"]
    b = ["y", "x", "w"]
    fused = reciprocal_rank_fusion(a, b)

    assert round(fused["x"], PLACES) == 0.032522
    assert round(fused["y"], PLACES) == 0.032522
    assert round(fused["z"], PLACES) == 0.015873
    assert round(fused["w"], PLACES) == 0.015873
    # x and y tie exactly (symmetric positions); both beat z and w.
    assert fused["x"] == pytest.approx(fused["y"], abs=1e-12)
    assert fused["x"] > fused["z"]


def test_id_in_only_some_rankings():
    """An id present in only one of three rankings contributes only from it.

    A = ["a", "b"]        -> a:1, b:2
    B = ["a", "c"]        -> a:1, c:2
    C = ["a", "d"]        -> a:1, d:2

    a = 1/61 + 1/61 + 1/61 = 3/61            = 0.049180327869
    b = 1/62 (only in A)                     = 0.016129032258
    c = 1/62 (only in B)                     = 0.016129032258
    d = 1/62 (only in C)                     = 0.016129032258
    """
    fused = reciprocal_rank_fusion(["a", "b"], ["a", "c"], ["a", "d"])

    assert round(fused["a"], PLACES) == round(3 / 61, PLACES)  # 0.049180
    assert round(fused["b"], PLACES) == 0.016129
    assert round(fused["c"], PLACES) == 0.016129
    assert round(fused["d"], PLACES) == 0.016129
    # The id appearing in all three rankings dominates.
    assert fused["a"] > fused["b"]
    assert set(fused) == {"a", "b", "c", "d"}


def test_custom_k():
    """k changes the damping; verify with k=1.

    A = ["p", "q"] -> p:1/(1+1)=0.5, q:1/(1+2)=0.333333
    """
    fused = reciprocal_rank_fusion(["p", "q"], k=1)
    assert round(fused["p"], PLACES) == 0.5
    assert round(fused["q"], PLACES) == round(1 / 3, PLACES)  # 0.333333


def test_single_ranking():
    """A single ranking just gives 1/(k+rank) per position, no crash."""
    fused = reciprocal_rank_fusion(["m", "n", "o"])
    assert round(fused["m"], PLACES) == 0.016393
    assert round(fused["n"], PLACES) == 0.016129
    assert round(fused["o"], PLACES) == 0.015873
    # Order is preserved by score: earlier rank -> higher score.
    assert fused["m"] > fused["n"] > fused["o"]


def test_empty_inputs():
    """No rankings, and empty rankings, both yield an empty dict (no crash)."""
    assert reciprocal_rank_fusion() == {}
    assert reciprocal_rank_fusion([], []) == {}


def test_duplicate_id_within_ranking_uses_first_position():
    """A duplicated id within one ranking counts only its first (best) rank.

    A = ["a", "b", "a"] -> a's best position is rank 1; the rank-3 dup is ignored.
    a = 1/61 = 0.016393 (NOT 1/61 + 1/63)
    b = 1/62 = 0.016129
    """
    fused = reciprocal_rank_fusion(["a", "b", "a"])
    assert round(fused["a"], PLACES) == 0.016393
    assert round(fused["b"], PLACES) == 0.016129
