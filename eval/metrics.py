"""Ranking metrics: NDCG@k, MAP, P@k — the scoring backbone.

Pure, dependency-light implementations of the EXACT metrics the challenge uses,
so our offline evaluation mirrors the hidden scoring. From the official
submission spec (``data/challenge/submission_spec.docx``, §4):

    Final composite = 0.50·NDCG@10 + 0.30·NDCG@50 + 0.15·MAP + 0.05·P@10

Definitions implemented here (all deterministic, stdlib-only):

- **DCG@k**   gain = ``2**rel - 1``; discount = ``1 / log2(rank + 1)`` with
              ``rank`` starting at 1. So the item at position ``i`` (0-based)
              contributes ``(2**rel_i - 1) / log2(i + 2)``.
- **NDCG@k**  ``DCG@k / IDCG@k`` where IDCG@k is the DCG of the ideal ordering
              (relevances sorted descending) of the SAME multiset. If
              ``IDCG@k == 0`` (e.g. all-zero relevances) NDCG is ``0.0`` — no
              division by zero.
- **AP**      graded labels are binarized at ``threshold`` (``rel >= threshold``
              is relevant); AP = mean of precision@i taken at each position ``i``
              where a relevant item appears; ``0.0`` if there are no relevant
              items. Note this normalizes by the count of relevant items present
              in the ranking (the standard "no fixed catalog size" convention).
- **P@k**     fraction of the top-k that are relevant (``rel >= threshold``).

NOTE on the official binary threshold: the spec defines P@10 as the fraction of
the top-10 that are "relevant (tier 3+)". That cutoff is named here as
``OFFICIAL_RELEVANCE_THRESHOLD = 3.0`` and is the default for ``evaluate_ranking``
(the official composite path). The low-level ``average_precision`` /
``precision_at_k`` keep ``threshold`` as a parameter (default ``1.0``) so the
same primitives serve looser binary diagnostics; NDCG is graded and ignores
``threshold`` entirely.

These are tested directly (``tests/test_metrics.py``) against hand-computed
fixtures because every tuning decision downstream trusts them.
"""

from __future__ import annotations

from math import log2

__all__ = [
    "dcg_at_k",
    "ndcg_at_k",
    "average_precision",
    "precision_at_k",
    "composite_score",
    "evaluate_ranking",
]

# Official composite weights (submission_spec.docx §4). Sum to 1.0.
COMPOSITE_WEIGHTS = {
    "ndcg@10": 0.50,
    "ndcg@50": 0.30,
    "map": 0.15,
    "p@10": 0.05,
}

# Binary relevance cutoff for the OFFICIAL metric. submission_spec.docx §4
# defines P@10 as the "Fraction of top-10 that are 'relevant' (tier 3+)", and §7
# pins honeypots to tier 0. So an item counts as relevant iff its graded tier is
# >= 3. This is the default for the official composite path (MAP and P@10); pass
# a different threshold to evaluate_ranking only for looser diagnostics.
OFFICIAL_RELEVANCE_THRESHOLD = 3.0


def dcg_at_k(relevances: list[float], k: int) -> float:
    """Discounted Cumulative Gain over the first ``k`` items.

    DCG@k = sum_{i=1..min(k,n)} (2**rel_i - 1) / log2(i + 1),
    where ``i`` is the 1-based rank position. ``k`` larger than the list length
    is handled by truncation (``min``).
    """
    k = min(k, len(relevances))
    total = 0.0
    for i in range(k):
        rel = relevances[i]
        # rank = i + 1, so discount denominator = log2(rank + 1) = log2(i + 2)
        total += (2.0**rel - 1.0) / log2(i + 2)
    return total


def ndcg_at_k(relevances: list[float], k: int) -> float:
    """Normalized DCG@k against the ideal ordering of the same relevances.

    NDCG@k = DCG@k / IDCG@k, where IDCG@k is the DCG of ``relevances`` sorted
    descending. Returns ``0.0`` when IDCG@k == 0 (e.g. all relevances are 0),
    avoiding division by zero. Graded: uses raw relevance values, not a
    binary threshold.
    """
    idcg = dcg_at_k(sorted(relevances, reverse=True), k)
    if idcg == 0.0:
        return 0.0
    return dcg_at_k(relevances, k) / idcg


def precision_at_k(relevances: list[float], k: int, threshold: float = 1.0) -> float:
    """Fraction of the top-``k`` items that are relevant (``rel >= threshold``).

    ``k`` is clamped to the list length, so a ``k`` larger than the ranking
    divides by the actual number of items present (and an empty ranking returns
    ``0.0``).
    """
    k = min(k, len(relevances))
    if k == 0:
        return 0.0
    hits = sum(1 for rel in relevances[:k] if rel >= threshold)
    return hits / k


def average_precision(relevances: list[float], threshold: float = 1.0) -> float:
    """Average Precision for a single ranked list with graded labels.

    Labels are binarized at ``threshold`` (``rel >= threshold`` is relevant).
    AP = mean over every position ``i`` holding a relevant item of the
    precision@i at that position. Returns ``0.0`` if no item is relevant.

    This is the standard single-query AP that normalizes by the number of
    relevant items found in the ranking itself (no external catalog size).
    """
    num_relevant = 0
    precision_sum = 0.0
    for i, rel in enumerate(relevances):
        if rel >= threshold:
            num_relevant += 1
            # precision at this position (1-based rank = i + 1)
            precision_sum += num_relevant / (i + 1)
    if num_relevant == 0:
        return 0.0
    return precision_sum / num_relevant


def composite_score(
    ndcg10: float, ndcg50: float, map_score: float, p10: float
) -> float:
    """Official weighted composite (submission_spec.docx §4).

    composite = 0.50·NDCG@10 + 0.30·NDCG@50 + 0.15·MAP + 0.05·P@10
    """
    return (
        COMPOSITE_WEIGHTS["ndcg@10"] * ndcg10
        + COMPOSITE_WEIGHTS["ndcg@50"] * ndcg50
        + COMPOSITE_WEIGHTS["map"] * map_score
        + COMPOSITE_WEIGHTS["p@10"] * p10
    )


def evaluate_ranking(
    ranked_ids: list[str],
    relevance_by_id: dict[str, float],
    threshold: float = OFFICIAL_RELEVANCE_THRESHOLD,
) -> dict:
    """Score a ranked id list against a relevance lookup (OFFICIAL composite).

    Maps each id in ``ranked_ids`` (in order) to its relevance via
    ``relevance_by_id``; an id missing from the dict counts as ``0.0`` relevance
    (it was ranked but is not in ground truth → not relevant). Returns the four
    official component metrics plus the weighted composite.

    ``threshold`` defaults to ``OFFICIAL_RELEVANCE_THRESHOLD`` (3.0, "tier 3+"
    per the spec) so this reproduces the official scoring out of the box. It
    only gates the binary metrics (MAP, P@10); NDCG is graded and unaffected.
    Override it (e.g. ``threshold=1.0``) for looser binary diagnostics.
    """
    relevances = [float(relevance_by_id.get(cid, 0.0)) for cid in ranked_ids]
    ndcg10 = ndcg_at_k(relevances, 10)
    ndcg50 = ndcg_at_k(relevances, 50)
    map_score = average_precision(relevances, threshold=threshold)
    p10 = precision_at_k(relevances, 10, threshold=threshold)
    return {
        "ndcg@10": ndcg10,
        "ndcg@50": ndcg50,
        "map": map_score,
        "p@10": p10,
        "composite": composite_score(ndcg10, ndcg50, map_score, p10),
    }
