"""Rank fusion across heterogeneous retrieval signals (ONLINE).

The cheap retrieval stage produces several *rankings* of the pool on
incomparable scales — bi-encoder cosine (~0.3–0.8, tight band) and lexical BM25
(unbounded, 0–30+). Summing those raw scores lets the larger-scale signal
silently dominate and needs fragile per-feature normalisation. Reciprocal Rank
Fusion (RRF) sidesteps that entirely: it uses only each candidate's *rank
position* within each list, so only ordinal agreement matters.

This is contract ``reciprocal_rank_fusion(*rankings, k=60) -> fused`` from
ARCHITECTURE.md §3 — used to fuse the retrieval rankings into the ~800-candidate
shortlist that the cross-encoder then reranks. Pure and deterministic.
"""

from __future__ import annotations


def reciprocal_rank_fusion(*rankings: list[str], k: int = 60) -> dict[str, float]:
    """Fuse ordered rankings into one ``{candidate_id: fused_score}`` map.

    RRF formula (Cormack et al., 2009)::

        score(id) = sum over each ranking R containing id of  1 / (k + rank_R(id))

    where ``rank_R(id)`` is the 1-based position of ``id`` in ranking ``R`` (best
    first), and ``k`` (default 60) damps the contribution so the top positions
    dominate while lower ranks taper off smoothly.

    Why RRF over summing raw scores: it is scale-free (only positions matter, so
    cosine and BM25 fuse cleanly without normalisation), the ``1/(k+rank)`` curve
    front-loads weight onto the head of each list (matching where NDCG@10/@50
    lives), and a candidate **missing from some rankings simply contributes
    nothing from those** rather than being penalised with a fake zero score.

    Args:
        *rankings: each a list of ``candidate_id`` strings, ordered best-first.
            Within a single ranking, the first occurrence of an id wins (a later
            duplicate is ignored) so the result is well-defined.
        k: RRF damping constant (default 60).

    Returns:
        ``{candidate_id: fused_score}`` for every id appearing in any ranking.
        Empty when no rankings (or only empty rankings) are passed.
    """
    fused: dict[str, float] = {}
    for ranking in rankings:
        seen: set[str] = set()
        for position, cid in enumerate(ranking):
            if cid in seen:
                # Honour only the best (first) position of a duplicated id.
                continue
            seen.add(cid)
            rank = position + 1  # 1-based
            fused[cid] = fused.get(cid, 0.0) + 1.0 / (k + rank)
    return fused
