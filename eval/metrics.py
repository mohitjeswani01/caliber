"""Ranking metrics: NDCG@k, MAP, P@k — the scoring backbone.

Pure, dependency-light implementations of the exact metrics the challenge uses,
so our offline evaluation mirrors the hidden scoring:

    composite = 0.50·NDCG@10 + 0.30·NDCG@50 + 0.15·MAP + 0.05·P@10

- NDCG@k with graded relevance tiers (standard log2 discount),
- MAP across relevance levels (tier 3+ counts as relevant for P@k),
- P@k.

These are tested directly (``tests/``) against known fixtures because every
tuning decision trusts them. Stub only — no logic yet.
"""
