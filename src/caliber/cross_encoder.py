"""Optional CPU cross-encoder re-rank of the shortlist (ONLINE, budgeted).

After fusion narrows the pool to a small shortlist (e.g. top few hundred), a
small local cross-encoder can re-score (JD, candidate-text) pairs for sharper
top-10/top-50 ordering — where 80% of the metric lives.

Strictly budget-aware: runs only over the shortlist (never the full 100K), CPU
only, model cached locally (no download at runtime). If enabling it risks the
5-minute wall-clock, it must be gated off — substance ranking must stand alone.
A re-ranker is a precision booster for the head of the list, not a crutch.
"""
