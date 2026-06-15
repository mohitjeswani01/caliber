"""Caliber — Intelligent Candidate Discovery & Ranking (Redrob Track 1).

Package root for the Caliber ranking system. Modules are split along the
two-phase architecture described in ``CLAUDE.md`` / ``docs/STRATEGY.md``:

OFFLINE (no time limit, may use an LLM for label generation only):
    jd_profile, embeddings, index  — build artifacts persisted to ``artifacts/``.

ONLINE (``rank.py``, the ≤5-min CPU-only budget; reads artifacts, no network):
    schema, io_utils, features, honeypot, behavioral, cross_encoder, ltr,
    fusion, scorer, reasoning, ranker.

Nothing here performs ranking yet — these are import-safe stubs that define the
module boundaries we will fill in incrementally.
"""

__version__ = "0.0.0"
