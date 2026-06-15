"""Online ranking orchestrator — the entry point ``rank.py`` calls.

Wires the ONLINE pipeline together within the ≤5-min, ≤16 GB, CPU-only,
no-network budget:

    1. stream candidates + load artifacts (io_utils, index)
    2. semantic score (index) + lexical BM25 score
    3. structured features with skill-gating (features)
    4. honeypot detection → floor (honeypot)
    5. behavioral multiplier (behavioral)
    6. fuse / LTR → composite → select top 100 (fusion/ltr, scorer)
    7. optional shortlist cross-encoder re-rank (cross_encoder)
    8. grounded reasoning (reasoning) → write + self-validate CSV (io_utils)

Owns orchestration and the determinism/timing contract only; the actual scoring
logic lives in the modules above. Thin top-level ``rank.py`` delegates here.
"""
