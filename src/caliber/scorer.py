"""Final composite scoring + deterministic top-100 selection (ONLINE).

Assembles the end-to-end score per candidate and selects the ranking:

    final = base_relevance (fusion or LTR) × behavioral_multiplier
    detected honeypots → forced to the score floor

Then selects the top 100, enforces non-increasing scores by rank, and breaks
ties deterministically (candidate_id ascending) to match the validator exactly.

This is where the pieces compose into the number that gets ranked; it owns the
score floor, the honeypot override, and the deterministic ordering contract.
"""
