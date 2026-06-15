"""OFFLINE silver-label generation — our own ground truth (our edge).

Builds a silver-standard relevance set so we can *measure* before submitting:

    1. draw a stratified sample (strong ML titles, adjacent titles, noise,
       suspected stuffers, suspected honeypots),
    2. score each against the JD-derived rubric — an LLM MAY assist here, this
       is offline and never on the ranking path,
    3. assign silver relevance tiers (0–N) and persist the labelled set.

These labels drive weight tuning and the LTR training target. We guard against
overfitting to the LLM's taste (Section 7). Stub only — no logic yet.
"""

if __name__ == "__main__":
    raise SystemExit("make_silver_labels.py is a stub — not implemented yet.")
