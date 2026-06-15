"""Central configuration: paths, model names, weights, and tunable constants.

Single source of truth for everything the rest of the package needs to agree on:

- Filesystem layout (``data/``, ``artifacts/``, ``models/`` locations).
- The local sentence-transformer model id used offline AND online (must be the
  same model, cached locally, so ``rank.py`` makes no network call).
- Hybrid-score component weights and the behavioral-multiplier envelope bounds
  (Section 4 & 6 of STRATEGY.md). These are *defaults* to be tuned against the
  silver labels, not frozen guesses.
- Determinism knobs: the global random seed and the canonical tie-break key.

Keeping these out of the logic modules makes the weights easy to sweep during
tuning and keeps ``rank.py`` reproducible.
"""
