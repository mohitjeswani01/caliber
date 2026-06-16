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

from __future__ import annotations

# --- determinism knobs (ARCHITECTURE.md §5) ---------------------------------
# The single global random seed. Every stochastic step in the project (silver
# sampling, any train/test split, LTR training) reads this so two runs are
# bit-identical. The canonical final tie-break is ``candidate_id`` ascending.
SEED = 42
TIE_BREAK_KEY = "candidate_id"

# The dataset's reference "now". The pool is a static snapshot, so honeypot
# date-consistency checks and recency logic must compare against a FIXED date
# (never the wall clock) to stay deterministic and reproducible. This matches
# the snapshot the candidates.jsonl pool was generated against.
REFERENCE_DATE = "2026-06-16"
