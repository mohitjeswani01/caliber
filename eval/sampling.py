"""STEP 1 — stratified, deterministic sampling over the candidate pool.

We do not want a silver set that is 68% noise; we want it dense in the HARD
cases (strong ML titles, adjacent hidden gems, suspected stuffers, suspected
honeypots) plus a random noise-floor slice. Stratum membership uses the shared
detectors in ``eval.heuristics`` so sampling and grading agree on definitions.

Determinism (ARCHITECTURE.md §5): a single seeded RNG draws from SORTED id lists
in a fixed stratum order, so the same pool + same seed yields identical ids.
"""

from __future__ import annotations

import random

from eval.heuristics import (
    _title,
    career_text,
    honeypot_reasons,
    stuffer_reasons,
    substance_areas_hit,
    title_class,
)

# config is pure constants (not part of the ONLINE ranking path), so importing
# it here does not couple eval to the ranker.
try:
    from src.caliber.config import SEED
except Exception:  # pragma: no cover - fallback if run with a different CWD
    SEED = 42

# --------------------------------------------------------------------------- #
# Sampling plan. Targets are *upper bounds*; a rare stratum that has fewer
# qualifying candidates than its target contributes all it has (the random pool
# then back-fills to TOTAL_TARGET). Order is fixed for determinism.
# --------------------------------------------------------------------------- #
TOTAL_TARGET = 400
SPECIAL_STRATA = ("strong_ml", "adjacent_gem", "suspected_stuffer", "suspected_honeypot")
STRATUM_TARGETS = {
    "strong_ml": 100,
    "adjacent_gem": 80,         # only ~67 exist in the pool -> we take them all
    "suspected_stuffer": 80,
    "suspected_honeypot": 80,   # only ~43 detectable -> we take them all
}
ALL_STRATA = SPECIAL_STRATA + ("random_pool",)


def classify_stratum(c, today):
    """Assign ONE special stratum (priority order) or None (-> noise floor).

    Priority: honeypot > stuffer > strong title > adjacent gem. Honeypots and
    stuffers take precedence because they are objective and are the traps we
    most need represented in the answer key.
    """
    if honeypot_reasons(c, today):
        return "suspected_honeypot"
    if stuffer_reasons(c):
        return "suspected_stuffer"
    tc = title_class(_title(c))
    if tc == "strong":
        return "strong_ml"
    # adjacent "hidden gem" = adjacent title whose DESCRIPTIONS show real
    # retrieval/ranking/recsys/search work (the Tier-5-who-didn't-say-it).
    if tc == "adjacent" and substance_areas_hit(career_text(c)):
        return "adjacent_gem"
    return None


def scan_and_bucket(records, today):
    """Single streaming pass: bucket candidate_ids by special stratum, and keep
    the rest as the noise floor. Holds only short id strings -> memory-safe."""
    buckets = {s: [] for s in SPECIAL_STRATA}
    non_special = []
    for c in records:
        cid = c.get("candidate_id")
        if cid is None:
            continue
        s = classify_stratum(c, today)
        if s:
            buckets[s].append(cid)
        else:
            non_special.append(cid)
    return buckets, non_special


def select_sample(buckets, non_special, seed=SEED, total_target=TOTAL_TARGET):
    """Deterministically draw the sample. Same inputs+seed => identical ids.

    Each stratum is sampled from its SORTED id list with a single seeded RNG in
    a fixed stratum order; the random pool back-fills to ``total_target``.
    """
    rng = random.Random(seed)
    chosen = {}
    for s in SPECIAL_STRATA:
        ids = sorted(buckets.get(s, []))
        k = min(STRATUM_TARGETS[s], len(ids))
        chosen[s] = sorted(rng.sample(ids, k)) if k else []
    used = sum(len(v) for v in chosen.values())
    rand_k = min(max(0, total_target - used), len(non_special))
    pool = sorted(non_special)
    chosen["random_pool"] = sorted(rng.sample(pool, rand_k)) if rand_k else []
    return chosen
