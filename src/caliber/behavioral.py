"""Behavioral multiplier from the 23 redrob_signals (ONLINE).

Converts platform engagement signals into a single bounded multiplier
(``config.BEHAVIORAL_MULTIPLIER_FLOOR``..``..CAP`` = 0.50..1.15) applied on top
of the base substance score, so a strong-on-paper but unavailable candidate is
pushed down without being erased (STRATEGY.md §3, §6).

Why a BOUNDED MULTIPLIER (not additive, not a hard filter):
  Availability is orthogonal to fit — it should re-weight substance, not stand in
  for it or veto it. An additive term would let behaviour swamp career substance;
  a hard filter would erase a genuinely strong senior who simply isn't active
  this month. Clamping to a tight envelope centred on 1.0 means behaviour can at
  most halve (0.50x) or modestly lift (1.15x) the score. The envelope is
  deliberately ASYMMETRIC — a 0.50x floor vs a 1.15x cap — so unavailability is
  penalised harder than engagement is rewarded (STRATEGY.md §3). This is what
  separates "behavioral twins": identical résumés, different availability.

Positive (lift toward the cap): recent last_active_date, open_to_work_flag, high
recruiter_response_rate, saved_by_recruiters_30d, high interview_completion_rate,
profile_completeness_score, verified_email/phone, healthy github_activity_score.
Negative (pull toward the floor): stale last_active_date (measured vs
REFERENCE_DATE), very low recruiter_response_rate, low interview_completion_rate,
very long notice_period_days.

Special handling:
  - ``github_activity_score == -1`` is the schema's "no GitHub linked" sentinel,
    and ~65% of the pool has no GitHub (STRATEGY.md §2). Treating -1 as a low
    score would wrongly sink the majority of the field for a signal they never
    provided, so -1 contributes NOTHING (neutral). Absence of evidence is not
    evidence of a weak signal.
  - ``offer_acceptance_rate == -1`` ("no offer history") is likewise neutral; we
    don't use that signal at all (STRATEGY.md §6 doesn't list it), so its
    sentinel needs no special code — it simply never contributes.

Determinism (ARCHITECTURE.md §5): recency math compares last_active_date against
``config.REFERENCE_DATE`` (the fixed snapshot date), read at call time, NEVER the
wall clock — so the multiplier is bit-reproducible regardless of when rank.py
runs. CPU-only, no network, no randomness: a pure function of the candidate.

Each signal is normalised into a small bounded contribution (an "envelope"), the
contributions are summed onto a 1.0 baseline, and the result is clamped to the
envelope. No single signal can blow past the bounds; the clamp is the hard
guarantee, the per-signal weights only shape the interior so a typical profile
lands ~1.0.
"""

from __future__ import annotations

import datetime as dt
from typing import Optional

from . import config
from .schema import Candidate

# --------------------------------------------------------------------------- #
# Envelope (shared contract — lives in config; see PR summary for the FLAG).
# --------------------------------------------------------------------------- #
_FLOOR = config.BEHAVIORAL_MULTIPLIER_FLOOR  # 0.50
_CAP = config.BEHAVIORAL_MULTIPLIER_CAP      # 1.15
_BASELINE = 1.0                              # an average candidate sits here

# Average days per calendar month, for last-active recency in months.
_DAYS_PER_MONTH = 30.44

# --------------------------------------------------------------------------- #
# Per-signal envelope weights. Each (NEG, POS) pair is the maximum downward /
# upward nudge that one signal may add to the baseline. Negatives are allowed to
# be far larger than positives on purpose: the summed worst-case positives
# (~0.19) and worst-case negatives (~0.55) both *overshoot* the [0.50, 1.15]
# envelope, and the final clamp is what enforces the bound. The interior shape is
# tuned so a median-everything candidate lands at ~1.0. Every mapping below is
# documented because we defend each one in the Stage-5 interview.
# --------------------------------------------------------------------------- #

# 1. last_active_date recency (vs REFERENCE_DATE) — the single strongest tell of
#    availability. Fresher than ~3 months lifts; staler ramps down to a year.
_ACTIVE_NEG, _ACTIVE_POS = 0.20, 0.04
_ACTIVE_NEUTRAL_M = 3.0    # ~3 months since active == neutral (0 contribution)
_ACTIVE_STALE_M = 12.0     # >=12 months since active == full penalty

# 2. recruiter_response_rate (0-1). Pool median ~0.44 (STRATEGY.md §2) == neutral.
_RR_NEUTRAL, _RR_NEG, _RR_POS = 0.44, 0.12, 0.03

# 3. interview_completion_rate (0-1). Neutral at the midpoint.
_IC_NEUTRAL, _IC_NEG, _IC_POS = 0.50, 0.10, 0.02

# 4. notice_period_days — negative-only. A short notice isn't really a *boost*
#    (everyone short-notice looks alike); a long one signals "far from joining".
_NOTICE_OK_D, _NOTICE_LONG_D, _NOTICE_NEG = 30.0, 120.0, 0.08

# 5. open_to_work_flag — a small, unambiguous availability lift; absence is not a
#    penalty (plenty of strong passive candidates never set the flag).
_OPEN_POS = 0.02

# 6. saved_by_recruiters_30d — positive-only interest signal; saturates fast.
_SAVED_FULL, _SAVED_POS = 5.0, 0.02

# 7. profile_completeness_score (0-1) — mostly a lift; a very sparse profile gets
#    a slight nudge down (harder to reason about / often abandoned).
_COMPLETE_LOW, _COMPLETE_NEUTRAL = 0.30, 0.70
_COMPLETE_NEG, _COMPLETE_POS = 0.03, 0.02

# 8. verified_email / verified_phone — tiny reachability lifts; absence neutral.
_VERIFIED_EMAIL_POS = 0.01
_VERIFIED_PHONE_POS = 0.01

# 9. github_activity_score (0-100, or -1 == no GitHub linked -> NEUTRAL). When
#    linked, a healthy score lifts; a linked-but-dead account nudges slightly
#    down. -1 is handled in the body (skipped entirely), never mapped here.
_GH_NEUTRAL, _GH_HIGH = 25.0, 70.0
_GH_NEG, _GH_POS = 0.02, 0.02


# --------------------------------------------------------------------------- #
# Small, pure envelope helpers.
# --------------------------------------------------------------------------- #
def _ramp(value: float, lo: float, hi: float) -> float:
    """Linear 0->1 as ``value`` moves ``lo``->``hi``, clamped to [0, 1]."""
    if hi <= lo:
        return 0.0
    return max(0.0, min(1.0, (value - lo) / (hi - lo)))


def _bidirectional(
    value: float, low: float, neutral: float, high: float, neg: float, pos: float
) -> float:
    """Signed contribution: 0 at ``neutral``, +``pos`` at ``high``, -``neg`` at
    ``low`` (piecewise-linear, clamped). ``low < neutral < high``."""
    if value >= neutral:
        return pos * _ramp(value, neutral, high)
    # below neutral: ramp the *distance* below neutral so low<neutral works.
    return -neg * _ramp(neutral - value, 0.0, neutral - low)


def _parse_date(s: Optional[str]) -> "dt.date | None":
    """Parse an ISO date string; None/blank/malformed -> None (treated neutral)."""
    if not s:
        return None
    try:
        return dt.date.fromisoformat(s)
    except (ValueError, TypeError):
        return None


def _reference_date() -> dt.date:
    """The static snapshot date (config.REFERENCE_DATE), read at CALL TIME so a
    monkeypatched wall clock can never shift a verdict (ARCHITECTURE.md §5)."""
    return dt.date.fromisoformat(config.REFERENCE_DATE)


def _months_since(date_str: Optional[str], ref: dt.date) -> Optional[float]:
    """Months between ``date_str`` and the reference snapshot. None if unknown.
    A future date (after the snapshot) yields a negative value == "very recent"."""
    d = _parse_date(date_str)
    if d is None:
        return None
    return (ref - d).days / _DAYS_PER_MONTH


def _recency_contribution(months: Optional[float]) -> float:
    """last_active recency -> signed contribution. Unknown date -> neutral (0).
    Fresher than the neutral window lifts (max at 0 months); staler ramps down to
    the full penalty at the stale horizon."""
    if months is None:
        return 0.0
    if months <= _ACTIVE_NEUTRAL_M:
        # fresher than neutral: lift, saturating at "active right at the snapshot".
        return _ACTIVE_POS * _ramp(_ACTIVE_NEUTRAL_M - months, 0.0, _ACTIVE_NEUTRAL_M)
    return -_ACTIVE_NEG * _ramp(months, _ACTIVE_NEUTRAL_M, _ACTIVE_STALE_M)


# --------------------------------------------------------------------------- #
# Public contract (ARCHITECTURE.md §3).
# --------------------------------------------------------------------------- #
def behavioral_multiplier(candidate: Candidate) -> float:
    """Bounded availability/engagement multiplier for ``candidate``.

    Returns a float in ``[config.BEHAVIORAL_MULTIPLIER_FLOOR,
    config.BEHAVIORAL_MULTIPLIER_CAP]`` (0.50..1.15). ~1.0 for an average
    candidate, near the cap for the clearly-available-and-engaged, near the floor
    for the clearly-inactive/unresponsive. Pure, deterministic, CPU-only: reads
    only ``candidate.redrob_signals`` and ``config.REFERENCE_DATE``.
    """
    s = candidate.redrob_signals
    ref = _reference_date()
    delta = 0.0

    # 1. last_active recency (vs the fixed snapshot, never the wall clock).
    delta += _recency_contribution(_months_since(s.last_active_date, ref))

    # 2. recruiter_response_rate. A negative value is a "no history" sentinel ->
    #    neutral (skip), never a penalty.
    if s.recruiter_response_rate is not None and s.recruiter_response_rate >= 0:
        delta += _bidirectional(
            s.recruiter_response_rate, 0.0, _RR_NEUTRAL, 1.0, _RR_NEG, _RR_POS
        )

    # 3. interview_completion_rate (same sentinel guard).
    if s.interview_completion_rate is not None and s.interview_completion_rate >= 0:
        delta += _bidirectional(
            s.interview_completion_rate, 0.0, _IC_NEUTRAL, 1.0, _IC_NEG, _IC_POS
        )

    # 4. notice_period_days — negative-only: long notice == far from joining.
    delta += -_NOTICE_NEG * _ramp(s.notice_period_days, _NOTICE_OK_D, _NOTICE_LONG_D)

    # 5. open_to_work_flag — small lift; absence is neutral (passive != bad).
    if s.open_to_work_flag:
        delta += _OPEN_POS

    # 6. saved_by_recruiters_30d — positive-only interest signal.
    delta += _SAVED_POS * _ramp(float(s.saved_by_recruiters_30d), 0.0, _SAVED_FULL)

    # 7. profile_completeness_score — mostly a lift, slight nudge down if sparse.
    delta += _bidirectional(
        s.profile_completeness_score,
        _COMPLETE_LOW, _COMPLETE_NEUTRAL, 1.0,
        _COMPLETE_NEG, _COMPLETE_POS,
    )

    # 8. verified_email / verified_phone — tiny reachability lifts.
    if s.verified_email:
        delta += _VERIFIED_EMAIL_POS
    if s.verified_phone:
        delta += _VERIFIED_PHONE_POS

    # 9. github_activity_score: -1 == NO GITHUB LINKED -> NEUTRAL (never penalise;
    #    ~65% of the pool has none). Only a *linked* score (>= 0) is mapped.
    if s.github_activity_score is not None and s.github_activity_score >= 0:
        delta += _bidirectional(
            s.github_activity_score, 0.0, _GH_NEUTRAL, _GH_HIGH, _GH_NEG, _GH_POS
        )

    # offer_acceptance_rate is intentionally unused (not in STRATEGY.md §6), so its
    # -1 "no history" sentinel needs no special handling — it never contributes.

    # The clamp is the hard guarantee: whatever the signals, the result lives in
    # the envelope. behaviour MODULATES substance, it cannot dominate it.
    return max(_FLOOR, min(_CAP, _BASELINE + delta))
