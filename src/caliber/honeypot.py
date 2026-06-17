"""Honeypot / internal-consistency detector — the CANONICAL source of truth.

The pool seeds ~80 deliberately-impossible profiles (STRATEGY.md §5): tenure that
exceeds the candidate's own career, "expert" skills with zero months of use,
self-contradicting dates. Ground truth pins them to relevance tier 0, and ranking
more than 10% of them into our top 100 is a Stage-3 disqualifier. So both halves
of the system must agree, to the candidate, on *which* profiles are impossible:

- the ONLINE ranker forces detected honeypots to the score floor, and
- the OFFLINE answer key (``eval/``) forces them to grade 0.

If those two used different definitions, every NDCG number we compute against the
silver labels would be measuring a ranker that plays by different rules than the
grader — fiction. To make that impossible, the detection rules live here, once.
``eval/heuristics.py`` imports :func:`honeypot_reasons` and delegates to it; the
ranker calls :func:`is_honeypot`. There is exactly one implementation.

Detection is by internal CONTRADICTION, never by keywords — a false positive that
buries a genuine fit is as costly as a miss, so we only trip on hard
impossibilities (the loose "a skill outlasts the career" signal fires on ~15% of
the pool by design and is deliberately NOT used).

Determinism (ARCHITECTURE.md §5): all date math compares against
``config.REFERENCE_DATE`` (the static snapshot date), NEVER the wall clock, so a
verdict is reproducible regardless of when ``rank.py`` runs. CPU-only, no network.

Public API:
    is_honeypot(candidate) -> tuple[bool, list[str]]      # the ranker contract
    honeypot_reasons(record, today) -> list[str]          # the eval-key contract
"""

from __future__ import annotations

import datetime as dt
from typing import Any, Mapping, Union

from . import config
from .schema import Candidate

# --------------------------------------------------------------------------- #
# Tolerant accessors over the raw candidate record. The canonical detector
# operates on a plain Mapping (not the typed Candidate) for two reasons: the
# eval answer key feeds it raw ``candidates.jsonl`` dicts directly, and keeping
# the core logic dict-based means the offline grader can import this module
# without dragging in any of the online ranking dependencies. A typed
# ``Candidate`` is projected to the same minimal shape (see ``_to_mapping``), so
# both entry points run the identical comparisons on the identical fields.
# --------------------------------------------------------------------------- #
def _profile(c: Mapping[str, Any]) -> Mapping[str, Any]:
    return c.get("profile", {}) or {}


def _roles(c: Mapping[str, Any]) -> list:
    return c.get("career_history", []) or []


def _skills(c: Mapping[str, Any]) -> list:
    return c.get("skills", []) or []


def _yoe(c: Mapping[str, Any]) -> float:
    try:
        return float(_profile(c).get("years_of_experience") or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _parse_date(s) -> "dt.date | None":
    if not s:
        return None
    try:
        return dt.date.fromisoformat(s)
    except (ValueError, TypeError):
        return None


# --------------------------------------------------------------------------- #
# The canonical contradiction checks (STRATEGY.md §5).
# --------------------------------------------------------------------------- #
def honeypot_reasons(c: Mapping[str, Any], today: dt.date) -> list[str]:
    """Return the list of internal-consistency violations (empty => clean).

    ``c`` is a raw candidate record (a Mapping with the ``candidates.jsonl``
    shape); ``today`` is the reference date the date checks compare against.
    Precision-tuned: only HARD impossibilities trip it. The reason strings are
    human-readable because they are reused verbatim for reasoning/audit and for
    the Stage-5 defence.

    Calibration note: the loose "a skill's duration exceeds total experience"
    signal fires on ~15% of candidates (it is noise by design) and is
    deliberately NOT used. We keep only hard impossibilities.
    """
    reasons: list[str] = []
    yoe_months = _yoe(c) * 12.0
    roles = _roles(c)
    skills = _skills(c)

    # 1. A single role lasting longer than the candidate's entire career (+6mo
    #    slack for rounding) is impossible.
    if yoe_months > 0:
        for r in roles:
            if r.get("duration_months", 0) > yoe_months + 6:
                reasons.append(
                    f"role tenure {r.get('duration_months')}mo exceeds total experience "
                    f"{yoe_months:.0f}mo"
                )
                break

    # 2. Summed role tenure wildly exceeds the career length (padded/overlapping).
    total_role_months = sum(r.get("duration_months", 0) for r in roles)
    if yoe_months > 0 and total_role_months > yoe_months * 1.5 + 24:
        reasons.append(
            f"summed role tenure {total_role_months}mo >> career {yoe_months:.0f}mo"
        )

    # 3. "Expert"/"advanced" in many skills with 0 months of use (STRATEGY §5).
    expert_zero = sum(
        1
        for s in skills
        if s.get("proficiency") in ("advanced", "expert") and s.get("duration_months", 0) == 0
    )
    if expert_zero >= 3:
        reasons.append(f"{expert_zero} advanced/expert skills with 0 months used")

    # 4. Impossible dates: end before start, or a start in the future.
    for r in roles:
        sd = _parse_date(r.get("start_date"))
        ed = _parse_date(r.get("end_date"))
        if sd and sd > today:
            reasons.append("role start date in the future")
            break
        if sd and ed and ed < sd:
            reasons.append("role end date precedes start date")
            break

    return reasons


# --------------------------------------------------------------------------- #
# Candidate -> minimal mapping projection for the typed (ranker) entry point.
# Only the fields the detector reads are copied; ``Skill.duration_months`` is
# Optional, and the raw-dict path treats an absent key as 0 (``.get(k, 0)``), so
# a ``None`` is mapped to "absent" — that way a typed Candidate parsed from a
# record and the record itself yield the IDENTICAL verdict.
# --------------------------------------------------------------------------- #
def _to_mapping(c: Candidate) -> dict[str, Any]:
    skills: list[dict[str, Any]] = []
    for s in c.skills:
        d: dict[str, Any] = {"proficiency": s.proficiency}
        if s.duration_months is not None:
            d["duration_months"] = s.duration_months
        skills.append(d)
    return {
        "profile": {"years_of_experience": c.profile.years_of_experience},
        "career_history": [
            {
                "duration_months": r.duration_months,
                "is_current": r.is_current,
                "start_date": r.start_date,
                "end_date": r.end_date,
            }
            for r in c.career_history
        ],
        "skills": skills,
    }


def _reference_date() -> dt.date:
    """The static snapshot date (config.REFERENCE_DATE), read at call time so a
    monkeypatched wall clock can never change a verdict (determinism)."""
    return dt.date.fromisoformat(config.REFERENCE_DATE)


def is_honeypot(candidate: Union[Candidate, Mapping[str, Any]]) -> tuple[bool, list[str]]:
    """The ranker contract (ARCHITECTURE.md §3): is this profile internally
    impossible?

    Returns ``(is_honeypot, reasons)``. Accepts a typed :class:`Candidate` (the
    online path) or a raw record :class:`Mapping` (convenience / tests). Date
    math uses :func:`config.REFERENCE_DATE`, never ``datetime.now()``, so the
    verdict is deterministic. Detected honeypots are forced to the score floor by
    the scorer; the reasons feed the grounded reasoning string.
    """
    rec = _to_mapping(candidate) if isinstance(candidate, Candidate) else candidate
    reasons = honeypot_reasons(rec, _reference_date())
    return (bool(reasons), reasons)
