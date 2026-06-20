"""Grounded, template-based reasoning generation (ONLINE, no LLM).

Produces the 1–2 sentence ``reasoning`` string for each ranked candidate,
deterministically from extracted profile facts — never invented. Per the Stage-4
manual-review checks, each entry must:

- cite specific facts (years, current title, named skills, signal values),
- connect to a specific JD requirement,
- honestly acknowledge real gaps/concerns (notice period, services-only, etc.),
- vary meaningfully between candidates, and
- match its tone to the rank (no glowing rank-95, no critical rank-5).

Pure templating over real fields is deliberately safer than an LLM here: it
*cannot* hallucinate a skill or employer the candidate doesn't have, which is
exactly what the review penalises.

HOW IT STAYS GROUNDED. Every clause is the human reading of ONE real signal on the
:class:`~caliber.scorer.CandidateScore`, and is emitted *only* when that signal is
present:

- a HONEYPOT reuses ``honeypot_reasons`` VERBATIM (e.g. "role tenure 120mo exceeds
  total experience 36mo") — the same strings the detector produced and the eval
  key grades on, so the note never contradicts the floor;
- every strength/gap is the reading of a real ``feature_dict`` value, each of which
  ``features.py`` computed from the candidate's own career text / fields
  (role_substance from the descriptions, experience_band from years_of_experience,
  location_fit from country+city, …);
- ``semantic_sim``, the cross-encoder, and ``behavioral_mult`` are read off the
  same score object.

A fixed map from real values to phrasing cannot invent a fact, so there is NO LLM
at runtime (online path, zero network). Tone follows the candidate's strength
(which is what placed them).

Determinism (ARCHITECTURE.md §5): feature values are deterministic, clause order is
fixed, no wall clock, no network → byte-identical strings run to run.

Public API (ARCHITECTURE.md §3):
    reasoning_for(candidate_score, candidate=None, *, max_len=...) -> str
"""

from __future__ import annotations

from typing import Optional

from .schema import Candidate
from .scorer import CandidateScore

# The CSV reasoning field is a concise note, not a paragraph (the sample rows run
# ~60-90 chars). We keep room for a few grounded clauses but cap hard so a row can
# never blow up the CSV; truncation is at a word boundary, never mid-word.
MAX_REASONING_LEN = 320

# Feature-value interpretation thresholds. A feature is "strong" enough to cite as
# a positive, or "weak" enough to flag as a gap, per these cutoffs. They mirror the
# discrete levels features.py actually emits (e.g. location_fit ∈ {1.0, 0.8, 0.45,
# 0.15}; production_recency ∈ {1.0, 0.6, 0.2, 0.1}), so the phrasing lines up with
# the real signal rather than guessing a continuum.
_STRONG = 0.8
_MID = 0.5


def _round1(x: float) -> str:
    """One-decimal, trailing-.0-trimmed number for compact factual leads."""
    return f"{x:.1f}".rstrip("0").rstrip(".")


# --------------------------------------------------------------------------- #
# Per-feature → (positive clause | gap clause). Each reads ONE real value off
# feature_dict and returns the recruiter-facing phrase, or None when the signal is
# neutral / uninformative (so we never pad the note with filler).
# --------------------------------------------------------------------------- #
def _role_substance_clause(v: float) -> tuple[Optional[str], Optional[str]]:
    if v >= _STRONG:
        return ("career history shows real retrieval/ranking/applied-ML substance", None)
    if v >= _MID:
        return ("some applied-ML substance in the role descriptions", None)
    if v < 0.25:
        return (None, "career descriptions show no retrieval/ranking/ML substance")
    return (None, "limited career substance for the role")


def _skill_corroboration_clause(v: float) -> tuple[Optional[str], Optional[str]]:
    # 1.0 means "few AI tags claimed, or fully corroborated" — not worth citing.
    # Only a LOW value is informative: many AI skill tags the career doesn't back.
    if v < _MID:
        return (None, "AI skills not corroborated by career history (keyword-stuffing signal)")
    return (None, None)


def _experience_band_clause(v: float) -> tuple[Optional[str], Optional[str]]:
    if v >= 0.999:
        return ("experience in the JD's target band", None)
    if v >= _MID:
        return ("experience close to the target band", None)
    # Symmetric feature: a low value is too-junior OR too-senior; we cannot tell
    # which from the value alone, so we stay honest and say "outside".
    return (None, "experience outside the target band")


def _nlp_ir_clause(v: float) -> tuple[Optional[str], Optional[str]]:
    if v >= 0.999:
        return ("NLP/IR background present", None)
    if v <= 0.001:
        return (None, "CV/speech/robotics-heavy with no NLP/IR (a JD negative)")
    return (None, None)  # 0.5 = unknown, not worth a clause


def _product_clause(v: float) -> tuple[Optional[str], Optional[str]]:
    if v >= 0.999:
        return ("product-company experience", None)
    if v <= 0.001:
        return (None, "career entirely at services/consulting firms")
    if v < _MID:
        return (None, "much of the career at services/consulting firms")
    return (None, None)


def _recency_clause(v: float) -> tuple[Optional[str], Optional[str]]:
    if v >= 0.999:
        return ("recently shipping production code", None)
    if v <= 0.2:
        return (None, "in a non-coding lead role with no recent shipping")
    return (None, None)  # 0.6 = neutral


def _tenure_clause(v: float) -> tuple[Optional[str], Optional[str]]:
    if v >= 0.999:
        return ("stable role tenure", None)
    if v <= 0.2:
        return (None, "short average tenure (title-chaser signal)")
    return (None, None)


def _external_clause(v: float) -> tuple[Optional[str], Optional[str]]:
    # 0.5 == no GitHub linked (neutral, ~65% of the pool) → never a gap.
    if v >= _STRONG:
        return ("strong external GitHub/open-source validation", None)
    if v > _MID:
        return ("some GitHub/open-source activity", None)
    return (None, None)


def _location_clause(v: float) -> tuple[Optional[str], Optional[str]]:
    if v >= 0.999:
        return ("India Tier-1 location", None)
    if v >= _STRONG:
        return ("India-based", None)
    if v >= 0.4:
        return (None, "non-India but relocation-willing")
    return (None, "non-India and not relocation-willing (JD down-weights)")


# feature key → clause fn, in the ORDER clauses should appear (substance first).
_FEATURE_CLAUSES = (
    ("role_substance", _role_substance_clause),
    ("experience_band", _experience_band_clause),
    ("nlp_ir_signal", _nlp_ir_clause),
    ("product_vs_consulting", _product_clause),
    ("production_recency", _recency_clause),
    ("tenure_stability", _tenure_clause),
    ("external_validation", _external_clause),
    ("location_fit", _location_clause),
    ("skill_corroboration", _skill_corroboration_clause),
)


def _semantic_clause(sem: float) -> Optional[str]:
    if sem >= 0.6:
        return "strong semantic match to the JD"
    return None  # cosine is relative; a low value is not a defensible "gap"


def _ce_clause(cs: CandidateScore) -> Optional[str]:
    if cs.ce_used and cs.ce_score is not None and cs.ce_score >= 0.6:
        return "cross-encoder confirms description-level fit"
    return None


def _behavioral_clause(mult: float) -> tuple[Optional[str], Optional[str]]:
    if mult >= 1.03:
        return ("engaged and available (behaviour lifts the score)", None)
    if mult <= 0.95:
        return (None, "low engagement / availability drags the score")
    return (None, None)


def _factual_lead(candidate: Optional[Candidate]) -> Optional[str]:
    """Optional grounded lead-in from the raw profile (title + years), used only
    when rank.py threads the Candidate. Pure profile facts, never inferred."""
    if candidate is None:
        return None
    title = (candidate.profile.current_title or "").strip()
    yoe = candidate.profile.years_of_experience
    bits: list[str] = []
    if title:
        bits.append(title)
    try:
        if yoe:
            bits.append(f"{_round1(float(yoe))} yrs")
    except (TypeError, ValueError):
        pass
    return ", ".join(bits) if bits else None


def _truncate(text: str, max_len: int) -> str:
    """Clamp to ``max_len`` at a word boundary, single line, with an ellipsis."""
    text = " ".join(text.split())  # collapse any newline/whitespace → CSV-safe
    if len(text) <= max_len:
        return text
    cut = text[: max_len - 1].rsplit(" ", 1)[0]
    return (cut.rstrip(",;. ") + "…") if cut else text[: max_len - 1] + "…"


def reasoning_for(
    candidate_score: CandidateScore,
    candidate: Optional[Candidate] = None,
    *,
    max_len: int = MAX_REASONING_LEN,
) -> str:
    """Build the grounded reasoning string for one placement.

    Grounded entirely in ``candidate_score`` (ARCHITECTURE.md §3 signature):
    ``feature_dict`` + ``honeypot_reasons`` + ``semantic_sim`` + ``ce_*`` +
    ``behavioral_mult``. ``candidate`` is OPTIONAL profile context — when present
    (rank.py may pass it) we prepend the literal current title + years; the string
    is fully defined without it. Deterministic, no LLM, no network, ≤ ``max_len``.
    """
    cs = candidate_score
    feats = cs.feature_dict or {}

    # A floored honeypot is explained by its detector reasons, VERBATIM — never on
    # merit (its merit signals are moot once it is forced to the floor).
    if cs.is_honeypot:
        reasons = "; ".join(cs.honeypot_reasons) if cs.honeypot_reasons else "internal inconsistency"
        return _truncate(
            f"Flagged as an internally inconsistent profile and forced to the score "
            f"floor: {reasons}. Not ranked on merit.",
            max_len,
        )

    positives: list[str] = []
    gaps: list[str] = []

    for key, fn in _FEATURE_CLAUSES:
        if key not in feats:
            continue
        pos, gap = fn(float(feats[key]))
        if pos:
            positives.append(pos)
        if gap:
            gaps.append(gap)

    sem = _semantic_clause(cs.semantic_sim)
    if sem:
        positives.append(sem)
    ce = _ce_clause(cs)
    if ce:
        positives.append(ce)
    beh_pos, beh_gap = _behavioral_clause(cs.behavioral_mult)
    if beh_pos:
        positives.append(beh_pos)
    if beh_gap:
        gaps.append(beh_gap)

    # Tone follows strength: a candidate carried by real substance leads "Strong
    # fit"; one with little leads "Limited fit". Read off the signal that actually
    # placed them, not a hand-set rank.
    substance = float(feats.get("role_substance", 0.0))
    if substance >= _STRONG and len(positives) >= 2:
        opener = "Strong fit"
    elif positives:
        opener = "Fit"
    else:
        opener = "Limited fit"

    segments: list[str] = []
    lead = _factual_lead(candidate)
    if lead:
        segments.append(lead)

    if positives:
        segments.append(f"{opener}: " + "; ".join(positives))
    else:
        segments.append(opener)
    if gaps:
        segments.append("Gaps: " + "; ".join(gaps))

    return _truncate(". ".join(segments) + ".", max_len)
