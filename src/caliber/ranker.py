"""Final selection + invariant enforcement (ONLINE) — the top-100 builder.

This is the deterministic core that turns the scorer's per-candidate breakdown
into the exact rows the submission CSV needs. It is kept PURE: it neither streams
the 100K pool nor writes a file (that I/O is ``rank.py``, which calls this). That
split keeps the make-or-break invariants — the ones a malformed value would get us
DISQUALIFIED for — in one small, fully unit-tested function.

    1. select       — sort by final_score DESC, tie-break candidate_id ASC, take N
    2. guardrail    — honeypot rate in the selection MUST stay < 10%
    3. reasoning    — attach a grounded note per row (reasoning.reasoning_for)
    4. assert       — every DQ-grade invariant, before any row leaves this module

What it guarantees (all asserted before any row is returned; ARCHITECTURE.md §5,
CLAUDE.md hard constraints):

  * exactly ``TOP_N`` (100) rows,
  * ``score`` NON-INCREASING down the ranking,
  * equal scores tie-broken by ``candidate_id`` ASCENDING (config.TIE_BREAK_KEY),
  * no duplicate ``candidate_id``,
  * ``rank`` contiguous 1..TOP_N,
  * HONEYPOT GUARDRAIL: < 10% of the top-100 are honeypots (CLAUDE.md: ">10% in
    our top 100 = disqualified"). If this ever trips we raise LOUDLY and emit
    nothing — a hard pre-write stop.

The score we sort on and the score we store are the SAME rounded value: we round
to ``SCORE_DECIMALS`` once, sort by ``(-rounded, candidate_id)``, and write that
rounded number. This closes a subtle DQ hole — if the ranker sorted on full
precision but the CSV later rounded for display, two distinct floats could collapse
to one printed value and silently violate the "equal score ⇒ candidate_id
ascending" rule. By rounding before the sort, the invariants hold on exactly the
bytes that get written. ``rank.py`` MUST write ``SubmissionRow.score`` with this
module's precision (use :data:`SCORE_DECIMALS`).

Honeypots are floored to ``scorer.HONEYPOT_FLOOR`` (-1.0) upstream, so in normal
operation they sort to the bottom and never reach the top-100 at all; the guardrail
is the independent safety net that fires even if a non-floored honeypot ever leaks
in. The default combiner is the scorer's hand-weights — LTR stays dormant (it is
swapped in at the scorer, not here), so nothing about LTR is wired in this module.

(The broader "stream pool + load artifacts + write CSV" orchestration that this
file's stub once sketched lives in ``rank.py``; this module is its pure selection
core.)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Optional, Union

from . import config
from . import reasoning as reasoning_mod
from .schema import Candidate
from .scorer import CandidateScore

# The submission rows always number this many (submission_spec / sample_submission
# has exactly 100 data rows; validate_submission enforces it).
TOP_N = 100

# CSV column order — quoted from data/challenge/sample_submission.csv and enforced
# by tests/validate_submission.py:REQUIRED_HEADER. rank.py writes the header from
# this so the two can never drift.
SUBMISSION_COLUMNS: tuple[str, ...] = ("candidate_id", "rank", "score", "reasoning")

# Rounding applied to the score BEFORE the final sort, and the precision rank.py
# must write with. 6 dp keeps near-ties distinct while guaranteeing the printed
# value and the sorted value are identical (see module docstring).
SCORE_DECIMALS = 6

# Guardrail threshold. CLAUDE.md: ">10% honeypot rate in our top 100 = disqualified"
# → we require the fraction to stay STRICTLY below this. At TOP_N=100 that means 10
# or more flagged honeypots in the selection trips the stop.
HONEYPOT_MAX_FRACTION = 0.10

_TIE_BREAK = config.TIE_BREAK_KEY  # "candidate_id" — documented join/sort key


class HoneypotGuardrailError(AssertionError):
    """Raised when the selected top-N would contain ≥ ``HONEYPOT_MAX_FRACTION``
    honeypots — a Stage-3 disqualifier. Subclasses ``AssertionError`` so it reads
    as the hard assertion it is, while staying catchable/identifiable. When this
    fires, NO rows are returned: the caller must not write a CSV."""


@dataclass(frozen=True)
class SubmissionRow:
    """One CSV-ready row. Field names ARE the submission columns; ``score`` is the
    rounded value to be written verbatim (do not re-round downstream)."""
    candidate_id: str
    rank: int
    score: float
    reasoning: str

    def as_dict(self) -> dict[str, object]:
        """The row as a ``{column: value}`` dict in SUBMISSION_COLUMNS order."""
        return {
            "candidate_id": self.candidate_id,
            "rank": self.rank,
            "score": self.score,
            "reasoning": self.reasoning,
        }


def _as_iterable(
    scores: Union[Mapping[str, CandidateScore], Iterable[CandidateScore]],
) -> list[CandidateScore]:
    """Accept either the scorer's ``{id: CandidateScore}`` map or a bare iterable
    of CandidateScore. Returns a list of the score objects."""
    if isinstance(scores, Mapping):
        return list(scores.values())
    return list(scores)


def select_top(
    scores: Union[Mapping[str, CandidateScore], Iterable[CandidateScore]],
    *,
    top_n: int = TOP_N,
) -> list[CandidateScore]:
    """Deterministically select the top ``top_n`` CandidateScores.

    Sort by ``final_score`` DESCENDING (on the value rounded to SCORE_DECIMALS),
    tie-broken by ``candidate_id`` ASCENDING, and take the first ``top_n``. Pure;
    no reasoning attached, no invariants asserted yet (that is
    ``build_submission_rows``).

    Raises ``ValueError`` if there are fewer than ``top_n`` distinct candidates to
    choose from — we cannot fabricate rows, and a short submission fails validation.
    """
    items = _as_iterable(scores)

    # Guard duplicates at the input boundary: a duplicated candidate_id from the
    # scorer would silently break the no-dupes invariant if we trusted the count.
    by_id: dict[str, CandidateScore] = {}
    for cs in items:
        cid = str(cs.candidate_id)
        if cid in by_id:
            raise ValueError(f"duplicate candidate_id in scorer output: {cid!r}")
        by_id[cid] = cs

    if len(by_id) < top_n:
        raise ValueError(
            f"need at least {top_n} scored candidates to fill the ranking, "
            f"got {len(by_id)}"
        )

    ordered = sorted(
        by_id.values(),
        key=lambda cs: (-round(float(cs.final_score), SCORE_DECIMALS), str(cs.candidate_id)),
    )
    return ordered[:top_n]


def _assert_invariants(rows: list[SubmissionRow], top_n: int) -> None:
    """Assert every DQ-grade contract on the final rows (see module docstring)."""
    assert len(rows) == top_n, f"expected {top_n} rows, got {len(rows)}"

    # ranks contiguous 1..top_n, in order.
    assert [r.rank for r in rows] == list(range(1, top_n + 1)), "ranks must be 1..N contiguous"

    # no duplicate candidate_ids.
    ids = [r.candidate_id for r in rows]
    assert len(set(ids)) == len(ids), "duplicate candidate_id in the ranking"

    # score non-increasing; equal scores ⇒ candidate_id ascending.
    for a, b in zip(rows, rows[1:]):
        assert a.score >= b.score, (
            f"score must be non-increasing: rank {a.rank} ({a.score}) < "
            f"rank {b.rank} ({b.score})"
        )
        if a.score == b.score:
            assert a.candidate_id <= b.candidate_id, (
                f"equal scores at ranks {a.rank}/{b.rank} must tie-break by "
                f"candidate_id ascending ({a.candidate_id!r} > {b.candidate_id!r})"
            )


def _assert_honeypot_guardrail(selected: list[CandidateScore], top_n: int) -> None:
    """Hard pre-write stop: < HONEYPOT_MAX_FRACTION of the selection may be
    honeypots. Raises ``HoneypotGuardrailError`` (no CSV) if violated."""
    hp = [cs for cs in selected if cs.is_honeypot]
    fraction = len(hp) / top_n if top_n else 0.0
    if fraction >= HONEYPOT_MAX_FRACTION:
        ids = ", ".join(cs.candidate_id for cs in hp)
        raise HoneypotGuardrailError(
            f"HONEYPOT GUARDRAIL TRIPPED: {len(hp)}/{top_n} "
            f"({fraction:.0%}) of the top-{top_n} are flagged honeypots — a Stage-3 "
            f"disqualifier (limit < {HONEYPOT_MAX_FRACTION:.0%}). Refusing to emit a "
            f"submission. Offending ids: {ids}"
        )


def build_submission_rows(
    scores: Union[Mapping[str, CandidateScore], Iterable[CandidateScore]],
    *,
    top_n: int = TOP_N,
    candidates: Optional[Mapping[str, Candidate]] = None,
) -> list[SubmissionRow]:
    """Produce the final, CSV-ready top-``top_n`` rows from scorer output.

    Steps: select (sort desc, tie-break candidate_id asc) → honeypot guardrail →
    attach grounded reasoning → assert every invariant → return. Pure and
    deterministic; does no file or pool I/O (``rank.py`` owns that).

    Args:
        scores: the scorer's ``{candidate_id: CandidateScore}`` map (or any iterable
            of CandidateScore).
        top_n: number of rows to emit (defaults to the spec's 100).
        candidates: optional ``{candidate_id: Candidate}`` map; when supplied, the
            reasoning string is enriched with the literal current title + years.
            Reasoning is fully grounded without it.

    Returns the ``top_n`` :class:`SubmissionRow`s, rank 1..top_n.

    Raises:
        ValueError: fewer than ``top_n`` candidates available, or a duplicate id in
            the scorer output.
        HoneypotGuardrailError: ≥ ``HONEYPOT_MAX_FRACTION`` of the selection are
            honeypots (nothing is returned — do NOT write a CSV).
    """
    selected = select_top(scores, top_n=top_n)

    # Guardrail BEFORE building rows: if it trips we emit nothing at all.
    _assert_honeypot_guardrail(selected, top_n)

    rows: list[SubmissionRow] = []
    for i, cs in enumerate(selected):
        cid = str(cs.candidate_id)
        cand = candidates.get(cid) if candidates else None
        rows.append(
            SubmissionRow(
                candidate_id=cid,
                rank=i + 1,
                score=round(float(cs.final_score), SCORE_DECIMALS),
                reasoning=reasoning_mod.reasoning_for(cs, cand),
            )
        )

    _assert_invariants(rows, top_n)
    return rows
