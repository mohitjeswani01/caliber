"""Typed access to a candidate record + schema-aware field extraction.

Wraps the raw ``candidates.jsonl`` record (shape defined in
``data/challenge/candidate_schema.json``) so the rest of the pipeline never
reaches into raw dicts. Responsibilities:

- Parse a single JSON line into a lightweight, attribute-accessible candidate
  object (profile, career_history, education, skills, redrob_signals).
- Provide safe accessors that tolerate missing/optional fields (certifications,
  languages, null end_dates) without raising.
- Normalise enums (company_size bands, proficiency, work_mode) into comparable
  forms used by ``features`` and ``honeypot``.

No scoring here — extraction and normalisation only.

NOTE (ownership): the ``Candidate`` dataclass and ``parse_candidate()`` are
owned by the pipeline/scoring half and are not implemented yet. The embedding
pipeline only needs the canonical text representation, so ``candidate_to_text``
is implemented here — in its contracted location (ARCHITECTURE.md §2) — to keep
it a single source of truth rather than duplicating it inside ``embeddings.py``.
It deliberately operates on the raw record dict so it has no dependency on the
not-yet-built dataclass; it can later accept a ``Candidate`` transparently.
"""

from typing import Any, Mapping


def _clean(value: Any) -> str:
    """Coerce a possibly-missing field to a stripped string ("" if absent)."""
    if value is None:
        return ""
    return str(value).strip()


def candidate_to_text(rec: Mapping[str, Any]) -> str:
    """Build the rich text representation used to embed a candidate.

    The string is intentionally **description-heavy**: headline + summary +
    every role's title *and free-text description* + skills-with-context. Role
    descriptions are the signal that surfaces the "plain-language Tier-5"
    candidates who do real retrieval/ranking/ML work but never type the
    buzzwords (STRATEGY.md §5). Skill tags come last and carry the least weight
    precisely because keyword-stuffers inflate them.

    Operates on the raw record dict (or any mapping with the same shape), so it
    is safe to call while streaming ``candidates.jsonl`` without constructing a
    ``Candidate`` first. Missing/optional fields are tolerated.
    """
    profile = rec.get("profile") or {}
    parts: list[str] = []

    headline = _clean(profile.get("headline"))
    if headline:
        parts.append(headline)

    summary = _clean(profile.get("summary"))
    if summary:
        parts.append(summary)

    current_title = _clean(profile.get("current_title"))
    current_company = _clean(profile.get("current_company"))
    if current_title or current_company:
        if current_title and current_company:
            parts.append(f"Current role: {current_title} at {current_company}.")
        else:
            parts.append(f"Current role: {current_title or current_company}.")

    # Career history — title + the free-text description for every role. This is
    # the bulk of the text and the part that actually distinguishes substance.
    for role in rec.get("career_history") or []:
        title = _clean(role.get("title"))
        company = _clean(role.get("company"))
        description = _clean(role.get("description"))
        segment: list[str] = []
        if title and company:
            segment.append(f"{title} at {company}.")
        elif title or company:
            segment.append(f"{title or company}.")
        if description:
            segment.append(description)
        if segment:
            parts.append(" ".join(segment))

    # Skills with their proficiency, as light context. Last on purpose: the
    # structured scorer gates skill credit behind career evidence, so we do not
    # want a long skill list dominating the embedding for a keyword-stuffer.
    skill_strs: list[str] = []
    for skill in rec.get("skills") or []:
        name = _clean(skill.get("name"))
        if not name:
            continue
        proficiency = _clean(skill.get("proficiency"))
        skill_strs.append(f"{name} ({proficiency})" if proficiency else name)
    if skill_strs:
        parts.append("Skills: " + ", ".join(skill_strs))

    return "\n".join(parts)
