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

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Union

# ---------------------------------------------------------------------------
# Enum value sets (mirrors candidate_schema.json). Kept as plain strings on the
# dataclasses rather than typing.Literal: parsing stays tolerant of any future
# value the data might add, and the real validation/normalisation of these
# bands lives in ``features``/``honeypot`` where it is actually used. These
# tuples document the contract and are importable by those modules.
# ---------------------------------------------------------------------------
COMPANY_SIZE_BANDS = (
    "1-10", "11-50", "51-200", "201-500",
    "501-1000", "1001-5000", "5001-10000", "10001+",
)
SKILL_PROFICIENCY_LEVELS = ("beginner", "intermediate", "advanced", "expert")
LANGUAGE_PROFICIENCY_LEVELS = ("basic", "conversational", "professional", "native")
EDUCATION_TIERS = ("tier_1", "tier_2", "tier_3", "tier_4", "unknown")
WORK_MODES = ("remote", "hybrid", "onsite", "flexible")


class SchemaError(ValueError):
    """Raised when a record is missing a field the schema marks required.

    A subclass of ``ValueError`` so callers can catch it specifically while it
    still behaves like a normal value error. The message always names the
    offending candidate (when known) and the missing field path.
    """


# ---------------------------------------------------------------------------
# Typed candidate structure (ARCHITECTURE.md §2; data/challenge/candidate_schema.json)
# ---------------------------------------------------------------------------


@dataclass
class Profile:
    anonymized_name: str
    headline: str
    summary: str
    location: str
    country: str
    years_of_experience: float
    current_title: str
    current_company: str
    current_company_size: str  # one of COMPANY_SIZE_BANDS
    current_industry: str


@dataclass
class Role:
    company: str
    title: str
    start_date: str               # ISO "YYYY-MM-DD" (kept as string; see module docstring)
    end_date: Optional[str]       # ISO date or None when current
    duration_months: int
    is_current: bool
    industry: str
    company_size: str             # one of COMPANY_SIZE_BANDS
    description: str


@dataclass
class Education:
    institution: str
    degree: str
    field_of_study: str
    start_year: int
    end_year: int
    grade: Optional[str] = None   # nullable per schema
    tier: Optional[str] = None    # one of EDUCATION_TIERS; absent on some records


@dataclass
class Skill:
    name: str
    proficiency: str              # one of SKILL_PROFICIENCY_LEVELS
    endorsements: int
    duration_months: Optional[int] = None  # months the skill has been used


@dataclass
class Certification:
    name: str
    issuer: str
    year: int


@dataclass
class Language:
    language: str
    proficiency: str              # one of LANGUAGE_PROFICIENCY_LEVELS


@dataclass
class SalaryRange:
    min: float
    max: float


@dataclass
class Signals:
    """The 23 redrob_signals (names/types mirror the schema exactly)."""
    profile_completeness_score: float
    signup_date: str
    last_active_date: str
    open_to_work_flag: bool
    profile_views_received_30d: int
    applications_submitted_30d: int
    recruiter_response_rate: float
    avg_response_time_hours: float
    skill_assessment_scores: dict[str, float]
    connection_count: int
    endorsements_received: int
    notice_period_days: int
    expected_salary_range_inr_lpa: SalaryRange
    preferred_work_mode: str      # one of WORK_MODES
    willing_to_relocate: bool
    github_activity_score: float  # 0-100, or -1 if no GitHub linked
    search_appearance_30d: int
    saved_by_recruiters_30d: int
    interview_completion_rate: float
    offer_acceptance_rate: float  # 0-1, or -1 if no offer history
    verified_email: bool
    verified_phone: bool
    linkedin_connected: bool


@dataclass
class Candidate:
    candidate_id: str
    profile: Profile
    career_history: list[Role]
    education: list[Education]
    skills: list[Skill]
    certifications: list[Certification]
    languages: list[Language]
    redrob_signals: Signals
    raw: dict = field(repr=False)  # original record, kept for reasoning/debug


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

# Required keys per object, straight from candidate_schema.json "required" lists.
_TOP_REQUIRED = (
    "candidate_id", "profile", "career_history", "education", "skills",
    "redrob_signals",
)
_PROFILE_REQUIRED = (
    "anonymized_name", "headline", "summary", "location", "country",
    "years_of_experience", "current_title", "current_company",
    "current_company_size", "current_industry",
)
_ROLE_REQUIRED = (
    "company", "title", "start_date", "end_date", "duration_months",
    "is_current", "industry", "company_size", "description",
)
_EDUCATION_REQUIRED = (
    "institution", "degree", "field_of_study", "start_year", "end_year",
)
_SKILL_REQUIRED = ("name", "proficiency", "endorsements")
_CERT_REQUIRED = ("name", "issuer", "year")
_LANGUAGE_REQUIRED = ("language", "proficiency")
_SALARY_REQUIRED = ("min", "max")
_SIGNALS_REQUIRED = (
    "profile_completeness_score", "signup_date", "last_active_date",
    "open_to_work_flag", "profile_views_received_30d",
    "applications_submitted_30d", "recruiter_response_rate",
    "avg_response_time_hours", "skill_assessment_scores", "connection_count",
    "endorsements_received", "notice_period_days",
    "expected_salary_range_inr_lpa", "preferred_work_mode",
    "willing_to_relocate", "github_activity_score", "search_appearance_30d",
    "saved_by_recruiters_30d", "interview_completion_rate",
    "offer_acceptance_rate", "verified_email", "verified_phone",
    "linkedin_connected",
)


def _check_required(rec: Mapping[str, Any], required: tuple[str, ...], ctx: str) -> None:
    """Raise SchemaError naming every required key absent from ``rec``."""
    if not isinstance(rec, Mapping):
        raise SchemaError(f"{ctx}: expected an object, got {type(rec).__name__}")
    missing = [k for k in required if k not in rec]
    if missing:
        raise SchemaError(f"{ctx}: missing required field(s) {', '.join(missing)}")


def _parse_profile(rec: Mapping[str, Any], ctx: str) -> Profile:
    _check_required(rec, _PROFILE_REQUIRED, ctx)
    return Profile(
        anonymized_name=rec["anonymized_name"],
        headline=rec["headline"],
        summary=rec["summary"],
        location=rec["location"],
        country=rec["country"],
        years_of_experience=rec["years_of_experience"],
        current_title=rec["current_title"],
        current_company=rec["current_company"],
        current_company_size=rec["current_company_size"],
        current_industry=rec["current_industry"],
    )


def _parse_role(rec: Mapping[str, Any], ctx: str) -> Role:
    _check_required(rec, _ROLE_REQUIRED, ctx)
    return Role(
        company=rec["company"],
        title=rec["title"],
        start_date=rec["start_date"],
        end_date=rec["end_date"],
        duration_months=rec["duration_months"],
        is_current=rec["is_current"],
        industry=rec["industry"],
        company_size=rec["company_size"],
        description=rec["description"],
    )


def _parse_education(rec: Mapping[str, Any], ctx: str) -> Education:
    _check_required(rec, _EDUCATION_REQUIRED, ctx)
    return Education(
        institution=rec["institution"],
        degree=rec["degree"],
        field_of_study=rec["field_of_study"],
        start_year=rec["start_year"],
        end_year=rec["end_year"],
        grade=rec.get("grade"),
        tier=rec.get("tier"),
    )


def _parse_skill(rec: Mapping[str, Any], ctx: str) -> Skill:
    _check_required(rec, _SKILL_REQUIRED, ctx)
    return Skill(
        name=rec["name"],
        proficiency=rec["proficiency"],
        endorsements=rec["endorsements"],
        duration_months=rec.get("duration_months"),
    )


def _parse_certification(rec: Mapping[str, Any], ctx: str) -> Certification:
    _check_required(rec, _CERT_REQUIRED, ctx)
    return Certification(name=rec["name"], issuer=rec["issuer"], year=rec["year"])


def _parse_language(rec: Mapping[str, Any], ctx: str) -> Language:
    _check_required(rec, _LANGUAGE_REQUIRED, ctx)
    return Language(language=rec["language"], proficiency=rec["proficiency"])


def _parse_signals(rec: Mapping[str, Any], ctx: str) -> Signals:
    _check_required(rec, _SIGNALS_REQUIRED, ctx)
    salary_raw = rec["expected_salary_range_inr_lpa"]
    _check_required(salary_raw, _SALARY_REQUIRED, f"{ctx}.expected_salary_range_inr_lpa")
    salary = SalaryRange(min=salary_raw["min"], max=salary_raw["max"])
    return Signals(
        profile_completeness_score=rec["profile_completeness_score"],
        signup_date=rec["signup_date"],
        last_active_date=rec["last_active_date"],
        open_to_work_flag=rec["open_to_work_flag"],
        profile_views_received_30d=rec["profile_views_received_30d"],
        applications_submitted_30d=rec["applications_submitted_30d"],
        recruiter_response_rate=rec["recruiter_response_rate"],
        avg_response_time_hours=rec["avg_response_time_hours"],
        skill_assessment_scores=dict(rec["skill_assessment_scores"]),
        connection_count=rec["connection_count"],
        endorsements_received=rec["endorsements_received"],
        notice_period_days=rec["notice_period_days"],
        expected_salary_range_inr_lpa=salary,
        preferred_work_mode=rec["preferred_work_mode"],
        willing_to_relocate=rec["willing_to_relocate"],
        github_activity_score=rec["github_activity_score"],
        search_appearance_30d=rec["search_appearance_30d"],
        saved_by_recruiters_30d=rec["saved_by_recruiters_30d"],
        interview_completion_rate=rec["interview_completion_rate"],
        offer_acceptance_rate=rec["offer_acceptance_rate"],
        verified_email=rec["verified_email"],
        verified_phone=rec["verified_phone"],
        linkedin_connected=rec["linkedin_connected"],
    )


def parse_candidate(rec: Mapping[str, Any]) -> Candidate:
    """Build a typed :class:`Candidate` from a raw record (a JSON-decoded dict).

    Tolerant of missing OPTIONAL fields per the schema: ``education``/``skills``
    may be empty lists; ``certifications``/``languages`` may be absent
    entirely; ``grade``/``tier``/``skill.duration_months`` may be missing or
    null. Missing REQUIRED fields raise :class:`SchemaError` naming the
    candidate and the field path. The original dict is preserved on
    ``Candidate.raw`` so downstream reasoning/debug can reach any field this
    typed view does not surface.
    """
    _check_required(rec, _TOP_REQUIRED, "candidate")
    cid = rec["candidate_id"]
    ctx = f"candidate {cid}"

    profile = _parse_profile(rec["profile"], f"{ctx}.profile")
    career_history = [
        _parse_role(r, f"{ctx}.career_history[{i}]")
        for i, r in enumerate(rec["career_history"] or [])
    ]
    education = [
        _parse_education(e, f"{ctx}.education[{i}]")
        for i, e in enumerate(rec["education"] or [])
    ]
    skills = [
        _parse_skill(s, f"{ctx}.skills[{i}]")
        for i, s in enumerate(rec["skills"] or [])
    ]
    certifications = [
        _parse_certification(c, f"{ctx}.certifications[{i}]")
        for i, c in enumerate(rec.get("certifications") or [])
    ]
    languages = [
        _parse_language(l, f"{ctx}.languages[{i}]")
        for i, l in enumerate(rec.get("languages") or [])
    ]
    signals = _parse_signals(rec["redrob_signals"], f"{ctx}.redrob_signals")

    return Candidate(
        candidate_id=cid,
        profile=profile,
        career_history=career_history,
        education=education,
        skills=skills,
        certifications=certifications,
        languages=languages,
        redrob_signals=signals,
        raw=dict(rec),
    )


def _candidate_to_mapping(c: Candidate) -> dict[str, Any]:
    """Project a :class:`Candidate` back to the minimal mapping shape that
    :func:`candidate_to_text` reads. Values are copied verbatim from the typed
    fields, so the text built from a Candidate is identical to the text built
    from its original raw record — the embedding pipeline gets one canonical
    string regardless of which form it holds.
    """
    return {
        "profile": {
            "headline": c.profile.headline,
            "summary": c.profile.summary,
            "current_title": c.profile.current_title,
            "current_company": c.profile.current_company,
        },
        "career_history": [
            {"title": r.title, "company": r.company, "description": r.description}
            for r in c.career_history
        ],
        "skills": [
            {"name": s.name, "proficiency": s.proficiency} for s in c.skills
        ],
    }


def _clean(value: Any) -> str:
    """Coerce a possibly-missing field to a stripped string ("" if absent)."""
    if value is None:
        return ""
    return str(value).strip()


def candidate_to_text(rec: Union["Candidate", Mapping[str, Any]]) -> str:
    """Build the rich text representation used to embed a candidate.

    The string is intentionally **description-heavy**: headline + summary +
    every role's title *and free-text description* + skills-with-context. Role
    descriptions are the signal that surfaces the "plain-language Tier-5"
    candidates who do real retrieval/ranking/ML work but never type the
    buzzwords (STRATEGY.md §5). Skill tags come last and carry the least weight
    precisely because keyword-stuffers inflate them.

    Accepts either a raw record dict (or any mapping with the same shape) **or**
    a typed :class:`Candidate`. The dict path is unchanged, so it stays safe to
    call while streaming ``candidates.jsonl`` without constructing a
    ``Candidate`` first (the embedding pipeline relies on this). A ``Candidate``
    is projected back to the same mapping shape, yielding an identical string.
    Missing/optional fields are tolerated.
    """
    if isinstance(rec, Candidate):
        rec = _candidate_to_mapping(rec)
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
