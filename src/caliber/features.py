"""Structured feature scorer (ONLINE) — STRATEGY.md §4 relevance drivers.

``structured_features(candidate, jd_profile) -> dict[str, float]`` turns a typed
:class:`~caliber.schema.Candidate` into a dict of NAMED, documented features,
each (unless noted) normalised to roughly ``[0, 1]``. These are RAW SIGNALS, one
per §4 driver — NOT a final weighted score. ``scorer.py`` (next batch) applies the
weights / the LTR model; we deliberately bake no weights in here.

Two design rules drive everything below:

1. **Substance > keyword (the skill-gate).** ``role_substance`` — the dominant
   §4 signal — is computed from the career *text* (headline + summary + role
   titles + role DESCRIPTIONS), NEVER from the skill-tag list. A listed skill
   therefore earns substance credit *only* when a role description corroborates
   it. A non-tech-title candidate who lists many AI skills with HR/sales
   descriptions scores ``role_substance == 0`` and sinks naturally — no hard
   floor needed. This is the ranker's half of the stuffer defence; the eval
   answer key's half is ``eval/heuristics.py:stuffer_reasons`` (a hard zero).
   Two intentional approaches to the same trap (honeypots get floored, stuffers
   get gated) — see the honeypot reconciliation note.

2. **Consistency with the answer key.** Every feature mirrors the corresponding
   component in ``eval/rubric.py`` / ``eval/heuristics.py`` so the ranker and the
   silver-label grader reason about a candidate the SAME way. The ranker may not
   import ``eval`` (the answer key must never depend on the path it grades — see
   ``scripts/make_silver_labels.py``), so the shared lexicons/logic are COPIED
   here verbatim. That copy is a divergence *risk*: the blocks below are marked
   "MIRROR of eval/…"; if one side changes, change both (or, better, lift the
   lexicons into a shared ``caliber`` module both import — a follow-up that would
   touch ``eval``, out of scope for this batch). Where a feature intentionally
   differs from eval, the difference is called out inline with "DIVERGENCE".

Pure, deterministic, CPU-only, no network. All recency/date math uses
``config.REFERENCE_DATE`` (the static snapshot date), never the wall clock.
"""

from __future__ import annotations

import datetime as dt
import re
from typing import Any, Mapping, Optional

from . import config
from .honeypot import is_honeypot
from .schema import Candidate

# --------------------------------------------------------------------------- #
# Tunable constants (kept here, not in config.py, so this batch touches no shared
# file). Surfaced at module top so a future sweep can find them.
# --------------------------------------------------------------------------- #
# experience_band: how far past the JD band (in years) the score decays to 0.
EXP_BAND_LOWER_SLACK = 2.0   # below (band.min - slack) -> 0
EXP_BAND_UPPER_SLACK = 4.0   # above (band.max + slack) -> 0
# production_recency: months in a non-shipping lead role before the deeper penalty.
RECENCY_STALE_MONTHS = 18
# external_validation: github_activity_score that earns the full bonus.
GITHUB_FULL_CREDIT = 70.0
# skill_corroboration: how many AI/ML skill tags count as "many" (mirrors the
# eval stuffer threshold) and how many substance areas fully back them.
STUFFER_AI_SKILL_THRESHOLD = 4
SKILL_CORROBORATION_FULL_AREAS = 2

DEFAULT_EXPERIENCE_BAND = {"min": 5, "max": 9, "ideal_min": 6, "ideal_max": 8}

# --------------------------------------------------------------------------- #
# Lexicons — MIRROR of eval/heuristics.py (title classes, substance areas, AI
# skill set) and eval/rubric.py (nlp/ir, cv/speech, build/ship, non-coding lead,
# tier-1 cities). Copied verbatim so feature definitions match the answer key.
# --------------------------------------------------------------------------- #
# -- from eval/heuristics.py --
_STRONG_TITLE_RE = re.compile(
    r"\b(ml engineer|machine learning engineer|ai engineer|applied (ml |ai )?scientist|"
    r"ai research(er| engineer)?|research engineer|data scientist|ml scientist|"
    r"nlp engineer|ml ?ops engineer|deep learning|staff ml|senior ml|senior ai|"
    r"ai specialist|search engineer|relevance engineer|recommendation engineer)\b",
    re.I,
)
_ADJACENT_TITLE_RE = re.compile(
    r"\b(data engineer|senior data engineer|analytics engineer|backend engineer|"
    r"software engineer|full[- ]?stack|sde\b|platform engineer|developer|programmer)\b",
    re.I,
)
_NONTECH_TITLE_RE = re.compile(
    r"\b(hr\b|human resource|recruit|talent acquisition|sales\b|marketing|content|"
    r"writer|graphic|design|account|finance|financial|mechanical|civil\b|"
    r"operations|teacher|nurse|doctor|lawyer|business analyst|customer\b|"
    r"support\b|administrat|product manager|project manager|consultant|executive\b)",
    re.I,
)
_SUBSTANCE_AREAS = {
    "retrieval_embeddings": re.compile(
        r"\b(embedding|embeddings|dense retrieval|semantic search|sentence[- ]transformer|"
        r"bge|e5|vector search|nearest neighbor|\bann\b|\bknn\b|faiss|pinecone|weaviate|qdrant|milvus)\b",
        re.I,
    ),
    "ranking_ltr": re.compile(
        r"\b(ranking|re-?rank|learning to rank|\bltr\b|ndcg|mrr|relevance tuning|"
        r"search relevance|search quality)\b",
        re.I,
    ),
    "recommendation": re.compile(
        r"\b(recommend|recommender|recommendation|recsys|personali[sz]ation|"
        r"collaborative filtering|matching engine)\b",
        re.I,
    ),
    "search_ir": re.compile(
        r"\b(information retrieval|inverted index|bm25|elasticsearch|opensearch|lucene|solr|"
        r"query understanding|hybrid search|full[- ]text search)\b",
        re.I,
    ),
    "applied_ml_prod": re.compile(
        r"\b(deployed|in production|productioni[sz]ed|served|serving|model serving|"
        r"trained and deployed|shipped a model|ml pipeline|feature store|a/b test|online metric)\b",
        re.I,
    ),
}
_AI_ML_SKILLS = {
    "machine learning", "deep learning", "nlp", "natural language processing", "llm",
    "llms", "large language models", "fine-tuning llms", "rag", "transformers",
    "computer vision", "image classification", "object detection", "pytorch",
    "tensorflow", "keras", "hugging face", "huggingface", "bert", "gpt",
    "reinforcement learning", "speech recognition", "tts", "lora", "langchain",
    "information retrieval", "recommender systems", "semantic search", "embeddings",
    "neural networks", "scikit-learn", "xgboost", "data science", "mlops",
    "prompt engineering", "generative ai", "stable diffusion", "opencv",
}
# -- from eval/rubric.py --
_NLP_IR_RE = re.compile(
    r"\b(nlp|natural language|information retrieval|text ranking|search relevance|"
    r"recommend|recommender|language model|word embedding|tf-?idf|question answering|"
    r"query understanding|semantic search|retrieval)\b",
    re.I,
)
_CV_SPEECH_ROBOTICS_RE = re.compile(
    r"\b(computer vision|image classification|object detection|segmentation|opencv|"
    r"speech recognition|\btts\b|\basr\b|text-to-speech|robotics|slam|lidar|autonomous|"
    r"point cloud|sensor fusion)\b",
    re.I,
)
_BUILD_SHIP_RE = re.compile(
    r"\b(built|build|shipped|deployed|implemented|developed|engineered|wrote|coded|"
    r"production|launched|delivered|owned)\b",
    re.I,
)
_NONCODING_LEAD_RE = re.compile(
    r"\b(architect|tech lead|technical lead|engineering manager|head of|director|"
    r"principal architect|leadership|strategy|roadmap|stakeholder)\b",
    re.I,
)
_TIER1_CITY_RE = re.compile(
    r"\b(pune|noida|hyderabad|mumbai|delhi|gurgaon|gurugram|bangalore|bengaluru)\b",
    re.I,
)


# --------------------------------------------------------------------------- #
# Typed-Candidate accessors. These reproduce, on the typed object, exactly what
# the eval heuristics read off the raw dict, so the SAME text/numbers go into the
# SAME comparisons on both sides.
# --------------------------------------------------------------------------- #
def _yoe(c: Candidate) -> float:
    try:
        return float(c.profile.years_of_experience or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _career_text(c: Candidate) -> str:
    """All free text where career SUBSTANCE lives (descriptions dominate).
    MIRROR of eval/heuristics.py:career_text — headline + summary + each role's
    title + description, joined the same way, so substance detection is identical.
    """
    parts = [c.profile.headline, c.profile.summary]
    for r in c.career_history:
        parts.append(r.title)
        parts.append(r.description)
    return "  ".join(x for x in parts if x)


def _title_class(title: str) -> str:
    """MIRROR of eval/heuristics.py:title_class (same precedence order)."""
    if _STRONG_TITLE_RE.search(title):
        return "strong"
    if _NONTECH_TITLE_RE.search(title):
        return "nontech"
    if _ADJACENT_TITLE_RE.search(title):
        return "adjacent"
    return "other"


def _substance_areas_hit(text: str) -> list[str]:
    """MIRROR of eval/heuristics.py:substance_areas_hit."""
    return [name for name, rx in _SUBSTANCE_AREAS.items() if rx.search(text)]


def _ai_skill_count(c: Candidate) -> int:
    """MIRROR of eval/heuristics.py:ai_skill_count — skill *names* in the AI set."""
    return sum(1 for s in c.skills if (s.name or "").strip().lower() in _AI_ML_SKILLS)


def _consulting_fraction(c: Candidate, consulting_firms) -> float:
    """MIRROR of eval/heuristics.py:consulting_fraction — share of roles at a
    named services/consulting firm (substring match, case-insensitive)."""
    roles = c.career_history
    if not roles:
        return 0.0
    firms = {f.lower() for f in consulting_firms}

    def is_consult(name: str) -> bool:
        n = (name or "").lower()
        return any(f in n for f in firms)

    return sum(1 for r in roles if is_consult(r.company)) / len(roles)


def _avg_completed_tenure(c: Candidate) -> Optional[float]:
    """MIRROR of eval/heuristics.py:avg_completed_tenure — avg months of NON-current
    roles; None if fewer than 2 completed roles (too little signal to judge)."""
    durs = [r.duration_months for r in c.career_history if not r.is_current]
    if len(durs) < 2:
        return None
    return sum(durs) / len(durs)


def _parse_date(s) -> "dt.date | None":
    if not s:
        return None
    try:
        return dt.date.fromisoformat(s)
    except (ValueError, TypeError):
        return None


def _months_between(start: dt.date, end: dt.date) -> int:
    """Whole months from ``start`` to ``end`` (calendar arithmetic, no wall clock)."""
    return (end.year - start.year) * 12 + (end.month - start.month)


# --------------------------------------------------------------------------- #
# The features (one per STRATEGY §4 driver).
# --------------------------------------------------------------------------- #
def _role_substance(c: Candidate) -> float:
    """Dominant §4 signal AND the skill-gate. MIRROR of eval/rubric.py:
    _score_role_substance. Substance is read from the career TEXT (descriptions),
    not the skill tags — so a strong title with empty descriptions cannot reach
    the top, and a plain adjacent title with real retrieval/ranking work can. A
    keyword-stuffer (non-tech title, AI skill TAGS, HR descriptions) lands at 0.0:
    non-tech title contributes 0.0 and the descriptions hit no substance area.
    """
    tc = _title_class(c.profile.current_title)
    title_score = {"strong": 1.0, "adjacent": 0.45, "other": 0.25, "nontech": 0.0}[tc]
    n_areas = len(_substance_areas_hit(_career_text(c)))
    desc_score = {0: 0.0, 1: 0.5, 2: 0.8}.get(n_areas, 1.0)
    return round(0.45 * title_score + 0.55 * desc_score, 4)


def _skill_corroboration(c: Candidate, text: str) -> float:
    """Explicit skill-gate signal (auxiliary; not in eval). 1.0 means "no
    unsubstantiated skill-stuffing detected"; 0.0 means "claims many AI skills
    with zero corroborating career substance" — the stuffer signature.

    Aligned with eval/heuristics.py:stuffer_reasons, which trips on >=4 AI skill
    tags with no substance: a candidate claiming fewer than that has nothing to
    gate (-> 1.0, no penalty, so plain-language Tier-5s who list few buzzwords are
    NOT punished); a candidate claiming many is credited by how many distinct
    substance areas their descriptions actually back.
    """
    if _ai_skill_count(c) < STUFFER_AI_SKILL_THRESHOLD:
        return 1.0
    n_areas = len(_substance_areas_hit(text))
    return round(min(1.0, n_areas / SKILL_CORROBORATION_FULL_AREAS), 4)


def _experience_band(yoe: float, band: Mapping[str, float]) -> float:
    """Fit to the JD experience band, SMOOTH (piecewise-linear) rather than the
    discrete steps eval uses.

    DIVERGENCE from eval/rubric.py:_score_experience_band (which returns 1.0/0.7/
    0.35/0.0 in hard steps). The prompt asks for a smooth ramp — better gradient
    for the LTR — so this peaks at 1.0 across the ideal range and decays linearly
    to 0 at ``band.min - LOWER_SLACK`` (too junior) and ``band.max + UPPER_SLACK``
    (too senior). Same shape as eval (peak in band, fall off both sides), finer
    values. Flagged for a joint decision on whether to align eval.
    """
    imin, imax = band["ideal_min"], band["ideal_max"]
    bmin, bmax = band["min"], band["max"]
    if imin <= yoe <= imax:
        return 1.0
    if yoe < imin:
        lo = bmin - EXP_BAND_LOWER_SLACK
        if yoe <= lo:
            return 0.0
        return round((yoe - lo) / (imin - lo), 4)
    hi = bmax + EXP_BAND_UPPER_SLACK
    if yoe >= hi:
        return 0.0
    return round((hi - yoe) / (hi - imax), 4)


def _nlp_ir_signal(text: str) -> float:
    """NLP/IR background vs primarily CV/speech/robotics. MIRROR of
    eval/rubric.py:_score_nlp_ir. NLP/IR present -> 1.0; otherwise CV/speech/
    robotics present (and no NLP/IR) -> 0.0 (an explicit §4 negative); neither
    mentioned -> 0.5 (unknown)."""
    if _NLP_IR_RE.search(text):
        return 1.0
    if _CV_SPEECH_ROBOTICS_RE.search(text):
        return 0.0
    return 0.5


def _product_vs_consulting(c: Candidate, consulting_firms) -> float:
    """Product-company experience vs career-long services/consulting. MIRROR of
    eval/rubric.py:_score_product_company. A career ENTIRELY at named consulting
    firms -> 0.0 (strong §4 negative); otherwise 1 - (consulting fraction)."""
    frac = _consulting_fraction(c, consulting_firms)
    if frac >= 0.999:
        return 0.0
    return round(1.0 - frac, 4)


def _production_recency(c: Candidate, today: dt.date) -> float:
    """Recently shipped/hands-on vs long in a non-coding architecture/TL role.
    MIRROR of eval/rubric.py:_score_production_recency, with one refinement.

    Looks at the current (or most recent) role: a non-coding lead title with no
    build/ship verbs in the description -> 0.2 (a senior who has stopped
    shipping); build/ship verbs present -> 1.0; otherwise 0.6.

    DIVERGENCE (refinement): when the non-shipping lead role has lasted
    ``RECENCY_STALE_MONTHS`` (18) or more — measured from its start_date to
    ``config.REFERENCE_DATE``, never the wall clock — the penalty deepens to 0.1
    (STRATEGY §4.5: "18+ months ... with no recent shipping"). eval applies a flat
    0.2; flagged for joint decision.
    """
    roles = c.career_history
    cur = [r for r in roles if r.is_current] or roles[:1]
    if not cur:
        return 0.5
    r = cur[0]
    desc = r.description or ""
    title = r.title or ""
    if _NONCODING_LEAD_RE.search(title) and not _BUILD_SHIP_RE.search(desc):
        sd = _parse_date(r.start_date)
        months_in_role = _months_between(sd, today) if sd else 0
        return 0.1 if months_in_role >= RECENCY_STALE_MONTHS else 0.2
    if _BUILD_SHIP_RE.search(desc):
        return 1.0
    return 0.6


def _tenure_stability(c: Candidate) -> float:
    """Penalise title-chasing. MIRROR of eval/rubric.py:_score_tenure_stability.
    Avg completed-role tenure >= 24mo -> 1.0; >= 18mo -> 0.6; < 18mo -> 0.2
    (title-chaser); too few completed roles to judge -> 0.7 (neutral)."""
    avg = _avg_completed_tenure(c)
    if avg is None:
        return 0.7
    if avg >= 24:
        return 1.0
    if avg >= 18:
        return 0.6
    return 0.2


def _external_validation(c: Candidate) -> float:
    """Open-source / GitHub signal. github_activity_score is 0-100, or -1 when no
    GitHub is linked.

    DIVERGENCE from eval/rubric.py:_score_external_validation (which maps both -1
    and 0 to 0.3). Per the prompt, a MISSING GitHub (-1) is NEUTRAL, not a
    penalty — ~65% of the pool has none (STRATEGY §2), and external validation is
    a *positive* §4 signal (~5%), so its absence must not push the majority down.
    So: no GitHub (-1) -> 0.5 (neutral); linked GitHub -> 0.5 + 0.5·min(1, gh/70),
    i.e. presence only ever adds. Flagged for joint decision on aligning eval.
    """
    gh = c.redrob_signals.github_activity_score
    try:
        gh = float(gh)
    except (TypeError, ValueError):
        gh = -1.0
    if gh < 0:
        return 0.5  # no GitHub linked -> neutral, not penalised
    return round(0.5 + 0.5 * min(1.0, gh / GITHUB_FULL_CREDIT), 4)


def _location_fit(c: Candidate) -> float:
    """India Tier-1 / relocation-willing vs non-India. MIRROR of
    eval/rubric.py:_score_location. India + Tier-1 city -> 1.0; India elsewhere ->
    0.8; non-India willing to relocate -> 0.45; non-India not willing -> 0.15.

    The Tier-1 city regex is the operational form of
    jd_profile.location_prefs.preferred_cities (with spelling variants); kept as
    the eval regex so ranker and key agree on which cities count.
    """
    loc = c.profile.location or ""
    country = (c.profile.country or "").strip().lower()
    relocate = bool(c.redrob_signals.willing_to_relocate)
    if country == "india":
        return 1.0 if _TIER1_CITY_RE.search(loc) else 0.8
    return 0.45 if relocate else 0.15


# --------------------------------------------------------------------------- #
# Public entry point.
# --------------------------------------------------------------------------- #
def structured_features(candidate: Candidate, jd_profile: Mapping[str, Any]) -> dict[str, float]:
    """Compute the named structured features for one candidate (see module
    docstring). ``jd_profile`` is the loaded ``artifacts/jd_profile.json`` dict.

    Returns a flat ``dict[str, float]`` of raw signals — no weights applied. Every
    feature is present for any schema-valid candidate. ``is_honeypot`` is exposed
    as a 0/1 feature (the scorer floors on it; we do NOT floor here, per the
    honeypot/feature separation in ARCHITECTURE.md §3).
    """
    band = jd_profile.get("experience_band", DEFAULT_EXPERIENCE_BAND)
    firms = jd_profile.get("consulting_firms", [])
    today = dt.date.fromisoformat(config.REFERENCE_DATE)
    text = _career_text(candidate)
    hp_flag, _ = is_honeypot(candidate)

    return {
        "role_substance": _role_substance(candidate),
        "skill_corroboration": _skill_corroboration(candidate, text),
        "experience_band": _experience_band(_yoe(candidate), band),
        "nlp_ir_signal": _nlp_ir_signal(text),
        "product_vs_consulting": _product_vs_consulting(candidate, firms),
        "production_recency": _production_recency(candidate, today),
        "tenure_stability": _tenure_stability(candidate),
        "external_validation": _external_validation(candidate),
        "location_fit": _location_fit(candidate),
        "is_honeypot": 1.0 if hp_flag else 0.0,
    }
