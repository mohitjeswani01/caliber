"""Shared candidate-fact extraction + stuffer detection + honeypot delegation.

This is the SINGLE source of the eval-side detection heuristics: tolerant parsing
accessors over the raw candidate dict, the shared lexicons, the keyword-stuffer
detector, and the answer-key entry point to honeypot detection. ``sampling`` and
``rubric`` both import from here so the two graders can't disagree about what a
honeypot or a stuffer is.

RECONCILED (Batch 2): honeypot detection is no longer implemented here. The
contradiction/impossibility rules now live ONCE in ``caliber.honeypot`` (the
canonical home, also used by the online ranker), and :func:`honeypot_reasons`
below is a thin pass-through to it. This guarantees the eval answer key and the
ranker can never drift on which profiles are impossible — they run identical code
on identical fields. The keyword-STUFFER detector stays here: it is a labeling
concept (the ranker handles stuffers via skill-gating in ``features.py``, not via
the honeypot floor), so it has no online counterpart to reconcile against.

The one online module eval imports is ``caliber.honeypot`` — a dependency-light
pure-Python module (datetime + config + schema, no models/faiss/torch), so the
answer key still pulls in none of the ranking-path's heavy machinery.
"""

from __future__ import annotations

import re

# Single source of truth for honeypot detection (see module docstring). Dual
# import path: ``caliber`` resolves under the test harness (conftest puts src/ on
# the path); ``src.caliber`` resolves when run from the repo root (the silver
# label script). Mirrors the pattern already used for ``config`` in sampling.py.
try:
    from caliber.honeypot import honeypot_reasons as _canonical_honeypot_reasons
except ImportError:  # pragma: no cover - exercised by the from-root run path
    from src.caliber.honeypot import honeypot_reasons as _canonical_honeypot_reasons

# --------------------------------------------------------------------------- #
# Lexicons / regexes. The substance/aspect terms are aligned with
# artifacts/jd_profile.json so the rule grader scores the SAME signals the JD
# profile encodes (career substance, not free-floating keywords).
# --------------------------------------------------------------------------- #
STRONG_TITLE_RE = re.compile(
    r"\b(ml engineer|machine learning engineer|ai engineer|applied (ml |ai )?scientist|"
    r"ai research(er| engineer)?|research engineer|data scientist|ml scientist|"
    r"nlp engineer|ml ?ops engineer|deep learning|staff ml|senior ml|senior ai|"
    r"ai specialist|search engineer|relevance engineer|recommendation engineer)\b",
    re.I,
)
ADJACENT_TITLE_RE = re.compile(
    r"\b(data engineer|senior data engineer|analytics engineer|backend engineer|"
    r"software engineer|full[- ]?stack|sde\b|platform engineer|developer|programmer)\b",
    re.I,
)
# Note on word boundaries: several real titles inflect ("Account" -> "Accountant",
# "Design" -> "Designer", "Recruit" -> "Recruiter"), so those stems intentionally
# omit a trailing \b. Stems that could collide with tech titles keep the trailing
# \b ("sales\b" matches "Sales Executive" but NOT "Salesforce Developer").
NONTECH_TITLE_RE = re.compile(
    r"\b(hr\b|human resource|recruit|talent acquisition|sales\b|marketing|content|"
    r"writer|graphic|design|account|finance|financial|mechanical|civil\b|"
    r"operations|teacher|nurse|doctor|lawyer|business analyst|customer\b|"
    r"support\b|administrat|product manager|project manager|consultant|executive\b)",
    re.I,
)

# Role-substance areas (STRATEGY §4.1, dominant). Each distinct AREA hit in the
# free-text career descriptions/summary is what we actually reward.
SUBSTANCE_AREAS = {
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

# Skill *names* (tags) that count as AI/ML for stuffer detection. Stuffers list
# many of these without any career substance backing them.
AI_ML_SKILLS = {
    "machine learning", "deep learning", "nlp", "natural language processing", "llm",
    "llms", "large language models", "fine-tuning llms", "rag", "transformers",
    "computer vision", "image classification", "object detection", "pytorch",
    "tensorflow", "keras", "hugging face", "huggingface", "bert", "gpt",
    "reinforcement learning", "speech recognition", "tts", "lora", "langchain",
    "information retrieval", "recommender systems", "semantic search", "embeddings",
    "neural networks", "scikit-learn", "xgboost", "data science", "mlops",
    "prompt engineering", "generative ai", "stable diffusion", "opencv",
}


# --------------------------------------------------------------------------- #
# Lightweight, tolerant accessors over the raw candidate dict.
# (schema.py is still a stub and, by design, eval must not depend on it.)
# --------------------------------------------------------------------------- #
def _profile(c):
    return c.get("profile", {}) or {}


def _roles(c):
    return c.get("career_history", []) or []


def _skills(c):
    return c.get("skills", []) or []


def _signals(c):
    return c.get("redrob_signals", {}) or {}


def _yoe(c):
    try:
        return float(_profile(c).get("years_of_experience") or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _title(c):
    return (_profile(c).get("current_title") or "").strip()


def career_text(c):
    """All free text where career SUBSTANCE lives (descriptions dominate)."""
    p = _profile(c)
    parts = [p.get("headline", ""), p.get("summary", "")]
    for r in _roles(c):
        parts.append(r.get("title", ""))
        parts.append(r.get("description", ""))
    return "  ".join(x for x in parts if x)


def title_class(title):
    """strong ML/AI title > adjacent tech title > non-tech > other."""
    if STRONG_TITLE_RE.search(title):
        return "strong"
    if NONTECH_TITLE_RE.search(title):
        return "nontech"
    if ADJACENT_TITLE_RE.search(title):
        return "adjacent"
    return "other"


def ai_skill_count(c):
    return sum(1 for s in _skills(c) if (s.get("name", "") or "").strip().lower() in AI_ML_SKILLS)


def substance_areas_hit(text):
    return [name for name, rx in SUBSTANCE_AREAS.items() if rx.search(text)]


def consulting_fraction(c, consulting_firms):
    roles = _roles(c)
    if not roles:
        return 0.0
    firms = {f.lower() for f in consulting_firms}

    def is_consult(name):
        n = (name or "").lower()
        return any(f in n for f in firms)

    return sum(1 for r in roles if is_consult(r.get("company"))) / len(roles)


def avg_completed_tenure(c):
    """Average duration (months) of NON-current roles; None if <2 completed."""
    durs = [r.get("duration_months", 0) for r in _roles(c) if not r.get("is_current")]
    if len(durs) < 2:
        return None
    return sum(durs) / len(durs)


# --------------------------------------------------------------------------- #
# Honeypot & stuffer detection (STRATEGY §5).
# Precision-tuned: a false positive that buries a real fit is as costly as a
# miss. We only trip on genuine INTERNAL contradictions, never on keywords.
# --------------------------------------------------------------------------- #
# Re-export the canonical detector under this module's name (same signature:
# ``honeypot_reasons(record, today) -> list[str]``). It is a direct alias, not a
# wrapper, so ``eval.heuristics.honeypot_reasons IS caliber.honeypot.honeypot_reasons``
# — there is one function object, impossible for a second copy to drift from it.
# ``sampling``/``rubric``/the silver-label tests keep importing it from here
# unchanged.
honeypot_reasons = _canonical_honeypot_reasons


def stuffer_reasons(c):
    """Keyword-stuffer (STRATEGY §5): non-tech title + many AI skills, with NO
    career substance to back any of them. The gate is the whole point — a skill
    counts only if the role history corroborates it."""
    reasons = []
    if title_class(_title(c)) != "nontech":
        return reasons
    n_ai = ai_skill_count(c)
    if n_ai < 4:
        return reasons
    if substance_areas_hit(career_text(c)):
        return reasons  # the history backs the skills -> not a stuffer
    reasons.append(
        f"non-tech title '{_title(c)}' lists {n_ai} AI/ML skills with zero corroborating "
        f"career substance"
    )
    return reasons
