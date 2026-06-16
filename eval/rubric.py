"""STEP 2 — the rule-based 0-4 grader (transparent, inspectable).

Implements the STRATEGY.md §4 relevance model as weighted components in [0,1],
mapped to a 0-4 tier with hard gates for the top tiers and disqualifier caps.
Honeypots/stuffers (detected in ``eval.heuristics``) are hard-forced to 0. Every
grade ships its full per-rule breakdown so we can defend it at Stage-5 review.

Shared parsing + detection lives in ``eval.heuristics`` (single source); only the
scoring-specific lexicons live here.
"""

from __future__ import annotations

import re

from eval.heuristics import (
    _profile,
    _roles,
    _signals,
    _title,
    _yoe,
    avg_completed_tenure,
    career_text,
    consulting_fraction,
    honeypot_reasons,
    stuffer_reasons,
    substance_areas_hit,
    title_class,
)

# --------------------------------------------------------------------------- #
# Scoring-only lexicons (the shared title/substance lexicons live in heuristics).
# --------------------------------------------------------------------------- #
NLP_IR_RE = re.compile(
    r"\b(nlp|natural language|information retrieval|text ranking|search relevance|"
    r"recommend|recommender|language model|word embedding|tf-?idf|question answering|"
    r"query understanding|semantic search|retrieval)\b",
    re.I,
)
CV_SPEECH_ROBOTICS_RE = re.compile(
    r"\b(computer vision|image classification|object detection|segmentation|opencv|"
    r"speech recognition|\btts\b|\basr\b|text-to-speech|robotics|slam|lidar|autonomous|"
    r"point cloud|sensor fusion)\b",
    re.I,
)
RESEARCH_ONLY_RE = re.compile(
    r"\b(phd|postdoc|published|publication|peer-reviewed|research paper|neurips|icml|"
    r"acl|cvpr|thesis|dissertation|academic)\b",
    re.I,
)
BUILD_SHIP_RE = re.compile(
    r"\b(built|build|shipped|deployed|implemented|developed|engineered|wrote|coded|"
    r"production|launched|delivered|owned)\b",
    re.I,
)
NONCODING_LEAD_RE = re.compile(
    r"\b(architect|tech lead|technical lead|engineering manager|head of|director|"
    r"principal architect|leadership|strategy|roadmap|stakeholder)\b",
    re.I,
)
LANGCHAIN_RE = re.compile(r"\b(langchain|llama-?index|openai api|gpt-4|prompt engineering)\b", re.I)
CLASSICAL_ML_RE = re.compile(
    r"\b(scikit|sklearn|xgboost|lightgbm|random forest|logistic regression|svm|"
    r"gradient boosting|feature engineering|classical ml|statistical model)\b",
    re.I,
)
TIER1_CITY_RE = re.compile(r"\b(pune|noida|hyderabad|mumbai|delhi|gurgaon|gurugram|bangalore|bengaluru)\b", re.I)

# §4 base-relevance weights (the behavioral multiplier is applied elsewhere, in
# the ranker, not in the silver tier). Normalised to sum to 1.0.
_RAW_WEIGHTS = {
    "role_substance": 35,
    "experience_band": 10,
    "nlp_ir": 10,
    "product_company": 10,
    "production_recency": 5,
    "tenure_stability": 5,
    "external_validation": 5,
    "location": 10,
}
_WSUM = sum(_RAW_WEIGHTS.values())
WEIGHTS = {k: v / _WSUM for k, v in _RAW_WEIGHTS.items()}


def _score_role_substance(c):
    """Dominant signal. Substance in the career text counts MORE than the title
    (substance > keyword): a strong title with empty descriptions cannot reach
    the top, and a plain adjacent title with real retrieval/ranking work can."""
    tc = title_class(_title(c))
    title_score = {"strong": 1.0, "adjacent": 0.45, "other": 0.25, "nontech": 0.0}[tc]
    n_areas = len(substance_areas_hit(career_text(c)))
    desc_score = {0: 0.0, 1: 0.5, 2: 0.8}.get(n_areas, 1.0)
    return round(0.45 * title_score + 0.55 * desc_score, 4)


def _score_experience_band(c, band):
    yoe = _yoe(c)
    if band["ideal_min"] <= yoe <= band["ideal_max"]:
        return 1.0
    if band["min"] <= yoe <= band["max"]:
        return 0.7
    if band["min"] - 1 <= yoe < band["min"] or band["max"] < yoe <= band["max"] + 2:
        return 0.35
    return 0.0


def _score_nlp_ir(text):
    if NLP_IR_RE.search(text):
        return 1.0
    if CV_SPEECH_ROBOTICS_RE.search(text):
        return 0.0  # primary CV/speech/robotics with no NLP/IR is a §4 negative
    return 0.5


def _score_product_company(c, consulting_firms):
    frac = consulting_fraction(c, consulting_firms)
    if frac >= 0.999:
        return 0.0  # career ENTIRELY at consulting/services -> strong negative
    return round(1.0 - frac, 4)


def _score_production_recency(c):
    cur = [r for r in _roles(c) if r.get("is_current")]
    cur = cur or _roles(c)[:1]
    if not cur:
        return 0.5
    r = cur[0]
    desc = r.get("description", "") or ""
    title = r.get("title", "") or ""
    if NONCODING_LEAD_RE.search(title) and not BUILD_SHIP_RE.search(desc):
        return 0.2  # senior who has stopped shipping
    if BUILD_SHIP_RE.search(desc):
        return 1.0
    return 0.6


def _score_tenure_stability(c):
    avg = avg_completed_tenure(c)
    if avg is None:
        return 0.7  # too few completed roles to judge -> neutral
    if avg >= 24:
        return 1.0
    if avg >= 18:
        return 0.6
    return 0.2  # < 1.5yr average -> title-chaser


def _score_external_validation(c):
    gh = _signals(c).get("github_activity_score", -1)
    try:
        gh = float(gh)
    except (TypeError, ValueError):
        gh = -1.0
    if gh <= 0:
        return 0.3  # no GitHub linked (-1) or zero activity -> mild neutral
    return round(0.5 + 0.5 * min(1.0, gh / 70.0), 4)


def _score_location(c):
    p = _profile(c)
    loc = p.get("location", "") or ""
    country = (p.get("country", "") or "").strip().lower()
    relocate = bool(_signals(c).get("willing_to_relocate"))
    if country == "india":
        return 1.0 if TIER1_CITY_RE.search(loc) else 0.8
    return 0.45 if relocate else 0.15


def _disqualifier_caps(c, text, consulting_firms):
    """Return {name: cap_grade} for each firing JD disqualifier (STRATEGY §4)."""
    caps = {}
    yoe = _yoe(c)
    if consulting_fraction(c, consulting_firms) >= 0.999:
        caps["consulting_only"] = 1
    if CV_SPEECH_ROBOTICS_RE.search(text) and not NLP_IR_RE.search(text):
        caps["cv_speech_robotics_no_nlp"] = 1
    avg = avg_completed_tenure(c)
    n_completed = sum(1 for r in _roles(c) if not r.get("is_current"))
    if avg is not None and avg < 18 and n_completed >= 3:
        caps["title_chaser"] = 2
    if RESEARCH_ONLY_RE.search(text) and not BUILD_SHIP_RE.search(text):
        caps["pure_research_no_prod"] = 1
    if LANGCHAIN_RE.search(text) and yoe < 3 and not CLASSICAL_ML_RE.search(text):
        caps["langchain_only_recent"] = 1
    if yoe < 4:
        caps["too_junior"] = 2
    elif yoe > 12:
        caps["too_senior"] = 2
    # Location: a candidate neither in India nor willing to relocate cannot take
    # this India role (no visa sponsorship). Down-weighted heavily (STRATEGY §4.8)
    # but not erased -> capped to "good fit", never the top tier.
    country = (_profile(c).get("country", "") or "").strip().lower()
    if country != "india" and not _signals(c).get("willing_to_relocate"):
        caps["non_india_no_relocate"] = 3
    return caps


def grade_rules(c, jd, today):
    """Grade a candidate 0-4 by the §4 rubric. Returns (grade, breakdown).

    The breakdown records every component score, its weight and contribution,
    the gates applied to the top tiers, the disqualifier caps, and any hard
    forced-zero reason — so every grade is fully inspectable and defensible.
    """
    text = career_text(c)
    band = jd.get("experience_band", {"min": 5, "max": 9, "ideal_min": 6, "ideal_max": 8})
    firms = jd.get("consulting_firms", [])

    comp = {
        "role_substance": _score_role_substance(c),
        "experience_band": _score_experience_band(c, band),
        "nlp_ir": _score_nlp_ir(text),
        "product_company": _score_product_company(c, firms),
        "production_recency": _score_production_recency(c),
        "tenure_stability": _score_tenure_stability(c),
        "external_validation": _score_external_validation(c),
        "location": _score_location(c),
    }
    contributions = {k: round(WEIGHTS[k] * comp[k], 4) for k in comp}
    base = round(sum(contributions.values()), 4)

    # Base tier from the weighted score.
    if base >= 0.72:
        grade = 4
    elif base >= 0.55:
        grade = 3
    elif base >= 0.38:
        grade = 2
    elif base >= 0.20:
        grade = 1
    else:
        grade = 0

    # Hard gates so the top tiers mean what STRATEGY says they mean (a 4 is the
    # CONJUNCTION of the must-haves, not a lucky weighted sum).
    gates = []
    yoe = _yoe(c)
    if grade >= 4 and not (
        comp["role_substance"] >= 0.65
        and band["min"] <= yoe <= band["max"]
        and comp["nlp_ir"] >= 0.5
        and comp["product_company"] > 0
        and comp["production_recency"] >= 0.5
    ):
        grade = 3
        gates.append("demoted 4->3: missing a must-have (substance/band/nlp-ir/product/recency)")
    if grade >= 3 and not (
        comp["role_substance"] >= 0.45 and comp["nlp_ir"] >= 0.5 and comp["product_company"] > 0
    ):
        grade = 2
        gates.append("demoted 3->2: missing core substance / nlp-ir / product signal")

    # Substance dominance (STRATEGY §4: role substance is THE driver). A profile
    # with little/no role substance cannot ride experience/location/github into a
    # high tier — substance caps the grade outright.
    rs = comp["role_substance"]
    if rs < 0.10:
        substance_cap = 0    # irrelevant (e.g. non-tech title, no substance at all)
    elif rs < 0.20:
        substance_cap = 1    # tangential
    elif rs < 0.40:
        substance_cap = 2    # partial / adjacent
    else:
        substance_cap = 4
    if grade > substance_cap:
        gates.append(f"substance cap {substance_cap}: role_substance={rs}")
        grade = substance_cap

    # Disqualifier caps.
    caps = _disqualifier_caps(c, text, firms)
    for name, cap in caps.items():
        if grade > cap:
            grade = cap

    # Hard forced-zero: honeypots and stuffers, regardless of everything above.
    forced = []
    hp = honeypot_reasons(c, today)
    st = stuffer_reasons(c)
    if hp:
        forced.append({"type": "honeypot", "reasons": hp})
    if st:
        forced.append({"type": "keyword_stuffer", "reasons": st})
    if forced:
        grade = 0

    breakdown = {
        "components": comp,
        "weights": {k: round(WEIGHTS[k], 4) for k in comp},
        "contributions": contributions,
        "base_score": base,
        "gates_applied": gates,
        "disqualifier_caps": caps,
        "forced_zero": forced,
        "final": grade,
    }
    return grade, breakdown
