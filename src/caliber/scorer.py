"""Final composite scoring (ONLINE) — the orchestration policy of the ranker.

This module composes every upstream signal into ONE explainable score per
candidate. It is deliberately the *only* place the retrieval / feature / rerank /
behavioural pieces meet, so the scoring policy lives in one auditable spot (we
defend each weight in the Stage-5 interview).

Pipeline (ARCHITECTURE.md §1; STRATEGY.md §4–§6), all CPU / offline / deterministic:

  1. SEMANTIC RETRIEVAL  — embed the JD aspect ``query_text``s (``embeddings``),
     search the FAISS index per aspect (``index.search``), aggregate the cosines
     WEIGHTED by the jd_profile aspect weights into one semantic ranking + a
     per-candidate ``semantic_sim`` signal, and a retrieval POOL (union of hits).
  2. LEXICAL RETRIEVAL   — BM25 (``rank_bm25``) over the retrieval pool's role
     DESCRIPTIONS, queried with the JD aspect text → one lexical ranking. This is
     how a plain-language Tier-5 (real retrieval/ranking work, weak buzzwords)
     still surfaces.
  3. FUSION SHORTLIST    — ``fusion.reciprocal_rank_fusion(semantic, lexical)`` →
     top ``SHORTLIST_SIZE`` (~800) = the candidate set we score in detail.
  4. STRUCTURED FEATURES — ``features.structured_features`` per shortlisted cand.
  5. HONEYPOT            — ``honeypot.is_honeypot`` ONCE per candidate (flag +
     reasons; reasons feed reasoning.py later). The flag drives the floor.
  6. CROSS-ENCODER       — ``cross_encoder.rerank`` over the top
     ``CROSS_ENCODER_SHORTLIST_SIZE`` of the shortlist; raw logits → sigmoid →
     [0,1]. Skipped gracefully (model absent / disabled) without crashing.
  7. COMPOSITE BASE      — assemble an ordered feature vector and ``combine`` it
     into ``base`` ∈ ~[0,1] (a tunable hand-weighted sum today; LTR-swappable).
  8. BEHAVIORAL          — ``final = base × behavioral_multiplier`` ∈ base·[0.50,1.15].
  9. HONEYPOT FLOOR      — detected honeypots forced to ``HONEYPOT_FLOOR`` (below
     every real score), applied LAST so a high behavioural multiplier can never
     lift a honeypot off the floor.

What this module does NOT do: it does not select the final 100, enforce the
non-increasing-score contract, or write the CSV — that is ``ranker.py``. It
returns a full per-candidate breakdown so ``ranker.py`` can sort and
``reasoning.py`` can explain every placement.

Determinism (ARCHITECTURE.md §5): pure CPU, no network at score time, no wall
clock (all date math is inside the consumed modules, pinned to
``config.REFERENCE_DATE``). Every internal sort breaks ties by
``config.TIE_BREAK_KEY`` (candidate_id) ascending, so two runs are identical.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence

from . import config
from .behavioral import behavioral_multiplier
from .features import structured_features
from .fusion import reciprocal_rank_fusion
from .honeypot import is_honeypot
from .schema import Candidate, candidate_to_text

# --------------------------------------------------------------------------- #
# Tunable retrieval sizes. These are NOT in config.py (config holds the
# cross-encoder + behavioral knobs); surfaced here as named constants so a future
# sweep can find them. CROSS_ENCODER_SHORTLIST_SIZE *is* in config and is read
# from there (never hardcoded).
# --------------------------------------------------------------------------- #
# Per-aspect FAISS retrieval depth. Generous: the union across the 6 aspects is
# the universe BM25 + fusion then narrow. 800 keeps recall high while the index
# search stays trivially within the CPU budget (flat 100K×384).
SEMANTIC_RETRIEVAL_K = 800
# Post-fusion shortlist we score in detail (features + behavioral; CE on its head).
SHORTLIST_SIZE = 800

# Detected honeypots are forced here — strictly below any real candidate's final
# score. Real finals are >= 0 (base ∈ [0,1] × behavioral ∈ [0.50,1.15]), so a
# negative floor guarantees honeypots sort to the bottom. They are floored, NOT
# removed (STRATEGY.md §5): ranker still sees them, with candidate_id tie-break.
HONEYPOT_FLOOR = -1.0

_TIE_BREAK = config.TIE_BREAK_KEY  # "candidate_id" (documented; ids ARE the key)

# --------------------------------------------------------------------------- #
# Composite weights — the PRIORS. STRATEGY.md §4 makes role_substance dominant;
# the cross-encoder and semantic similarity carry the description-level substance
# signal that the structured features cannot see. These are *starting priors to
# be tuned against the silver labels*, NOT frozen. The LTR model (ltr.py) later
# replaces `combine` with learned weights over the SAME ordered feature vector,
# so retuning never touches the feature assembly below.
#
# is_honeypot is intentionally absent from the composite: it is handled by the
# hard FLOOR (step 9), not as a soft additive feature.
# --------------------------------------------------------------------------- #
COMPOSITE_FEATURE_NAMES: tuple[str, ...] = (
    "role_substance",          # §4.1 dominant — career-text substance (gated)
    "skill_corroboration",     # the skill-gate auxiliary (stuffer penalty)
    "experience_band",         # §4.2
    "nlp_ir_signal",           # §4.3
    "product_vs_consulting",   # §4.4
    "production_recency",      # §4.5
    "tenure_stability",        # §4.6
    "external_validation",     # §4.7
    "location_fit",            # §4.8
    "ce_score",                # cross-encoder pair score (sigmoid-normalized)
    "semantic_sim",            # aspect-weighted bi-encoder cosine, clamped [0,1]
)

# TUNED (2026-06) via the multi-seed silver sweep (eval/sweep.py): selected on the
# TRAIN composite, validated on a held-out half across 5 deterministic stratified
# splits (seeds 1/7/42/123/2024). The change vs the original ce=0.25 / sem=0.20
# prior — CUT the cross-encoder weight and shift that mass into the structured
# substance backbone — beat that prior 5/5 on BOTH validation composite and
# NDCG@10 (the gain is NDCG-driven; MAP/P@10 flat). The isolation curve showed the
# old 0.25 CE was consistently too high.
#
# ce_score = 0.10 is a DELIBERATELY LOW, NON-ZERO CE, not a fragile tuned constant:
# the silver data marginally favours 0.0, but we keep a little CE because its proven
# keyword-stuffer discrimination matters on the real pool in ways our silver set
# cannot measure. role_substance stays dominant per §4; the eight small structured
# features are scaled together (× 0.57/0.32) to fill the remainder so the vector
# still sums to 1.0. Still PRIORS, not frozen — re-sweep when the silver set grows.
DEFAULT_WEIGHTS: dict[str, float] = {
    # Structured substance backbone (~0.80 total; role_substance dominant per §4).
    "role_substance": 0.23,
    "skill_corroboration": 0.07125,        # ~0.0713  (0.04 × 0.57/0.32)
    "experience_band": 0.0890625,          # ~0.0891  (0.05 × 0.57/0.32)
    "nlp_ir_signal": 0.0890625,            # ~0.0891
    "product_vs_consulting": 0.0890625,    # ~0.0891
    "production_recency": 0.0534375,       # ~0.0534  (0.03 × 0.57/0.32)
    "tenure_stability": 0.0534375,         # ~0.0534
    "external_validation": 0.035625,       # ~0.0356  (0.02 × 0.57/0.32)
    "location_fit": 0.0890625,             # ~0.0891
    # Description-level relevance (~0.20 total). CE sharpens the head (where
    # NDCG@10/@50 lives) but at a low weight; semantic_sim is the broader signal.
    "ce_score": 0.10,
    "semantic_sim": 0.10,
}
assert abs(sum(DEFAULT_WEIGHTS.values()) - 1.0) < 1e-9
assert set(DEFAULT_WEIGHTS) == set(COMPOSITE_FEATURE_NAMES)


# --------------------------------------------------------------------------- #
# Per-candidate breakdown (the scorer's output contract).
# --------------------------------------------------------------------------- #
@dataclass
class CandidateScore:
    """Everything needed to rank AND explain one candidate's placement.

    ``ranker.py`` consumes ``final_score`` (+ candidate_id tie-break);
    ``reasoning.py`` consumes ``feature_dict`` + ``honeypot_reasons`` + the
    retrieval/CE signals. Kept deliberately rich so we can defend every rank.
    """
    candidate_id: str
    final_score: float
    base_score: float
    behavioral_mult: float
    is_honeypot: bool
    honeypot_reasons: list[str]
    feature_dict: dict[str, float]            # the 10 structured features
    ce_score: Optional[float]                 # sigmoid-normalized CE, None if not reranked
    ce_used: bool                             # did THIS candidate get a CE score?
    semantic_sim: float                       # aspect-weighted cosine, clamped [0,1]
    rrf_score: float                          # fusion score (debug / audit)


# --------------------------------------------------------------------------- #
# Small pure helpers.
# --------------------------------------------------------------------------- #
def _sigmoid(x: float) -> float:
    """Numerically-stable logistic sigmoid → (0, 1). Overflow-safe for large |x|
    so the cross-encoder's ~[-11, +11] logits never produce inf/NaN."""
    if x >= 0.0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


def _clamp01(x: float) -> float:
    """Clamp to [0, 1] (keeps the cosine signal a well-defined [0,1] feature)."""
    if x != x:  # NaN guard
        return 0.0
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)


_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric tokenization — deterministic, shared by the BM25
    corpus and query so the term spaces match."""
    return _TOKEN_RE.findall((text or "").lower())


def combine(values: Mapping[str, Optional[float]], weights: Mapping[str, float]) -> float:
    """Hand-weighted base-relevance blend over the ordered composite features.

    A weighted sum that RENORMALIZES over the present (non-None) features, so the
    result stays in ~[0,1] whether or not the cross-encoder ran: when ``ce_score``
    is ``None`` (CE skipped / candidate past the rerank head) its weight is simply
    dropped and the remaining weights are rescaled to sum to 1. That keeps the
    composite well-defined and comparable across candidates either way.

    This is the swappable combination step (STEP 7 / decision #5): ``ltr.py`` can
    supply ``predict`` over the same ``COMPOSITE_FEATURE_NAMES`` vector as a
    drop-in ``combine_fn`` without touching the feature assembly.
    """
    num = 0.0
    den = 0.0
    for name in COMPOSITE_FEATURE_NAMES:
        v = values.get(name)
        if v is None:
            continue
        w = weights.get(name, 0.0)
        num += w * float(v)
        den += w
    if den <= 0.0:
        return 0.0
    return num / den


def load_jd_profile_artifact(path: Optional[Any] = None) -> dict:
    """Convenience: load ``artifacts/jd_profile.json`` (the offline → online
    handoff). Plain JSON read — NOT a dependency on ``jd_profile.load_jd_profile``
    (which is not implemented yet). ``ranker.py`` may call this until the offline
    loader lands.
    """
    if path is None:
        path = config.ARTIFACTS_DIR / config.JD_PROFILE_FILE
    with open(Path(path), "r", encoding="utf-8") as fh:
        return json.load(fh)


def _aspects_sorted(jd_profile: Mapping[str, Any]) -> list[tuple[str, dict]]:
    """Aspects in a DETERMINISTIC order (sorted by name) so the query batch /
    weight vector / search rows always align run-to-run regardless of JSON order.
    """
    aspects = jd_profile.get("aspects", {}) or {}
    return sorted(aspects.items(), key=lambda kv: kv[0])


def _build_jd_text(jd_profile: Mapping[str, Any]) -> str:
    """One JD string for the cross-encoder: the role plus every aspect query_text.
    The CE scores each candidate against this whole-JD description."""
    parts: list[str] = []
    role = jd_profile.get("role")
    if role:
        parts.append(str(role))
    for _name, aspect in _aspects_sorted(jd_profile):
        qt = (aspect or {}).get("query_text")
        if qt:
            parts.append(str(qt))
    return "  ".join(parts)


def _bm25_query_tokens(jd_profile: Mapping[str, Any]) -> list[str]:
    """BM25 query terms = tokens of every aspect query_text + its keywords."""
    toks: list[str] = []
    for _name, aspect in _aspects_sorted(jd_profile):
        aspect = aspect or {}
        toks.extend(_tokenize(aspect.get("query_text", "")))
        for kw in aspect.get("keywords", []) or []:
            toks.extend(_tokenize(kw))
    return toks


def _role_description_text(c: Candidate) -> str:
    """The lexical document for BM25: a candidate's role DESCRIPTIONS (+ headline
    and summary, which are free-text career substance too), joined. Skill TAGS are
    excluded on purpose — BM25 over tags would reward the keyword-stuffer."""
    parts = [c.profile.headline or "", c.profile.summary or ""]
    parts.extend(r.description or "" for r in c.career_history)
    return "  ".join(p for p in parts if p)


# --------------------------------------------------------------------------- #
# Retrieval stages.
# --------------------------------------------------------------------------- #
def _semantic_retrieval(
    jd_profile: Mapping[str, Any],
    candidate_ids: Sequence,
    faiss_index: Any,
    encode_query_fn: Callable,
    search_fn: Callable,
    semantic_k: int,
) -> tuple[list[str], dict[str, float]]:
    """STEP 1. Encode the aspect queries, search per aspect, aggregate cosines
    weighted by aspect weight into one semantic ranking + a {id: semantic_sim} map.

    Aggregation method: for each candidate, ``semantic_sim = Σ_aspect (aspect_weight
    × cosine(aspect_query, candidate))`` over the aspects whose top-k retrieved
    that candidate (a missing aspect contributes 0 for that candidate). Aspect
    weights sum to 1.0, cosines ≤ 1, so the aggregate lives in ~[0,1]. The semantic
    RANKING is this map sorted desc, ties by candidate_id asc.
    """
    aspects = _aspects_sorted(jd_profile)
    if not aspects:
        return [], {}

    query_texts = [(a or {}).get("query_text", "") for _n, a in aspects]
    weights = [float((a or {}).get("weight", 0.0)) for _n, a in aspects]

    n_cand = len(candidate_ids)
    k = max(1, min(semantic_k, n_cand))

    query_emb = encode_query_fn(query_texts, is_query=True)
    scores, indices = search_fn(faiss_index, query_emb, k)

    sem: dict[str, float] = {}
    for a_idx in range(len(aspects)):
        w = weights[a_idx]
        if w == 0.0:
            # Still contributes to the pool via the ranking below? No: a zero-weight
            # aspect adds nothing to semantic_sim and nothing to the ranking. Skip.
            continue
        row_scores = scores[a_idx]
        row_indices = indices[a_idx]
        for j in range(len(row_indices)):
            idx = int(row_indices[j])
            if idx < 0 or idx >= n_cand:
                continue  # FAISS pads with -1 when k > ntotal
            cid = str(candidate_ids[idx])
            sem[cid] = sem.get(cid, 0.0) + w * float(row_scores[j])

    ranking = sorted(sem.keys(), key=lambda cid: (-sem[cid], cid))
    return ranking, sem


def _lexical_retrieval(
    jd_profile: Mapping[str, Any],
    pool_ids: list[str],
    candidates_by_id: Mapping[str, Candidate],
) -> list[str]:
    """STEP 2. BM25 over the retrieval pool's role-description text → a lexical
    ranking (best first). Deterministic; ties by candidate_id asc. Degrades to an
    empty ranking (semantic-only fusion) if rank_bm25 is unavailable or the pool /
    query is empty — never crashes."""
    if not pool_ids:
        return []
    query_tokens = _bm25_query_tokens(jd_profile)
    if not query_tokens:
        return []

    # Deterministic corpus order so get_scores() aligns to a stable id list.
    ordered_ids = sorted(pool_ids)
    corpus = [
        _tokenize(_role_description_text(candidates_by_id[cid]))
        for cid in ordered_ids
        if cid in candidates_by_id
    ]
    ordered_ids = [cid for cid in ordered_ids if cid in candidates_by_id]
    if not corpus or all(len(doc) == 0 for doc in corpus):
        return []

    try:
        from rank_bm25 import BM25Okapi
    except ImportError:
        print("[scorer] rank_bm25 unavailable — skipping lexical ranking.")
        return []

    bm25 = BM25Okapi(corpus)
    scores = bm25.get_scores(query_tokens)
    # rank desc by score, ties by candidate_id ascending (deterministic).
    order = sorted(
        range(len(ordered_ids)),
        key=lambda i: (-float(scores[i]), ordered_ids[i]),
    )
    return [ordered_ids[i] for i in order]


# --------------------------------------------------------------------------- #
# Cross-encoder stage.
# --------------------------------------------------------------------------- #
def _cross_encoder_scores(
    jd_text: str,
    shortlist_ids: list[str],
    candidates_by_id: Mapping[str, Candidate],
    rerank_fn: Callable,
    ce_shortlist_size: int,
    ce_enabled: bool,
) -> dict[str, float]:
    """STEP 6. Rerank the top ``ce_shortlist_size`` of the shortlist; return
    {id: sigmoid(logit)} for those scored. Empty dict when CE is disabled or the
    model is absent (FileNotFoundError) — the caller then leaves ce_score=None and
    combine() renormalizes. Decisions #1 (sigmoid) and #3 (graceful degradation).
    """
    if not ce_enabled or not shortlist_ids:
        return {}

    ce_n = min(ce_shortlist_size, len(shortlist_ids))
    head_ids = shortlist_ids[:ce_n]
    texts = [candidate_to_text(candidates_by_id[cid]) for cid in head_ids]

    try:
        # top_k=None: every passed text is scored (we pass only the head subset,
        # so there is no tail sentinel to reason about).
        logits = rerank_fn(jd_text, texts, top_k=None)
    except FileNotFoundError:
        # Model dir absent — the documented "CE unavailable" signal. Substance
        # ranking must stand on its own (cross_encoder.py module docstring).
        print("[scorer] cross-encoder model absent — skipping rerank (ce_used=False).")
        return {}

    return {head_ids[i]: _sigmoid(float(logits[i])) for i in range(len(head_ids))}


# --------------------------------------------------------------------------- #
# Public entry point.
# --------------------------------------------------------------------------- #
def score_candidates(
    *,
    jd_profile: Mapping[str, Any],
    candidate_ids: Sequence,
    faiss_index: Any = None,
    candidates_by_id: Optional[Mapping[str, Candidate]] = None,
    candidates_path: Optional[Any] = None,
    encode_query_fn: Optional[Callable] = None,
    search_fn: Optional[Callable] = None,
    rerank_fn: Optional[Callable] = None,
    combine_fn: Callable = combine,
    weights: Mapping[str, float] = DEFAULT_WEIGHTS,
    ce_enabled: bool = True,
    jd_text: Optional[str] = None,
    semantic_k: int = SEMANTIC_RETRIEVAL_K,
    shortlist_size: int = SHORTLIST_SIZE,
    ce_shortlist_size: int = config.CROSS_ENCODER_SHORTLIST_SIZE,
) -> dict[str, CandidateScore]:
    """Score the pool and return ``{candidate_id: CandidateScore}``.

    Args:
        jd_profile: the loaded ``artifacts/jd_profile.json`` dict (aspects with
            weights summing to 1.0, experience_band, consulting_firms, ...).
        candidate_ids: ids in FAISS row order — row ``i`` of the index is
            ``candidate_ids[i]`` (the join key). May be a numpy str array; ids are
            coerced to ``str`` at every dict/set boundary.
        faiss_index: the loaded FAISS index, passed through to ``search_fn``.
        candidates_by_id: optional pre-built ``{id: Candidate}`` map. If omitted,
            the pool is streamed ONCE from ``candidates_path`` (or
            ``config.CANDIDATES_PATH``) via ``io_utils.stream_candidates``.
        candidates_path: source for streaming when ``candidates_by_id`` is None.
        encode_query_fn / search_fn / rerank_fn: dependency-injection seams
            (default to ``embeddings.encode_texts`` / ``index.search`` /
            ``cross_encoder.rerank``, resolved lazily so importing this module
            pulls in no torch/faiss). Tests inject light fakes.
        combine_fn: the base-relevance combiner (default ``combine``; LTR-swappable).
        weights: composite weights passed to ``combine_fn`` (default priors).
        ce_enabled: budget/availability flag; False skips the cross-encoder.
        jd_text: optional override of the CE JD string (default built from the
            profile).
        semantic_k / shortlist_size / ce_shortlist_size: stage sizes.

    Pure, deterministic, CPU-only, no network at score time.
    """
    # Lazy default resolution — keeps this module importable without faiss/torch.
    if encode_query_fn is None:
        from .embeddings import encode_texts as encode_query_fn  # type: ignore
    if search_fn is None:
        from .index import search as search_fn  # type: ignore
    if rerank_fn is None:
        from .cross_encoder import rerank as rerank_fn  # type: ignore

    # Stream the pool once into an id->Candidate map (memory-safe single pass).
    if candidates_by_id is None:
        from .io_utils import stream_candidates
        path = candidates_path if candidates_path is not None else config.CANDIDATES_PATH
        candidates_by_id = {str(c.candidate_id): c for c in stream_candidates(path)}
    else:
        candidates_by_id = {str(cid): c for cid, c in candidates_by_id.items()}

    if jd_text is None:
        jd_text = _build_jd_text(jd_profile)

    # STEP 1 — semantic retrieval (ranking + per-candidate weighted cosine).
    semantic_ranking, semantic_sim = _semantic_retrieval(
        jd_profile, candidate_ids, faiss_index, encode_query_fn, search_fn, semantic_k
    )

    # The retrieval POOL is everything any aspect surfaced; BM25 reorders within it.
    pool_ids = list(semantic_sim.keys())

    # STEP 2 — lexical retrieval (BM25 over the pool's descriptions).
    lexical_ranking = _lexical_retrieval(jd_profile, pool_ids, candidates_by_id)

    # STEP 3 — fusion → shortlist (best-first), deterministic tie-break.
    fused = reciprocal_rank_fusion(semantic_ranking, lexical_ranking)
    shortlist_ids = sorted(fused.keys(), key=lambda cid: (-fused[cid], cid))
    shortlist_ids = shortlist_ids[:shortlist_size]

    # STEP 6 — cross-encoder over the shortlist head (sigmoid-normalized).
    ce_by_id = _cross_encoder_scores(
        jd_text, shortlist_ids, candidates_by_id, rerank_fn, ce_shortlist_size, ce_enabled
    )

    # STEPS 4,5,7,8,9 — features, honeypot, composite, behavioral, floor.
    results: dict[str, CandidateScore] = {}
    for cid in shortlist_ids:
        cand = candidates_by_id.get(cid)
        if cand is None:
            # An id from retrieval not present in the pool map: skip defensively
            # (cannot score what we cannot read). Should not happen when
            # candidate_ids and the pool come from the same artifacts.
            continue

        feats = structured_features(cand, jd_profile)       # STEP 4 (10 keys)
        hp_flag, hp_reasons = is_honeypot(cand)             # STEP 5 (once)

        ce_score = ce_by_id.get(cid)                        # None if not reranked
        sem_sim = _clamp01(semantic_sim.get(cid, 0.0))

        # STEP 7 — assemble the ordered composite vector, then combine -> base.
        composite_values: dict[str, Optional[float]] = {
            "role_substance": feats["role_substance"],
            "skill_corroboration": feats["skill_corroboration"],
            "experience_band": feats["experience_band"],
            "nlp_ir_signal": feats["nlp_ir_signal"],
            "product_vs_consulting": feats["product_vs_consulting"],
            "production_recency": feats["production_recency"],
            "tenure_stability": feats["tenure_stability"],
            "external_validation": feats["external_validation"],
            "location_fit": feats["location_fit"],
            "ce_score": ce_score,
            "semantic_sim": sem_sim,
        }
        base = combine_fn(composite_values, weights)

        # STEP 8 — behavioral multiplier (bounded; modulates, never dominates).
        beh = behavioral_multiplier(cand)
        final = base * beh

        # STEP 9 — honeypot floor LAST, so behaviour can't lift it off the floor.
        if hp_flag:
            final = HONEYPOT_FLOOR

        results[cid] = CandidateScore(
            candidate_id=cid,
            final_score=final,
            base_score=base,
            behavioral_mult=beh,
            is_honeypot=hp_flag,
            honeypot_reasons=hp_reasons,
            feature_dict=feats,
            ce_score=ce_score,
            ce_used=ce_score is not None,
            semantic_sim=sem_sim,
            rrf_score=fused.get(cid, 0.0),
        )

    return results
